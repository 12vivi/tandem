from __future__ import annotations

import dataclasses

import pytest

from conftest import make_engine, prompts
from tandem.config import HybridConfig
from tandem.hybrid.assign import assign_experts
from tandem.model_config import ModelConfig
from tandem.planner.cost_model import ExpertCostModel, stream_crossover_tokens
from tandem.planner.hardware import PRESETS, CPUSpec, HardwareProfile
from tandem.planner.model_presets import MODEL_PRESETS
from tandem.planner.placement import make_plan
from tandem.sampling_params import SamplingParams

HW = PRESETS["rtx4090-desktop"]
SLOW_CPU = dataclasses.replace(HW, cpu=CPUSpec("slow", ["avx2"], 4, 1, 0.01, 0.0001))
SLOW_PCIE = dataclasses.replace(HW, pcie_gbps=0.001)


# ------------------------------------------------------------------ assignment
def test_assignment_prefers_cpu_when_pcie_is_slow():
    cost = ExpertCostModel(SLOW_PCIE, expert_params=3 * 2048 * 768, bytes_per_param=0.5)
    a = assign_experts({1: 1, 2: 1, 3: 1}, resident=set(), cost=cost, max_page_in=8)
    assert a.cpu == [1, 2, 3] and a.page_in == []


def test_assignment_pages_in_high_load_experts_first():
    cost = ExpertCostModel(HW, expert_params=3 * 2048 * 768, bytes_per_param=0.5)
    loads = {e: 1 for e in range(8)}
    loads[5] = 4096  # prefill-sized load: CPU compute-bound, page-in wins
    a = assign_experts(loads, resident={0}, cost=cost, max_page_in=4)
    assert a.resident == [0]
    assert 5 in a.page_in
    assert set(a.cpu) | set(a.page_in) == set(range(1, 8))


def test_assignment_respects_cache_capacity():
    cost = ExpertCostModel(SLOW_CPU, expert_params=1000, bytes_per_param=2)
    a = assign_experts({e: 1 for e in range(6)}, resident=set(), cost=cost, max_page_in=2)
    assert len(a.page_in) == 2 and len(a.cpu) == 4


# ---------------------------------------------------------- hybrid end-to-end
@pytest.mark.parametrize(
    "hw,expect",
    [(SLOW_PCIE, "cpu_experts"), (SLOW_CPU, "page_in_experts")],
    ids=["cpu-path", "page-in-path"],
)
def test_hybrid_engine_matches_hf(tiny_model, tmp_path, hw, expect):
    tm = tiny_model("qwen3_moe")
    profile = tmp_path / "hw.json"
    profile.write_text(hw.to_json(), encoding="utf-8")
    eng = make_engine(tm.path, hybrid=HybridConfig(enabled=True, gpu_expert_slots=2, hw_profile=str(profile)))
    ps = prompts(10, [7, 15])
    outs = eng.generate(ps, SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))
    for p, out in zip(ps, outs):
        assert out.output_token_ids == tm.greedy(p, 8)
    stats = eng.moe_executor.stats()
    assert stats.get(expect, 0) > 0
    assert stats.get("resident_experts", 0) > 0


# --------------------------------------------------------------------- planner
def test_crossover_matches_formula():
    assert stream_crossover_tokens(HW, 0.5625) == pytest.approx(1.0e12 * 0.5625 / (2 * 25e9))


def test_plan_modes():
    qwen3_8b = ModelConfig.from_dict(MODEL_PRESETS["Qwen3-8B"])
    assert make_plan(qwen3_8b, HW, bytes_per_param=2).mode == "gpu"

    moe = ModelConfig.from_dict(MODEL_PRESETS["Qwen3-30B-A3B"])
    plan = make_plan(moe, HW, bytes_per_param=0.5, vram_budget_gb=12)
    assert plan.mode == "hybrid-moe" and 0 < plan.expert_slots_per_layer < 128

    llama = ModelConfig.from_dict(MODEL_PRESETS["Llama-3.3-70B"])
    plan = make_plan(llama, HW, bytes_per_param=0.5)
    assert plan.mode == "hybrid-dense" and 0 < plan.gpu_layers < 80

    cpu_only = HardwareProfile("cpu", CPUSpec("x", ["avx2"], 8, 1, 50, 0.5))
    assert make_plan(qwen3_8b, cpu_only).mode == "cpu"


def test_model_presets_parameter_counts():
    from tandem.planner.placement import footprint

    totals = {name: footprint(ModelConfig.from_dict(cfg)).total / 1e9 for name, cfg in MODEL_PRESETS.items()}
    assert totals["Qwen3-30B-A3B"] == pytest.approx(30.5, abs=0.5)
    assert totals["Qwen3-235B-A22B"] == pytest.approx(235, abs=2)
    assert totals["Qwen3-8B"] == pytest.approx(8.2, abs=0.2)
    assert totals["Llama-3.3-70B"] == pytest.approx(70.6, abs=0.5)
