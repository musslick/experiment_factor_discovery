"""Regression tests for direct information-leakage guards."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.discovery.candidate_generator import (
    generate_candidates,
    validate_candidate_structure,
)
from src.discovery.factor_registry import CandidateFactor, FactorRegistry
from src.discovery.predicate_synthesizer import _parse_synthesis_response
from src.discovery.sandbox import run_predicate
from src.discovery.strategies.llm_genetic_evolver import _parse_offspring
from src.discovery.within_round_search import _build_context, _build_obs_desc_str


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOVED_ANSWER_EXAMPLES = {
    "candidate_generation_system.txt": [
        "whether the ink colour matches the word meaning",
        "task_transition (width=2)",
        "congruency_2_back (width=3)",
        "recent_outcome:",
    ],
    "candidate_refinement_system.txt": [
        "whether the ink colour matches the word meaning",
        "task_transition (width=2)",
        "congruency_2_back (width=3)",
    ],
    "predicate_synthesis_system.txt": [
        "example 1 — within_trial + discrete: congruency",
        "example 2 — window + discrete, width=2: task_transition",
        "recent_congruency_proportion",
    ],
    "genetic_evolution_system.txt": [
        "example: congruency: colour == word",
        "example: task_transition (width=2)",
    ],
    "effect_ranking_system.txt": [
        "congruency × task_transition",
        "congruency × previous_congruency",
    ],
}


class StubLLM:
    def __init__(self, response: str):
        self.response = response

    def complete(self, **_kwargs):
        return self.response


def _candidate(name: str, depends_on: list[str]) -> CandidateFactor:
    return CandidateFactor(
        name=name,
        description="test",
        factor_type="within_trial",
        levels=["a", "b"],
        depends_on=depends_on,
    )


def test_runtime_prompts_omit_removed_answer_examples():
    for filename, removed_phrases in REMOVED_ANSWER_EXAMPLES.items():
        text = (PROJECT_ROOT / "prompts" / filename).read_text().lower()
        for phrase in removed_phrases:
            assert phrase not in text, f"{phrase!r} leaked by {filename}"


def test_dependency_prompt_rules_target_outcomes_not_response_factors():
    for filename in (
        "candidate_refinement_system.txt",
        "genetic_evolution_system.txt",
    ):
        text = (PROJECT_ROOT / "prompts" / filename).read_text().lower()
        assert "configured outcome variable" in text
        assert "correctness, accuracy, response" not in text

    synthesis = (
        PROJECT_ROOT / "prompts" / "predicate_synthesis_system.txt"
    ).read_text()
    assert "compute_factor may access only keys listed under Depends on." in synthesis
    assert "trial dict contains only the keys listed under Depends on" in synthesis
    assert "full list of available trial dict keys" not in synthesis
    synthesis_user = (
        PROJECT_ROOT / "prompts" / "predicate_synthesis_user.txt"
    ).read_text()
    assert "Only the keys named under Depends on are present" in synthesis_user
    assert "Available keys" not in synthesis_user
    assert "participant_id" not in synthesis_user
    assert "trial_index" not in synthesis_user


def test_prompt_output_contracts_are_consistent():
    for filename in (
        "candidate_generation_system.txt",
        "candidate_refinement_system.txt",
    ):
        text = (PROJECT_ROOT / "prompts" / filename).read_text()
        assert 'Use "window_width": 2 for within_trial factors' in text
        assert "omit it for" not in text

    ranking = (PROJECT_ROOT / "prompts" / "effect_ranking_system.txt").read_text()
    assert '"priority"' not in ranking


def test_synthesis_parser_rejects_wrong_json_shapes_for_retry():
    invalid = (
        "null",
        "[]",
        '{"compute_factor_code": null, "sweetpea_code": ""}',
        '{"compute_factor_code": "def compute_factor(x): return x"}',
    )
    assert all(_parse_synthesis_response(raw) is None for raw in invalid)


def test_candidate_generation_allows_design_and_discovered_dependencies_only():
    response = json.dumps([
        None,
        {"name": "malformed", "factor_type": "within_trial", "levels": None, "depends_on": ["task"]},
        {"name": "design_safe", "factor_type": "within_trial", "levels": ["a", "b"], "depends_on": ["correct_response"]},
        {"name": "discovered_safe", "factor_type": "within_trial", "levels": ["a", "b"], "depends_on": ["derived_context"]},
        {"name": "outcome_leak", "factor_type": "within_trial", "levels": ["a", "b"], "depends_on": ["correct"]},
        {"name": "unknown", "factor_type": "within_trial", "levels": ["a", "b"], "depends_on": ["hidden_factor"]},
        {"name": "empty", "factor_type": "within_trial", "levels": ["a", "b"], "depends_on": []},
    ])
    discovered = SimpleNamespace(
        column_name="derived_context",
        candidate=_candidate("derived_context", ["task"]),
    )
    candidates = generate_candidates(
        llm=StubLLM(response),
        observable_factors=["task", "color", "correct_response"],
        discovered_so_far=[discovered],
        rejected_so_far=[],
        round_num=1,
        max_candidates=7,
        temperature=0.0,
    )
    assert [candidate.name for candidate in candidates] == [
        "design_safe",
        "discovered_safe",
    ]


def test_genetic_offspring_uses_the_same_dependency_allowlist():
    response = json.dumps([
        "malformed",
        {"name": "null_fields", "factor_type": "within_trial", "levels": ["a", "b"], "depends_on": None},
        {"name": "safe", "factor_type": "within_trial", "levels": ["a", "b"], "depends_on": ["task"]},
        {"name": "leak", "factor_type": "within_trial", "levels": ["a", "b"], "depends_on": ["correct"]},
    ])
    offspring = _parse_offspring(response, 1, {"task"})
    assert [candidate.name for candidate in offspring] == ["safe"]


def test_candidate_structure_validator_enforces_search_constraints():
    constraints = {
        "existing_columns": {"task", "correct"},
        "allowed_factor_types": {"within_trial", "window"},
        "allowed_factor_classes": {"discrete", "continuous"},
        "max_window_width": 4,
    }
    valid = _candidate("T1_previous", ["task"])
    valid.window_width = 999  # Irrelevant for within-trial factors.
    assert validate_candidate_structure(valid, **constraints) == (True, None)

    invalid_candidates = [
        (_candidate("Not-Snake", ["task"]), "unsafe_name"),
        (_candidate("task", ["task"]), "name_collision"),
        (_candidate("future_task", ["task"]), "factor_type_not_allowed"),
        (_candidate("ordinal_task", ["task"]), "factor_class_not_allowed"),
        (_candidate("wide_window", ["task"]), "invalid_window_width"),
        (_candidate("one_level", ["task"]), "invalid_levels"),
        (_candidate("scaled_task", ["task"]), "invalid_levels"),
    ]
    invalid_candidates[2][0].factor_type = "future"
    invalid_candidates[3][0].factor_class = "ordinal"
    invalid_candidates[4][0].factor_type = "window"
    invalid_candidates[4][0].window_width = 5
    invalid_candidates[5][0].levels = ["same", " ", "same"]
    invalid_candidates[6][0].factor_class = "continuous"

    for candidate, expected_reason in invalid_candidates:
        is_valid, reason = validate_candidate_structure(candidate, **constraints)
        assert is_valid is False
        assert reason is not None and expected_reason in reason


def test_context_excludes_outcome_but_preserves_correct_response():
    discovery = SimpleNamespace(
        seeding_strategy=SimpleNamespace(n_candidates=4),
        evolution_strategy=SimpleNamespace(n_candidates=2, top_k=1),
        allowed_factor_types=["within_trial", "window"],
        allowed_factor_classes=["discrete"],
        max_window_width=5,
    )
    config = SimpleNamespace(
        discovery=discovery,
        base_factors=[
            SimpleNamespace(name="task", dtype="categorical", levels=["a", "b"]),
            SimpleNamespace(name="correct_response", dtype="categorical", levels=["left", "right"]),
            SimpleNamespace(name="correct", dtype="categorical", levels=["0", "1"]),
        ],
        outcome_variable_defs=[SimpleNamespace(name="correct")],
    )

    context = _build_context(
        registry=FactorRegistry(),
        config=config,
        round_num=1,
        iteration=0,
        scored=[],
        all_scored=[],
        task_context="test",
        observable_descriptions={"correct": "0 | 1"},
    )
    names = [factor["name"] for factor in context.observable_factors]
    assert names == ["task", "correct_response"]


def test_outcome_is_excluded_from_predicate_prompt_descriptions():
    description = _build_obs_desc_str(
        ["participant_id", "task", "correct_response", "correct"],
        {
            "task": "a | b",
            "correct_response": "left | right",
            "correct": "0 | 1",
        },
        excluded_names={"correct"},
    )
    assert "task: a | b" in description
    assert "correct_response: left | right" in description
    assert "  correct: 0 | 1" not in description


def test_sandbox_exposes_only_declared_dependencies():
    df = pd.DataFrame({
        "participant_id": [1, 1],
        "block_index": [0, 0],
        "trial_index": [0, 1],
        "correct_response": ["left", "right"],
        "correct": [1, 0],
    })
    for factor_type, forbidden_key in (
        ("within_trial", "correct"),
        ("within_trial", "participant_id"),
        ("window", "trial_index"),
        ("window", "block_index"),
    ):
        argument = "trial" if factor_type == "within_trial" else "window"
        row = argument if factor_type == "within_trial" else "window[-1]"
        result = run_predicate(
            predicate_code=(
                f"def compute_factor({argument}):\n"
                f"    return 'yes' if {row}[{forbidden_key!r}] else 'no'"
            ),
            df=df,
            factor_type=factor_type,
            depends_on=["correct_response"],
        )
        assert not result.success
        assert result.error_type == "runtime_error"


def test_sandbox_provides_prompt_promised_modules():
    df = pd.DataFrame({
        "participant_id": [1],
        "trial_index": [0],
        "color": ["red"],
    })
    code = """
def compute_factor(trial):
    values = list(itertools.chain([trial["color"]]))
    count = functools.reduce(lambda total, _: total + 1, values, 0)
    matched = re.fullmatch(r".+", values[0]) is not None
    encoded = json.dumps({"count": count})
    counted = collections.Counter(values)
    return "ok" if math.isfinite(count) and matched and encoded and counted else "bad"
"""
    result = run_predicate(code, df, "within_trial", depends_on=["color"])
    assert result.success
    assert result.values == ["ok"]
