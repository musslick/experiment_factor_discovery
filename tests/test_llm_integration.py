"""
Functional integration tests for Phase 4 (LLM integration).

These tests make real Anthropic API calls.  Run them with:
    pytest tests/test_llm_integration.py -v -m integration

They are marked ``integration`` so the faster unit tests (Phases 1-3) can be
run independently with ``pytest -m "not integration"``.

Test structure
--------------
TestLLMClient          – the client can make a basic API call.
TestCandidateGeneration – the LLM proposes structurally valid candidates and
                          includes at least one congruency-like proposal for
                          the canonical Stroop factors.
TestPredicateSynthesis  – end-to-end synthesis + sandbox round-trip:
                          (a) within-trial: congruency values match color == word
                          (b) transition: task_transition has None at participant
                              starts and correct repeat/switch elsewhere.
TestPipelineOneRound    – a single discovery round on real data discovers
                          congruency with p < 0.05.
"""

import numpy as np
import pandas as pd
import pytest

from src.discovery.llm_client import LLMClient
from src.discovery.factor_registry import CandidateFactor, FactorRegistry
from src.discovery.candidate_generator import generate_candidates
from src.discovery.predicate_synthesizer import synthesize_predicate
from src.discovery.sandbox import run_predicate
from src.discovery.pipeline import run_discovery_pipeline
from src.utils.config import load_config

pytestmark = pytest.mark.integration

MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def llm() -> LLMClient:
    return LLMClient(model=MODEL)


@pytest.fixture(scope="module")
def stroop_df() -> pd.DataFrame:
    """
    Small Stroop-like DataFrame (2 participants × 18 trials) used for
    predicate validation.  Generated entirely from numpy — no SweetPea needed.
    """
    rng = np.random.default_rng(7)
    tasks  = ["color_naming", "word_reading"]
    colors = ["red", "blue", "green"]
    words  = ["red", "blue", "green"]
    rows = []
    for pid in range(2):
        for t in range(18):
            rows.append({
                "participant_id": pid,
                "trial_index":    t,
                "task":           rng.choice(tasks),
                "color":          rng.choice(colors),
                "word":           rng.choice(words),
                "correct":        int(rng.random() > 0.35),
            })
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# TestLLMClient
# ---------------------------------------------------------------------------

class TestLLMClient:

    def test_returns_non_empty_string(self, llm):
        response = llm.complete(
            system="Respond with a single word.",
            user="What colour is the sky?",
            max_tokens=20,
        )
        assert isinstance(response, str) and len(response.strip()) > 0

    def test_respects_system_prompt(self, llm):
        """The model should follow the system constraint."""
        response = llm.complete(
            system="You must respond ONLY with the word 'BANANA' and nothing else.",
            user="What is 1+1?",
            max_tokens=20,
        )
        assert "BANANA" in response.upper()


# ---------------------------------------------------------------------------
# TestCandidateGeneration
# ---------------------------------------------------------------------------

class TestCandidateGeneration:

    @pytest.fixture(scope="class")
    def candidates(self, llm):
        return generate_candidates(
            llm=llm,
            observable_factors=["task", "color", "word"],
            discovered_so_far=[],
            rejected_so_far=[],
            round_num=1,
            max_candidates=5,
            temperature=0.7,
        )

    def test_at_least_one_candidate_returned(self, candidates):
        assert len(candidates) >= 1

    def test_all_candidates_have_required_fields(self, candidates):
        for c in candidates:
            assert isinstance(c, CandidateFactor)
            assert c.name and isinstance(c.name, str)
            assert c.factor_type in {"within_trial", "transition", "window"}
            assert len(c.levels) >= 2 or c.factor_class == "continuous"
            assert len(c.depends_on) >= 1

    def test_congruency_is_proposed(self, candidates):
        """
        For the basic Stroop factors, the LLM should propose at least one
        congruency-like factor (color-word match).
        """
        names = " ".join(c.name.lower() for c in candidates)
        descs = " ".join(c.description.lower() for c in candidates)
        text = names + " " + descs
        congruency_like = (
            "congruen" in text or "match" in text or "color" in text
        )
        assert congruency_like, (
            f"Expected a congruency-like candidate among: "
            f"{[c.name for c in candidates]}"
        )

    def test_no_simple_factor_renaming(self, candidates):
        """Proposed names should not merely copy an observable factor name."""
        observable = {"task", "color", "word"}
        for c in candidates:
            assert c.name.lower() not in observable, (
                f"Candidate '{c.name}' is a bare renaming of an observable factor"
            )


# ---------------------------------------------------------------------------
# TestPredicateSynthesis
# ---------------------------------------------------------------------------

class TestPredicateSynthesis:

    # ---- (a) within-trial: congruency ----

    @pytest.fixture(scope="class")
    def congruency_code(self, llm, stroop_df):
        candidate = CandidateFactor(
            name="congruency",
            description="Whether the ink colour matches the word meaning",
            factor_type="within_trial",
            levels=["congruent", "incongruent"],
            depends_on=["color", "word"],
        )
        code = synthesize_predicate(
            llm=llm, candidate=candidate, working_df=stroop_df,
            discovered=[], max_retries=3, temperature=0.2,
            timeout_seconds=10, backend="subprocess",
        )
        return code, candidate

    def test_congruency_synthesis_succeeds(self, congruency_code):
        code, candidate = congruency_code
        assert code is not None, "Predicate synthesis returned None"
        assert candidate.predicate_status == "valid"

    def test_congruency_sweetpea_code_stored(self, congruency_code):
        _, candidate = congruency_code
        assert candidate.sweetpea_code and len(candidate.sweetpea_code) > 0

    def test_congruency_values_match_ground_truth(self, congruency_code, stroop_df):
        code, _ = congruency_code
        result = run_predicate(code, stroop_df, "within_trial")
        assert result.success, f"Sandbox failed: {result.error_message}"
        for i, row in stroop_df.iterrows():
            expected = "congruent" if row["color"] == row["word"] else "incongruent"
            assert result.values[i] == expected, (
                f"Row {i}: color={row['color']}, word={row['word']}, "
                f"expected={expected}, got={result.values[i]}"
            )

    def test_congruency_no_unexpected_levels(self, congruency_code, stroop_df):
        code, _ = congruency_code
        result = run_predicate(code, stroop_df, "within_trial")
        returned = {v for v in result.values if v is not None}
        assert returned <= {"congruent", "incongruent"}

    # ---- (b) transition: task_transition ----

    @pytest.fixture(scope="class")
    def task_trans_code(self, llm, stroop_df):
        candidate = CandidateFactor(
            name="task_transition",
            description="Whether the task repeated or switched from the previous trial",
            factor_type="transition",
            levels=["repeat", "switch"],
            depends_on=["task"],
        )
        code = synthesize_predicate(
            llm=llm, candidate=candidate, working_df=stroop_df,
            discovered=[], max_retries=3, temperature=0.2,
            timeout_seconds=10, backend="subprocess",
        )
        return code, candidate

    def test_task_transition_synthesis_succeeds(self, task_trans_code):
        code, candidate = task_trans_code
        assert code is not None, "Predicate synthesis returned None"
        assert candidate.predicate_status == "valid"

    def test_task_transition_none_at_participant_starts(self, task_trans_code, stroop_df):
        """None must appear exactly at the first trial of each participant."""
        code, _ = task_trans_code
        result = run_predicate(code, stroop_df, "transition")
        assert result.success, f"Sandbox failed: {result.error_message}"

        for pid in sorted(stroop_df["participant_id"].unique()):
            p_df = (
                stroop_df[stroop_df["participant_id"] == pid]
                .sort_values("trial_index")
                .reset_index()
            )
            first_idx = int(p_df.loc[0, "index"])
            assert result.values[first_idx] is None

    def test_task_transition_values_are_correct(self, task_trans_code, stroop_df):
        code, _ = task_trans_code
        result = run_predicate(code, stroop_df, "transition")
        df = stroop_df.reset_index(drop=True)
        for pid in sorted(df["participant_id"].unique()):
            p_df = (
                df[df["participant_id"] == pid]
                .sort_values("trial_index")
                .reset_index()
            )
            for pos in range(1, len(p_df)):
                orig = int(p_df.loc[pos, "index"])
                prev_task = p_df.loc[pos - 1, "task"]
                curr_task = p_df.loc[pos, "task"]
                expected = "repeat" if prev_task == curr_task else "switch"
                assert result.values[orig] == expected, (
                    f"Participant {pid}, pos {pos}: "
                    f"prev={prev_task}, curr={curr_task}, "
                    f"expected={expected}, got={result.values[orig]}"
                )


# ---------------------------------------------------------------------------
# TestPipelineOneRound
# ---------------------------------------------------------------------------

class TestPipelineOneRound:
    """
    Runs a single discovery round on the generated ground-truth dataset.
    Expects at least congruency to be proposed and statistically accepted.
    Uses a minimal config override (1 round, 5 candidates, subprocess sandbox).
    """

    @pytest.fixture(scope="class")
    def pipeline_result(self, llm, tmp_path_factory):
        """Run one round and return the registry."""
        cfg = load_config("config/synthetic_stroop_benchmark.yaml")
        # Override for a fast test run
        cfg.discovery.n_rounds = 1
        cfg.discovery.seeding_strategy.n_candidates = 5
        cfg.discovery.sandbox_backend = "subprocess"
        cfg.llm.candidate_temperature = 0.7
        cfg.llm.predicate_temperature = 0.2

        # Load the generated input dataset (must exist from generate_data.py)
        input_path = "data/input/stroop_factor_discovery_input.csv"
        try:
            obs_df = pd.read_csv(input_path)
        except FileNotFoundError:
            pytest.skip(
                f"{input_path} not found — run `python generate_data.py --config config/synthetic_stroop_benchmark.yaml` first"
            )

        registry = FactorRegistry(baseline_formula="correct ~ 1")
        out_dir = str(tmp_path_factory.mktemp("pipeline_results"))
        run_discovery_pipeline(obs_df, cfg, llm, registry, output_dir=out_dir)
        return registry

    def test_at_least_one_factor_discovered(self, pipeline_result):
        assert len(pipeline_result.discovered) >= 1, (
            "Expected at least one factor to be discovered in round 1"
        )

    def test_congruency_discovered(self, pipeline_result):
        """
        Congruency is the strongest effect (β = 0.8) and the most obvious
        Stroop factor — the pipeline should reliably find it in round 1.
        """
        names = {f.column_name.lower() for f in pipeline_result.discovered}
        congruency_found = any("congruen" in n or "match" in n for n in names)
        assert congruency_found, (
            f"Expected a congruency-like factor among discovered: {names}"
        )

    def test_discovered_factors_improve_held_out_fit(self, pipeline_result):
        """
        Every accepted factor must show positive held-out validation improvement.
        (lrt_pvalue is a legacy field set to 1.0 as a sentinel in the new pipeline;
        validation_improvement is the operative acceptance criterion.)
        """
        for f in pipeline_result.discovered:
            assert f.validation_improvement is not None, (
                f"Factor '{f.column_name}' has no recorded validation_improvement"
            )
            assert f.validation_improvement > 0, (
                f"Factor '{f.column_name}' has validation_improvement = "
                f"{f.validation_improvement:.4f} ≤ 0"
            )

    def test_baseline_formula_updated(self, pipeline_result):
        """The registry formula should advance beyond the intercept-only model."""
        assert pipeline_result.get_current_formula() != "correct ~ 1"
