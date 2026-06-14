import pandas as pd
import pytest

from src.analysis.evaluation import compute_level_recovery, match_factors_bijection
from src.discovery.factor_registry import CandidateFactor, DiscoveredFactor
from src.utils.config import GroundTruthFactor


def _gt(name, levels, factor_type="window"):
    return GroundTruthFactor(
        name=name,
        type=factor_type,
        levels=levels,
        factor_class="discrete",
    )


def _disc(name, levels, values, factor_type="window", factor_class="discrete"):
    candidate = CandidateFactor(
        name=name,
        description="test",
        factor_type=factor_type,
        factor_class=factor_class,
        levels=levels,
        depends_on=[],
    )
    formula_term = name if factor_class == "continuous" else f"C({name})"
    return DiscoveredFactor(
        candidate=candidate,
        column_name=name,
        column_values=pd.Series(values),
        lrt_statistic=0.0,
        lrt_pvalue=1.0,
        lrt_dof=0,
        formula_with=f"correct ~ {formula_term}",
    )


def test_aba_vs_nonaba_recovers_only_aba_level_not_full_factor():
    df = pd.DataFrame({
        "n2_task_inhibition": [
            "aba_return", "cba_nonreturn", "other",
            "aba_return", "other", "cba_nonreturn",
            "aba_return", "other", "cba_nonreturn",
        ]
    })
    discovered_values = df["n2_task_inhibition"].map({
        "aba_return": "aba_like",
        "cba_nonreturn": "not_aba",
        "other": "not_aba",
    })
    discovered = _disc("aba_contrast", ["aba_like", "not_aba"], discovered_values)
    gt = _gt(
        "n2_task_inhibition",
        ["aba_return", "cba_nonreturn", "other"],
    )

    report = match_factors_bijection([gt], [discovered], df, threshold=0.95)

    assert report.recall == pytest.approx(0.0)
    assert report.unmatched_ground_truth == ["n2_task_inhibition"]
    assert report.unmatched_discovered == ["aba_contrast"]

    level_recovery = report.level_recovery
    assert level_recovery.total_count == 3
    assert level_recovery.recovered_count == 1
    assert level_recovery.recall == pytest.approx(1 / 3)

    by_level = {r.ground_truth_level: r for r in level_recovery.level_results}
    assert by_level["aba_return"].recovered
    assert by_level["aba_return"].best_match.discovered_name == "aba_contrast"
    assert by_level["aba_return"].best_match.discovered_subset == ["aba_like"]
    assert by_level["aba_return"].best_match.f1 == pytest.approx(1.0)
    assert not by_level["cba_nonreturn"].recovered
    assert not by_level["other"].recovered


def test_full_relabelled_factor_recovers_all_levels():
    df = pd.DataFrame({
        "n2_task_inhibition": [
            "aba_return", "cba_nonreturn", "other",
            "aba_return", "other", "cba_nonreturn",
        ]
    })
    discovered_values = df["n2_task_inhibition"].map({
        "aba_return": "x",
        "cba_nonreturn": "y",
        "other": "z",
    })
    discovered = _disc("generic_eqpart", ["x", "y", "z"], discovered_values)
    gt = _gt(
        "n2_task_inhibition",
        ["aba_return", "cba_nonreturn", "other"],
    )

    report = match_factors_bijection([gt], [discovered], df, threshold=0.95)

    assert report.recall == pytest.approx(1.0)
    assert report.level_recovery.total_count == 3
    assert report.level_recovery.recovered_count == 3
    assert report.level_recovery.recall == pytest.approx(1.0)


def test_subset_match_handles_discovered_factor_that_oversplits_a_true_level():
    df = pd.DataFrame({"gt_factor": ["A", "A", "B", "B", "A", "B"]})
    discovered = _disc(
        "oversplit_factor",
        ["x", "y", "z"],
        ["x", "y", "z", "z", "x", "z"],
        factor_type="within_trial",
    )
    gt = _gt("gt_factor", ["A", "B"], factor_type="within_trial")

    level_recovery = compute_level_recovery([gt], [discovered], df, threshold=0.95)

    assert level_recovery.total_count == 2
    assert level_recovery.recovered_count == 2
    by_level = {r.ground_truth_level: r for r in level_recovery.level_results}
    assert by_level["A"].best_match.discovered_subset == ["x", "y"]
    assert by_level["A"].best_match.f1 == pytest.approx(1.0)
    assert by_level["B"].best_match.discovered_subset == ["z"]
    assert by_level["B"].best_match.f1 == pytest.approx(1.0)


def test_continuous_ground_truth_factors_do_not_enter_level_recovery():
    df = pd.DataFrame({"difficulty": [0.1, 0.2, 0.8, 0.9]})
    gt = GroundTruthFactor(
        name="difficulty",
        type="within_trial",
        levels=[],
        factor_class="continuous",
    )
    discovered = _disc(
        "found_difficulty",
        [],
        [0.1, 0.2, 0.8, 0.9],
        factor_type="within_trial",
        factor_class="continuous",
    )

    level_recovery = compute_level_recovery([gt], [discovered], df)

    assert level_recovery.total_count == 0
    assert level_recovery.recovered_count == 0
    assert level_recovery.recall == pytest.approx(1.0)
