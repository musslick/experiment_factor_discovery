import pandas as pd
import pytest

from src.discovery.sandbox import run_predicate


pytestmark = pytest.mark.integration


def test_docker_backend_runs_with_configured_image():
    df = pd.DataFrame(
        [
            {"participant_id": 1, "trial_index": 0, "color": "red", "word": "red"},
            {"participant_id": 1, "trial_index": 1, "color": "blue", "word": "red"},
        ]
    )
    code = """
def compute_factor(trial):
    return "congruent" if trial["color"] == trial["word"] else "incongruent"
"""

    result = run_predicate(
        code,
        df,
        "within_trial",
        timeout_seconds=30,
        backend="docker",
        docker_image="python:3.9-slim",
    )

    missing_optional_dependency = (
        "llm-sandbox Docker backend is not installed" in (result.error_message or "")
        or "Docker backend requires" in (result.error_message or "")
    )
    if not result.success and missing_optional_dependency:
        pytest.skip(result.error_message)

    assert result.success, result.error_message
    assert result.values == ["congruent", "incongruent"]
