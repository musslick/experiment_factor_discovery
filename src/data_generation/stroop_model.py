"""
Thin shim kept for backward compatibility: delegates to base.sample_outcome.

The Stroop ground-truth model is now expressed via the generic LogisticModelConfig
(intercept + terms list) so that run_benchmark.py and tests can use the shared
sampler without task-specific field names.
"""

import pandas as pd

from src.data_generation.base import sample_outcome
from src.utils.config import LogisticModelConfig


def sample_accuracy(
    df: pd.DataFrame,
    model: LogisticModelConfig,
    seed: int,
) -> pd.DataFrame:
    """
    Add a binary 'correct' column by sampling from the logistic model.
    Delegates to base.sample_outcome.
    """
    return sample_outcome(df, model, outcome_var="correct", seed=seed)
