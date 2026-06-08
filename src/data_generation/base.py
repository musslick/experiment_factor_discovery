"""
Abstract base class for benchmark data generators and a generic logistic
outcome sampler shared by all concrete generators.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.utils.config import BenchmarkConfig, LogisticModelConfig


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sample_outcome(
    df: pd.DataFrame,
    model: LogisticModelConfig,
    outcome_var: str,
    seed: int,
) -> pd.DataFrame:
    """
    Add a binary outcome column to df by sampling from a logistic model.

    For each ModelTerm:
      - If term.level is None  → continuous predictor; values are z-scored
        (mean 0, sd 1) across all non-NaN rows before multiplying by coefficient.
      - If term.level is a str → discrete indicator: 1 where df[factor]==level.

    NaN factor values contribute 0 to the linear predictor.

    Parameters
    ----------
    df          : DataFrame that already contains all hidden factor columns.
    model       : LogisticModelConfig with intercept and terms.
    outcome_var : Name of the binary column to add (e.g. "correct", "chose_left").
    seed        : RNG seed for reproducibility.

    Returns
    -------
    df with an additional binary column named outcome_var.
    """
    rng   = np.random.default_rng(seed)
    df    = df.copy()
    logit = np.full(len(df), model.intercept, dtype=float)

    for term in model.terms:
        if term.factor not in df.columns:
            continue
        col = df[term.factor]

        if term.level is None:
            # Continuous predictor — z-score before applying coefficient
            vals = pd.to_numeric(col, errors="coerce").fillna(np.nan)
            mu   = vals.mean()
            sd   = vals.std()
            if sd == 0 or np.isnan(sd):
                continue
            z = ((vals - mu) / sd).fillna(0.0).values
            logit += term.coefficient * z
        else:
            # Discrete indicator
            indicator = (col == term.level).astype(float).fillna(0.0).values
            logit += term.coefficient * indicator

    p_outcome = _sigmoid(logit)
    df[outcome_var] = rng.binomial(1, p_outcome).astype(int)
    return df


class BenchmarkDataGenerator(ABC):
    """
    Interface that every benchmark data generator must satisfy.

    Concrete subclasses implement build_dataset() to produce a DataFrame that
    contains all columns — both the observable base factors AND the hidden
    derived factors — but does NOT yet include the outcome column.

    The generate() convenience method adds the outcome via sample_outcome()
    and strips hidden columns to produce the discovery-pipeline input.
    """

    @abstractmethod
    def build_dataset(
        self,
        n_participants: int,
        n_blocks_per_participant: int,
        seed: int,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with participant_id, trial_index, all base factor
        columns, and all hidden derived factor columns.  No outcome column yet.
        """

    @property
    @abstractmethod
    def observable_columns(self) -> List[str]:
        """
        Ordered list of columns to expose to the discovery pipeline, including
        participant_id, trial_index, base factors, and the outcome variable.
        Hidden derived factor columns are NOT listed here.
        """

    def generate(self, cfg: BenchmarkConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build the full dataset, add the outcome, and return (full_df, input_df).

        full_df  — all columns including hidden factors and outcome.
        input_df — observable_columns only (what the pipeline sees).
        """
        dg    = cfg.data_generation
        df    = self.build_dataset(dg.n_participants, dg.n_blocks_per_participant, cfg.seed)
        df    = sample_outcome(df, dg.logistic_model, dg.outcome_variable, cfg.seed)
        obs   = [c for c in self.observable_columns if c in df.columns]
        input_df = df[obs].copy()
        return df, input_df
