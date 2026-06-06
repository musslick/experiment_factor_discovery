"""
Converts raw predicate output (a list of strings / None values from the
sandbox executor) into a pandas Series aligned to the original DataFrame,
and validates that every declared level appears a minimum number of times.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def encode_factor(
    raw_values: list,
    df_index: pd.Index,
    declared_levels: List[str],
    min_level_count: int = 5,
) -> Tuple[pd.Series, bool, str]:
    """
    Convert raw predicate output to a pandas Series aligned to df_index.

    Parameters
    ----------
    raw_values      : list of strings (or None) produced by the sandbox executor.
                      Length must equal len(df_index).
    df_index        : index of the original trial DataFrame (for alignment).
    declared_levels : level names declared in the candidate factor proposal.
    min_level_count : minimum number of occurrences required for each level
                      (guards against perfect separation in logistic regression).

    Returns
    -------
    (series, is_valid, reason)
        series    : dtype=object Series; None and '' from raw_values are
                    converted to NaN (expected at transition-factor block starts).
        is_valid  : True only when all validation checks pass.
        reason    : "ok" on success, or a description of the failure.
    """
    if len(raw_values) != len(df_index):
        return (
            pd.Series(dtype=object),
            False,
            f"Length mismatch: got {len(raw_values)} values, "
            f"expected {len(df_index)}",
        )

    # Build Series; convert None and empty string to NaN
    s = pd.Series(raw_values, index=df_index, dtype=object)
    s = s.where(s.notna() & (s != ""), other=np.nan)

    # Only declared levels (and NaN) are permitted
    non_nan = s.dropna()
    unexpected = set(non_nan.unique()) - set(declared_levels)
    if unexpected:
        return (
            s,
            False,
            f"Unexpected level values returned by predicate: {unexpected}",
        )

    # Every declared level must appear at least min_level_count times
    counts = non_nan.value_counts()
    for level in declared_levels:
        n = int(counts.get(level, 0))
        if n < min_level_count:
            return (
                s,
                False,
                f"Level '{level}' has only {n} occurrence(s) "
                f"(minimum required: {min_level_count})",
            )

    return s, True, "ok"
