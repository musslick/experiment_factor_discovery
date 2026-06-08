"""
Converts raw predicate output (a list of values / None from the sandbox
executor) into a pandas Series aligned to the original DataFrame, and
validates the output according to factor class:

  discrete   : validates string level membership and minimum level counts.
  continuous : validates numeric return type; no level-count check.
"""

import math
import numpy as np
import pandas as pd
from typing import List, Tuple


def encode_discrete_factor(
    raw_values: list,
    df_index: pd.Index,
    declared_levels: List[str],
    min_level_count: int = 5,
) -> Tuple[pd.Series, bool, str]:
    """
    Convert raw predicate output to a pandas Series for a discrete factor.

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
        series    : dtype=object Series; None and '' are converted to NaN.
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

    s = pd.Series(raw_values, index=df_index, dtype=object)
    s = s.where(s.notna() & (s != ""), other=np.nan)

    non_nan = s.dropna()
    unexpected = set(non_nan.unique()) - set(declared_levels)
    if unexpected:
        return (
            s,
            False,
            f"Unexpected level values returned by predicate: {unexpected}",
        )

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


def encode_continuous_factor(
    raw_values: list,
    df_index: pd.Index,
) -> Tuple[pd.Series, bool, str]:
    """
    Convert raw predicate output to a float64 pandas Series for a continuous factor.

    Parameters
    ----------
    raw_values : list of numeric values (int/float) or None produced by the sandbox.
                 Length must equal len(df_index).
    df_index   : index of the original trial DataFrame (for alignment).

    Returns
    -------
    (series, is_valid, reason)
        series    : float64 Series with NaN where raw_values had None.
        is_valid  : True when all non-None values are numeric.
        reason    : "ok" on success, or a description of the failure.
    """
    if len(raw_values) != len(df_index):
        return (
            pd.Series(dtype=float),
            False,
            f"Length mismatch: got {len(raw_values)} values, "
            f"expected {len(df_index)}",
        )

    bad = [
        v for v in raw_values
        if v is not None and (
            not isinstance(v, (int, float))
            or (isinstance(v, float) and math.isnan(v))
        )
    ]
    if bad:
        return (
            pd.Series(dtype=float),
            False,
            f"Expected numeric values for continuous factor, got non-numeric: "
            f"{bad[:3]}",
        )

    float_vals = [float(v) if v is not None else float("nan") for v in raw_values]
    s = pd.Series(float_vals, index=df_index, dtype=float)
    return s, True, "ok"


def encode_factor(
    raw_values: list,
    df_index: pd.Index,
    declared_levels: List[str],
    min_level_count: int = 5,
    factor_class: str = "discrete",
) -> Tuple[pd.Series, bool, str]:
    """
    Dispatch to encode_discrete_factor or encode_continuous_factor based on factor_class.

    Parameters
    ----------
    raw_values      : list produced by the sandbox executor.
    df_index        : index of the original trial DataFrame.
    declared_levels : level names (discrete only; ignored for continuous).
    min_level_count : minimum occurrences per level (discrete only).
    factor_class    : "discrete" (default) or "continuous".

    Returns
    -------
    (series, is_valid, reason)
    """
    if factor_class == "continuous":
        return encode_continuous_factor(raw_values, df_index)
    return encode_discrete_factor(raw_values, df_index, declared_levels, min_level_count)
