"""
Automated level-contrast generation for discrete candidate factors.

The contrast search is intentionally narrow: for an already-valid multi-level
candidate, generate one binary level-vs-rest candidate per declared level.  The
within-round search scores these derived candidates against the same baseline
as the parent, allowing a parsimonious active contrast to beat the full
categorical factor before winner selection.
"""

import re
import textwrap
from typing import List, Optional, Tuple

import pandas as pd

from src.discovery.factor_registry import CandidateFactor


def _safe_name_fragment(value: object) -> str:
    fragment = re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip().lower())
    fragment = re.sub(r"_+", "_", fragment).strip("_")
    if not fragment:
        fragment = "level"
    if fragment[0].isdigit():
        fragment = f"level_{fragment}"
    return fragment


def _rename_parent_compute_function(parent_code: str) -> Optional[str]:
    renamed, count = re.subn(
        r"^(\s*)def\s+compute_factor\s*\(",
        r"\1def _parent_compute_factor(",
        parent_code,
        count=1,
        flags=re.MULTILINE,
    )
    return renamed if count == 1 else None


def _make_contrast_compute_code(
    parent_code: str,
    target_level: str,
    positive_label: str,
    negative_label: str,
) -> Optional[str]:
    parent = _rename_parent_compute_function(parent_code)
    if parent is None:
        return None

    wrapper = f"""

def compute_factor(x):
    value = _parent_compute_factor(x)
    if value is None:
        return None
    return {positive_label!r} if value == {target_level!r} else {negative_label!r}
"""
    return parent.rstrip() + "\n" + textwrap.dedent(wrapper)


def _contrast_series(
    parent_series: pd.Series,
    target_level: str,
    positive_label: str,
    negative_label: str,
) -> pd.Series:
    def map_value(value):
        if pd.isna(value):
            return pd.NA
        return positive_label if value == target_level else negative_label

    return parent_series.map(map_value)


def generate_level_vs_rest_contrasts(
    parent: CandidateFactor,
    parent_series: pd.Series,
    max_contrasts: Optional[int] = None,
) -> List[Tuple[CandidateFactor, pd.Series]]:
    """
    Return binary contrast candidates and aligned series for a parent factor.

    Only discrete candidates with at least three declared levels are eligible.
    The parent must have compute_code so accepted contrasts can be recomputed on
    validation/full datasets through the normal sandbox path.
    """
    if parent.factor_class != "discrete":
        return []
    if len(parent.levels) < 3:
        return []
    if not parent.compute_code:
        return []

    levels = [str(level) for level in parent.levels]
    if max_contrasts is not None:
        levels = levels[:max(0, int(max_contrasts))]

    contrasts: List[Tuple[CandidateFactor, pd.Series]] = []
    used_fragments: set = set()
    for idx, target_level in enumerate(levels, 1):
        fragment = _safe_name_fragment(target_level)
        if fragment in used_fragments:
            fragment = f"{fragment}_{idx}"
        used_fragments.add(fragment)

        positive_label = f"is_{fragment}"
        negative_label = f"not_{fragment}"
        compute_code = _make_contrast_compute_code(
            parent.compute_code,
            target_level,
            positive_label,
            negative_label,
        )
        if compute_code is None:
            continue

        name = f"{parent.name}__{fragment}_vs_rest"
        contrast = CandidateFactor(
            name=name,
            description=(
                f"Binary contrast isolating level '{target_level}' from "
                f"all other levels of {parent.name}."
            ),
            factor_type=parent.factor_type,
            factor_class="discrete",
            window_width=parent.window_width,
            window_stride=parent.window_stride,
            levels=[positive_label, negative_label],
            depends_on=list(parent.depends_on),
            round_num=parent.round_num,
            compute_code=compute_code,
            sweetpea_code=None,
            predicate_status="valid",
            contrast_of=parent.name,
            contrast_positive_levels=[target_level],
        )
        series = _contrast_series(parent_series, target_level, positive_label, negative_label)
        contrasts.append((contrast, series))

    return contrasts
