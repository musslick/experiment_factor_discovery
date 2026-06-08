"""
Generates candidate derived factors by prompting the LLM with the current
state of the discovery pipeline (known factors, already-discovered factors,
previously rejected candidates).
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.discovery.llm_client import LLMClient
from src.discovery.factor_registry import CandidateFactor, DiscoveredFactor

_PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"


def _load(filename: str) -> str:
    return (_PROMPT_DIR / filename).read_text(encoding="utf-8")


def _fill(template: str, **subs: str) -> str:
    """Replace <<key>> placeholders in template."""
    for key, value in subs.items():
        template = template.replace(f"<<{key}>>", value)
    return template


def _format_observable(factor_names: List[str],
                        descriptions: Optional[dict] = None) -> str:
    """
    Format the observable factor list for the LLM prompt.

    Parameters
    ----------
    factor_names  : ordered list of column names.
    descriptions  : mapping name → description string.  When supplied, used
                    instead of the built-in Stroop-specific fallback.
    """
    desc = descriptions or {}
    lines = []
    for name in factor_names:
        if name in desc:
            lines.append(f"  {name}: {desc[name]}")
        else:
            lines.append(f"  {name}")
    return "\n".join(lines) if lines else "  (none)"


def _format_discovered(discovered: List[DiscoveredFactor]) -> str:
    if not discovered:
        return "  (none yet)"
    lines = []
    for d in discovered:
        if d.candidate.factor_class == "continuous":
            lines.append(
                f"  {d.column_name} ({d.candidate.factor_type}, continuous): "
                f"returns float — {d.candidate.description}"
            )
        else:
            levels_str = ", ".join(f'"{lv}"' for lv in d.candidate.levels)
            type_str = d.candidate.factor_type
            if d.candidate.factor_type == "window":
                type_str += f" width={d.candidate.window_width}"
            lines.append(
                f"  {d.column_name} ({type_str}, discrete): "
                f"levels=[{levels_str}] — {d.candidate.description}"
            )
    return "\n".join(lines)


def _format_rejected(rejected: List[CandidateFactor]) -> str:
    if not rejected:
        return "  (none)"
    lines = []
    for r in rejected:
        reason = r.rejection_reason or "unknown"
        lines.append(f"  {r.name} — rejected reason: {reason}")
    return "\n".join(lines)


def _parse_candidates(raw: str, round_num: int) -> List[CandidateFactor]:
    """
    Extract a JSON array from the LLM response.
    Handles both raw JSON and JSON wrapped in markdown code fences.
    """
    # Strip markdown fences if present
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        # Take the first [...] block in the response
        bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
        if bracket_match:
            text = bracket_match.group(0)

    try:
        items = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse candidate list as JSON: {exc}\nRaw: {raw[:400]}") from exc

    candidates = []
    for item in items:
        try:
            candidates.append(
                CandidateFactor(
                    name=str(item["name"]).strip(),
                    description=str(item.get("description", "")).strip(),
                    factor_type=str(item["factor_type"]).strip(),
                    factor_class=str(item.get("factor_class", "discrete")).strip(),
                    window_width=int(item.get("window_width") or 2),
                    levels=[str(lv).strip() for lv in item.get("levels", [])],
                    depends_on=[str(d).strip() for d in item.get("depends_on", [])],
                    round_num=round_num,
                )
            )
        except KeyError as exc:
            # Skip malformed items rather than crashing
            print(f"  [candidate_generator] Skipping malformed candidate (missing {exc}): {item}")
    return candidates


def _format_scored_candidates(scored_candidates: list, top_k: int):
    """
    Format a list of ScoredCandidate objects for the refinement prompt.

    Parameters
    ----------
    scored_candidates : list of objects with .candidate (CandidateFactor) and
                        .cv_score (CVScore) attributes.
    top_k             : how many top candidates to highlight separately.

    Returns
    -------
    (all_str, top_str) — both ready for <<scored_candidates>> / <<top_candidates>>.
    """
    sorted_sc = sorted(
        scored_candidates,
        key=lambda sc: sc.cv_score.mean_ll_improvement,
        reverse=True,
    )

    all_lines = []
    for rank, sc in enumerate(sorted_sc, 1):
        levels_str = ", ".join(f'"{lv}"' for lv in sc.candidate.levels)
        all_lines.append(
            f"  {rank}. {sc.candidate.name} ({sc.candidate.factor_type}, {sc.candidate.factor_class})"
            f" | levels=[{levels_str}]"
            f" | cv_mean={sc.cv_score.mean_ll_improvement:.4f}"
            f" ± se={sc.cv_score.se_ll_improvement:.4f}"
            f"\n     {sc.candidate.description}"
        )

    top_lines = []
    for sc in sorted_sc[:top_k]:
        levels_str = ", ".join(f'"{lv}"' for lv in sc.candidate.levels)
        top_lines.append(
            f"  {sc.candidate.name} ({sc.candidate.factor_type})"
            f"\n     Description : {sc.candidate.description}"
            f"\n     Levels      : [{levels_str}]"
            f"\n     Factor class: {sc.candidate.factor_class}"
            f"\n     Depends on  : {sc.candidate.depends_on}"
            f"\n     CV score    : mean={sc.cv_score.mean_ll_improvement:.4f},"
            f" se={sc.cv_score.se_ll_improvement:.4f}"
        )

    return (
        "\n".join(all_lines) if all_lines else "  (none evaluated yet)",
        "\n".join(top_lines) if top_lines else "  (none)",
    )


def refine_candidates(
    llm: LLMClient,
    scored_candidates: list,
    hard_rejected: List[CandidateFactor],
    top_k: int,
    observable_factors: List[str],
    discovered_so_far: List[DiscoveredFactor],
    round_num: int,
    iteration_num: int,
    n_to_generate: int,
    temperature: float,
    max_tokens: int = 2000,
    max_window_width: int = 5,
    task_context: str = "",
    observable_descriptions: Optional[Dict[str, str]] = None,
) -> List[CandidateFactor]:
    """
    Ask the LLM to propose refined or alternative candidates based on CV scores.

    Only hard_rejected factors are permanently banned; soft-rejected candidates
    from earlier rounds may be refined or reintroduced.
    """
    system = _load("candidate_refinement_system.txt")
    user_template = _load("candidate_refinement_user.txt")

    scored_str, top_str = _format_scored_candidates(scored_candidates, top_k)

    user = _fill(
        user_template,
        task_context=task_context.strip(),
        observable_factors=_format_observable(observable_factors, observable_descriptions),
        discovered_factors=_format_discovered(discovered_so_far),
        hard_rejected_factors=_format_rejected(hard_rejected),
        scored_candidates=scored_str,
        top_candidates=top_str,
        iteration=str(iteration_num),
        round_num=str(round_num),
        n_to_generate=str(n_to_generate),
        max_window_width=str(max_window_width),
    )

    raw = llm.complete(system=system, user=user,
                       max_tokens=max_tokens, temperature=temperature)

    try:
        return _parse_candidates(raw, round_num)
    except ValueError as exc:
        print(f"  [refine_candidates] Parse error: {exc}")
        return []


def generate_candidates(
    llm: LLMClient,
    observable_factors: List[str],
    discovered_so_far: List[DiscoveredFactor],
    rejected_so_far: List[CandidateFactor],
    round_num: int,
    max_candidates: int,
    temperature: float,
    max_tokens: int = 2000,
    max_window_width: int = 5,
    task_context: str = "",
    observable_descriptions: Optional[Dict[str, str]] = None,
) -> List[CandidateFactor]:
    """
    Ask the LLM to propose up to max_candidates new derived factor candidates.

    Returns a list of CandidateFactor objects (may be empty on parse error).
    """
    system = _load("candidate_generation_system.txt")
    user_template = _load("candidate_generation_user.txt")

    user = _fill(
        user_template,
        task_context=task_context.strip(),
        observable_factors=_format_observable(observable_factors, observable_descriptions),
        discovered_factors=_format_discovered(discovered_so_far),
        rejected_factors=_format_rejected(rejected_so_far),
        max_candidates=str(max_candidates),
        max_window_width=str(max_window_width),
    )

    raw = llm.complete(system=system, user=user,
                       max_tokens=max_tokens, temperature=temperature)

    try:
        return _parse_candidates(raw, round_num)
    except ValueError as exc:
        print(f"  [candidate_generator] Parse error: {exc}")
        return []
