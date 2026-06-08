"""
Phase 2 of each discovery round: candidate effect search.

After the LLM-driven factor search (Phase 1) accepts or rejects a new factor,
this module exhaustively enumerates pairwise interaction terms from the current
factor library, scores them via the same CV infrastructure used in Phase 1, and
registers accepted interactions as formula extensions — without synthesizing any
new column.

Key design points
-----------------
* Interaction terms are pure formula extensions: C(f_i):C(f_j) is appended to
  the baseline formula.  Statsmodels/patsy computes the term from existing
  columns at fit time.
* Candidate pairs are evaluated in priority-tier order (Tier 1: pairs involving
  the newly accepted factor; Tier 2: pairs queued by the decomposition check;
  Tier 3: all remaining untested pairs, if a full pass is configured).
* A preliminary scoring pass against the current baseline_formula determines
  evaluation order.  Once an interaction is accepted and working_formula
  advances, subsequent candidates are re-scored against the updated formula
  (greedy forward selection).
* Interaction rejection is formula-state-aware: a pair is only skipped if it
  was tested under the current formula hash.  A model change (new factor or
  new interaction accepted) makes previously rejected pairs eligible again.
* Marginality is enforced: if either factor's main effect is absent from the
  working formula, it is added before the interaction term.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple

import math
import numpy as np
import pandas as pd

from src.analysis.model_comparison import CVScore, evaluate_on_held_out, score_candidate_cv
from src.discovery.factor_registry import (
    DiscoveredEffect,
    FactorRegistry,
    TestedInteraction,
)
from src.discovery.llm_client import LLMClient
from src.utils.config import BenchmarkConfig

_PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class EffectSearchResult:
    round_num: int
    accepted_effects: List[DiscoveredEffect]
    all_tested: List[TestedInteraction]   # full audit trail for this round's Phase 2


# ---------------------------------------------------------------------------
# Formula helpers
# ---------------------------------------------------------------------------

def _formula_hash(formula: str) -> str:
    return hashlib.md5(formula.encode()).hexdigest()[:12]


def _factor_term(name: str, registry) -> str:
    """
    Return the patsy term for a discovered factor:
      - continuous → bare name (treated as a numeric predictor)
      - discrete   → C(name)  (treatment-coded categorical)
    Falls back to C(name) for factors not found in the registry.
    """
    for d in registry.discovered:
        if d.column_name == name:
            if d.candidate.factor_class == "continuous":
                return name
            return f"C({name})"
    return f"C({name})"


def _term_present(term: str, name: str, formula: str) -> bool:
    """
    Check whether a factor's main-effect term is already in the formula.
    For a bare name (continuous), checks as a word boundary match.
    For C(name) (discrete), checks for the exact C(...) token.
    """
    if term.startswith("C("):
        return bool(re.search(r'\bC\(' + re.escape(name) + r'\)', formula))
    return bool(re.search(r'(?<![:\w])' + re.escape(name) + r'\b', formula))


def _build_interaction_formula(
    base_formula: str,
    f_i_name: str,
    f_j_name: str,
    registry,
) -> str:
    """
    Append the interaction term to base_formula, adding missing main effects
    first to preserve the marginality hierarchy.

    Continuous factors use bare names; discrete factors use C().
    """
    f_i_term = _factor_term(f_i_name, registry)
    f_j_term = _factor_term(f_j_name, registry)

    missing = []
    for name, term in [(f_i_name, f_i_term), (f_j_name, f_j_term)]:
        if not _term_present(term, name, base_formula):
            missing.append(term)
            print(
                f"  [effect_search] Warning: main effect {term} absent "
                f"from formula — adding for marginality"
            )

    interaction_term = f"{f_i_term}:{f_j_term}"
    additions = missing + [interaction_term]
    lhs, rhs = base_formula.split("~", 1)
    rhs = rhs.strip()
    if rhs == "1":
        return f"{lhs.strip()} ~ {' + '.join(additions)}"
    return f"{lhs.strip()} ~ {rhs} + {' + '.join(additions)}"


# ---------------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------------

def _get_candidate_pairs(
    registry: FactorRegistry,
    new_factor_name: Optional[str],
    run_full_pass: bool,
) -> List[Tuple[str, str]]:
    """
    Assemble interaction pairs in priority-tier order.

    Tier 1: pairs involving the newly accepted factor (fastest signal).
    Tier 2: pending pairs queued by the decomposition check.
    Tier 3: all remaining untested pairs (only if run_full_pass is True).

    A pair is excluded if it is already accepted or already tested under the
    current formula state.
    """
    discovered_names = [f.column_name for f in registry.discovered]
    if len(discovered_names) < 2:
        return []

    all_pairs = {
        tuple(sorted(pair))
        for pair in combinations(discovered_names, 2)
    }

    accepted = {
        tuple(sorted(e.factor_names))
        for e in registry.discovered_effects
    }

    current_hash = _formula_hash(registry.baseline_formula)
    tested_now = {
        tuple(sorted(t.factor_names))
        for t in registry.tested_interactions
        if t.formula_hash == current_hash
    }

    eligible: set = all_pairs - accepted - tested_now

    # Tier 1
    tier1: List[Tuple[str, str]] = []
    if new_factor_name:
        tier1 = sorted(p for p in eligible if new_factor_name in p)
    tier1_set = set(tier1)

    # Tier 2
    tier2: List[Tuple[str, str]] = []
    for pending in registry.pending_interactions:
        pair = tuple(sorted(pending))
        if pair in eligible and pair not in tier1_set:
            tier2.append(pair)
    tier2_set = set(tier2)

    # Tier 3
    tier3: List[Tuple[str, str]] = []
    if run_full_pass:
        tier3 = sorted(
            p for p in eligible
            if p not in tier1_set and p not in tier2_set
        )

    return tier1 + tier2 + tier3


# ---------------------------------------------------------------------------
# Optional LLM ranking
# ---------------------------------------------------------------------------

def _llm_rank_interactions(
    pairs: List[Tuple[str, str]],
    registry: FactorRegistry,
    llm: LLMClient,
    config: BenchmarkConfig,
) -> Tuple[List[Tuple[str, str]], dict]:
    """
    Ask the LLM to rank candidate interaction pairs by psychological plausibility.

    Returns (reordered_pairs, rationale_map) where rationale_map maps
    (f_i, f_j) → str rationale.  On any failure, returns the original order
    and an empty rationale map.
    """
    rationale_map: dict = {}
    if not pairs:
        return pairs, rationale_map

    try:
        system = (_PROMPT_DIR / "effect_ranking_system.txt").read_text(encoding="utf-8")
        user_template = (_PROMPT_DIR / "effect_ranking_user.txt").read_text(encoding="utf-8")
    except FileNotFoundError:
        return pairs, rationale_map

    pairs_text = "\n".join(
        f"  {i+1}. {p[0]} × {p[1]}"
        for i, p in enumerate(pairs)
    )
    discovered_text = "\n".join(
        f"  {d.column_name} ({d.candidate.factor_type}): {d.candidate.description}"
        for d in registry.discovered
    )
    user = (
        user_template
        .replace("<<candidate_interactions>>", pairs_text)
        .replace("<<discovered_factors>>", discovered_text)
    )

    try:
        raw = llm.complete(
            system=system,
            user=user,
            max_tokens=config.discovery.max_tokens_interaction_ranking,
            temperature=0.3,
        )
    except Exception as exc:
        print(f"  [effect_search] LLM ranking failed ({exc}); using default order")
        return pairs, rationale_map

    # Parse JSON array from response
    try:
        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        else:
            bracket = re.search(r"\[.*\]", text, re.DOTALL)
            if bracket:
                text = bracket.group(0)
        items = json.loads(text)
    except Exception as exc:
        print(f"  [effect_search] Could not parse LLM ranking ({exc}); using default order")
        return pairs, rationale_map

    pair_set = {p: i for i, p in enumerate(pairs)}
    ranked: List[Tuple[str, str]] = []
    seen: set = set()
    for item in items:
        try:
            fs = sorted(item["factors"])
            p: Tuple[str, str] = (fs[0], fs[1])
            if p in pair_set and p not in seen:
                ranked.append(p)
                seen.add(p)
                rationale_map[p] = str(item.get("rationale", ""))
        except (KeyError, IndexError, TypeError):
            continue

    # Append any pairs the LLM omitted (preserve coverage)
    for p in pairs:
        if p not in seen:
            ranked.append(p)

    return ranked, rationale_map


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_effect_search(
    search_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: BenchmarkConfig,
    llm: LLMClient,
    registry: FactorRegistry,
    round_num: int,
    new_factor_name: Optional[str],
) -> EffectSearchResult:
    """
    Run Phase 2 (candidate effect search) for one discovery round.

    Parameters
    ----------
    search_df        : DataFrame with discovered factor columns already present.
    validation_df    : Fixed held-out DataFrame; same discovered factor columns.
    config           : Full benchmark configuration.
    llm              : Configured LLMClient (used for optional LLM ranking).
    registry         : Current FactorRegistry (mutated in place).
    round_num        : 1-based round index.
    new_factor_name  : Column name of the factor accepted in Phase 1, or None.

    Returns
    -------
    EffectSearchResult with accepted effects and full audit trail.
    """
    disc_cfg = config.discovery
    accepted_effects: List[DiscoveredEffect] = []
    all_tested_this_round: List[TestedInteraction] = []

    # Decide whether to run a full Tier 3 pass this round
    run_full = disc_cfg.effect_search_full_pass
    if not run_full and disc_cfg.effect_search_full_pass_interval is not None:
        run_full = (round_num % disc_cfg.effect_search_full_pass_interval == 0)

    candidate_pairs = _get_candidate_pairs(registry, new_factor_name, run_full)

    if not candidate_pairs:
        print("  [Phase 2] No interaction pairs to test this round.")
        return EffectSearchResult(
            round_num=round_num,
            accepted_effects=[],
            all_tested=[],
        )

    print(f"\n  [Phase 2] Testing {len(candidate_pairs)} interaction pair(s)")

    # ------------------------------------------------------------------
    # Optional LLM ranking (preliminary ordering only)
    # ------------------------------------------------------------------
    rationale_map: dict = {}
    if disc_cfg.llm_rank_interactions:
        candidate_pairs, rationale_map = _llm_rank_interactions(
            candidate_pairs, registry, llm, config
        )

    # ------------------------------------------------------------------
    # Preliminary scoring against baseline_formula (for ordering only)
    # ------------------------------------------------------------------
    baseline_formula = registry.baseline_formula
    baseline_hash = _formula_hash(baseline_formula)

    @dataclass
    class _Prelim:
        pair: Tuple[str, str]
        prelim_score: float

    prelim_scores: List[_Prelim] = []
    for pair in candidate_pairs:
        f_i, f_j = pair
        formula_alt = _build_interaction_formula(baseline_formula, f_i, f_j, registry)
        try:
            cv = score_candidate_cv(
                df=search_df,
                formula_null=baseline_formula,
                formula_alt=formula_alt,
                participant_col="participant_id",
                n_folds=disc_cfg.cv_n_folds,
                random_state=config.seed,
            )
            prelim_scores.append(_Prelim(pair=pair, prelim_score=cv.mean_ll_improvement))
        except Exception as exc:
            print(f"    [Phase 2] Preliminary score failed for {f_i}×{f_j}: {exc}")
            prelim_scores.append(_Prelim(pair=pair, prelim_score=-np.inf))

    prelim_scores.sort(key=lambda x: x.prelim_score, reverse=True)

    # ------------------------------------------------------------------
    # Greedy forward selection
    # ------------------------------------------------------------------
    working_formula = baseline_formula

    for ps in prelim_scores:
        if len(accepted_effects) >= disc_cfg.max_interactions_per_round:
            break

        f_i, f_j = ps.pair
        formula_null = working_formula
        formula_alt = _build_interaction_formula(working_formula, f_i, f_j, registry)

        if formula_alt == working_formula:
            # Interaction already present (from a prior accepted effect this round)
            continue

        working_hash = _formula_hash(working_formula)
        term = f"{_factor_term(f_i, registry)}:{_factor_term(f_j, registry)}"

        print(
            f"    → {f_i} × {f_j}  (prelim={ps.prelim_score:.4f})",
            end=" ",
            flush=True,
        )

        try:
            cv = score_candidate_cv(
                df=search_df,
                formula_null=formula_null,
                formula_alt=formula_alt,
                participant_col="participant_id",
                n_folds=disc_cfg.cv_n_folds,
                random_state=config.seed,
            )
        except Exception as exc:
            print(f"✗ cv_error: {exc}")
            registry.record_interaction_test(
                factor_names=[f_i, f_j],
                term=term,
                formula_hash=working_hash,
                formula_snapshot=working_formula,
                cv_score_mean=None,
                cv_score_se=None,
                outcome="skipped",
                round_num=round_num,
            )
            all_tested_this_round.append(registry.tested_interactions[-1])
            continue

        print(
            f"cv_mean={cv.mean_ll_improvement:.4f} "
            f"±se={cv.se_ll_improvement:.4f} "
            f"(n={cv.n_participants})",
            end=" ",
            flush=True,
        )

        if not (math.isfinite(cv.mean_ll_improvement)
                and cv.mean_ll_improvement >= disc_cfg.effect_search_min_cv_improvement):
            print("→ below CV threshold")
            registry.record_interaction_test(
                factor_names=[f_i, f_j],
                term=term,
                formula_hash=working_hash,
                formula_snapshot=working_formula,
                cv_score_mean=cv.mean_ll_improvement,
                cv_score_se=cv.se_ll_improvement,
                outcome="below_cv_threshold",
                round_num=round_num,
            )
            all_tested_this_round.append(registry.tested_interactions[-1])
            continue

        # Held-out validation
        improvement = evaluate_on_held_out(
            df_train=search_df,
            df_test=validation_df,
            formula_null=formula_null,
            formula_alt=formula_alt,
        )

        accepted = (
            math.isfinite(improvement)
            and improvement >= disc_cfg.effect_search_min_validation_improvement
        )
        status = "ACCEPTED" if accepted else (
            f"REJECTED (improvement={improvement:.4f} "
            f"< threshold={disc_cfg.effect_search_min_validation_improvement})"
        )
        print(f"→ val={improvement:.4f} {status}")

        outcome = "accepted" if accepted else "failed_validation"
        registry.record_interaction_test(
            factor_names=[f_i, f_j],
            term=term,
            formula_hash=working_hash,
            formula_snapshot=working_formula,
            cv_score_mean=cv.mean_ll_improvement,
            cv_score_se=cv.se_ll_improvement,
            outcome=outcome,
            round_num=round_num,
        )
        all_tested_this_round.append(registry.tested_interactions[-1])

        if accepted:
            # Advance working_formula so subsequent candidates are scored conditionally
            working_formula = formula_alt
            effect = DiscoveredEffect(
                term=term,
                factor_names=sorted([f_i, f_j]),
                effect_type="interaction",
                effect_order=2,
                cv_score_mean=cv.mean_ll_improvement,
                cv_score_se=cv.se_ll_improvement,
                n_participants=cv.n_participants,
                validation_improvement=improvement,
                round_num=round_num,
                formula_with=working_formula,
                source="effect_search",
                llm_rationale=rationale_map.get(ps.pair),
            )
            registry.register_effect(effect)
            accepted_effects.append(effect)
            print(
                f"  ✓ Interaction '{term}' added to model. "
                f"New formula: {registry.get_current_formula()}"
            )

    return EffectSearchResult(
        round_num=round_num,
        accepted_effects=accepted_effects,
        all_tested=all_tested_this_round,
    )
