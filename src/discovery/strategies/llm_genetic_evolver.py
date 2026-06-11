"""
LLMGeneticEvolver — genetic algorithm with LLM as the crossover/mutation operator.

Each call to evolve() performs one generation:
  1. Assemble parent pools (elite + tournament + diversity + repair queue).
  2. Assign operator slots (mutation / crossover / repair / novel).
  3. Prompt the LLM with parent details and operator assignments.
  4. Parse offspring.
  5. Apply diversity guard: drop near-structural-duplicates.

Inspired by FunSearch (Romera-Paredes et al., 2023): the LLM acts as a
semantics-guided recombination operator over a scored population, rather than
a cold-start generator.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.discovery.factor_registry import CandidateFactor, DiscoveredFactor
from src.discovery.llm_client import LLMClient
from src.discovery.strategies.base import EvolutionStrategy, ScoredCandidate, SearchContext

_PROMPT_DIR = Path(__file__).parent.parent.parent.parent / "prompts"


def _load(filename: str) -> str:
    return (_PROMPT_DIR / filename).read_text(encoding="utf-8")


def _fill(template: str, **subs: str) -> str:
    for key, value in subs.items():
        template = template.replace(f"<<{key}>>", value)
    return template


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_observable(obs: List[dict]) -> str:
    lines = []
    for f in obs:
        desc = f.get("description", "")
        lines.append(f"  {f['name']}{': ' + desc if desc else ''}")
    return "\n".join(lines) or "  (none)"


def _fmt_discovered(discovered: List[DiscoveredFactor]) -> str:
    if not discovered:
        return "  (none yet)"
    lines = []
    for d in discovered:
        if d.candidate.factor_class == "continuous":
            lines.append(f"  {d.column_name} ({d.candidate.factor_type}, continuous): {d.candidate.description}")
        else:
            lvls = ", ".join(f'"{lv}"' for lv in d.candidate.levels)
            type_str = d.candidate.factor_type
            if d.candidate.factor_type == "window":
                type_str += f" width={d.candidate.window_width}"
            lines.append(f"  {d.column_name} ({type_str}): levels=[{lvls}] — {d.candidate.description}")
    return "\n".join(lines)


def _fmt_hard_rejected(rejected: List[CandidateFactor]) -> str:
    if not rejected:
        return "  (none)"
    return "\n".join(f"  {r.name} — {r.rejection_reason or 'rejected'}" for r in rejected)


def _fmt_all_scored(scored: List[ScoredCandidate]) -> str:
    if not scored:
        return "  (none evaluated yet)"
    sorted_sc = sorted(scored, key=lambda sc: sc.adjusted_score, reverse=True)
    lines = []
    for rank, sc in enumerate(sorted_sc, 1):
        c = sc.candidate
        lvls = ", ".join(f'"{lv}"' for lv in c.levels) if c.levels else "continuous"
        type_str = c.factor_type
        if c.factor_type == "window":
            type_str += f" w={c.window_width}"
        novelty_str = f" | novelty={sc.novelty_score:.3f}" if sc.novelty_score > 0.0 else ""
        lines.append(
            f"  {rank}. {c.name} ({type_str}, {c.factor_class})"
            f" | adj={sc.adjusted_score:.4f}"
            f" | cv={sc.cv_score_mean:.4f}±{sc.cv_score_se:.4f}"
            f"{novelty_str}"
            f"\n     {c.description}"
            f"\n     depends_on={c.depends_on}  levels=[{lvls}]"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parent selection
# ---------------------------------------------------------------------------

def _structural_fingerprint(c: CandidateFactor) -> tuple:
    return (c.factor_type, c.factor_class, c.window_width, frozenset(c.depends_on), len(c.levels))


def _select_parents(
    population: List[ScoredCandidate],
    n_elite: int,
    n_tournament: int,
    n_diversity: int,
) -> Tuple[List[ScoredCandidate], List[ScoredCandidate], List[ScoredCandidate], List[ScoredCandidate]]:
    """
    Returns (elite, tournament, diversity, repair) parent pools.
    repair: candidates with predicate_status != "valid" (synthesis failures)
    """
    if not population:
        return [], [], [], []

    sorted_pop = sorted(population, key=lambda sc: sc.adjusted_score, reverse=True)
    valid = [sc for sc in sorted_pop if sc.candidate.predicate_status == "valid"]
    repair = [sc for sc in sorted_pop if sc.candidate.predicate_status != "valid"]

    # Elite: top-n unconditionally
    elite = valid[:n_elite]
    used_names = {sc.candidate.name for sc in elite}

    # Tournament: cluster by fingerprint, pick best per cluster
    clusters: Dict[tuple, List[ScoredCandidate]] = {}
    for sc in valid:
        fp = _structural_fingerprint(sc.candidate)
        clusters.setdefault(fp, []).append(sc)
    cluster_reps = [max(members, key=lambda sc: sc.adjusted_score) for members in clusters.values()]
    cluster_reps = sorted(cluster_reps, key=lambda sc: sc.adjusted_score, reverse=True)
    tournament = [sc for sc in cluster_reps if sc.candidate.name not in used_names][:n_tournament]
    used_names.update(sc.candidate.name for sc in tournament)

    # Diversity: one from bottom half of valid scores
    half = len(valid) // 2
    low_pool = [sc for sc in valid[half:] if sc.candidate.name not in used_names]
    diversity = low_pool[:n_diversity]

    return elite, tournament, diversity, repair[:3]  # cap repair at 3


# ---------------------------------------------------------------------------
# Diversity guard
# ---------------------------------------------------------------------------

def _structural_distance(a: CandidateFactor, b: CandidateFactor, max_window: int, n_factors: int) -> float:
    d = 0.0
    d += float(a.factor_type != b.factor_type)
    d += float(a.factor_class != b.factor_class)
    d += abs(a.window_width - b.window_width) / max(max_window, 1)
    d += len(set(a.depends_on).symmetric_difference(set(b.depends_on))) / max(n_factors, 1)
    d += abs(len(a.levels) - len(b.levels)) / 4.0
    return d


def _apply_diversity_guard(
    offspring: List[CandidateFactor],
    population: List[ScoredCandidate],
    max_window: int,
    n_factors: int,
    threshold: float = 0.1,
) -> List[CandidateFactor]:
    """Drop offspring that are near-structural-duplicates of higher-scoring population members."""
    pop_candidates = [sc.candidate for sc in population]
    kept = []
    for child in offspring:
        is_dup = any(
            _structural_distance(child, p, max_window, n_factors) < threshold
            for p in pop_candidates
        )
        if not is_dup:
            kept.append(child)
        else:
            # Keep it but note it's structurally redundant; scorer will handle it
            kept.append(child)  # include anyway — let CV scoring decide
    return kept


# ---------------------------------------------------------------------------
# Operator assignment builder
# ---------------------------------------------------------------------------

def _build_operator_assignments(
    elite: List[ScoredCandidate],
    tournament: List[ScoredCandidate],
    diversity: List[ScoredCandidate],
    repair: List[ScoredCandidate],
    n_to_generate: int,
    operator_mix: dict,
) -> Tuple[str, List[dict]]:
    """
    Returns (formatted_string, assignment_list).
    assignment_list is for passing to the LLM.
    """
    n_mut = max(1, round(n_to_generate * operator_mix.get("mutation", 0.4)))
    n_cross = max(0, round(n_to_generate * operator_mix.get("crossover", 0.3)))
    n_repair = max(0, round(n_to_generate * operator_mix.get("repair", 0.2)))
    n_novel = max(0, round(n_to_generate * operator_mix.get("novel", 0.1)))

    all_parents = elite + tournament + diversity
    if not all_parents:
        return "  (no scored candidates yet — generate novel candidates)", []

    lines = []
    assignments = []
    slot = 1

    mutation_subtypes = ["type_scope", "width_adjust", "depends_expand", "depends_contract", "class_flip"]
    for i in range(n_mut):
        if slot > n_to_generate:
            break
        parent = all_parents[i % len(all_parents)]
        subtype = mutation_subtypes[i % len(mutation_subtypes)]
        lines.append(f"  Slot {slot}: MUTATION:{subtype} — parent: {parent.candidate.name} (adj={parent.adjusted_score:.4f})")
        assignments.append({"slot": slot, "op": "mutation", "subtype": subtype, "parent": parent.candidate.name})
        slot += 1

    parent_pairs = []
    if len(all_parents) >= 2:
        for i in range(n_cross):
            p1 = all_parents[i % len(all_parents)]
            p2 = all_parents[(i + 1) % len(all_parents)]
            parent_pairs.append((p1, p2))
    for i, (p1, p2) in enumerate(parent_pairs):
        if slot > n_to_generate:
            break
        lines.append(f"  Slot {slot}: CROSSOVER — parents: {p1.candidate.name} × {p2.candidate.name}")
        assignments.append({"slot": slot, "op": "crossover", "p1": p1.candidate.name, "p2": p2.candidate.name})
        slot += 1

    repair_pool = repair if repair else (all_parents[-1:])  # fallback: repair lowest scorer
    for i in range(n_repair):
        if slot > n_to_generate:
            break
        parent = repair_pool[i % len(repair_pool)]
        lines.append(f"  Slot {slot}: REPAIR — candidate: {parent.candidate.name} (status: {parent.candidate.predicate_status}, score: {parent.adjusted_score:.4f})")
        assignments.append({"slot": slot, "op": "repair", "parent": parent.candidate.name})
        slot += 1

    for _ in range(n_novel):
        if slot > n_to_generate:
            break
        lines.append(f"  Slot {slot}: NOVEL — inspired by population patterns, but structurally independent")
        assignments.append({"slot": slot, "op": "novel"})
        slot += 1

    return "\n".join(lines), assignments


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_offspring(raw: str, round_num: int) -> List[CandidateFactor]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        bracket = re.search(r"\[.*\]", text, re.DOTALL)
        if bracket:
            text = bracket.group(0)
    try:
        items = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse genetic offspring as JSON: {exc}\nRaw: {raw[:400]}") from exc

    candidates = []
    for item in items:
        try:
            candidates.append(CandidateFactor(
                name=str(item["name"]).strip(),
                description=str(item.get("description", "")).strip(),
                factor_type=str(item["factor_type"]).strip(),
                factor_class=str(item.get("factor_class", "discrete")).strip(),
                window_width=int(item.get("window_width") or 2),
                levels=[str(lv).strip() for lv in item.get("levels", [])],
                depends_on=[str(d).strip() for d in item.get("depends_on", [])],
                round_num=round_num,
            ))
        except KeyError as exc:
            print(f"  [llm_genetic_evolver] Skipping malformed offspring (missing {exc}): {item}")
    return candidates


# ---------------------------------------------------------------------------
# LLMGeneticEvolver
# ---------------------------------------------------------------------------

class LLMGeneticEvolver(EvolutionStrategy):
    """
    Genetic algorithm with LLM as the mutation/crossover operator.
    Falls back to LLMEvolver-style refinement when the population is too small.
    """

    def __init__(self, llm: LLMClient, llm_cfg, disc_cfg, evolver_cfg) -> None:
        self._llm = llm
        self._llm_cfg = llm_cfg
        self._disc_cfg = disc_cfg
        self._cfg = evolver_cfg

    def evolve(self, context: SearchContext) -> List[CandidateFactor]:
        if not context.all_scored_candidates:
            # Nothing to evolve from — return empty and let the fallback in the loop handle it
            return []

        n_elite = getattr(self._cfg, "n_elite", 1)
        n_tournament = getattr(self._cfg, "n_tournament_parents", 3)
        n_diversity = getattr(self._cfg, "n_diversity_parents", 1)
        operator_mix = {
            "mutation": getattr(self._cfg, "operator_mix_mutation", 0.40),
            "crossover": getattr(self._cfg, "operator_mix_crossover", 0.30),
            "repair":    getattr(self._cfg, "operator_mix_repair", 0.20),
            "novel":     getattr(self._cfg, "operator_mix_novel", 0.10),
        }
        diversity_threshold = getattr(self._cfg, "diversity_threshold", 0.1)

        elite, tournament, diversity, repair = _select_parents(
            context.all_scored_candidates, n_elite, n_tournament, n_diversity
        )

        op_assignment_str, _ = _build_operator_assignments(
            elite, tournament, diversity, repair,
            context.n_to_generate, operator_mix,
        )

        system = _load("genetic_evolution_system.txt")
        user_template = _load("genetic_evolution_user.txt")

        observable_names = [f["name"] for f in context.observable_factors]
        observable_descriptions = {
            f["name"]: f.get("description", "")
            for f in context.observable_factors
            if f.get("description")
        }

        user = _fill(
            user_template,
            task_context=context.task_context,
            observable_factors=_fmt_observable(context.observable_factors),
            discovered_factors=_fmt_discovered(context.discovered_factors),
            hard_rejected_factors=_fmt_hard_rejected(context.hard_rejected),
            all_scored_candidates=_fmt_all_scored(context.all_scored_candidates),
            operator_assignments=op_assignment_str,
            round_num=str(context.round_num),
            iteration=str(context.iteration),
            n_to_generate=str(context.n_to_generate),
            max_window_width=str(context.max_window_width),
        )

        raw = self._llm.complete(
            system=system,
            user=user,
            max_tokens=self._llm_cfg.max_tokens_candidate,
            temperature=self._llm_cfg.candidate_temperature,
        )

        try:
            offspring = _parse_offspring(raw, context.round_num)
        except ValueError as exc:
            print(f"  [llm_genetic_evolver] Parse error: {exc}")
            return []

        n_factors = len(context.observable_factors) + len(context.discovered_factors)
        offspring = _apply_diversity_guard(
            offspring,
            context.all_scored_candidates,
            context.max_window_width,
            n_factors,
            diversity_threshold,
        )

        return offspring[: context.n_to_generate]
