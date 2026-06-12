"""
RandomSeeder         — SweetPea-aware template sampling (T1-T11).
RandomLookupSeeder   — template-free categorical random lookup tables.
FactorTemplateLibrary— enumerates all valid template instantiations.

Both seeders set compute_code directly on every CandidateFactor they produce,
bypassing LLM predicate synthesis entirely.  The predicate synthesis fast path
in within_round_search.py skips the synthesiser when compute_code is not None.

Template families
-----------------
T1  equality_match      within_trial  discrete   binary        ≥2 discrete parents with overlapping levels
T2  membership_test     within_trial  discrete   binary        1 discrete parent
T3  repeat_detect       window        discrete   binary        1 discrete parent × window_width
T4  joint_repeat        window        discrete   binary        2 discrete parents, width=2
T5  streak_detect       window        discrete   binary        1 discrete parent × window_width ≥ 3
T6  proportion          window        continuous n/a           1 discrete parent × level × window_width
T7  continuous_transform within_trial continuous n/a           ≥1 continuous parent (constraint)
T8  running_stat        window        continuous n/a           ≥1 continuous parent (constraint)
T9  random_partition    within_trial  discrete   binary        fallback; random lookup table
T10 equality_partition  window        discrete   3-level       sparse width-3 equality-signature partitions
T11 lagged_select      window        continuous n/a           previous trial selected continuous value
"""

import hashlib
import itertools
import random
import textwrap
from typing import List, Optional, Tuple

from src.discovery.factor_registry import CandidateFactor
from src.discovery.strategies.base import SearchContext, SeedingStrategy


def _rng_for_context(seed: Optional[int], context: SearchContext) -> random.Random:
    if seed is None:
        return random.Random()
    derived_seed = int(seed) + 1009 * int(context.round_num) + 9176 * int(context.iteration)
    return random.Random(derived_seed)


# ---------------------------------------------------------------------------
# Code generators (one per template)
# ---------------------------------------------------------------------------

def _gen_t1(f1: str, f2: str) -> str:
    return textwrap.dedent(f"""\
        def compute_factor(trial: dict) -> str:
            return 'match' if trial['{f1}'] == trial['{f2}'] else 'mismatch'
        """)


def _gen_t2(f: str, target_levels: List[str]) -> str:
    lvl_set = repr(set(target_levels))
    return textwrap.dedent(f"""\
        def compute_factor(trial: dict) -> str:
            return 'target' if trial['{f}'] in {lvl_set} else 'non_target'
        """)


def _gen_t3(f: str) -> str:
    return textwrap.dedent(f"""\
        def compute_factor(window: list) -> str:
            return 'repeat' if window[-1]['{f}'] == window[0]['{f}'] else 'switch'
        """)


def _gen_t4(f1: str, f2: str) -> str:
    return textwrap.dedent(f"""\
        def compute_factor(window: list) -> str:
            if window[-1]['{f1}'] == window[0]['{f1}'] and window[-1]['{f2}'] == window[0]['{f2}']:
                return 'both_repeat'
            return 'any_switch'
        """)


def _gen_t5(f: str) -> str:
    return textwrap.dedent(f"""\
        def compute_factor(window: list) -> str:
            vals = [w['{f}'] for w in window]
            return 'streak' if len(set(vals)) == 1 else 'broken'
        """)


_T10_SIGNATURES = ("aaa", "aab", "aba", "abb", "abc")


def _t10_partitions() -> List[Tuple[Tuple[str, ...], ...]]:
    """Sparse 3-way partitions of width-3 equality signatures.

    The grammar singles out two equality signatures and groups all remaining
    signatures into an ``other`` level. Binary repeat/switch atoms are already
    covered by simpler templates, so T10 focuses on higher-order motifs that
    need at least three levels.
    """
    partitions: List[Tuple[Tuple[str, ...], ...]] = []

    for first_idx, first in enumerate(_T10_SIGNATURES):
        for second in _T10_SIGNATURES[first_idx + 1:]:
            other = tuple(sig for sig in _T10_SIGNATURES if sig not in {first, second})
            partitions.append(((first,), (second,), other))

    return partitions


def _gen_t10_equality_partition(f: str, partition: Tuple[Tuple[str, ...], ...]) -> str:
    lookup = {
        signature: f"level_{group_idx + 1}"
        for group_idx, group in enumerate(partition)
        for signature in group
    }
    lookup_repr = repr(lookup)
    return textwrap.dedent(f"""\
        def compute_factor(window: list) -> str:
            two_back = window[0]['{f}']
            previous = window[-2]['{f}']
            current = window[-1]['{f}']
            if two_back == previous == current:
                signature = 'aaa'
            elif two_back == previous:
                signature = 'aab'
            elif two_back == current:
                signature = 'aba'
            elif previous == current:
                signature = 'abb'
            else:
                signature = 'abc'
            return {lookup_repr}[signature]
        """)


def _has_discovered_window_factor(
    discovered_factors: List[object],
    parent_name: str,
    window_width: int,
) -> bool:
    for discovered in discovered_factors:
        candidate = getattr(discovered, "candidate", discovered)
        if candidate.factor_type != "window":
            continue
        if candidate.window_width != window_width:
            continue
        if candidate.depends_on == [parent_name]:
            return True
    return False


def _gen_t6(f: str, level: str) -> str:
    return textwrap.dedent(f"""\
        def compute_factor(window: list) -> float:
            return sum(1 for w in window if w['{f}'] == '{level}') / len(window)
        """)


def _gen_t7_difference(f1: str, f2: str) -> str:
    return textwrap.dedent(f"""\
        def compute_factor(trial: dict) -> float:
            return float(trial['{f1}']) - float(trial['{f2}'])
        """)


def _gen_t7_ratio(f1: str, f2: str) -> str:
    return textwrap.dedent(f"""\
        def compute_factor(trial: dict) -> float:
            denom = float(trial['{f2}'])
            if denom == 0.0:
                return 0.0
            return float(trial['{f1}']) / denom
        """)


def _gen_t7_inversion(f: str) -> str:
    return textwrap.dedent(f"""\
        def compute_factor(trial: dict) -> float:
            return 1.0 - float(trial['{f}'])
        """)


def _gen_t7_conditional_select(discrete_f: str, mapping: dict) -> str:
    mapping_repr = repr(mapping)
    return textwrap.dedent(f"""\
        def compute_factor(trial: dict) -> float:
            _map = {mapping_repr}
            return float(trial[_map[trial['{discrete_f}']]])
        """)


def _gen_t7_conditional_select_inv(discrete_f: str, mapping: dict) -> str:
    mapping_repr = repr(mapping)
    return textwrap.dedent(f"""\
        def compute_factor(trial: dict) -> float:
            _map = {mapping_repr}
            return 1.0 - float(trial[_map[trial['{discrete_f}']]])
        """)


def _gen_t8_lag(f: str) -> str:
    return textwrap.dedent(f"""\
        def compute_factor(window: list) -> float:
            v = window[0]['{f}']
            return float(v) if v is not None else float('nan')
        """)


def _gen_t8_mean(f: str) -> str:
    return textwrap.dedent(f"""\
        def compute_factor(window: list) -> float:
            vals = [float(w['{f}']) for w in window if w['{f}'] is not None]
            return sum(vals) / len(vals) if vals else float('nan')
        """)


def _gen_t8_delta(f: str) -> str:
    return textwrap.dedent(f"""\
        def compute_factor(window: list) -> float:
            return float(window[-1]['{f}']) - float(window[0]['{f}'])
        """)


def _gen_t8_max(f: str) -> str:
    return textwrap.dedent(f"""\
        def compute_factor(window: list) -> float:
            vals = [float(w['{f}']) for w in window if w['{f}'] is not None]
            return max(vals) if vals else float('nan')
        """)


def _gen_t8_min(f: str) -> str:
    return textwrap.dedent(f"""\
        def compute_factor(window: list) -> float:
            vals = [float(w['{f}']) for w in window if w['{f}'] is not None]
            return min(vals) if vals else float('nan')
        """)


def _gen_t11_lagged_conditional_select(discrete_f: str, mapping: dict, invert: bool) -> str:
    mapping_repr = repr(mapping)
    return_expr = "1.0 - float(value)" if invert else "float(value)"
    return textwrap.dedent(f"""\
        def compute_factor(window: list) -> float:
            _map = {mapping_repr}
            previous = window[0]
            value = previous[_map[previous['{discrete_f}']]]
            return {return_expr}
        """)


def _gen_t9_within_trial(depends_on: List[str], lookup: dict) -> str:
    """Generate an if-elif chain for a within-trial random partition."""
    lines = ["def compute_factor(trial: dict) -> str:"]
    keys = sorted(lookup.keys())
    first = True
    for key in keys:
        conditions = " and ".join(
            f"trial['{f}'] == {repr(v)}" for f, v in zip(depends_on, key)
        )
        keyword = "if" if first else "elif"
        lines.append(f"    {keyword} {conditions}:")
        lines.append(f"        return {repr(lookup[key])}")
        first = False
    # default: most common output level
    default_level = max(set(lookup.values()), key=list(lookup.values()).count)
    lines.append(f"    else:")
    lines.append(f"        return {repr(default_level)}")
    return "\n".join(lines) + "\n"


def _gen_t9_window(depends_on: List[str], window_width: int, lookup: dict) -> str:
    """Generate an if-elif chain for a window random partition."""
    # Keys are tuples of (w0_f0_val, w0_f1_val, ..., w1_f0_val, ...) flattened
    lines = ["def compute_factor(window: list) -> str:"]
    keys = sorted(lookup.keys())
    first = True
    for key in keys:
        # key is a tuple of length len(depends_on) * window_width
        # layout: (w[0][f0], w[0][f1], ..., w[1][f0], ..., w[W-1][f0], ...)
        conditions = []
        idx = 0
        for wi in range(window_width):
            for f in depends_on:
                conditions.append(f"window[{wi}]['{f}'] == {repr(key[idx])}")
                idx += 1
        keyword = "if" if first else "elif"
        lines.append(f"    {keyword} {' and '.join(conditions)}:")
        lines.append(f"        return {repr(lookup[key])}")
        first = False
    default_level = max(set(lookup.values()), key=list(lookup.values()).count)
    lines.append(f"    else:")
    lines.append(f"        return {repr(default_level)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# FactorTemplateLibrary
# ---------------------------------------------------------------------------

class FactorTemplateLibrary:
    """
    Enumerates all valid CandidateFactor stubs from templates T1-T11.
    Each stub has compute_code set so no LLM synthesis is needed.
    """

    def enumerate(
        self,
        context: SearchContext,
        rng: Optional[random.Random] = None,
    ) -> List[CandidateFactor]:
        """Return all valid template instantiations for this context."""
        candidates: List[CandidateFactor] = []
        obs = context.observable_factors
        discrete_obs = [f for f in obs if f.get("dtype", "categorical") == "categorical"]
        continuous_obs = [f for f in obs if f.get("dtype") == "continuous"]
        allow_disc = "discrete" in context.allowed_factor_classes
        allow_cont = "continuous" in context.allowed_factor_classes
        allow_wt = "within_trial" in context.allowed_factor_types
        allow_win = "window" in context.allowed_factor_types

        if allow_wt and allow_disc:
            candidates.extend(self._t1(discrete_obs, context.round_num))
            candidates.extend(self._t2(discrete_obs, context.round_num))
        if allow_win and allow_disc:
            candidates.extend(self._t3(discrete_obs, context.max_window_width, context.round_num))
            candidates.extend(self._t4(discrete_obs, context.round_num))
            candidates.extend(self._t5(discrete_obs, context.max_window_width, context.round_num))
            candidates.extend(self._t10(discrete_obs, context.max_window_width, context.round_num, context.discovered_factors))
        if allow_win and allow_cont:
            # T6: proportion of discrete levels — no continuous parent required
            candidates.extend(self._t6(discrete_obs, context.max_window_width, context.round_num))
        if allow_wt and allow_cont and continuous_obs:
            candidates.extend(self._t7(obs, continuous_obs, context.round_num))
        if allow_win and allow_cont and continuous_obs:
            candidates.extend(self._t8(continuous_obs, context.max_window_width, context.round_num))
            candidates.extend(self._t11(obs, continuous_obs, context.round_num))

        return candidates

    def sample_t9(
        self,
        context: SearchContext,
        rng: random.Random,
        n: int,
        max_depends: int = 2,
        max_partition_size: int = 16,
    ) -> List[CandidateFactor]:
        """Generate n random binary partitions as T9 fallback candidates."""
        discrete_obs = [
            f for f in context.observable_factors
            if f.get("dtype", "categorical") == "categorical"
        ]
        if not discrete_obs:
            return []
        banned = {c.name for c in context.hard_rejected}
        existing = {sc.candidate.name for sc in context.all_scored_candidates}
        results = []
        attempts = 0
        while len(results) < n and attempts < n * 20:
            attempts += 1
            k = rng.randint(1, min(max_depends, len(discrete_obs)))
            parents = rng.sample(discrete_obs, k)
            parent_names = [f["name"] for f in parents]
            parent_levels = [f.get("levels", []) for f in parents]
            if any(len(lvls) == 0 for lvls in parent_levels):
                continue
            combos = list(itertools.product(*parent_levels))
            if len(combos) > max_partition_size:
                combos = rng.sample(combos, max_partition_size)
            output_levels = ["level_a", "level_b"]
            lookup = {combo: rng.choice(output_levels) for combo in combos}
            # Ensure both levels appear at least once
            if len(set(lookup.values())) < 2 and len(combos) >= 2:
                k2 = rng.randrange(len(combos))
                key2 = list(lookup.keys())[k2]
                lookup[key2] = output_levels[1] if lookup[key2] == output_levels[0] else output_levels[0]
            lookup_hash = hashlib.md5(repr(sorted(lookup.items())).encode()).hexdigest()[:6]
            name = "rnd_" + "_".join(parent_names) + f"_p{lookup_hash}"
            if name in banned or name in existing:
                continue
            desc = f"Random binary partition of ({', '.join(parent_names)}) level combinations"
            code = _gen_t9_within_trial(parent_names, lookup)
            results.append(CandidateFactor(
                name=name,
                description=desc,
                factor_type="within_trial",
                factor_class="discrete",
                levels=output_levels,
                depends_on=parent_names,
                round_num=context.round_num,
                compute_code=code,
                predicate_status="pending",
            ))
        return results

    # ------------------------------------------------------------------
    # Per-template enumeration helpers
    # ------------------------------------------------------------------

    def _t1(self, discrete_obs: List[dict], round_num: int) -> List[CandidateFactor]:
        out = []
        for f1, f2 in itertools.combinations(discrete_obs, 2):
            if not (set(f1.get("levels", [])) & set(f2.get("levels", []))):
                continue
            name = f"{f1['name']}_{f2['name']}_match"
            out.append(CandidateFactor(
                name=name,
                description=f"Whether {f1['name']} and {f2['name']} share the same level on this trial",
                factor_type="within_trial",
                factor_class="discrete",
                levels=["match", "mismatch"],
                depends_on=[f1["name"], f2["name"]],
                round_num=round_num,
                compute_code=_gen_t1(f1["name"], f2["name"]),
                predicate_status="pending",
            ))
        return out

    def _t2(self, discrete_obs: List[dict], round_num: int) -> List[CandidateFactor]:
        out = []
        for f in discrete_obs:
            levels = f.get("levels", [])
            if len(levels) < 2:
                continue
            # Non-trivial, non-complementary subsets: only generate singletons
            # for L=2, and all singletons for L>2 (complement pairs are skipped)
            seen = set()
            for i, lv in enumerate(levels):
                subset = frozenset([lv])
                complement = frozenset(levels) - subset
                if subset in seen or complement in seen:
                    continue
                seen.add(subset)
                name = f"{f['name']}_is_{lv}"
                out.append(CandidateFactor(
                    name=name,
                    description=f"Whether {f['name']} is '{lv}' on this trial",
                    factor_type="within_trial",
                    factor_class="discrete",
                    levels=["target", "non_target"],
                    depends_on=[f["name"]],
                    round_num=round_num,
                    compute_code=_gen_t2(f["name"], [lv]),
                    predicate_status="pending",
                ))
        return out

    def _t3(self, discrete_obs: List[dict], max_width: int, round_num: int) -> List[CandidateFactor]:
        out = []
        for f in discrete_obs:
            for w in range(2, max_width + 1):
                name = f"{f['name']}_transition_w{w}"
                out.append(CandidateFactor(
                    name=name,
                    description=f"Whether {f['name']} repeated or switched from {w - 1} trial(s) ago",
                    factor_type="window",
                    factor_class="discrete",
                    window_width=w,
                    levels=["repeat", "switch"],
                    depends_on=[f["name"]],
                    round_num=round_num,
                    compute_code=_gen_t3(f["name"]),
                    predicate_status="pending",
                    priority="task" in f["name"].lower() and w == 2,
                ))
        return out

    def _t4(self, discrete_obs: List[dict], round_num: int) -> List[CandidateFactor]:
        out = []
        for f1, f2 in itertools.combinations(discrete_obs, 2):
            name = f"{f1['name']}_{f2['name']}_joint_repeat"
            out.append(CandidateFactor(
                name=name,
                description=f"Whether both {f1['name']} and {f2['name']} repeated from the previous trial",
                factor_type="window",
                factor_class="discrete",
                window_width=2,
                levels=["both_repeat", "any_switch"],
                depends_on=[f1["name"], f2["name"]],
                round_num=round_num,
                compute_code=_gen_t4(f1["name"], f2["name"]),
                predicate_status="pending",
            ))
        return out

    def _t5(self, discrete_obs: List[dict], max_width: int, round_num: int) -> List[CandidateFactor]:
        out = []
        for f in discrete_obs:
            for w in range(3, max_width + 1):
                name = f"{f['name']}_streak_w{w}"
                out.append(CandidateFactor(
                    name=name,
                    description=f"Whether {f['name']} held the same level across the last {w} trials",
                    factor_type="window",
                    factor_class="discrete",
                    window_width=w,
                    levels=["streak", "broken"],
                    depends_on=[f["name"]],
                    round_num=round_num,
                    compute_code=_gen_t5(f["name"]),
                    predicate_status="pending",
                ))
        return out

    def _t10(
        self,
        discrete_obs: List[dict],
        max_width: int,
        round_num: int,
        discovered_factors: List[object],
    ) -> List[CandidateFactor]:
        if max_width < 3:
            return []
        out = []
        for f in discrete_obs:
            if len(f.get("levels", [])) < 3:
                continue
            if not _has_discovered_window_factor(discovered_factors, f["name"], window_width=2):
                continue
            for partition in _t10_partitions():
                partition_key = "|".join(",".join(group) for group in partition)
                partition_hash = hashlib.md5(partition_key.encode()).hexdigest()[:6]
                name = f"{f['name']}_eqpart_w3_{partition_hash}"
                levels = [f"level_{idx + 1}" for idx in range(len(partition))]
                groups_desc = "; ".join(
                    f"level_{idx + 1}={{{','.join(group)}}}"
                    for idx, group in enumerate(partition)
                )
                out.append(CandidateFactor(
                    name=name,
                    description=(
                        f"Generic width-3 equality-pattern partition over {f['name']}: "
                        f"{groups_desc}"
                    ),
                    factor_type="window",
                    factor_class="discrete",
                    window_width=3,
                    levels=levels,
                    depends_on=[f["name"]],
                    round_num=round_num,
                    compute_code=_gen_t10_equality_partition(f["name"], partition),
                    predicate_status="pending",
                    priority="task" in f["name"].lower() and len(partition) == 3,
                ))
        return out

    def _t6(self, discrete_obs: List[dict], max_width: int, round_num: int) -> List[CandidateFactor]:
        out = []
        for f in discrete_obs:
            levels = f.get("levels", [])
            for lv in levels:
                for w in range(2, max_width + 1):
                    name = f"{f['name']}_{lv}_proportion_w{w}"
                    out.append(CandidateFactor(
                        name=name,
                        description=f"Proportion of the last {w} trials where {f['name']} was '{lv}'",
                        factor_type="window",
                        factor_class="continuous",
                        window_width=w,
                        levels=[],
                        depends_on=[f["name"]],
                        round_num=round_num,
                        compute_code=_gen_t6(f["name"], lv),
                        predicate_status="pending",
                    ))
        return out

    def _t7(
        self,
        all_obs: List[dict],
        continuous_obs: List[dict],
        round_num: int,
    ) -> List[CandidateFactor]:
        """Continuous within-trial transforms. Requires ≥1 continuous parent."""
        out = []
        cont_names = [f["name"] for f in continuous_obs]
        discrete_obs = [f for f in all_obs if f.get("dtype", "categorical") == "categorical"]

        # Inversion: 1 - f for each continuous factor
        for f in continuous_obs:
            name = f"{f['name']}_inv"
            out.append(CandidateFactor(
                name=name,
                description=f"Inverted value of {f['name']} (1 - {f['name']})",
                factor_type="within_trial",
                factor_class="continuous",
                levels=[],
                depends_on=[f["name"]],
                round_num=round_num,
                compute_code=_gen_t7_inversion(f["name"]),
                predicate_status="pending",
            ))

        # Difference/ratio: pairs of continuous factors
        for f1, f2 in itertools.combinations(continuous_obs, 2):
            name_diff = f"{f1['name']}_minus_{f2['name']}"
            out.append(CandidateFactor(
                name=name_diff,
                description=f"Difference {f1['name']} - {f2['name']} on this trial",
                factor_type="within_trial",
                factor_class="continuous",
                levels=[],
                depends_on=[f1["name"], f2["name"]],
                round_num=round_num,
                compute_code=_gen_t7_difference(f1["name"], f2["name"]),
                predicate_status="pending",
            ))
            name_ratio = f"{f1['name']}_over_{f2['name']}"
            out.append(CandidateFactor(
                name=name_ratio,
                description=f"Ratio {f1['name']} / {f2['name']} on this trial",
                factor_type="within_trial",
                factor_class="continuous",
                levels=[],
                depends_on=[f1["name"], f2["name"]],
                round_num=round_num,
                compute_code=_gen_t7_ratio(f1["name"], f2["name"]),
                predicate_status="pending",
            ))

        # Conditional select: discrete factor selects which continuous factor to use
        for disc_f in discrete_obs:
            disc_levels = disc_f.get("levels", [])
            if len(disc_levels) == len(continuous_obs):
                # Positional mapping: disc_levels[i] → continuous_obs[i].name
                mapping = {lv: cf["name"] for lv, cf in zip(disc_levels, continuous_obs)}
                name_sel = f"{disc_f['name']}_sel_coherence"
                out.append(CandidateFactor(
                    name=name_sel,
                    description=f"Continuous value selected by {disc_f['name']} (one continuous factor per level)",
                    factor_type="within_trial",
                    factor_class="continuous",
                    levels=[],
                    depends_on=[disc_f["name"]] + cont_names,
                    round_num=round_num,
                    compute_code=_gen_t7_conditional_select(disc_f["name"], mapping),
                    predicate_status="pending",
                ))
                name_sel_inv = f"{disc_f['name']}_sel_difficulty"
                out.append(CandidateFactor(
                    name=name_sel_inv,
                    description=f"Inverted continuous value selected by {disc_f['name']} (difficulty proxy)",
                    factor_type="within_trial",
                    factor_class="continuous",
                    levels=[],
                    depends_on=[disc_f["name"]] + cont_names,
                    round_num=round_num,
                    compute_code=_gen_t7_conditional_select_inv(disc_f["name"], mapping),
                    predicate_status="pending",
                    priority="task" in disc_f["name"].lower(),
                ))
        return out

    def _t8(self, continuous_obs: List[dict], max_width: int, round_num: int) -> List[CandidateFactor]:
        """Running statistics over continuous factors. Requires ≥1 continuous parent."""
        out = []
        subtypes = {
            "lag":   ("lag",   _gen_t8_lag,   "Previous trial's value of"),
            "mean":  ("mean",  _gen_t8_mean,  "Running mean of"),
            "delta": ("delta", _gen_t8_delta, "Change in"),
            "max":   ("max",   _gen_t8_max,   "Running maximum of"),
            "min":   ("min",   _gen_t8_min,   "Running minimum of"),
        }
        for f in continuous_obs:
            for subtype_key, (suffix, gen_fn, desc_prefix) in subtypes.items():
                for w in range(2, max_width + 1):
                    if subtype_key == "lag" and w != 2:
                        continue  # lag only makes sense at width=2
                    name = f"{f['name']}_{suffix}_w{w}"
                    out.append(CandidateFactor(
                        name=name,
                        description=f"{desc_prefix} {f['name']} across the last {w} trials",
                        factor_type="window",
                        factor_class="continuous",
                        window_width=w,
                        levels=[],
                        depends_on=[f["name"]],
                        round_num=round_num,
                        compute_code=gen_fn(f["name"]),
                        predicate_status="pending",
                    ))
        return out

    def _t11(
        self,
        all_obs: List[dict],
        continuous_obs: List[dict],
        round_num: int,
    ) -> List[CandidateFactor]:
        """Lagged conditional continuous selection over a width-2 window."""
        out = []
        cont_names = [f["name"] for f in continuous_obs]
        discrete_obs = [f for f in all_obs if f.get("dtype", "categorical") == "categorical"]

        for disc_f in discrete_obs:
            disc_levels = disc_f.get("levels", [])
            if len(disc_levels) != len(continuous_obs):
                continue

            mapping = {lv: cf["name"] for lv, cf in zip(disc_levels, continuous_obs)}
            name_sel = f"{disc_f['name']}_sel_coherence_lag_w2"
            out.append(CandidateFactor(
                name=name_sel,
                description=f"Previous trial continuous value selected by {disc_f['name']}",
                factor_type="window",
                factor_class="continuous",
                window_width=2,
                levels=[],
                depends_on=[disc_f["name"]] + cont_names,
                round_num=round_num,
                compute_code=_gen_t11_lagged_conditional_select(disc_f["name"], mapping, invert=False),
                predicate_status="pending",
            ))
            name_sel_inv = f"{disc_f['name']}_sel_difficulty_lag_w2"
            out.append(CandidateFactor(
                name=name_sel_inv,
                description=f"Previous trial inverted continuous value selected by {disc_f['name']}",
                factor_type="window",
                factor_class="continuous",
                window_width=2,
                levels=[],
                depends_on=[disc_f["name"]] + cont_names,
                round_num=round_num,
                compute_code=_gen_t11_lagged_conditional_select(disc_f["name"], mapping, invert=True),
                predicate_status="pending",
                priority="task" in disc_f["name"].lower(),
            ))
        return out


# ---------------------------------------------------------------------------
# RandomSeeder
# ---------------------------------------------------------------------------

class RandomSeeder(SeedingStrategy):
    """
    Samples from the FactorTemplateLibrary (T1-T11).
    Falls back to T9 random partitions when the template pool is exhausted.
    """

    def __init__(self, disc_cfg, seeder_cfg, seed: Optional[int] = None) -> None:
        self._disc_cfg = disc_cfg
        self._cfg = seeder_cfg
        self._seed = seed
        self._library = FactorTemplateLibrary()

    def seed(self, context: SearchContext) -> List[CandidateFactor]:
        rng = _rng_for_context(self._seed, context)
        banned = {c.name for c in context.hard_rejected}
        scored_names = {sc.candidate.name for sc in context.all_scored_candidates}

        all_templates = self._library.enumerate(context, rng)

        # Filter already-seen and hard-rejected
        pool = [c for c in all_templates if c.name not in banned and c.name not in scored_names]
        priority = [c for c in pool if c.priority]
        pool = [c for c in pool if c not in priority]

        seed_multiplier = getattr(self._cfg, "seed_multiplier", 1.0)
        n_want = round(context.n_to_generate * seed_multiplier)
        template_bias = getattr(self._cfg, "template_bias", "uniform")
        selected = priority[:n_want]
        n_remaining = max(n_want - len(selected), 0)

        if template_bias == "cognitive":
            # Upweight T1 (equality_match) and T3 (repeat_detect), downweight T9
            def weight(c: CandidateFactor) -> float:
                if "match" in c.name or "transition" in c.name:
                    return 3.0
                return 1.0
            weights = [weight(c) for c in pool]
            total = sum(weights)
            probs = [w / total for w in weights]
            n_sample = min(n_remaining, len(pool))
            if n_sample == 0:
                sampled = []
            else:
                indices = rng.choices(range(len(pool)), weights=probs, k=n_sample)
                seen_idx = set()
                sampled = []
                for i in indices:
                    if i not in seen_idx:
                        seen_idx.add(i)
                        sampled.append(pool[i])
            selected.extend(sampled)
        else:
            # Uniform sampling without replacement
            n_sample = min(n_remaining, len(pool))
            selected.extend(rng.sample(pool, n_sample) if n_sample > 0 else [])

        # Fallback: T9 random partitions if pool is exhausted
        if len(selected) < n_want:
            still_need = n_want - len(selected)
            t9_candidates = self._library.sample_t9(context, rng, still_need)
            selected.extend(t9_candidates[:still_need])

        for c in selected:
            c.round_num = context.round_num

        return selected[:context.n_to_generate]


# ---------------------------------------------------------------------------
# RandomLookupSeeder
# ---------------------------------------------------------------------------

class RandomLookupSeeder(SeedingStrategy):
    """
    Template-free categorical seeding.  Samples random (factor_type, depends_on,
    window_width, n_output_levels) combinations and generates if-elif predicates
    via exhaustive enumeration of input key combinations.

    Discrete factors only.  Window key spaces are capped at max_table_size.
    """

    def __init__(self, disc_cfg, seeder_cfg, seed: Optional[int] = None) -> None:
        self._disc_cfg = disc_cfg
        self._cfg = seeder_cfg
        self._seed = seed

    def seed(self, context: SearchContext) -> List[CandidateFactor]:
        rng = _rng_for_context(self._seed, context)
        max_depends = getattr(self._cfg, "max_depends_on", 2)
        max_output_levels = getattr(self._cfg, "max_output_levels", 2)
        max_table_size = getattr(self._cfg, "max_table_size", 64)
        allow_window = getattr(self._cfg, "allow_window", True)

        discrete_obs = [
            f for f in context.observable_factors
            if f.get("dtype", "categorical") == "categorical"
            and len(f.get("levels", [])) >= 2
        ]
        if not discrete_obs:
            return []

        banned = {c.name for c in context.hard_rejected}
        scored_names = {sc.candidate.name for sc in context.all_scored_candidates}
        output_vocab = ["level_a", "level_b", "level_c", "level_d"]

        candidates: List[CandidateFactor] = []
        attempts = 0
        while len(candidates) < context.n_to_generate and attempts < context.n_to_generate * 30:
            attempts += 1

            # Sample structural parameters
            k = rng.randint(1, min(max_depends, len(discrete_obs)))
            parents = rng.sample(discrete_obs, k)
            parent_names = [f["name"] for f in parents]
            parent_levels = [f["levels"] for f in parents]
            n_out = rng.randint(2, max_output_levels)
            out_levels = output_vocab[:n_out]

            if allow_window and "window" in context.allowed_factor_types and rng.random() < 0.4:
                # Window factor
                w = rng.randint(2, context.max_window_width)
                # Key space: product of parent levels repeated w times
                single_combo_size = 1
                for lvls in parent_levels:
                    single_combo_size *= len(lvls)
                full_size = single_combo_size ** w
                if full_size > max_table_size:
                    # Sample a subset of keys
                    all_single = list(itertools.product(*parent_levels))
                    sampled_positions = [rng.sample(all_single, k=min(single_combo_size, single_combo_size)) for _ in range(w)]
                    # Generate max_table_size random sequences
                    keys = []
                    for _ in range(max_table_size):
                        seq = tuple(rng.choice(all_single) for _ in range(w))
                        flat = tuple(v for step in seq for v in step)
                        keys.append(flat)
                    keys = list(set(keys))  # deduplicate
                else:
                    per_step = list(itertools.product(*parent_levels))
                    keys = []
                    for combo in itertools.product(*([per_step] * w)):
                        flat = tuple(v for step in combo for v in step)
                        keys.append(flat)

                lookup = {key: rng.choice(out_levels) for key in keys}
                # Ensure all output levels appear at least once
                for i, lv in enumerate(out_levels):
                    if lv not in lookup.values() and keys:
                        lookup[keys[i % len(keys)]] = lv

                lookup_hash = hashlib.md5(repr(sorted(lookup.items())).encode()).hexdigest()[:6]
                name = "rlookup_" + "_".join(parent_names) + f"_w{w}_{lookup_hash}"
                if name in banned or name in scored_names:
                    continue
                desc = f"Random lookup table over ({', '.join(parent_names)}) window width {w} → {n_out} levels"
                code = _gen_t9_window(parent_names, w, lookup)
                c = CandidateFactor(
                    name=name,
                    description=desc,
                    factor_type="window",
                    factor_class="discrete",
                    window_width=w,
                    levels=out_levels,
                    depends_on=parent_names,
                    round_num=context.round_num,
                    compute_code=code,
                    predicate_status="pending",
                )
            else:
                # Within-trial factor
                combos = list(itertools.product(*parent_levels))
                if len(combos) > max_table_size:
                    combos = rng.sample(combos, max_table_size)
                lookup = {combo: rng.choice(out_levels) for combo in combos}
                for i, lv in enumerate(out_levels):
                    if lv not in lookup.values() and combos:
                        lookup[combos[i % len(combos)]] = lv

                lookup_hash = hashlib.md5(repr(sorted(lookup.items())).encode()).hexdigest()[:6]
                name = "rlookup_" + "_".join(parent_names) + f"_wt_{lookup_hash}"
                if name in banned or name in scored_names:
                    continue
                desc = f"Random lookup table over ({', '.join(parent_names)}) within trial → {n_out} levels"
                code = _gen_t9_within_trial(parent_names, lookup)
                c = CandidateFactor(
                    name=name,
                    description=desc,
                    factor_type="within_trial",
                    factor_class="discrete",
                    levels=out_levels,
                    depends_on=parent_names,
                    round_num=context.round_num,
                    compute_code=code,
                    predicate_status="pending",
                )

            candidates.append(c)

        return candidates[: context.n_to_generate]
