"""
Factory functions for building seeding and evolution strategies from config.

Usage in pipeline.py:
    from src.discovery.strategies import build_seeding_strategy, build_evolution_strategy
    seeder  = build_seeding_strategy(config, llm)
    evolver = build_evolution_strategy(config, llm)
"""

from typing import List, Optional, Tuple

from src.discovery.strategies.base import EvolutionStrategy, SeedingStrategy


def build_seeding_strategy(config, llm) -> SeedingStrategy:
    """
    Construct a SeedingStrategy from BenchmarkConfig.
    config.discovery.seeding_strategy.type determines the class.
    """
    from src.discovery.strategies.llm_seeder import LLMSeeder
    from src.discovery.strategies.random_seeder import RandomSeeder, RandomLookupSeeder
    from src.discovery.strategies.mixed import MixedSeeder

    disc_cfg = config.discovery
    llm_cfg  = config.llm
    cfg      = disc_cfg.seeding_strategy
    strategy_seed = getattr(config, "seed", None)

    if cfg.type == "llm":
        return LLMSeeder(llm=llm, llm_cfg=llm_cfg, disc_cfg=disc_cfg)

    if cfg.type == "random":
        return RandomSeeder(disc_cfg=disc_cfg, seeder_cfg=cfg, seed=strategy_seed)

    if cfg.type == "random_lookup":
        return RandomLookupSeeder(disc_cfg=disc_cfg, seeder_cfg=cfg, seed=strategy_seed)

    if cfg.type == "mixed":
        components: List[Tuple[SeedingStrategy, int]] = []
        for idx, comp in enumerate(cfg.components):
            n = comp.get("n_candidates", cfg.n_candidates)
            component_seed = None if strategy_seed is None else strategy_seed + idx
            sub_type = comp.get("type", "llm")
            sub_cfg = _make_sub_seeder_cfg(comp)
            if sub_type == "llm":
                components.append((LLMSeeder(llm=llm, llm_cfg=llm_cfg, disc_cfg=disc_cfg), n))
            elif sub_type == "random":
                components.append((RandomSeeder(disc_cfg=disc_cfg, seeder_cfg=sub_cfg, seed=component_seed), n))
            elif sub_type == "random_lookup":
                components.append((RandomLookupSeeder(disc_cfg=disc_cfg, seeder_cfg=sub_cfg, seed=component_seed), n))
            else:
                raise ValueError(f"Unknown mixed seeder component type: {sub_type}")
        return MixedSeeder(components=components)

    raise ValueError(f"Unknown seeding strategy type: {cfg.type!r}")


def build_evolution_strategy(config, llm) -> EvolutionStrategy:
    """
    Construct an EvolutionStrategy from BenchmarkConfig.
    config.discovery.evolution_strategy.type determines the class.
    """
    from src.discovery.strategies.llm_evolver import LLMEvolver
    from src.discovery.strategies.llm_genetic_evolver import LLMGeneticEvolver
    from src.discovery.strategies.mutation_evolver import MutationEvolver
    from src.discovery.strategies.mixed import MixedEvolver

    disc_cfg = config.discovery
    llm_cfg  = config.llm
    cfg      = disc_cfg.evolution_strategy

    if cfg.type == "llm":
        return LLMEvolver(llm=llm, llm_cfg=llm_cfg, disc_cfg=disc_cfg)

    if cfg.type == "llm_genetic":
        return LLMGeneticEvolver(llm=llm, llm_cfg=llm_cfg, disc_cfg=disc_cfg, evolver_cfg=cfg)

    if cfg.type == "mutation":
        return MutationEvolver()

    if cfg.type == "mixed":
        components: List[Tuple[EvolutionStrategy, int]] = []
        for comp in cfg.components:
            n = comp.get("n_candidates", cfg.n_candidates)
            sub_type = comp.get("type", "llm")
            sub_cfg = _make_sub_evolver_cfg(comp)
            if sub_type == "llm":
                components.append((LLMEvolver(llm=llm, llm_cfg=llm_cfg, disc_cfg=disc_cfg), n))
            elif sub_type == "llm_genetic":
                components.append((LLMGeneticEvolver(llm=llm, llm_cfg=llm_cfg, disc_cfg=disc_cfg, evolver_cfg=sub_cfg), n))
            elif sub_type == "mutation":
                components.append((MutationEvolver(), n))
            else:
                raise ValueError(f"Unknown mixed evolver component type: {sub_type}")
        return MixedEvolver(components=components)

    raise ValueError(f"Unknown evolution strategy type: {cfg.type!r}")


# ---------------------------------------------------------------------------
# Sub-config helpers (build lightweight objects from component dicts)
# ---------------------------------------------------------------------------

class _SubCfg:
    """Lightweight attribute container built from a dict."""
    def __init__(self, d: dict) -> None:
        for k, v in d.items():
            setattr(self, k, v)

    def __getattr__(self, name: str):
        return None  # return None for any missing attribute


def _make_sub_seeder_cfg(comp: dict) -> _SubCfg:
    return _SubCfg(comp)


def _make_sub_evolver_cfg(comp: dict) -> _SubCfg:
    # Flatten operator_mix dict into individual attributes for LLMGeneticEvolver
    d = dict(comp)
    if "operator_mix" in d and isinstance(d["operator_mix"], dict):
        om = d.pop("operator_mix")
        d["operator_mix_mutation"] = om.get("mutation", 0.40)
        d["operator_mix_crossover"] = om.get("crossover", 0.30)
        d["operator_mix_repair"]    = om.get("repair", 0.20)
        d["operator_mix_novel"]     = om.get("novel", 0.10)
    return _SubCfg(d)
