"""
Factory for benchmark data generators and empirical dataset loader.
"""

from src.data_generation.base import BenchmarkDataGenerator, sample_outcome
from src.data_generation.empirical_loader import load_empirical_data
from src.data_generation.stroop_simon_builder import build_stroop_simon_dataset
from src.data_generation.rdk_builder import build_rdk_dataset
from src.data_generation.prospect_theory_builder import build_prospect_theory_dataset
from src.data_generation.sweetpea_builder import build_stroop_dataset
from src.data_generation.stroop_model import sample_accuracy
from src.utils.config import BenchmarkConfig

from typing import List, Tuple
import pandas as pd


class StroopDataGenerator(BenchmarkDataGenerator):
    @property
    def observable_columns(self) -> List[str]:
        return ["participant_id", "block_index", "trial_index", "task", "color", "word", "correct"]

    def build_dataset(self, n_participants, n_blocks_per_participant, seed):
        from src.data_generation.sweetpea_builder import build_stroop_dataset
        return build_stroop_dataset(n_participants, n_blocks_per_participant, seed)


class StroopSimonDataGenerator(BenchmarkDataGenerator):
    @property
    def observable_columns(self) -> List[str]:
        return [
            "participant_id", "block_index", "trial_index",
            "word", "color", "stimulus_location", "correct_response",
            "correct",
        ]

    def build_dataset(self, n_participants, n_blocks_per_participant, seed):
        return build_stroop_simon_dataset(n_participants, n_blocks_per_participant, seed)


class RDKDataGenerator(BenchmarkDataGenerator):
    @property
    def observable_columns(self) -> List[str]:
        return [
            "participant_id", "block_index", "trial_index",
            "task", "motion", "color", "orientation",
            "motion_coherence", "color_coherence", "orientation_coherence",
            "correct_response",
            "correct",
        ]

    def build_dataset(self, n_participants, n_blocks_per_participant, seed):
        return build_rdk_dataset(n_participants, n_blocks_per_participant, seed)


class ProspectTheoryDataGenerator(BenchmarkDataGenerator):
    @property
    def observable_columns(self) -> List[str]:
        return [
            "participant_id", "trial_index",
            "left_gain", "left_loss", "left_gain_probability",
            "right_gain", "right_loss", "right_gain_probability",
            "chose_left",
        ]

    def build_dataset(self, n_participants, n_blocks_per_participant, seed):
        return build_prospect_theory_dataset(n_participants, n_blocks_per_participant, seed)


_REGISTRY = {
    "stroop":             StroopDataGenerator,
    "stroop_simon":       StroopSimonDataGenerator,
    "rdk_task_switching": RDKDataGenerator,
    "prospect_theory":    ProspectTheoryDataGenerator,
}


def get_data_generator(benchmark_type: str) -> BenchmarkDataGenerator:
    """Return a fresh BenchmarkDataGenerator for the given benchmark type."""
    if benchmark_type not in _REGISTRY:
        raise ValueError(
            f"Unknown benchmark_type '{benchmark_type}'. "
            f"Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[benchmark_type]()
