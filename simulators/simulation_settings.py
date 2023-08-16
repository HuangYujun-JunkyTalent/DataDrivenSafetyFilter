from __future__ import annotations

from enum import Enum, auto, unique
from typing import Tuple

@unique
class SafetyFilterTypes(Enum):
    """Different types of safety filters
    """
    INDIRECT_FITTING_TERMINAL = auto()
    INDIRECT_FIX_MU = auto()
    INDIRECT_ZERO = auto()
    INDIRECT_ZERO_V = auto()
    INDIRECT_STOP = auto()
    INDIRECT_FIX_MU_ADD_DATA = auto()
    DIRECT_ZERO_TERMINAL = auto()

    @classmethod
    @property
    def fix_data_types(self) -> Tuple[SafetyFilterTypes]:
        return (
            SafetyFilterTypes.INDIRECT_FITTING_TERMINAL,
            SafetyFilterTypes.INDIRECT_FIX_MU,
            SafetyFilterTypes.INDIRECT_ZERO,
            SafetyFilterTypes.INDIRECT_ZERO_V,
            SafetyFilterTypes.INDIRECT_STOP,
            SafetyFilterTypes.DIRECT_ZERO_TERMINAL,
        )

    @classmethod
    @property
    def add_data_types(self) -> Tuple[SafetyFilterTypes]:
        return (
            SafetyFilterTypes.INDIRECT_FIX_MU_ADD_DATA,
        )


class TrackFilterTypes(Enum):
    """Different types of track filters
    """
    SINGLE_SEGMENT = auto()
    SINGLE_SEGMENT_ADD_DATA = auto()


class SimulationInputRule(Enum):
    SINE_WAVE = 1
    MAX_THROTTLE_SINE_STEER = 2
