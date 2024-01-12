from __future__ import annotations

from enum import Enum, auto, unique
from typing import Tuple

@unique
class SafetyFilterTypes(Enum):
    """Different types of safety filters
    """
    INDIRECT_FITTING_TERMINAL = 'fitting steady state'
    INDIRECT_FIX_MU = 'fix mu'
    INDIRECT_FIX_MU_WEIGHTING = 'fix mu with weighting'
    INDIRECT_FIX_MU_WEIGHTING_ADD_DATA = 'fix mu with\n weighting and add data'
    INDIRECT_FIX_MU_WEIGHTING_ADD_DATA_WITH_L = 'fix mu with\n weighting and add data\n with l'
    INDIRECT_ZERO = 'all zero'
    INDIRECT_ZERO_V = 'stop at \n center line'
    INDIRECT_ZERO_V_WEIGHTING = 'stop at \n center line \n with weighting'
    INDIRECT_ZERO_V_WEIGHTING_ADD_DATA = 'stop at \n center line \n with weighting \n and add data'
    INDIRECT_STOP = 'stop anywhere \n on the track'
    INDIRECT_FIX_MU_ADD_DATA = auto()
    INDIRECT_FIX_MU_ADD_DATA_LATERAL = 'fix mu and \n add data'
    DIRECT_ZERO_TERMINAL = auto()

    @classmethod
    @property
    def direct_types(self) -> Tuple[SafetyFilterTypes]:
        return (
            SafetyFilterTypes.DIRECT_ZERO_TERMINAL,
        )

    @classmethod
    @property
    def add_data_types(self) -> Tuple[SafetyFilterTypes]:
        return (
            SafetyFilterTypes.INDIRECT_FIX_MU_WEIGHTING_ADD_DATA,
            SafetyFilterTypes.INDIRECT_FIX_MU_ADD_DATA_LATERAL,
            SafetyFilterTypes.INDIRECT_FIX_MU_ADD_DATA,
            SafetyFilterTypes.INDIRECT_ZERO_V_WEIGHTING_ADD_DATA,
        )

    @classmethod
    @property
    def feedin_xi_types(self) -> Tuple[SafetyFilterTypes]:
        return (
            SafetyFilterTypes.INDIRECT_FITTING_TERMINAL,
            SafetyFilterTypes.INDIRECT_FIX_MU,
            SafetyFilterTypes.INDIRECT_FIX_MU_WEIGHTING,
            SafetyFilterTypes.INDIRECT_FIX_MU_WEIGHTING_ADD_DATA,
            SafetyFilterTypes.INDIRECT_FIX_MU_WEIGHTING_ADD_DATA_WITH_L,
            SafetyFilterTypes.INDIRECT_ZERO,
            SafetyFilterTypes.INDIRECT_ZERO_V,
            SafetyFilterTypes.INDIRECT_STOP,
            SafetyFilterTypes.INDIRECT_FIX_MU_ADD_DATA_LATERAL,
            SafetyFilterTypes.DIRECT_ZERO_TERMINAL,
            SafetyFilterTypes.INDIRECT_ZERO_V_WEIGHTING,
            SafetyFilterTypes.INDIRECT_ZERO_V_WEIGHTING_ADD_DATA,
            ControllerTypes.INDIRECT_FIX_MU_MAX_L,
        )

    @classmethod
    @property
    def feedin_list_state_types(self) -> Tuple[SafetyFilterTypes]:
        return (
            SafetyFilterTypes.INDIRECT_FIX_MU_ADD_DATA,
        )
    
    @classmethod
    @property
    def output_hankel_matrix_types(self) -> Tuple[SafetyFilterTypes]:
        return (
            SafetyFilterTypes.INDIRECT_FIX_MU_WEIGHTING,
            SafetyFilterTypes.INDIRECT_FIX_MU_WEIGHTING_ADD_DATA,
            SafetyFilterTypes.INDIRECT_ZERO_V_WEIGHTING,
            SafetyFilterTypes.INDIRECT_ZERO_V_WEIGHTING_ADD_DATA,
        )


class ControllerTypes(Enum):
    INDIRECT_FIX_MU_MAX_L = auto()

class TrackFilterTypes(Enum):
    """Different types of track filters
    """
    SINGLE_SEGMENT = auto()
    SINGLE_SEGMENT_ADD_DATA = auto()


class SimulationInputRule(Enum):
    SINE_WAVE = 1
    MAX_THROTTLE_SINE_STEER = 2
    MAX_THROTTLE = 3
    SINE_WITH_MEAN = 4
    SINE_WITH_MEAN_RANDOM = 5
    RANDOM_WITH_MEAN = 6
    SINE_THROTTLE_RANDOM_STEER = 7
    RANDOM_WITH_MEAN_LARGE_THROTTLE = 8
    OLD_SINE_WITH_MEAN_RANDOM = 9


class ModelType(Enum):
    KINEMATIC = auto()
    DYNAMIC = auto()
    DYNAMIC_FEW_OUTPUT = auto()
    DYNAMIC_WITH_L = auto()

    @classmethod
    @property
    def dynamic_models(self) -> Tuple[ModelType]:
        return (
            ModelType.DYNAMIC,
            ModelType.DYNAMIC_FEW_OUTPUT,
            ModelType.DYNAMIC_WITH_L,
        )
    
    @classmethod
    @property
    def dynamic_with_l_models(self) -> Tuple[ModelType]:
        return (
            ModelType.DYNAMIC_WITH_L,
        )

    @classmethod
    @property
    def kinematic_models(self) -> Tuple[ModelType]:
        return (
            ModelType.KINEMATIC,
        )
    
    @classmethod
    @property
    def few_output_models(self) -> Tuple[ModelType]:
        return (
            ModelType.KINEMATIC,
            ModelType.DYNAMIC_FEW_OUTPUT,
        )
