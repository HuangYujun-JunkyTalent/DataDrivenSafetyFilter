from tools.simple_track_generator import trackGenerator, line, arc
from tools.track_functions import demo_track

# from SafetyFilters.Indirect_Nominal_Fitting import IndirectNominalFittingParams, IndirectNominalFittingFilter
from SafetyFilters.DDSF import DDSafetyFilter
from IOData.IODataWith_l import IODataWith_l
from System.ErrorKinematicAcceLATI import LinearizedErrorKinematicAcceModel
from simulators.simulation_settings import SafetyFilterTypes

import math
import numpy as np
import numpy.linalg as npl
from typing import List, Callable, Mapping, Union, Optional, Tuple
from warnings import warn


class SafetyFilterForTrack:
    '''
    Safety Filter defined for a certain track. Make sure the track is closed!
    '''
    _track_generator: trackGenerator # track  generator, contains information of the track
    _safety_filters: List[DDSafetyFilter] # list of safety filters
    _systems: List[LinearizedErrorKinematicAcceModel] # list of systems, each also contains information of the segment!
    _i: int # index of current track segment
    # global trajectory of the vehicle
    # list of tuple (input:[a, delta], state: [x_p, y_p, Psi, v])
    # Used for initial condition of safety filter
    _previous_traj: List[Tuple[np.matrix, np.ndarray]]
    # slack value for determining segment index
    _slack: float

    def __init__(self, 
                 track_fun: Callable[[trackGenerator, float, np.ndarray], trackGenerator],
                 t: float, # track width
                 track_start: np.ndarray, # initial position and orientation of track, [x_0, y_0, Psi_0]
                 density: int, #parameter used for generating points on track, not used here
                 filter_list: List[DDSafetyFilter], # list of safety filters
                #  io_datas: Mapping[float, IODataWith_l], # map from curvature to dataset
                 systems: List[LinearizedErrorKinematicAcceModel], # a list of systems, each also contains information of the segment!
                 slack: float = 1e-1,
                 position_0: Optional[np.ndarray] = None, # initial position of the vehicle, only used for initialize _i [x_p0, y_p0]
                 i: Optional[int] = None, # optionally, just specify the initial segment
                ) -> None:
        self._track_generator = trackGenerator(density, t)
        track_fun(self._track_generator, t, track_start) # generate the track

        self._slack = slack

        self._safety_filters = filter_list
        self._systems = systems

        if i is not None:
            self._i = i
        else:
            self._i = -1
        for idx, segment in enumerate(self._track_generator.chainOfSegments):
            if isinstance(segment, line):
                # self._safety_filters.append(IndirectNominalFittingFilter(
                #     systems[idx], io_datas[0], filter_params[idx]
                #     ))
                if self._i == -1: # still not initialized yet
                    if self.contains(segment, position_0[0:2], t, self._slack): # check if the vehicle is in the segment
                        self._i = idx
            if isinstance(segment, arc):
                # if segment.theta_f > segment.theta_s: # left turn, curvature > 0 in filter
                #     self._safety_filters.append(IndirectNominalFittingFilter(
                #         systems[idx], io_datas[segment.curvature], filter_params[idx]
                #         ))
                # else: # right turn, curvature < 0 in filter
                #     self._safety_filters.append(IndirectNominalFittingFilter(
                #         systems[idx], io_datas[-segment.curvature], filter_params[idx]
                #         ))
                if self._i == -1: # still not initialized yet
                    if self.contains(segment, position_0[0], position_0[1], t): # check if the vehicle is in the segment
                        self._i = idx
        if self._i is None or self._i == -1:
            warn('Vehicle is not in any segment of the track!')
        
        self._previous_traj = []
        self.first_filtering = True

    def add_point(self, u: np.matrix, global_state: np.ndarray) -> None:
        '''
        global trajectory of the vehicle
        list of tuple (input:[a, delta], state: [x_p, y_p, Psi, v])
        Used for initial condition of safety filter
        '''
        self._previous_traj.append((u, global_state))

    def filter(self, u_obj: np.matrix) -> Tuple[np.matrix, str, float]:
        '''
        Return tuple of (filtered control input, status, optimal value)
        use pre-stored global trajectory as initial condition
        '''
        # get current segment index, based on last point of global trajectory
        segment_changed = self.update_index(self._previous_traj[-1][1], self._slack)
        safety_filter = self._safety_filters[self._i]
        if self.first_filtering:
            segment_changed = True
            self.first_filtering = False
        system = self._systems[self._i]

        if len(self._previous_traj) < safety_filter._lag:
            raise ValueError('Not long enough global trajectory for the safety filter!')
        elif len(self._previous_traj) > safety_filter._lag:
            warn('Too long global trajectory for the safety filter!')
        
        # construct initial condition xi_t for safety filter
        us = np.matrix(np.zeros((0,1)))
        ys = np.matrix(np.zeros((0,1)))
        for i in range(1,safety_filter._lag+1):
            us = np.vstack((self._previous_traj[-i][0], us))
            t_lag = (safety_filter._lag+1)*system.kinematic_model.Ts
            # set proper round to get proper initial l value
            a_max = system.a_max
            system.set_kinematic_model_state(self._previous_traj[-i][1], round=-t_lag*(system._v_0+a_max*t_lag))
            error_state_no_v0 = system.state - np.array([0,0,system._v_0,0])
            ys = np.vstack((np.matrix(error_state_no_v0[0:3]).transpose(),
                            ys))
        xi_t = np.vstack((us, ys))
        self._previous_traj = [] # clear the global trajectory
        return safety_filter.filter(xi_t, u_obj, segment_changed)
    
    def update_index(self, global_state: np.ndarray, slack: float) -> bool:
        """Update index of segment based on given global state
        :param global_state: [x_p, y_p, Psi, v]
        :param slack: parameter to make segments "extend beyond beginning"
        return True if segment changed
        """
        self._systems[self._i].set_kinematic_model_state(global_state, round=-2*slack)
        l = self._systems[self._i].state[3] # error state: [e_lat, mu, v, l]
        if l > self._track_generator.chainOfSegments[self._i].length-slack:
            self._i = self._i + 1
            if self._i >= len(self._track_generator.chainOfSegments):
                self._i = 0
            print(f"vehicle moved to segment {self._i}!")
            return True
        elif l < -slack:
            self._i = self._i - 1
            if self._i < 0:
                self._i = len(self._track_generator.chainOfSegments)-1
            print(f"vehicle traveld to previous segment {self._i}!")
            return True
        return False

    @staticmethod
    def contains(segment: Union[line, arc], p:np.ndarray, t: float, slack: float) -> bool:
        '''
        Check if a point p=[x,y] is contained in a segment with width t
        :param slack: parameter to make segments "extend beyond beginning"
        '''
        if isinstance(segment, line):
            unit_local_x = np.array([segment.x1-segment.x0, segment.y1-segment.y0])
            unit_local_x = unit_local_x / np.linalg.norm(unit_local_x)
            unit_local_y = np.array([-unit_local_x[1], unit_local_x[0]])
            x_local = np.inner(p, unit_local_x)
            y_local = np.inner(p, unit_local_y)
            return (x_local >= -slack) and (x_local <= segment.length) and (y_local >= -t) and (y_local <= t)
        if isinstance(segment, arc):
            delta_theta = segment.theta_f - segment.theta_s
            if delta_theta > 0: # left turn, curvature > 0 in filter
                unit_from_center = np.array([np.cos(segment.theta_s), np.sin(segment.theta_s)])
                unit_orthogonal = np.array([-unit_from_center[1], unit_from_center[0]])
                center_to_p = p - np.array([segment.x0, segment.y0])
                r = npl.norm(center_to_p)
                theta = np.arctan2(np.inner(center_to_p, unit_from_center), np.inner(center_to_p, unit_orthogonal))
                return (theta >= -slack/segment.radius) and (theta <= delta_theta) and (r >= segment.radius-t) and (r <= segment.radius+t)
            else: # right turn, curvature < 0 in filter
                unit_from_center = np.array([np.cos(segment.theta_s), np.sin(segment.theta_s)])
                unit_orthogonal = np.array([-unit_from_center[1], unit_from_center[0]])
                center_to_p = p - np.array([segment.x0, segment.y0])
                r = npl.norm(center_to_p)
                theta = np.arctan2(np.inner(center_to_p, unit_from_center), np.inner(center_to_p, unit_orthogonal))
                return (theta <= slack/segment.radius) and (theta >= delta_theta) and (r >= segment.radius-t) and (r <= segment.radius+t)
        raise ValueError("segment must be either line or arc" )
