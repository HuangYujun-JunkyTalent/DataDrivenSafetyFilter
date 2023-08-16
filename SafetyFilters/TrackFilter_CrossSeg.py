from tools.simple_track_generator import trackGenerator, line, arc
from tools.track_functions import demo_track

from SafetyFilters.Indirect_Nominal_Fitting import NominalFewSlackTightenedDDSafetyFilter, NominalFewSlackTightenedSFParams
from IOData.IODataWith_l import IODataWith_l
from System.ErrorKinematicAcceLATI import LinearizedErrorKinematicAcceModel
from .DDSF import DDSafetyFilter, SFParams

import math
import numpy as np
import numpy.linalg as npl
import cvxpy as cp
from typing import List, Callable, Mapping, Union, Optional, Tuple
from warnings import warn
from dataclasses import dataclass

@dataclass
class CrossSegmentFilterParams:
    '''
    Parameters for CrossSegmentFilter, mainly regarding optimisation part
    '''
    L: int
    lag: int # approximation of lag of system
    R: np.matrix
    # lam_alph: float
    lam_sig: float
    # epsilon: float
    c: List[List[float]]
    sf_params: SFParams = SFParams()
    

class CrossSegmentFilter(DDSafetyFilter):
    '''
    Safety Filter defined for a certain track. Make sure the track is closed!
    '''
    _track_generator: trackGenerator # track  generator, contains information of the track
    # CURRENTLY using part of prediction matrix to make predictions, no causal constraints posed
    _prediction_matrices: List[np.matrix] # list of prediction matrices, each for a segment. takes in [u_[-l,-1]; u[0, L-1]; y[-l, -1], 1], returns y[0, L-1]
    _systems: List[LinearizedErrorKinematicAcceModel] # list of systems, each also contains information of the segment!
    _i: int # index of current track segment
    _L_1: int # length of predictions that stay in same segment
    # global trajectory of the vehicle
    # list of tuple (input:[a, delta], state: [x_p, y_p, Psi, v])
    # Used for initial condition of safety filter
    _previous_traj: List[Tuple[np.matrix, np.ndarray]]
    # CURRENTLY using same constraint tightening for all segments and cross-semgent predictions.
    _input_output_constraints: List # constraints that will work for all segments
    _obj: cp.Expression # objective for optimization problem

    # optimization variables
    _y: cp.Variable # output variable
    _sigma: cp.Variable # slack variables for compensating noise
    _lam_sig: float # regulizing term for sigma
    _epsilon: float # noise level
    _R: np.matrix
    _c_sum: List[List[float]] # tightening constants

    _u_s: Mapping[float, np.ndarray] # steasy state input, given 
    _y_s: Mapping[float, np.ndarray] # steasy state output, estimated from data
    

    def __init__(self, 
                 track_fun: Callable[[trackGenerator, float, np.ndarray], trackGenerator],
                 t: float, # track width
                 track_start: np.ndarray, # initial position and orientation of track, [x_0, y_0, Psi_0]
                 density: int, #parameter used for generating points on track, only for visualization
                 io_datas: Mapping[float, IODataWith_l], # map from curvature to dataset
                 systems: List[LinearizedErrorKinematicAcceModel], # a list of systems, each also contains information of the segment!
                 params: CrossSegmentFilterParams, 
                 position_0: Optional[np.ndarray] = None, # initial position of the vehicle, only used for initialize _i [x_p0, y_p0]
                 i: Optional[int] = None, # optionally, just specify the initial segment
                ) -> None:
        self._track_generator = trackGenerator(density, t)
        track_fun(self._track_generator, t, track_start) # generate the track

        self._systems = systems
        self._m, self._n, self._p = systems[0].m, systems[0].n, systems[0].p
        self._L, self._lag = params.L, params.lag

        # initialize segment index
        if i is not None:
            self._i = i
        else:
            self._i = -1
        for idx, segment in enumerate(self._track_generator.chainOfSegments):
            if self._i == -1: # still not initialized yet
                if self.contains(segment, position_0[0], position_0[1], t): # check if the vehicle is in the segment
                    self._i = idx
        if self._i is None or self._i == -1:
            warn('Vehicle is not in any segment of the track!')
        
        # initialize prediction matrices
        self._prediction_matrices = []
        prediction_matrices_dict = {}
        for cur, io_data in io_datas.items():
            width_H = io_data.H_input.shape[1]
            H_uy1_noised: np.matrix = np.vstack( (io_data.H_input, io_data.H_output_noised_part((0, self._lag)), np.ones((1, width_H))) )
            H_uy1_noised_inv = npl.pinv(H_uy1_noised)
            A = io_data.H_output_noised_part((self._lag, self._lag+self._L)) @ H_uy1_noised_inv
            prediction_matrices_dict[cur] = A
        for segment in self._track_generator.chainOfSegments:
            if isinstance(segment, line):
                self._prediction_matrices.append(prediction_matrices_dict[0])
            elif isinstance(segment, arc):
                self._prediction_matrices.append(prediction_matrices_dict[segment.curvature])
        
        # initialize safety filter
        DDSafetyFilter.__init__(self, params.sf_params)
        
        # initialize filter parameters, 
            
        self._previous_traj = []

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
        slack = 1e-1
        self.update_index(self._previous_traj[-1][1], slack)
        safety_filter = self._safety_filters[self._i]
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
            system.set_kinematic_model_state(self._previous_traj[-i][1], -2*slack)
            error_state_no_v0 = system.state - np.array([0,0,system._v_0,0])
            ys = np.vstack((np.matrix(error_state_no_v0[0:3]).transpose(),
                            ys))
        xi_t = np.vstack((us, ys))
        self._previous_traj = [] # clear the global trajectory
        return safety_filter.filter(xi_t, u_obj)
    
    def update_index(self, global_state: np.ndarray, slack: float) -> None:
        """Update index of segment based on given global state
        :param global_state: [x_p, y_p, Psi, v]
        :param slack: parameter to make segments "extend beyond beginning"
        """
        self._systems[self._i].set_kinematic_model_state(global_state, round=-2*slack)
        l = self._systems[self._i].state[3] # error state: [e_lat, mu, v, l]
        if l > self._track_generator.chainOfSegments[self._i].length-slack:
            self._i = self._i + 1
            if self._i >= len(self._track_generator.chainOfSegments):
                self._i = 0
            print(f"vehicle moved to segment {self._i}!")
        elif l < -slack:
            warn("vehicle traveld to previous segment!")
            self._i = self._i - 1
            if self._i < 0:
                self._i = len(self._track_generator.chainOfSegments)-1

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
