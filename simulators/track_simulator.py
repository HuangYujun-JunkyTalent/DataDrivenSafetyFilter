from typing import List, Dict, Any, Tuple, Union
import math
import os
import pickle
from enum import Enum
from copy import deepcopy
from itertools import product
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from IOData.IOData import IOData, InputRule
from IOData.IODataWith_l import IODataWith_l
from System.ErrorKinematicAcceLATI import LinearizedErrorKinematicAcceModel, LinearizedErrorKinematicAcceModelParams, KinematicAcceModelParams, KinematicAcceModel
from System.DynamicModel import DynamicModel, DynamicModelParams, DynamicModelFewOutput
from System.DynamicErrorModel import DynamicErrorModelFewOutput, DynamicErrorModelParams
from tools.simualtion_results import Results
from tools.simple_track_generator import trackGenerator, arc, line

from SafetyFilters.DDSF import SFParams
from SafetyFilters.Indirect_Nominal_Fitting import IndirectNominalFittingFilter, IndirectNominalFittingParams
from SafetyFilters.Indirect_Nominal_FixMu import IndirectNominalFixMuFilter, IndirectNominalFixMuParams
from SafetyFilters.Indirect_Nominal_FixMu_Weighting import IndirectNominalFixMuWeightingFilter, IndirectNominalFixMuWeightingParams
from SafetyFilters.Indirect_Nominal_FixMu_Weighting_Add_Data import IndirectNominalFixMuWeightingAddDataFilter, IndirectNominalFixMuWeightingAddDataParams
from SafetyFilters.Indirect_Nominal_Stop import IndirectNominalStopFilter, IndirectNominalStopParams
from SafetyFilters.Indirect_Nominal_Zero import IndirectNominalZeroFilter, IndirectNominalZeroParams
from SafetyFilters.Indirect_Nominal_ZeroV import IndirectNominalZeroVFilter, IndirectNominalZeroVParams
from SafetyFilters.Indirect_Nominal_FixMu_AddData import IndirectNominalFixMuAddDataFilter, IndirectNominalFixMuAddDataParams
from SafetyFilters.Indirect_Nominal_FixMu_AddDataLateral import IndirectNominalFixMuAddDataLateralFilter, IndirectNominalFixMuAddDataLateralParams
from SafetyFilters.Direct_Nominal_Zero import DirectNominalZeroFilter, DirectNominalZeroParams
from SafetyFilters.TrackFilter import SafetyFilterForTrack
from SafetyFilters.TrackFilter_AddData import SafetyFilterForTrackAddData

from simulators.simulation_settings import SafetyFilterTypes, TrackFilterTypes, SimulationInputRule, ModelType
from typing import Mapping

FewOutPutErrorSystem = Union[LinearizedErrorKinematicAcceModel, DynamicErrorModelFewOutput]


def oval_track(gen: trackGenerator, t: float, track_start_point: np.ndarray) -> trackGenerator:
    next = gen.straight(track_start_point, 3*t)
    next = gen.left_turn(next, t, np.pi)
    next = gen.straight(next, 6*t)
    next = gen.left_turn(next, t, np.pi)
    next = gen.straight(next, 3*t)
    return gen

def larger_oval_track(gen: trackGenerator, t: float, track_start_point: np.ndarray) -> trackGenerator:
    next = gen.straight(track_start_point, 3*t)
    next = gen.left_turn(next, 3*t, np.pi)
    next = gen.straight(next, 6*t)
    next = gen.left_turn(next, 3*t, np.pi)
    next = gen.straight(next, 3*t)
    return gen

def round_track(gen: trackGenerator, t: float, track_start_point: np.ndarray) -> trackGenerator:
    next = gen.left_turn(track_start_point, t, 2*np.pi)
    return gen

track_func = oval_track


class TrackSimulator:
    """Simulator for a certain track"""
    # parameters for simulation
    # parameters for kinematic model
    l_r = 0.052
    l_f = 0.038
    # additional parameters for dynamic model
    m = 0.181
    Iz = 0.000505
    Df = 0.65
    Cf = 1.5
    Bf = 5.2
    Dr = 1.0
    Cr = 1.45
    Br = 8.5
    Cm1 = 0.98028992
    Cm2 = 0.01814131
    Cd = 0.02750696
    Croll = 0.08518052
    # control and output constraints
    a_min, a_max = -1.1, 5.5
    delta_max = 0.4
    d_a_max = 27.62
    d_delta_max = 40
    v_min, v_max = -3, 5
    mu_min, mu_max = -0.5*math.pi, 0.5*math.pi

    # track parameters
    cur = 1/0.3
    track_width = 0.5
    track_start = np.array([0,0,0]) # [x_p0, y_p0, Psi_0]
    # steady state speed
    v_0 = 1
    # track density and generator
    density = 300
    track_generator = trackGenerator(density, track_width)
    track_fun_name = 'round_track'

    # noise level for online observation
    n_x = 0.001
    n_y = 0.001
    n_psi = 0.01
    n_v = 0.005
    # noise level for dataset
    n_e_lat_max = 0.002
    n_mu_max = 0.01
    n_v_max = 0.005
    n_l_max = 0.002

    use_saved_data = False # whether to use saved dataset
    save_data = True # whether to save dataset
    # maximum control inputs for collecting dataset
    a_d_max = 3
    delta_d_max = 0.2
    t_data = 10 # time horizon for dataset, in seconds
    data_model_type = ModelType.DYNAMIC_FEW_OUTPUT # model type used for collecting dataset
    data_input_rule = InputRule.RANDOM_2_WITH_MEAN # input rule used for collecting dataset

    # random seed for generating noise
    random_seed = 0
    # model type used for simulation
    simulate_model_type = ModelType.DYNAMIC_FEW_OUTPUT
    # intial state of system
    global_initial_state = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0]) # [x_p0, y_p0, yaw_0, v_x_0, v_y_0, yaw_rate_0], will be transformed if necessary
    simulation_input_type = SimulationInputRule.SINE_WAVE
    # magnitute of control inputs for simulation
    a_sim = 6
    delta_sim = 0.12*math.pi
    # simulation time
    t_sim = 1
    num_predicted_traj = 5 # number of predicted trajectories to be saved
    first_save_time = 0.6 # time when the first predicted trajectory is saved

    # filter parameters, those are important also for simulation
    filter_model_type = ModelType.KINEMATIC
    track_filter_type = TrackFilterTypes.SINGLE_SEGMENT
    filter_type_list = [SafetyFilterTypes.INDIRECT_FITTING_TERMINAL]
    # sampling time
    Ts = 1/100
    L = 50 # prediction horizon
    lag = 15 # steps used for initial condition of filter
    steps = 4 # number of inputs applied to the system for each filtering
    slack = 1e-1 # slack variable for determining the segment index

    # variables to be used for generating filters
    systems: List[LinearizedErrorKinematicAcceModel]
    filter_params: List
    io_data_dict: Mapping[float, IODataWith_l]

    def get_filter_params(self, cur: float, filter_type: SafetyFilterTypes, **kwargs):
        half_track_width = self.track_width/2
        epsilon = max(self.n_e_lat_max, self.n_mu_max, self.n_v_max)
        if filter_type == SafetyFilterTypes.INDIRECT_FITTING_TERMINAL:
            lam_sig = 50000
            c = [[0.3, 0.1, 0.05, 0.01], [0.3, 0.1, 0.05, 0.01]]
            R = np.matrix('1 0; 0 1')
            lam_sig = kwargs.get('lam_sig', lam_sig)
            c = kwargs.get('c', c)
            R = kwargs.get('R', R)
            epsilon = kwargs.get('epsilon', epsilon)
            filter_params = IndirectNominalFittingParams(
                L=self.L, lag=self.lag, R=R, lam_sig=lam_sig, epsilon=None,
                c=[
                    [c_i*half_track_width for c_i in c[0]],
                    [c_i*self.mu_max for c_i in c[1]],
                ],
                sf_params=SFParams(steps=self.steps, verbose=False, solver=cp.MOSEK,
                            solver_args={'mosek_params':{
                                'MSK_DPAR_INTPNT_CO_TOL_PFEAS':1e-4,
                                'MSK_DPAR_INTPNT_CO_TOL_DFEAS':1e-4,
                                'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':5e-4,
                                'MSK_DPAR_INTPNT_CO_TOL_INFEAS':1e-5,
                                'MSK_IPAR_INTPNT_MAX_ITERATIONS': 30000}}),
            )
        elif filter_type == SafetyFilterTypes.INDIRECT_FIX_MU:
            lam_sig = 50000
            c = [[0.3, 0.1, 0.05, 0.01], [0.3, 0.1, 0.05, 0.01]]
            R = np.matrix('1 0; 0 1')
            lam_sig = kwargs.get('lam_sig', lam_sig)
            c = kwargs.get('c', c)
            R = kwargs.get('R', R)
            epsilon = kwargs.get('epsilon', epsilon)

            filter_params = IndirectNominalFixMuParams(
                L=self.L, lag=self.lag, R=R, lam_sig=lam_sig, epsilon=None,
                c=[
                    [c_i*half_track_width for c_i in c[0]],
                    [c_i*self.mu_max for c_i in c[1]],
                ],
                sf_params=SFParams(steps=self.steps, verbose=False, solver=cp.MOSEK,
                                    solver_args={'mosek_params':{
                                        'MSK_DPAR_INTPNT_CO_TOL_PFEAS':1e-5,
                                        'MSK_DPAR_INTPNT_CO_TOL_DFEAS':1e-5,
                                        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-3,
                                        'MSK_DPAR_INTPNT_CO_TOL_INFEAS':1e-5,
                                        'MSK_IPAR_INTPNT_MAX_ITERATIONS': 30000}})
            )
        elif filter_type == SafetyFilterTypes.INDIRECT_FIX_MU_WEIGHTING:
            lam_sig = 50000
            c = [[0.3, 0.1, 0.05, 0.01], [0.3, 0.1, 0.05, 0.01]]
            R = np.matrix('1 0; 0 1')
            lam_sig = kwargs.get('lam_sig', lam_sig)
            c = kwargs.get('c', c)
            R = kwargs.get('R', R)
            epsilon = kwargs.get('epsilon', epsilon)

            filter_params = IndirectNominalFixMuWeightingParams(
                L=self.L, lag=self.lag, R=R, lam_sig=lam_sig, epsilon=None,
                c=[
                    [c_i*half_track_width for c_i in c[0]],
                    [c_i*self.mu_max for c_i in c[1]],
                ],
                sf_params=SFParams(steps=self.steps, verbose=False, solver=cp.MOSEK,
                                    solver_args={'mosek_params':{
                                        'MSK_DPAR_INTPNT_CO_TOL_PFEAS':1e-5,
                                        'MSK_DPAR_INTPNT_CO_TOL_DFEAS':1e-5,
                                        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-3,
                                        'MSK_DPAR_INTPNT_CO_TOL_INFEAS':1e-5,
                                        'MSK_IPAR_INTPNT_MAX_ITERATIONS': 30000}})
            )
        elif filter_type == SafetyFilterTypes.INDIRECT_FIX_MU_WEIGHTING_ADD_DATA:
            lam_sig = 50000
            c = [[0.3, 0.1, 0.05, 0.01], [0.3, 0.1, 0.05, 0.01]]
            R = np.matrix('1 0; 0 1')
            lam_sig = kwargs.get('lam_sig', lam_sig)
            c = kwargs.get('c', c)
            R = kwargs.get('R', R)
            epsilon = kwargs.get('epsilon', epsilon)

            filter_params = IndirectNominalFixMuWeightingAddDataParams(
                L=self.L, lag=self.lag, R=R, lam_sig=lam_sig, epsilon=None,
                c=[
                    [c_i*half_track_width for c_i in c[0]],
                    [c_i*self.mu_max for c_i in c[1]],
                ],
                sf_params=SFParams(steps=self.steps, verbose=False, solver=cp.MOSEK,
                                    solver_args={'mosek_params':{
                                        'MSK_DPAR_INTPNT_CO_TOL_PFEAS':1e-5,
                                        'MSK_DPAR_INTPNT_CO_TOL_DFEAS':1e-5,
                                        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-3,
                                        'MSK_DPAR_INTPNT_CO_TOL_INFEAS':1e-5,
                                        'MSK_IPAR_INTPNT_MAX_ITERATIONS': 30000}})
            )
        elif filter_type == SafetyFilterTypes.INDIRECT_ZERO_V:
            lam_sig = 50000
            c = [[0.3, 0.1, 0.05, 0.01], [0.3, 0.1, 0.05, 0.01]]
            R = np.matrix('1 0; 0 1')
            lam_sig = kwargs.get('lam_sig', lam_sig)
            c = kwargs.get('c', c)
            R = kwargs.get('R', R)
            epsilon = kwargs.get('epsilon', epsilon)

            filter_params = IndirectNominalZeroVParams(
                L=self.L, lag=self.lag, R=R, lam_sig=lam_sig, epsilon=None,
                c=[
                    [c_i*half_track_width for c_i in c[0]],
                    [c_i*self.mu_max for c_i in c[1]],
                ],
                sf_params=SFParams(steps=self.steps, verbose=False, solver=cp.MOSEK,
                                    solver_args={'mosek_params':{
                                        'MSK_DPAR_INTPNT_CO_TOL_PFEAS':1e-5,
                                        'MSK_DPAR_INTPNT_CO_TOL_DFEAS':1e-5,
                                        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-3,
                                        'MSK_DPAR_INTPNT_CO_TOL_INFEAS':1e-5,
                                        'MSK_IPAR_INTPNT_MAX_ITERATIONS': 30000}})
            )
        elif filter_type == SafetyFilterTypes.INDIRECT_ZERO:
            lam_sig = 50000
            c = [[0.3, 0.1, 0.05, 0.01], [0.3, 0.1, 0.05, 0.01]]
            R = np.matrix('1 0; 0 1')
            lam_sig = kwargs.get('lam_sig', lam_sig)
            c = kwargs.get('c', c)
            R = kwargs.get('R', R)
            epsilon = kwargs.get('epsilon', epsilon)

            filter_params = IndirectNominalZeroParams(
                L=self.L, lag=self.lag, R=R, lam_sig=lam_sig, epsilon=None,
                c=[
                    [c_i*half_track_width for c_i in c[0]],
                    [c_i*self.mu_max for c_i in c[1]],
                ],
                sf_params=SFParams(steps=self.steps, verbose=False, solver=cp.MOSEK,
                                    solver_args={'mosek_params':{
                                        'MSK_DPAR_INTPNT_CO_TOL_PFEAS':1e-5,
                                        'MSK_DPAR_INTPNT_CO_TOL_DFEAS':1e-5,
                                        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-3,
                                        'MSK_DPAR_INTPNT_CO_TOL_INFEAS':1e-5,
                                        'MSK_IPAR_INTPNT_MAX_ITERATIONS': 30000}})
            )
        elif filter_type == SafetyFilterTypes.INDIRECT_STOP:
            lam_sig = 50000
            c = [[0.3, 0.1, 0.05, 0.01], [0.3, 0.1, 0.05, 0.01]]
            R = np.matrix('1 0; 0 1')
            lam_sig = kwargs.get('lam_sig', lam_sig)
            c = kwargs.get('c', c)
            R = kwargs.get('R', R)
            epsilon = kwargs.get('epsilon', epsilon)

            filter_params = IndirectNominalStopParams(
                L=self.L, lag=self.lag, R=R, lam_sig=lam_sig, epsilon=None,
                c=[
                    [c_i*half_track_width for c_i in c[0]],
                    [c_i*self.mu_max for c_i in c[1]],
                ],
                sf_params=SFParams(steps=self.steps, verbose=False, solver=cp.MOSEK,
                                    solver_args={'mosek_params':{
                                        'MSK_DPAR_INTPNT_CO_TOL_PFEAS':1e-5,
                                        'MSK_DPAR_INTPNT_CO_TOL_DFEAS':1e-5,
                                        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-3,
                                        'MSK_DPAR_INTPNT_CO_TOL_INFEAS':1e-5,
                                        'MSK_IPAR_INTPNT_MAX_ITERATIONS': 30000}})
            )
        elif filter_type == SafetyFilterTypes.INDIRECT_FIX_MU_ADD_DATA:
            lam_sig = 50000
            c = [[0.3, 0.1, 0.05, 0.01], [0.3, 0.1, 0.05, 0.01]]
            R = np.matrix('1 0; 0 1')
            lam_sig = kwargs.get('lam_sig', lam_sig)
            c = kwargs.get('c', c)
            R = kwargs.get('R', R)
            epsilon = kwargs.get('epsilon', epsilon)

            filter_params = IndirectNominalFixMuAddDataParams(
                L=self.L, lag=self.lag, R=R, lam_sig=lam_sig, epsilon=None,
                c=[
                    [c_i*half_track_width for c_i in c[0]],
                    [c_i*self.mu_max for c_i in c[1]],
                ],
                sf_params=SFParams(steps=self.steps, verbose=False, solver=cp.MOSEK,
                                    solver_args={'mosek_params':{
                                        'MSK_DPAR_INTPNT_CO_TOL_PFEAS':1e-5,
                                        'MSK_DPAR_INTPNT_CO_TOL_DFEAS':1e-5,
                                        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-3,
                                        'MSK_DPAR_INTPNT_CO_TOL_INFEAS':1e-5,
                                        'MSK_IPAR_INTPNT_MAX_ITERATIONS': 30000}})
            )
        elif filter_type == SafetyFilterTypes.INDIRECT_FIX_MU_ADD_DATA_LATERAL:
            lam_sig = 50000
            c = [[0.3, 0.1, 0.05, 0.01], [0.3, 0.1, 0.05, 0.01]]
            R = np.matrix('1 0; 0 1')
            lam_sig = kwargs.get('lam_sig', lam_sig)
            c = kwargs.get('c', c)
            R = kwargs.get('R', R)
            epsilon = kwargs.get('epsilon', epsilon)

            filter_params = IndirectNominalFixMuAddDataLateralParams(
                L=self.L, lag=self.lag, R=R, lam_sig=lam_sig, epsilon=None,
                c=[
                    [c_i*half_track_width for c_i in c[0]],
                    [c_i*self.mu_max for c_i in c[1]],
                ],
                sf_params=SFParams(steps=self.steps, verbose=False, solver=cp.MOSEK,
                                    solver_args={'mosek_params':{
                                        'MSK_DPAR_INTPNT_CO_TOL_PFEAS':1e-5,
                                        'MSK_DPAR_INTPNT_CO_TOL_DFEAS':1e-5,
                                        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-3,
                                        'MSK_DPAR_INTPNT_CO_TOL_INFEAS':1e-5,
                                        'MSK_IPAR_INTPNT_MAX_ITERATIONS': 30000}})
            )
        elif filter_type == SafetyFilterTypes.DIRECT_ZERO_TERMINAL:
            lam_sig = 1500
            lam_alph = 1500/epsilon
            c = [[0.35, 0.1, 0.1, 0.05], [0.4, 0.1, 0.1, 0.05]]
            R = np.matrix('1 0; 0 1')
            lam_sig = kwargs.get('lam_sig', lam_sig)
            lam_alph = kwargs.get('lam_alph', lam_alph)
            c = kwargs.get('c', c)
            R = kwargs.get('R', R)
            epsilon = kwargs.get('epsilon', epsilon)
            filter_params = DirectNominalZeroParams(
                L=self.L, lag=self.lag, R=R, lam_alph=lam_alph, lam_sig=lam_sig, epsilon=epsilon,
                c=[
                    [c_i*half_track_width for c_i in c[0]],
                    [c_i*self.mu_max for c_i in c[1]],
                ],
                sf_params=SFParams(steps=self.steps, verbose=True, solver=cp.MOSEK, 
                                   solver_args={'mosek_params':{
                                        'MSK_DPAR_INTPNT_CO_TOL_PFEAS':3e-5,
                                        'MSK_DPAR_INTPNT_CO_TOL_DFEAS':3e-5,
                                        'MSK_DPAR_INTPNT_CO_TOL_INFEAS':3e-6,
                                        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-4}})
            )
        return filter_params
    
    def get_single_filter(self, 
                          cur: float, filter_type: SafetyFilterTypes,
                          system, io_data, **kwargs):
        params = self.get_filter_params(cur, filter_type, **kwargs)
        FilterClass = {SafetyFilterTypes.INDIRECT_FITTING_TERMINAL: IndirectNominalFittingFilter,
                        SafetyFilterTypes.INDIRECT_FIX_MU: IndirectNominalFixMuFilter,
                        SafetyFilterTypes.INDIRECT_FIX_MU_WEIGHTING: IndirectNominalFixMuWeightingFilter,
                        SafetyFilterTypes.INDIRECT_FIX_MU_WEIGHTING_ADD_DATA: IndirectNominalFixMuWeightingAddDataFilter,
                        SafetyFilterTypes.INDIRECT_ZERO_V: IndirectNominalZeroVFilter,
                        SafetyFilterTypes.INDIRECT_ZERO: IndirectNominalZeroFilter,
                        SafetyFilterTypes.INDIRECT_STOP: IndirectNominalStopFilter,
                        SafetyFilterTypes.INDIRECT_FIX_MU_ADD_DATA: IndirectNominalFixMuAddDataFilter,
                        SafetyFilterTypes.INDIRECT_FIX_MU_ADD_DATA_LATERAL: IndirectNominalFixMuAddDataLateralFilter,
                        SafetyFilterTypes.DIRECT_ZERO_TERMINAL: DirectNominalZeroFilter,
                        }.get(filter_type, IndirectNominalFittingFilter)
        return FilterClass(system, io_data, params)
    
    def simulate_multi(self,
                       random_seends: List[float],
                       track_filter_types: List[TrackFilterTypes],
                       filter_type_lists: List[SafetyFilterTypes],
                       filter_params: Dict[SafetyFilterTypes, List[Dict[str, Any]]],
                       simualtion_input_types: List[SimulationInputRule],) -> Dict[Any, List[Results]]:
        dict_results = {}
        self.noise_list_dict = {}
        for random_seed in random_seends:
            self.random_seed = random_seed
            self.noise_list_dict[random_seed] = self.get_noise()
        for random_seed, track_filter_type, filter_type, simulation_input_type in product(random_seends, track_filter_types, filter_type_lists, simualtion_input_types):
            self.track_filter_type = track_filter_type
            self.filter_type_list = [filter_type] # reserve for using different filter types for segments
            self.simulation_input_type = simulation_input_type
            dict_results[(random_seed, track_filter_type, filter_type, simulation_input_type)] = []
            print(f"\n \n Simulating random_seed={random_seed}, track_filter_type={track_filter_type}, filter_type={filter_type}, simulation_input_type={simulation_input_type}")
            for filter_param in filter_params.get(filter_type, [{}]):
                results = self.simulate_once(random_seed, **filter_param)
                dict_results[(random_seed, track_filter_type, filter_type, simulation_input_type)].append(results)
        
        self.dict_results = dict_results
        return dict_results
    
    def simulate_once(self, random_seed: int, **kwargs) -> Results:
        self.random_seed = random_seed
        # store parameters, after single simualtion they should be restored
        Ts, L, steps, lag, slack = self.Ts, self.L, self.steps, self.lag, self.slack
    
        self.set_params_from_dict(**kwargs)
        
        self.track_func = {
            'oval_track': oval_track,
            'round_track': round_track,
            'large_oval_track': larger_oval_track,
        }.get(self.track_fun_name, oval_track)
        self.track_generator = self.propogate_track_gen()

        self.systems = self.get_system_list(system_type=self.filter_model_type)

        self.io_data_dict = self.get_io_data_dic()

        if random_seed not in self.noise_list_dict.keys():
            print(f"Noise with random_seed={random_seed} not generated, generating it online")
            self.noise_list = self.get_noise()
        else:
            self.noise_list = self.noise_list_dict[random_seed]

        self.filter = self.get_filter(**kwargs)

        if self.filter is None:
            self.Ts, self.L, self.steps, self.lag, self.slack = Ts, L, steps, lag, slack
            return None

        results = Results(self.Ts)
        # stores the segment index for each time step
        self.segment_index_list = []

        # initialize global model
        if self.simulate_model_type == ModelType.KINEMATIC:
            kinematic_model_params = KinematicAcceModelParams(self.l_r, self.l_f, self.m, self.Ts)
            v_0 = math.sqrt(self.global_initial_state[3]**2 + self.global_initial_state[4]**2)
            initial_state = np.hstack((self.global_initial_state[0:3],np.array([v_0])))
            crs_model = KinematicAcceModel(kinematic_model_params, initial_state)
        elif self.simulate_model_type == ModelType.DYNAMIC:
            initial_state = self.global_initial_state
            dynamic_model_params = DynamicModelParams(
                self.Ts, self.l_f, self.l_r, self.m, self.Iz, self.Bf, self.Cf, self.Df, self.Br, self.Cr, self.Dr, self.Croll, self.Cm1, self.Cm2, self.Cd,
            )
            crs_model = DynamicModel(dynamic_model_params, self.global_initial_state)
        elif self.simulate_model_type == ModelType.DYNAMIC_FEW_OUTPUT:
            initial_state = self.global_initial_state
            dynamic_model_params = DynamicModelParams(
                self.Ts, self.l_f, self.l_r, self.m, self.Iz, self.Bf, self.Cf, self.Df, self.Br, self.Cr, self.Dr, self.Croll, self.Cm1, self.Cm2, self.Cd,
            )
            crs_model = DynamicModelFewOutput(dynamic_model_params, self.global_initial_state)
            v_0 = math.sqrt(self.global_initial_state[3]**2 + self.global_initial_state[4]**2)
            # initial state for safety filter, here it should be few output model
            initial_state = np.hstack((self.global_initial_state[0:3],np.array([v_0])))

        # initial states for the safety filter. If initial state is non-zero, careful with consistency!
        initial_states_filter: List[np.ndarray] = [initial_state] * self.lag
        initial_inputs_filter: List[np.ndarray] = [np.matrix([[0],[0]])] * self.lag
        
        # prepare log and block number for simulation
        N = int(self.t_sim/self.Ts)
        N_blocks = int(N/self.steps)
        first_block_to_save = int(self.first_save_time/self.Ts/self.steps)
        N_slices = int((N_blocks-first_block_to_save)/self.num_predicted_traj) # save predicted trajectory every N_slices blocks

        for i_block in range(N_blocks):
            t_block = i_block*self.steps*self.Ts
            # get objective inputs
            u_obj = self.get_input_obj(t_block, crs_model.state)
            
            # try solving the optimization problem
            try:
                start = timer()
                # update initial conditions for filter
                for u_init, y_init, n_init in zip(initial_inputs_filter[-self.lag:], initial_states_filter[-self.lag:], self.noise_list[i_block*self.steps:i_block*self.steps+self.lag]):
                    self.filter.add_point(u_init, y_init + n_init)
                
                # apply the safety filter
                u_i, status_i, opt_i = self.filter.filter(u_obj)
                end = timer()
            except Exception as e:
                print(f"Exception {e} raised during optimization, returning partial results")
                # calculate root mean square intervention and calcualtion time
                results.calculate_intervention()
                results.calculate_mean_calculation_time()
                results.calculate_sigma_infty_value()

                # reset parameters
                self.Ts, self.L, self.steps, self.lag, self.slack = Ts, L, steps, lag, slack
                return results
            results.add_calculation_time(end-start)
            results.add_opt_value(opt_i)
            i_seg = self.filter._i
            results.add_sigma_value(self.filter._safety_filters[i_seg]._sigma.value)

            # save predicted trajectory every N_slices blocks
            if (i_block-first_block_to_save)%N_slices == 0:
                if self.filter_type_list[i_seg] in SafetyFilterTypes.direct_types:
                    predicted_traj = np.split(self.filter._safety_filters[i_seg]._y.value[self.lag*self.filter._p:], self.L)
                    predicted_with_slack = self.filter._safety_filters[i_seg]._y.value[self.lag*self.filter._p:] - self.filter._safety_filters[i_seg]._sigma.value[self.lag*self.filter._p:]
                    predicted_with_slack = np.split(predicted_with_slack, self.L)
                # elif self.filter_type_list[i_seg] == SafetyFilterTypes.INDIRECT_FITTING_TERMINAL:
                else:
                    predicted_traj = np.split(self.filter._safety_filters[i_seg]._y.value, self.L)
                    predicted_with_slack = self.filter._safety_filters[i_seg]._y.value - self.filter._safety_filters[i_seg]._sigma.value
                    predicted_with_slack = np.split(predicted_with_slack, self.L)
                # transforms for readability
                for y in predicted_traj:
                    y[1] = y[1] * 180 / np.pi # from rad to deg
                    y[2] = y[2] + self.v_0 # from deviation from steady state to actual velocity
                for y in predicted_with_slack:
                    y[1] = y[1] * 180 / np.pi # from rad to deg
                    y[2] = y[2] + self.v_0 # from deviation from steady state to actual velocity
                # get and save real output when the proposed inputs are applied
                sys_for_output = self.get_system(cur=self.systems[i_seg].cur, start_point=self.systems[i_seg].segment_start, system_type=self.simulate_model_type)
                sys_for_output.set_kinematic_model_state(crs_model._state)
                real_output_list = []
                if self.filter_type_list[i_seg] in SafetyFilterTypes.direct_types:
                    u_proposed = self.filter._safety_filters[i_seg]._u.value[self.lag*self.filter._m:]
                else:
                    u_proposed = self.filter._safety_filters[i_seg]._u.value
                for i in range(self.L):
                    try:
                        u = u_proposed[i*sys_for_output.m:(i+1)*sys_for_output.m]
                        u = np.matrix(np.reshape(u, (sys_for_output.m, 1)))
                        y, e_lin, n = sys_for_output.step_lin(u)
                        y = np.array(y).flatten()
                        y[1] = y[1] * 180 / np.pi # from rad to deg
                        y[2] = y[2] + self.v_0 # from deviation from steady state to actual velocity
                        real_output_list.append(y)
                    except RuntimeError as e:
                        print(e, "during simulation of real system, returning partial results")
                        break
                results.add_predicted_error_slice(i_block*self.steps*self.Ts, predicted_traj)
                results.add_predicted_error_slack_slice(i_block*self.steps*self.Ts, predicted_with_slack)
                results.add_error_slice(i_block*self.steps*self.Ts, real_output_list)
            
            # save and update system state, for `step` times
            for j in range(self.steps):
                i_steps = i_block*self.steps + j
                # save system states and inputs first
                # global kinematic model state
                global_state = deepcopy(crs_model.state)
                # error system state
                self.systems[i_seg].set_kinematic_model_state(global_state, -self.slack)
                error_dynamics_state = deepcopy(self.systems[i_seg].state)
                global_state[2] = global_state[2] * 180 / np.pi # from rad to deg
                error_dynamics_state[1] = error_dynamics_state[1] * 180 / np.pi # from rad to deg
                u_obj_t = u_obj[j*self.systems[i_seg].m:(j+1)*self.systems[i_seg].m]
                u_t = u_i[j*self.systems[i_seg].m:(j+1)*self.systems[i_seg].m]

                # try step global system
                try:
                    crs_model.step(np.array(u_t).flatten())
                except RuntimeError as e:
                    print(f"Exception {e} raised during system simulation, returning partial results")
                    # calculate root mean square intervention and calcualtion time
                    results.calculate_intervention()
                    results.calculate_mean_calculation_time()
                    results.calculate_sigma_infty_value()

                    # reset parameters
                    self.Ts, self.L, self.steps, self.lag, self.slack = Ts, L, steps, lag, slack
                    return results

                # save states and inputs for filter
                initial_inputs_filter.append(u_t)
                initial_states_filter.append(crs_model.state)

                # save to results and trajectory, for initial condition of filter later
                results.add_point(u_obj_t, u_t, 
                                  global_state, self.noise_list[i_steps+self.lag], 
                                  error_dynamics_state, np.zeros(error_dynamics_state.shape),)

        # calculate root mean square intervention and calcualtion time
        results.calculate_intervention()
        results.calculate_mean_calculation_time()
        results.calculate_sigma_infty_value()

        # reset parameters
        self.Ts, self.L, self.steps, self.lag, self.slack = Ts, L, steps, lag, slack
        return results
    
    def propogate_track_gen(self) -> trackGenerator:
        self.track_generator = trackGenerator(self.density, self.track_width)
        # track_func = {
        #     'oval_track': oval_track,
        #     'round_track': round_track,
        # }.get(self.track_fun_name, oval_track)
        self.track_func(self.track_generator, t=1/self.cur, track_start_point=self.track_start)
        self.track_generator.populatePointsAndArcLength()
        return self.track_generator
    
    def get_input_obj(self, t: float, global_state: np.ndarray) -> np.matrix:
        """get next `steps` inputs

        :param t: simulation time
        :param xi_t: extended state of system
        :return: proposed u_obj for next `steps`
        """
        u_obj = np.matrix(np.ndarray((0,1)))
        throttle_sim = self.a_sim * self.m
        for j in range(self.steps):
            t_j = t + j*self.Ts

            if self.simulation_input_type == SimulationInputRule.SINE_WAVE:
                u_obj_k = np.matrix([[throttle_sim*math.sin(t_j*math.pi)],[self.delta_sim*math.sin(3*t_j*math.pi)]])
            elif self.simulation_input_type == SimulationInputRule.MAX_THROTTLE_SINE_STEER:
                u_obj_k = np.matrix([[throttle_sim],[self.delta_sim*math.sin(3*t_j*math.pi)]])
            elif self.simulation_input_type == SimulationInputRule.MAX_THROTTLE:
                u_obj_k = np.matrix([[throttle_sim],[0]])
            u_obj = np.vstack( (u_obj, u_obj_k) )
        return u_obj
    
    def get_system_list(self, system_type: ModelType = ModelType.KINEMATIC) -> List[FewOutPutErrorSystem]:
        system_list = []
        for segment in self.track_generator.chainOfSegments:
            if isinstance(segment, line):
                psi_0 = math.atan2(segment.y1-segment.y0, segment.x1-segment.x0, )
                start_point = np.array([segment.x0, segment.y0, psi_0])
                system_list.append(self.get_system(cur=0, start_point=start_point, system_type=system_type))
            elif isinstance(segment, arc):
                unit_from_center = np.array([np.cos(segment.theta_s), np.sin(segment.theta_s)])
                x_p0 = segment.x0 + unit_from_center[0]*segment.radius
                y_p0 = segment.y0 + unit_from_center[1]*segment.radius
                if segment.theta_f > segment.theta_s: # left hand turn
                    cur = segment.curvature
                    psi_0 = segment.theta_s + 0.5*math.pi
                else: # right hand turn
                    cur = -segment.curvature
                    psi_0 = segment.theta_s - 0.5*math.pi
                system_list.append(self.get_system(cur=cur, start_point=np.array([x_p0, y_p0, psi_0]), system_type=system_type))
        return system_list
    
    def get_system(self, cur: float, start_point: np.ndarray, system_type: ModelType = ModelType.KINEMATIC) -> FewOutPutErrorSystem:
        """get Linearized system for given curvature and start point
        
        start_point: array of the form [x, y, psi]
        """
        half_track_width = self.track_width/2
        if system_type == ModelType.KINEMATIC:
            system_params = LinearizedErrorKinematicAcceModelParams(
                    KinematicAcceModelParams(l_r=self.l_r,l_f=self.l_f,m=self.m,Ts=self.Ts),
                    cur = cur,
                    state_s0 = start_point,
                    v_0 = self.v_0,
                    A_u = np.matrix('1 0; 0 1; -1 0; 0 -1'),
                    b_u = np.matrix([[self.a_max*self.m],[self.delta_max],[-self.a_min*self.m],[self.delta_max]]),
                    A_y = np.matrix('1 0 0; 0 1 0; -1 0 0; 0 -1 0'),
                    b_y = np.matrix([[half_track_width],[self.mu_max],[half_track_width],[-self.mu_min]]),
                    A_n = np.matrix('1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1'),
                    b_n = np.matrix([[self.n_e_lat_max],[self.n_mu_max],[self.n_v_max],[self.n_e_lat_max],[self.n_mu_max],[self.n_v_max]]),
                )
            # for this system, the global initial state does not matter
            return LinearizedErrorKinematicAcceModel(params=system_params, state_0=np.array([0,0,0,0]))
        elif system_type == ModelType.DYNAMIC_FEW_OUTPUT:
            system_params = DynamicErrorModelParams(
                    DynamicModelParams(Ts=self.Ts, l_f=self.l_f, l_r=self.l_r, m=self.m, Iz=self.Iz, Bf=self.Bf, Cf=self.Cf, Df=self.Df, Br=self.Br, Cr=self.Cr, Dr=self.Dr, Croll=self.Croll, Cm1=self.Cm1, Cm2=self.Cm2, Cd=self.Cd),
                    cur = cur,
                    segment_start = start_point,
                    v_0 = self.v_0,
                    A_u = np.matrix('1 0; 0 1; -1 0; 0 -1'),
                    b_u = np.matrix([[self.a_max*self.m],[self.delta_max],[-self.a_min*self.m],[self.delta_max]]),
                    A_y = np.matrix('1 0 0; 0 1 0; -1 0 0; 0 -1 0'),
                    b_y = np.matrix([[half_track_width],[self.mu_max],[half_track_width],[-self.mu_min]]),
                    A_n = np.matrix('1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1'),
                    b_n = np.matrix([[self.n_e_lat_max],[self.n_mu_max],[self.n_v_max],[self.n_e_lat_max],[self.n_mu_max],[self.n_v_max]]),
            )
            # for this system, the global initial state does not matter
            return DynamicErrorModelFewOutput(params=system_params, initial_state=np.array([0,0,0,0,0,0]))
    
    def get_io_data_dic(self) -> Dict[float, IODataWith_l]:
        if self.use_saved_data: # use saved data
            with open(os.path.join(os.getcwd(), 'datasets', f'io_datas_{self.t_data}_{self.Ts}.pkl'), 'rb') as read_file:
                io_data_dict: Dict[float, IODataWith_l] = pickle.load(read_file)
        else: # generate new data
            io_data_dict = {}
            np.random.seed(self.random_seed)
            for segment in self.track_generator.chainOfSegments:
                if isinstance(segment, line):
                    cur = 0
                elif segment.theta_f < segment.theta_s: # right hand turn
                    cur = -segment.curvature
                elif segment.theta_f > segment.theta_s: # left hand turn
                    cur = segment.curvature
                sys = self.get_system(cur=cur, start_point=np.array([0,0,0]), system_type=self.data_model_type)
                zero_input = sys.get_zero_input()
                zero_state = sys.get_zero_state()
                sys.set_error_state(zero_state)
                length = int(self.t_data/self.Ts)
                u_0_max = self.a_d_max*self.m
                io_data = IODataWith_l(
                            self.L+self.lag, sys=sys, input_rule=self.data_input_rule, mean_input=np.matrix(zero_input).transpose(), length=length,
                            A_u_d=np.matrix('1 0; 0 1; -1 0; 0 -1'), b_u_d=np.matrix([[u_0_max],[self.delta_d_max],[u_0_max],[self.delta_d_max]]),
                            d_u_max=np.matrix([[self.a_d_max*self.m],[self.delta_d_max]]),
                            n_l = self.n_l_max, lag = self.lag, L = self.L, N_l = int(length/2), K_l = 5)
                io_data_dict[cur] = io_data
                if self.save_data:
                    with open(os.path.join(os.getcwd(), 'datasets', f'io_datas_{self.t_data}_{self.Ts}.pkl'), 'wb') as file:
                        pickle.dump(io_data_dict, file)
        return io_data_dict

    def get_filter(self, **kwargs):
        if self.track_filter_type == TrackFilterTypes.SINGLE_SEGMENT:
            filter_list = []
            num_segments = len(self.track_generator.chainOfSegments)
            num_given_filter_types = len(self.filter_type_list)
            if num_given_filter_types < num_segments:
                self.filter_type_list = self.filter_type_list + [self.filter_type_list[-1]] * (num_segments - num_given_filter_types)
            for segment, filter_type, system in zip(
                self.track_generator.chainOfSegments, self.filter_type_list, self.systems
                ):
                if filter_type not in SafetyFilterTypes.feedin_xi_types:
                    print(f"Warning: filter type {filter_type} is not supported for track filter type {self.track_filter_type}.")
                    return None
                if isinstance(segment, line):
                    filter_list.append(self.get_single_filter(0, filter_type, system, self.io_data_dict[0], **kwargs))
                elif isinstance(segment, arc):
                    if segment.theta_f > segment.theta_s: # left hand turn
                        cur = segment.curvature
                    elif segment.theta_f < segment.theta_s: # right hand turn
                        cur = -segment.curvature
                    filter_list.append(self.get_single_filter(0, filter_type, system, self.io_data_dict[cur], **kwargs))
            self.filter_list = filter_list
            filter = SafetyFilterForTrack(
                track_fun=self.track_func,
                t = 1/self.cur,
                track_start = self.track_start,
                density = self.density,
                filter_list=filter_list,
                # io_datas=self.io_data_dict,
                systems=self.systems,
                slack=self.slack,
                position_0=self.global_initial_state
            )
        elif self.track_filter_type == TrackFilterTypes.SINGLE_SEGMENT_ADD_DATA:
            filter_list = []
            num_segments = len(self.track_generator.chainOfSegments)
            num_given_filter_types = len(self.filter_type_list)
            if num_given_filter_types < num_segments:
                self.filter_type_list = self.filter_type_list + [self.filter_type_list[-1]] * (num_segments - num_given_filter_types)
            for segment, filter_type, system in zip(
                self.track_generator.chainOfSegments, self.filter_type_list, self.systems
                ):
                if filter_type not in SafetyFilterTypes.feedin_list_state_types:
                    print(f"Warning: filter type {filter_type} is not supported for track filter type {self.track_filter_type}.")
                    return None
                if isinstance(segment, line):
                    filter_list.append(self.get_single_filter(0, filter_type, system, self.io_data_dict[0], **kwargs))
                elif isinstance(segment, arc):
                    if segment.theta_f > segment.theta_s: # left hand turn
                        cur = segment.curvature
                    elif segment.theta_f < segment.theta_s: # right hand turn
                        cur = -segment.curvature
                    filter_list.append(self.get_single_filter(0, filter_type, system, self.io_data_dict[cur], **kwargs))
            self.filter_list = filter_list
            filter = SafetyFilterForTrackAddData(
                track_fun=self.track_func,
                t = 1/self.cur,
                track_start = self.track_start,
                density = self.density,
                filter_list=filter_list,
                # io_datas=self.io_data_dict,
                systems=self.systems,
                slack=self.slack,
                position_0=self.global_initial_state
            )
        
        return filter

    def get_noise(self) -> List[np.ndarray]:
        length = self.lag + int(self.t_sim/self.Ts)
        np.random.seed(self.random_seed)
        if self.simulate_model_type in ModelType.few_output_models:
            return [np.array([
                                self.n_x*(2*np.random.rand()-1),
                                self.n_y*(2*np.random.rand()-1),
                                self.n_psi*(2*np.random.rand()-1),
                                self.n_v*(2*np.random.rand()-1),
                            ]) for _ in range(length)]
    
    def set_params_from_dict(self, **kwargs) -> None:
        """
        Get parameters based on default values and kwargs \\
        return (Ts, L, steps, lag, slack)
        """
        self.Ts = kwargs.get('Ts', self.Ts)
        self.L = kwargs.get('L', self.L)
        self.steps = kwargs.get('steps', self.steps)
        self.lag = kwargs.get('lag', self.lag)
        self.slack = kwargs.get('slack', self.slack)
