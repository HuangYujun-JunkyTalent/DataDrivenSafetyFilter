from typing import List, Dict, Any, Tuple, Union, Optional
import math
import os
import datetime
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
from System.DynamicErrorModel import DynamicErrorModelFewOutput, DynamicErrorModelParams, DynamicErrorModelVxy
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
from SafetyFilters.Indirect_Nominal_ZeroV_Weighting import IndirectNominalZeroVWeightingFilter, IndirectNominalZeroVWeightingParams
from SafetyFilters.Indirect_Nominal_ZeroV_Weighting_Add_Data import IndirectNominalZeroVWeightingAddDataFilter, IndirectNominalZeroVWeightingAddDataParams
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
    v_x_min, v_x_max = 0.05, 3.2
    v_y_min, v_y_max = -2.0, 2.0
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
    num_prediction_buffer = 50 # save the previous preidcted trajectory and proposed inputs, for later uasge
    buffer_y_value: List[Tuple[float,List[np.ndarray]]] = [] #[[time step, List of state],]
    buffer_sigma_value: List[Tuple[float,List[np.ndarray]]] = []
    buffer_u_value: List[Tuple[float,List[np.ndarray]]] = []
    buffer_i_seg: List[Tuple[float,int]] = []
    buffer_global_state: List[Tuple[float,np.ndarray]] = []
    out_of_track: bool = False # whether the car is out of track, we only want to see what happens at the first steps!
    save_predicted_traj_regular = True # whether to save predicted trajectory
    stop_after_out_of_track = False # whether to stop simulation after the car is out of track
    save_dataset_after = True # whether to save dataset after simulation
    delete_dataset_after = True # whether to delete dataset after reading it

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
    io_data_dict_stored: Mapping[float, IODataWith_l] = {} # stored data, not modified after created

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
            c_max_list = [half_track_width, self.mu_max, (self.v_x_max-self.v_x_min)/2, (self.v_y_max-self.v_y_min)/2]

            filter_params = IndirectNominalFixMuWeightingParams(
                L=self.L, lag=self.lag, R=R, lam_sig=lam_sig, epsilon=None,
                c=[
                    [c_i*c_max for c_i in c_sub] for c_max,c_sub in zip(c_max_list, c)
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
            t_new_data = kwargs.get('t_new_data', 6.0)
            c_max_list = [half_track_width, self.mu_max, (self.v_x_max-self.v_x_min)/2, (self.v_y_max-self.v_y_min)/2]

            min_dist = kwargs.get('min_dist', 0.1)
            min_num_slices = kwargs.get('min_num_slices', 300)
            min_portion_slices = kwargs.get('min_portion_slices', 0.5)
            f = kwargs.get('f', lambda x: 1/x**2)

            filter_params = IndirectNominalFixMuWeightingAddDataParams(
                L=self.L, lag=self.lag, R=R, lam_sig=lam_sig, epsilon=None,
                t_new_data=t_new_data,
                min_dist=min_dist,
                min_num_slices=min_num_slices,
                min_portion_slices=min_portion_slices,
                f = f,
                c=[
                    [c_i*c_max for c_i in c_sub] for c_max,c_sub in zip(c_max_list, c)
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
        elif filter_type == SafetyFilterTypes.INDIRECT_ZERO_V_WEIGHTING:
            lam_sig = 50000
            c = [[0.3, 0.1, 0.05, 0.01], [0.3, 0.1, 0.05, 0.01]]
            R = np.matrix('1 0; 0 1')
            lam_sig = kwargs.get('lam_sig', lam_sig)
            c = kwargs.get('c', c)
            R = kwargs.get('R', R)
            epsilon = kwargs.get('epsilon', epsilon)
            t_new_data = kwargs.get('t_new_data', 6.0)
            c_max_list = [half_track_width, self.mu_max, (self.v_x_max-self.v_x_min)/2, (self.v_y_max-self.v_y_min)/2]

            filter_params = IndirectNominalZeroVWeightingParams(
                L=self.L, lag=self.lag, R=R, lam_sig=lam_sig, epsilon=None,
                c=[
                    [c_i*c_max for c_i in c_sub] for c_max,c_sub in zip(c_max_list, c)
                ],
                sf_params=SFParams(steps=self.steps, verbose=False, solver=cp.MOSEK,
                                    solver_args={'mosek_params':{
                                        'MSK_DPAR_INTPNT_CO_TOL_PFEAS':1e-5,
                                        'MSK_DPAR_INTPNT_CO_TOL_DFEAS':1e-5,
                                        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-3,
                                        'MSK_DPAR_INTPNT_CO_TOL_INFEAS':1e-5,
                                        'MSK_IPAR_INTPNT_MAX_ITERATIONS': 30000}})
            )
        elif filter_type == SafetyFilterTypes.INDIRECT_ZERO_V_WEIGHTING_ADD_DATA:
            lam_sig = 50000
            c = [[0.3, 0.1, 0.05, 0.01], [0.3, 0.1, 0.05, 0.01]]
            R = np.matrix('1 0; 0 1')
            lam_sig = kwargs.get('lam_sig', lam_sig)
            c = kwargs.get('c', c)
            R = kwargs.get('R', R)
            epsilon = kwargs.get('epsilon', epsilon)
            t_new_data = kwargs.get('t_new_data', 6.0)
            c_max_list = [half_track_width, self.mu_max, (self.v_x_max-self.v_x_min)/2, (self.v_y_max-self.v_y_min)/2]

            filter_params = IndirectNominalZeroVWeightingAddDataParams(
                L=self.L, lag=self.lag, R=R, lam_sig=lam_sig, epsilon=None,
                t_new_data=t_new_data,
                c=[
                    [c_i*c_max for c_i in c_sub] for c_max,c_sub in zip(c_max_list, c)
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
                        SafetyFilterTypes.INDIRECT_ZERO_V_WEIGHTING: IndirectNominalZeroVWeightingFilter,
                        SafetyFilterTypes.INDIRECT_ZERO_V_WEIGHTING_ADD_DATA: IndirectNominalZeroVWeightingAddDataFilter,
                        SafetyFilterTypes.INDIRECT_ZERO: IndirectNominalZeroFilter,
                        SafetyFilterTypes.INDIRECT_STOP: IndirectNominalStopFilter,
                        SafetyFilterTypes.INDIRECT_FIX_MU_ADD_DATA: IndirectNominalFixMuAddDataFilter,
                        SafetyFilterTypes.INDIRECT_FIX_MU_ADD_DATA_LATERAL: IndirectNominalFixMuAddDataLateralFilter,
                        SafetyFilterTypes.DIRECT_ZERO_TERMINAL: DirectNominalZeroFilter,
                        }.get(filter_type, IndirectNominalFittingFilter)
        return FilterClass(system, io_data, params)
    
    def simulate_with_separate_collection(self,
                                        random_seed: int,
                                        track_filter_type: TrackFilterTypes,
                                        filter_type: SafetyFilterTypes,
                                        filter_params: Dict[str, Any],
                                        collection_input_type: SimulationInputRule,
                                        simulation_input_types: List[SimulationInputRule],
                                        max_run_turns: int = 10,
                                        ) -> List[Results]:
        """Simulate with separate data-collection and filtering steps."""
        # setup needed options
        self.stop_after_out_of_track = True

        self.noise_list_dict = {}
        self.random_seed = random_seed
        self.noise_list_dict[random_seed] = self.get_noise()
        self.track_filter_type = track_filter_type
        self.filter_type_list = [filter_type] # reserve for using different filter types for segments

        results_list: List[Results] = []

        print("\nRunning with initial dataset")
        empty_result = Results(Ts = self.Ts)
        results_list.append(empty_result)
        for input_rule in simulation_input_types:
            print(f"\nSimulating with input rule {input_rule}")
            self.stop_after_out_of_track = False
            self.out_of_track = False
            self.save_dataset_after = False
            self.simulation_input_type = input_rule
            self.get_utilities_for_simualtion(random_seed, **filter_params)
            print(f"And with first dataset list {[data.length for data in self.filter._safety_filters[0]._io_data_list]}")
            result = self.simulate_once(random_seed, **filter_params)
            results_list.append(result)

        print("\nCollecting data for the first time")
        self.stop_after_out_of_track = True
        self.get_utilities_for_simualtion(random_seed, **filter_params)
        print(f"\nSimulating with input rule {collection_input_type}")
        self.save_dataset_after = True
        self.simulation_input_type = collection_input_type
        print(f"And with first dataset list {[data.length for data in self.filter._safety_filters[0]._io_data_list]}")
        dataset_result = self.simulate_once(random_seed, **filter_params)
        results_list.append(dataset_result)

        all_good = False
        while not all_good and len(results_list) < max_run_turns*(len(simulation_input_types)+1):
        # while len(results_list) < max_run_turns:
            print("\nContinue to simulate")
            for input_rule in simulation_input_types:
                print(f"\nSimulating with input rule {input_rule}")
                self.stop_after_out_of_track = False
                self.out_of_track = False
                self.save_dataset_after = False
                self.simulation_input_type = input_rule
                self.get_utilities_for_simualtion(random_seed, **filter_params)
                self.load_datasets(dataset_result.saved_dataset_name, delete_after_load=False)
                print(f"And with first dataset list {[data.length for data in self.filter._safety_filters[0]._io_data_list]}")
                result = self.simulate_once(random_seed, **filter_params)
                results_list.append(result)
            all_good = True # whether constraints not violated for all simulation input rules
            for result in results_list[-len(simulation_input_types):]:
                if len(result.violating_time_steps) != 0:
                    all_good = False
                    if len(results_list) < max_run_turns*(len(simulation_input_types)+1):
                        # collect new datasets with proper input rule
                        print(f"\nCollecting dataset with input rule {collection_input_type}. \n")
                        self.stop_after_out_of_track = True
                        self.out_of_track = False
                        self.save_dataset_after = True
                        self.simulation_input_type = collection_input_type
                        self.get_utilities_for_simualtion(random_seed, **filter_params)
                        self.load_datasets(dataset_result.saved_dataset_name, delete_after_load=False)
                        print(f"And with first dataset list {[data.length for data in self.filter._safety_filters[0]._io_data_list]}")
                        dataset_result = self.simulate_once(random_seed, **filter_params)
                        results_list.append(dataset_result)
                    break
        
        print(f"\nConcluding simulation, with {len(results_list)} results collected.\n")

        return results_list

    def simulate_with_dataset_update(self,
                                     random_seed: int,
                                     track_filter_type: TrackFilterTypes,
                                     filter_type: SafetyFilterTypes,
                                     filter_params: Dict[str, Any],
                                     simulation_input_types: List[SimulationInputRule],
                                     max_run_turns: int = 10,
                                     ) -> List[Results]:
        # setup needed options
        self.save_dataset_after = True
        self.stop_after_out_of_track = True

        self.noise_list_dict = {}
        self.random_seed = random_seed
        self.noise_list_dict[random_seed] = self.get_noise()
        self.track_filter_type = track_filter_type
        self.filter_type_list = [filter_type] # reserve for using different filter types for segments
        self.simulation_input_type = simulation_input_types[0]

        results_list: List[Results] = []
        
        print("Simulating for the first time")
        self.get_utilities_for_simualtion(random_seed, **filter_params)
        print(f"\nSimulating with input rule {simulation_input_types[0]}")
        print(f"And with first dataset list {[data.length for data in self.filter._safety_filters[0]._io_data_list]}")
        result = self.simulate_once(random_seed, **filter_params)
        results_list.append(result)
        i = 1
        for input_rule in simulation_input_types[1:]:
            self.out_of_track = False
            print(f"\nSimulating with input rule {input_rule}")
            self.simulation_input_type = input_rule
            self.get_utilities_for_simualtion(random_seed, **filter_params)
            self.load_datasets(results_list[-1].saved_dataset_name)
            print(f"And with first dataset list {[data.length for data in self.filter._safety_filters[0]._io_data_list]}")
            result = self.simulate_once(random_seed, **filter_params)
            results_list.append(result)
            i += 1
        all_good = True # good for all simulation input rules
        for result in results_list[-len(simulation_input_types):]:
            if len(result.violating_time_steps) != 0:
                all_good = False
                break
        while not all_good and len(results_list) < max_run_turns:
        # while len(results_list) < max_run_turns:
            print("\nContinue to simulate since violated constraints last time")
            for input_rule in simulation_input_types:
                print(f"\nSimulating with input rule {input_rule}")
                self.out_of_track = False
                self.simulation_input_type = input_rule
                self.get_utilities_for_simualtion(random_seed, **filter_params)
                self.load_datasets(results_list[-1].saved_dataset_name)
                print(f"And with first dataset list {[data.length for data in self.filter._safety_filters[0]._io_data_list]}")
                result = self.simulate_once(random_seed, **filter_params)
                results_list.append(result)
            all_good = True # good for all simulation input rules
            for result in results_list[-len(simulation_input_types):]:
                if len(result.violating_time_steps) != 0:
                    all_good = False
                    break
        
        return results_list

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
                self.get_utilities_for_simualtion(random_seed, **filter_param)
                results = self.simulate_once(random_seed, **filter_param)
                dict_results[(random_seed, track_filter_type, filter_type, simulation_input_type)].append(results)
        
        self.dict_results = dict_results
        return dict_results
    
    def get_utilities_for_simualtion(self, random_seed: int, **kwargs):
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

        if self.io_data_dict_stored:
            self.io_data_dict = deepcopy(self.io_data_dict_stored)
        else:
            self.io_data_dict_stored = self.get_io_data_dic()
            self.io_data_dict = deepcopy(self.io_data_dict_stored)

        if random_seed not in self.noise_list_dict.keys():
            print(f"Noise with random_seed={random_seed} not generated, generating it online")
            self.noise_list = self.get_noise()
        else:
            self.noise_list = self.noise_list_dict[random_seed]

        self.filter = self.get_filter(**kwargs)
        self.Ts, self.L, self.steps, self.lag, self.slack = Ts, L, steps, lag, slack

    def simulate_once(self, random_seed: int, **kwargs) -> Results:

        self.random_seed = random_seed
        # store parameters, after single simualtion they should be restored
        Ts, L, steps, lag, slack = self.Ts, self.L, self.steps, self.lag, self.slack
    
        self.set_params_from_dict(**kwargs)

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
                if self.save_dataset_after:
                    file_name = self.save_datasets()
                    results.saved_dataset_name = file_name
                results.end_simulation()

                # reset parameters
                self.Ts, self.L, self.steps, self.lag, self.slack = Ts, L, steps, lag, slack
                return results
            results.add_calculation_time(end-start)
            results.add_opt_value(opt_i)
            i_seg = self.filter._i
            results.add_sigma_value(self.filter._safety_filters[i_seg]._sigma.value)
            self.update_predicted_buffer(self.Ts*self.steps*i_block, self.filter_type_list[i_seg], self.filter._safety_filters[i_seg], i_seg, deepcopy(crs_model.state))

            # save predicted trajectory every N_slices blocks
            if (i_block-first_block_to_save)%N_slices == 0 and self.save_predicted_traj_regular:
                # get predicted outputs
                predicted_traj, predicted_with_slack = self.get_predicted_outputs_from_buffer(-1)
                i_seg_buffer = self.buffer_i_seg[-1][1]
                global_initial_state = self.buffer_global_state[-1][1]
                # get and save real output when the proposed inputs are applied
                real_output_list = self.get_real_trajectory_from_proposed_buffer(-1, i_seg_buffer, global_initial_state)
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

                # if constraint not satisfied, save predicted values at previous steps
                A = self.systems[i_seg].A_y[::self.systems[i_seg].p] # only take e_lat
                b = self.systems[i_seg].b_y[::self.systems[i_seg].p]
                x = np.matrix(self.systems[i_seg].state).T
                x = x[:self.filter._safety_filters[i_seg]._p]
                x[2] = x[2] - self.v_0 # from actual velocity to deviation from steady state
                if not np.all(A*x <= b) and not self.out_of_track: # first time constraint not satisfied
                    self.out_of_track = True
                    results.violating_time_steps.append(self.Ts*(i_block*self.steps+j))
                    index_to_save_list = [-1, -5, -10]
                    print(f"Constraint not satisfied at time {self.Ts*(i_block*self.steps+j)}.")
                    try:
                        for index_to_save in index_to_save_list:
                            results.mark_time_steps.append((i_block+1+index_to_save)*self.steps-1)
                            # save predicted trajectory at previous steps
                            predicted_traj, predicted_with_slack = self.get_predicted_outputs_from_buffer(index_to_save)
                            i_seg_buffer = self.buffer_i_seg[index_to_save][1]
                            global_initial_state = self.buffer_global_state[index_to_save][1]
                            # get and save real output when the proposed inputs are applied
                            real_output_list = self.get_real_trajectory_from_proposed_buffer(index_to_save, i_seg_buffer, global_initial_state)
                            results.add_predicted_error_slice(self.buffer_u_value[index_to_save][0], predicted_traj)
                            results.add_predicted_error_slack_slice(self.buffer_u_value[index_to_save][0], predicted_with_slack)
                            results.add_error_slice(self.buffer_u_value[index_to_save][0], real_output_list)
                            results._proposed_input_slices.append(self.buffer_u_value[index_to_save])
                    except Exception as e:
                        print(f"Exception {e} raised during saving constraint violation history.")
                    if self.save_dataset_after:
                        file_name = self.save_datasets()
                        results.saved_dataset_name = file_name
                    if self.stop_after_out_of_track:
                        print(f"Stop simulation after constraint not satisfied!")
                        results.end_simulation()

                        # reset parameters
                        self.Ts, self.L, self.steps, self.lag, self.slack = Ts, L, steps, lag, slack
                        return results
                elif np.all(A*x <= b) and self.out_of_track:
                    print(f"Constraint satisfied again at time {self.Ts*(i_block*self.steps+j)}.")
                    self.out_of_track = False

                # try step global system
                try:
                    crs_model.step(np.array(u_t).flatten())
                except RuntimeError as e:
                    print(f"Exception {e} raised during system simulation, returning partial results")
                    if self.save_dataset_after:
                        file_name = self.save_datasets()
                        results.saved_dataset_name = file_name
                    # calculate root mean square intervention and calcualtion time
                    results.end_simulation()

                    # reset parameters
                    self.Ts, self.L, self.steps, self.lag, self.slack = Ts, L, steps, lag, slack
                    return results

                # save states and inputs for filter
                initial_inputs_filter.append(u_t)
                initial_states_filter.append(crs_model.state)

                # save to results and trajectory, for initial condition of filter later
                results.add_point(u_obj_t, u_t, 
                                  global_state, self.noise_list[i_steps+self.lag], 
                                  error_dynamics_state, np.zeros(error_dynamics_state.shape),
                                  i_seg,)

        # calculate root mean square intervention and calcualtion time
        if self.save_dataset_after:
            file_name = self.save_datasets()
            results.saved_dataset_name = file_name
        results.end_simulation()
        if self.filter_type_list[0] in SafetyFilterTypes.output_hankel_matrix_types:
            results.H_uy, results.H_y_future = self.filter._safety_filters[0].get_datasets_hankel_matrix()

        # reset parameters
        self.Ts, self.L, self.steps, self.lag, self.slack = Ts, L, steps, lag, slack
        return results
    
    def save_datasets(self) -> Optional[str]:
        """Save the list of current datasets to a pickle file"""
        time = f"{datetime.datetime.now():%m-%d-%H-%M-%S}"
        file_name = time + "-" + self.track_fun_name + "-" + self.filter_type_list[0].name + ".pkl"
        io_data_list_list = []
        for single_filter in self.filter._safety_filters:
            io_data_list_list.append(single_filter._io_data_list)
        with open(os.path.join(os.getcwd(), 'datasets', file_name), 'wb') as f:
            pickle.dump(io_data_list_list, f)
        return file_name

    def load_datasets(self, file_name: str, delete_after_load: bool = True) -> bool:
        """
        Load the list of current datasets from a pickle file
        Directly replace current datasets in the filter
        """
        with open(os.path.join(os.getcwd(), 'datasets', file_name), 'rb') as read_file:
                io_data_list_list: List[List[IODataWith_l]] = pickle.load(read_file)
        if len(io_data_list_list) != len(self.filter._safety_filters):
            raise RuntimeError(f"Number of filters and io_data lists are different, cannot load datasets")
        for single_filter, io_data_list in zip(self.filter._safety_filters, io_data_list_list):
            single_filter._io_data_list = io_data_list
        if self.delete_dataset_after and delete_after_load:
            os.remove(os.path.join(os.getcwd(), 'datasets', file_name))

    def get_real_trajectory_from_proposed_buffer(self, i: int, i_seg: int, global_initial_state: np.ndarray) -> List[np.ndarray]:
        """
        get list of outputs if the proposed inputs are applied, with index i
        """
        sys_for_output = self.get_system(cur=self.systems[i_seg].cur, start_point=self.systems[i_seg].segment_start, system_type=self.simulate_model_type)
        sys_for_output.set_kinematic_model_state(global_initial_state)
        real_output_list = []
        u_proposed_list = self.buffer_u_value[i][1]
        for u in u_proposed_list:
            try:
                u = np.matrix(np.reshape(u, (sys_for_output.m, 1)))
                y, e_lin, n = sys_for_output.step_lin(u)
                y = np.array(y + e_lin).flatten()
                y[1] = y[1] * 180 / np.pi # from rad to deg
                y[2] = y[2] + self.v_0 # from deviation from steady state to actual velocity
                real_output_list.append(y)
            except RuntimeError as e:
                print(e, "during simulation of real system, returning partial results")
                break
        return real_output_list

    def get_predicted_outputs_from_buffer(self, i: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        get list of predicted outputs from the buffer, with index i
        return Tuple[list of predicted trajectory, list of predicted trajectory with slack]
        """
        predicted_traj = self.buffer_y_value[i][1]
        predicted_with_slack = [y-s for y,s in zip(self.buffer_y_value[i][1], self.buffer_sigma_value[i][1])]
        # transforms for readability
        for y in predicted_traj:
            y[1] = y[1] * 180 / np.pi # from rad to deg
            y[2] = y[2] + self.v_0 # from deviation from steady state to actual velocity
        for y in predicted_with_slack:
            y[1] = y[1] * 180 / np.pi # from rad to deg
            y[2] = y[2] + self.v_0 # from deviation from steady state to actual velocity
        return predicted_traj, predicted_with_slack

    def separate_proposed_u(self, m: int, filter_type: SafetyFilterTypes, u: np.ndarray) -> List[np.ndarray]:
        """Separate a large array containing proposed inputs into a list of inputs
        m: size of input
        filter_type: type of filter"""
        if filter_type in SafetyFilterTypes.direct_types:
            return np.split(u[m*self.lag:], self.L)
        else:
            return np.split(u, self.L)
    
    def separate_predicted_value(self, p: int, filter_type: SafetyFilterTypes, y: np.ndarray) -> List[np.ndarray]:
        """Separate a large array containing predicted trajectory into a list of outputs
        m: size of input
        filter_type: type of filter"""
        if filter_type in SafetyFilterTypes.direct_types:
            return np.split(y[self.lag*p:], self.L)
        else:
            return np.split(y, self.L)

    def update_predicted_buffer(self, t: float, filter_type: SafetyFilterTypes, safety_filter, i_seg: int, global_state: np.ndarray) -> None:
        """
        update the buffer used to store predicted trajectory and proposed inputs
        params are variables from the filter, will be separated according to the type of filter
        """
        u_value = safety_filter._u.value
        y_value = safety_filter._y.value
        sigma_value = safety_filter._sigma.value
        
        if len(self.buffer_y_value) >= self.num_prediction_buffer:
            self.buffer_y_value.pop(0)
            self.buffer_u_value.pop(0)
            self.buffer_sigma_value.pop(0)
            self.buffer_i_seg.pop(0)
            self.buffer_global_state.pop(0)
        
        self.buffer_y_value.append( (t, self.separate_predicted_value(safety_filter._p, filter_type, y_value)) )
        self.buffer_sigma_value.append( (t, self.separate_predicted_value(safety_filter._p, filter_type, sigma_value)) )
        self.buffer_u_value.append( (t, self.separate_proposed_u(safety_filter._m, filter_type, u_value)) )
        self.buffer_i_seg.append( (t, i_seg) )
        self.buffer_global_state.append( (t, global_state) )

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
                u_obj_k = np.matrix([[throttle_sim],[self.delta_sim*math.sin(t_j*math.pi)]])
            elif self.simulation_input_type == SimulationInputRule.MAX_THROTTLE:
                u_obj_k = np.matrix([[throttle_sim],[0]])
            elif self.simulation_input_type == SimulationInputRule.SINE_WITH_MEAN:
                u_obj_k = np.matrix([
                    [0.4*throttle_sim + throttle_sim*math.sin(t_j*math.pi)],
                    [self.delta_sim*math.sin(t_j*math.pi)]])
            elif self.simulation_input_type == SimulationInputRule.SINE_WITH_MEAN_RANDOM:
                if t_j==0:
                # if True:
                    self.phi_delta = 0.5*math.pi*np.random.rand()
                    self.phi_tau = 0.5*math.pi*np.random.rand()
                    self.omega_delta = 3*np.pi*np.random.rand() + 2*np.pi
                    self.omega_tau = 2*np.pi*np.random.rand() + 2*np.pi
                u_obj_k = np.matrix([
                    [0.4*throttle_sim + throttle_sim*math.sin(t_j*self.omega_tau + self.phi_tau)],
                    [self.delta_sim*math.sin(t_j*self.omega_delta + self.phi_delta)]])
            elif self.simulation_input_type == SimulationInputRule.RANDOM_WITH_MEAN:
                u_obj_k = np.matrix([
                    [0.4*throttle_sim + throttle_sim*np.random.rand()],
                    [self.delta_sim*np.random.rand()]])
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
                    A_y = np.matrix('1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1'),
                    b_y = np.matrix([[half_track_width],[self.mu_max],[self.v_x_max-self.v_0],[half_track_width],[-self.mu_min], [-self.v_x_min+self.v_0]]),
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
                    A_y = np.matrix('1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1'),
                    b_y = np.matrix([[half_track_width],[self.mu_max],[self.v_x_max-self.v_0],[half_track_width],[-self.mu_min], [-self.v_x_min+self.v_0]]),
                    A_n = np.matrix('1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1'),
                    b_n = np.matrix([[self.n_e_lat_max],[self.n_mu_max],[self.n_v_max],[self.n_e_lat_max],[self.n_mu_max],[self.n_v_max]]),
            )
            # for this system, the global initial state does not matter
            return DynamicErrorModelFewOutput(params=system_params, initial_state=np.array([0,0,0,0,0,0]))
        elif system_type == ModelType.DYNAMIC:
            system_params = DynamicErrorModelParams(
                    DynamicModelParams(Ts=self.Ts, l_f=self.l_f, l_r=self.l_r, m=self.m, Iz=self.Iz, Bf=self.Bf, Cf=self.Cf, Df=self.Df, Br=self.Br, Cr=self.Cr, Dr=self.Dr, Croll=self.Croll, Cm1=self.Cm1, Cm2=self.Cm2, Cd=self.Cd),
                    cur = cur,
                    segment_start = start_point,
                    v_0 = self.v_0,
                    A_u = np.matrix('1 0; 0 1; -1 0; 0 -1'),
                    b_u = np.matrix([[self.a_max*self.m],[self.delta_max],[-self.a_min*self.m],[self.delta_max]]),
                    A_y = np.matrix('1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1; -1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 -1'),
                    b_y = np.matrix([[half_track_width],[self.mu_max],[self.v_x_max-self.v_0], [self.v_y_max], [half_track_width],[-self.mu_min],[-self.v_x_min+self.v_0],[-self.v_y_min]]),
                    A_n = np.matrix('1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1; -1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 -1'),
                    b_n = np.matrix([[self.n_e_lat_max],[self.n_mu_max],[self.n_v_max/1.2],[self.n_v_max/1.2],[self.n_e_lat_max],[self.n_mu_max],[self.n_v_max/1.2],[self.n_v_max/1.2]]),
            )
            # for this system, the global initial state does not matter
            return DynamicErrorModelVxy(params=system_params, initial_state=np.array([0,0,0,0,0,0]))
    
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
                    filter_list.append(self.get_single_filter(0, filter_type, system, deepcopy(self.io_data_dict[0]), **kwargs))
                elif isinstance(segment, arc):
                    if segment.theta_f > segment.theta_s: # left hand turn
                        cur = segment.curvature
                    elif segment.theta_f < segment.theta_s: # right hand turn
                        cur = -segment.curvature
                    filter_list.append(self.get_single_filter(0, filter_type, system, deepcopy(self.io_data_dict[cur]), **kwargs))
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
                    filter_list.append(self.get_single_filter(0, filter_type, system, deepcopy(self.io_data_dict[0]), **kwargs))
                elif isinstance(segment, arc):
                    if segment.theta_f > segment.theta_s: # left hand turn
                        cur = segment.curvature
                    elif segment.theta_f < segment.theta_s: # right hand turn
                        cur = -segment.curvature
                    filter_list.append(self.get_single_filter(0, filter_type, system, deepcopy(self.io_data_dict[cur]), **kwargs))
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
        else:
            return [np.array([
                                self.n_x*(2*np.random.rand()-1),
                                self.n_y*(2*np.random.rand()-1),
                                self.n_psi*(2*np.random.rand()-1),
                                self.n_v*(2*np.random.rand()-1)/1.2,
                                self.n_v*(2*np.random.rand()-1)/1.2,
                                0,
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
