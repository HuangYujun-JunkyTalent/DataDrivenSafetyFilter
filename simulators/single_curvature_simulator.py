from typing import List, Dict, Any
import math
import os
import pickle
from enum import Enum
from copy import deepcopy
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from IOData.IOData import IOData, InputRule
from IOData.IODataWith_l import IODataWith_l
from System.ErrorKinematicAcceLATI import LinearizedErrorKinematicAcceModel, LinearizedErrorKinematicAcceModelParams, KinematicAcceModelParams
from tools.simualtion_results import Results
from tools.simple_track_generator import trackGenerator

from SafetyFilters.DDSF import SFParams
from SafetyFilters.Indirect_Nominal_Fitting import IndirectNominalFittingFilter, IndirectNominalFittingParams
from SafetyFilters.Indirect_Nominal_FixMu import IndirectNominalFixMuFilter, IndirectNominalFixMuParams
from SafetyFilters.Indirect_Nominal_Stop import IndirectNominalStopFilter, IndirectNominalStopParams
from SafetyFilters.Indirect_Nominal_Zero import IndirectNominalZeroFilter, IndirectNominalZeroParams
from SafetyFilters.Indirect_Nominal_ZeroV import IndirectNominalZeroVFilter, IndirectNominalZeroVParams
from SafetyFilters.Direct_Nominal_Zero import DirectNominalZeroFilter, DirectNominalZeroParams

from simulators.simulation_settings import SafetyFilterTypes, SimulationInputRule


class SingleCurvatureSimulator:
    """Simulator for a single curvature
    """
    # parameters for simualtion
    # parameters for kinematic model
    l_r = 0.052
    l_f = 0.038
    # sampling time
    Ts = 1/100
    # control and output constraints
    a_min, a_max = -3, 6
    delta_max = 0.35*math.pi
    v_min, v_max = -3, 5
    mu_min, mu_max = -0.5*math.pi, 0.5*math.pi

    # track parameters
    cur = 1/0.3
    track_width = 0.5
    track_start = [0,0,0] # [x_p0, y_p0, Psi_0]
    # steady state speed
    v_0 = 1.5

    # noise level for simulation
    n_e_lat_max = 0.002
    n_mu_max = 0.01
    n_v_max = 0.005
    n_l_max = 0.002

    use_saved_data = True # whether to use saved dataset
    # maximum control inputs for collecting dataset
    a_d_max = 4
    delta_d_max = 2e-2
    t_data = 10.0 # time horizon for dataset, in seconds

    # random seed for generating noise
    random_seed = 0
    # intial state of system
    global_initial_state = np.array([0,0,0,0]) # [x_p0, y_p0, v_0, Psi_0]
    simulation_input_type = SimulationInputRule.SINE_WAVE
    # maximum control inputs for simulation
    a_sim = 10
    delta_sim = 0.12*math.pi
    # simulation time
    t_sim = 3
    num_predicted_traj = 5 # number of predicted trajectories to be saved

    # filter parameters, those are important also for simulation
    filter_type = SafetyFilterTypes.INDIRECT_FITTING_TERMINAL
    L = 50 # prediction horizon
    lag = 9 # steps used for initial condition of filter
    steps = 3 # number of inputs applied to the system for each filtering

    def get_filter_params(self, **kargs):
        half_track_width = self.track_width/2
        epsilon = max(self.n_e_lat_max, self.n_mu_max, self.n_v_max)
        if self.filter_type == SafetyFilterTypes.INDIRECT_FITTING_TERMINAL:
            lam_sig = 50000
            c = [[0.3, 0.1, 0.05, 0.01], [0.3, 0.1, 0.05, 0.01]]
            R = np.matrix('1 0; 0 100')
            lam_sig = kargs.get('lam_sig', lam_sig)
            c = kargs.get('c', c)
            R = kargs.get('R', R)
            epsilon = kargs.get('epsilon', epsilon)

            filter_params = IndirectNominalFittingParams(
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
        elif self.filter_type == SafetyFilterTypes.INDIRECT_FIX_MU:
            lam_sig = 50000
            c = [[0.3, 0.1, 0.05, 0.01], [0.3, 0.1, 0.05, 0.01]]
            R = np.matrix('1 0; 0 100')
            lam_sig = kargs.get('lam_sig', lam_sig)
            c = kargs.get('c', c)
            R = kargs.get('R', R)
            epsilon = kargs.get('epsilon', epsilon)

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
        elif self.filter_type == SafetyFilterTypes.INDIRECT_ZERO_V:
            lam_sig = 50000
            c = [[0.3, 0.1, 0.05, 0.01], [0.3, 0.1, 0.05, 0.01]]
            R = np.matrix('1 0; 0 100')
            lam_sig = kargs.get('lam_sig', lam_sig)
            c = kargs.get('c', c)
            R = kargs.get('R', R)
            epsilon = kargs.get('epsilon', epsilon)

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
        elif self.filter_type == SafetyFilterTypes.INDIRECT_ZERO:
            lam_sig = 50000
            c = [[0.3, 0.1, 0.05, 0.01], [0.3, 0.1, 0.05, 0.01]]
            R = np.matrix('1 0; 0 100')
            lam_sig = kargs.get('lam_sig', lam_sig)
            c = kargs.get('c', c)
            R = kargs.get('R', R)
            epsilon = kargs.get('epsilon', epsilon)

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
        elif self.filter_type == SafetyFilterTypes.INDIRECT_STOP:
            lam_sig = 50000
            c = [[0.3, 0.1, 0.05, 0.01], [0.3, 0.1, 0.05, 0.01]]
            R = np.matrix('1 0; 0 100')
            lam_sig = kargs.get('lam_sig', lam_sig)
            c = kargs.get('c', c)
            R = kargs.get('R', R)
            epsilon = kargs.get('epsilon', epsilon)

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
        elif self.filter_type == SafetyFilterTypes.DIRECT_ZERO_TERMINAL:
            lam_sig = 1500
            lam_alph = 1500/epsilon
            c = [[0.35, 0.1, 0.1, 0.05], [0.4, 0.1, 0.1, 0.05]]
            R = np.matrix('1 0; 0 100')
            lam_alph = kargs.get('lam_alph', lam_alph)
            lam_sig = kargs.get('lam_sig', lam_sig)
            c = kargs.get('c', c)
            R = kargs.get('R', R)
            epsilon = kargs.get('epsilon', epsilon)

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

    def get_input_obj(self, t: float, xi_t: np.matrix) -> np.matrix:
        """get next `steps` inputs

        :param t: simulation time
        :param xi_t: extended state of system
        :return: proposed u_obj for next `steps`
        """
        u_obj = np.matrix(np.ndarray((0,1)))
        for j in range(self.steps):
            t_j = t + j*self.Ts
            if self.simulation_input_type == SimulationInputRule.SINE_WAVE:
                u_obj_k = np.matrix([[self.a_sim*math.sin(t_j*math.pi)],[self.delta_sim*math.sin(3*t_j*math.pi)]])
            elif self.simulation_input_type == SimulationInputRule.MAX_THROTTLE_SINE_STEER:
                u_obj_k = np.matrix([[self.a_sim],[self.delta_sim*math.sin(3*t_j*math.pi)]])
            u_obj = np.vstack( (u_obj, u_obj_k) )
        return u_obj

    def simulate_multi(self,
                       random_seends: List[float],
                       filter_types: List[SafetyFilterTypes],
                       filter_params: Dict[SafetyFilterTypes, List[Dict[str, Any]]],
                       simualtion_input_types: List[SimulationInputRule],) -> Dict[Any, List[Results]]:
        dict_results = {}
        for random_seed, filter_type, simulation_input_type in product(random_seends, filter_types, simualtion_input_types):
            self.filter_type = filter_type
            self.simulation_input_type = simulation_input_type
            dict_results[(random_seed, filter_type, simulation_input_type)] = []
            for add_filter_param in filter_params.get(filter_type, [{}]):
                dict_results[(random_seed, filter_type, simulation_input_type)].append(self.simulate_once(random_seed, **add_filter_param))
        
        self.dict_results = dict_results
        return dict_results

    def simulate_once(self, random_seed=0, **kwargs) -> Results:
        """Simulate the system under given controller, dataset and safety filter
        """
        # np.random.seed(0)
        self.set_params_from_dict(**kwargs)
        self.system = self.get_system()

        self.random_seed = random_seed
        self._noise_trajectory = self.get_noise(self.system)
        
        # note: self.system is needed to generate new dataset
        self.io_data = self.get_io_data()
        # self.data_length = self.io_data.length
        # self.t_data = self.data_length*self.Ts
        
        filter_params = self.get_filter_params(**kwargs)
        FilterClass = {SafetyFilterTypes.INDIRECT_FITTING_TERMINAL: IndirectNominalFittingFilter,
                        SafetyFilterTypes.INDIRECT_FIX_MU: IndirectNominalFixMuFilter,
                        SafetyFilterTypes.INDIRECT_ZERO_V: IndirectNominalZeroVFilter,
                        SafetyFilterTypes.INDIRECT_ZERO: IndirectNominalZeroFilter,
                        SafetyFilterTypes.INDIRECT_STOP: IndirectNominalStopFilter,
                        SafetyFilterTypes.DIRECT_ZERO_TERMINAL: DirectNominalZeroFilter,
                        }.get(self.filter_type, IndirectNominalFittingFilter)
        self.filter = FilterClass(self.system, self.io_data, filter_params)
        
        # prepare log and block number for simulation
        N = int(self.t_sim/self.Ts)
        N_blocks = int(N/self.steps)
        N_slices = int(N_blocks/self.num_predicted_traj) # save predicted trajectory every N_slices blocks
        initial_state = self.system.state
        initial_y = np.matrix(initial_state[:self.system.p]-np.array([0,0,self.v_0])).transpose()
        traj_u: List[np.matrix] = [np.matrix([[0] for __ in range(self.system.m)]) for _ in range(self.lag)]
        traj_y: List[np.matrix] = [initial_y for _ in range(self.lag)]
        # traj_noise: List[np.matrix] = [np.matrix([[0] for __ in range(self.system.p)]) for _ in range(self.lag)]
        results = Results(self.Ts)

        # simulate
        for i in range(N_blocks):
            # construct extended system state
            k = i*self.steps
            xi_i = np.vstack(tuple( traj_u[k:k+self.lag] + [traj_y[_] + self._noise_trajectory[_] for _ in range(k,k+self.lag)] ))
            # get next input objectives
            u_obj = self.get_input_obj(i*self.steps*self.Ts, xi_i)

            # apply safety filter
            u_i, status_i, opt_i = self.filter.filter(xi_i, u_obj)

            # save predicted trajectory every N_slices blocks
            if i%N_slices == 0:
                if self.filter_type == SafetyFilterTypes.DIRECT_ZERO_TERMINAL:
                    predicted_traj = np.split(self.filter._y.value[self.lag*self.filter._p:], self.L)
                else:
                    predicted_traj = np.split(self.filter._y.value, self.L)
                # transforms for readability
                for y in predicted_traj:
                    y[1] = y[1] * 180 / np.pi # from rad to deg
                    y[2] = y[2] + self.v_0 # from deviation from steady state to actual velocity
                results.add_predicted_error_slice(i*self.steps*self.Ts, predicted_traj)

            # save and update system state, for `step` times
            for j in range(self.steps):
                # save system states and inputs first
                global_state = deepcopy(self.system.kinematic_model.state)
                error_dynamics_state = deepcopy(self.system.state)
                global_state[2] = global_state[2] * 180 / np.pi # from rad to deg
                error_dynamics_state[1] = error_dynamics_state[1] * 180 / np.pi # from rad to deg
                u_obj_t = u_obj[j*self.system.m:(j+1)*self.system.m]
                u_t = u_i[j*self.system.m:(j+1)*self.system.m]

                # step the system
                y_t, e_lin_t, n_t = self.system.step_lin(u_t)

                # save to results and trajectory
                results.add_point(np.array(u_obj_t).flatten(), np.array(u_t).flatten(), 
                                  global_state, np.zeros(global_state.shape), 
                                  error_dynamics_state, np.zeros(error_dynamics_state.shape),)
                traj_y.append(y_t+e_lin_t)
                traj_u.append(u_t)
                # traj_noise.append(n_t)
        
        return results

    def get_track_generator(self, density: int) -> trackGenerator:
        gen = trackGenerator(density, self.track_width)
        if self.cur == 0:
            # get distance traveled by vehicle
            l_list = [error_state[-1] for error_state in self.results._error_trajectory]
            dist = max(l_list) - min(l_list)
            gen.addLine(x0=self.track_start[0], y0=self.track_start[1],
                        x1=self.track_start[0]+np.cos(self.track_start[2]), y1=self.track_start[1]+np.sin(self.track_start[2]),
                        dist=dist)
        elif self.cur > 0:
            gen.left_turn(self.track_start, 1/self.cur, 2*np.pi)
        else:
            gen.left_turn(self.track_start, -1/self.cur, 2*np.pi)
        gen.populatePointsAndArcLength()

        return gen

    def get_system(self) -> LinearizedErrorKinematicAcceModel:
        half_track_width = self.track_width/2
        system_params = LinearizedErrorKinematicAcceModelParams(
                KinematicAcceModelParams(l_r=self.l_r,l_f=self.l_f,Ts=self.Ts),
                cur = self.cur,
                state_s0 = self.track_start,
                v_0 = self.v_0,
                A_u = np.matrix('1 0; 0 1; -1 0; 0 -1'),
                b_u = np.matrix([[self.a_max],[self.delta_max],[-self.a_min],[self.delta_max]]),
                # A_y = np.matrix('1 0 0 0; 0 1 0 0; -1 0 0 0; 0 -1 0 0'),
                A_y = np.matrix('1 0 0; 0 1 0; -1 0 0; 0 -1 0'),
                # b_y = np.matrix([[t],[t]]),
                b_y = np.matrix([[half_track_width],[self.mu_max],[half_track_width],[-self.mu_min]]),
                A_n = np.matrix('1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1'),
                # b_n = np.matrix([[n_x_max],[n_y_max],[n_psi_max],[n_v_max],[n_x_max],[n_y_max],[n_psi_max],[n_v_max]]),
                # b_n = np.matrix([[n_x_max],[n_y_max],[n_psi_max],[n_x_max],[n_y_max],[n_psi_max]]),
                b_n = np.matrix([[self.n_e_lat_max],[self.n_mu_max],[self.n_v_max],[self.n_e_lat_max],[self.n_mu_max],[self.n_v_max]]),
                # b_n = np.matrix([[n_e_lat_max],[n_mu_max],[n_e_lat_max],[n_mu_max]]),
                # b_n = np.matrix([[n_e_lat_max],[n_e_lat_max]]),
            )
        return LinearizedErrorKinematicAcceModel(params=system_params, state_0=self.global_initial_state)
    
    def get_io_data(self) -> IOData:
        if self.use_saved_data: # use saved data
            with open(os.path.join(os.getcwd(), 'datasets', f'io_datas_{self.t_data}_{self.Ts}.pkl'), 'rb') as read_file:
                io_datas: Dict[float, IODataWith_l] = pickle.load(read_file)
            io_data = io_datas[self.cur]
            io_data.update_depth(self.lag+self.L)
            io_data.update_l_estimation_matrix(self.L, self.lag)
            self.data_length = io_data.length
        else: # generate new data
            np.random.seed(self.random_seed)
            self.data_length = int(self.t_data/self.Ts)
            # store system state, need to be reset afterwards
            global_system_state = self.system.kinematic_model.state

            # collect dataset around steady state, use additional random inputs
            steady_input = self.system.get_zero_input()
            steady_state = self.system.get_zero_state()
            self.system.set_error_state(steady_state)
            io_data = IODataWith_l(self.L+self.lag,
                                sys=self.system,
                                input_rule=InputRule.RANDOM_2_WITH_MEAN,
                                mean_input=np.matrix(steady_input).transpose(),
                                length=self.data_length,
                                A_u_d=np.matrix('1 0; 0 1; -1 0; 0 -1'),
                                b_u_d=np.matrix([[self.a_d_max],[self.delta_d_max],[self.a_d_max],[self.delta_d_max]]),
                                n_l = self.n_l_max, lag = self.lag, L = self.L, N_l = int(self.data_length/2), K_l = 5)
            # remember to reset system state
            self.system.set_kinematic_model_state(global_system_state)
        return io_data
    
    def get_noise(self, system: LinearizedErrorKinematicAcceModel) -> List[np.matrix]:
        length = self.lag + int(self.t_sim/self.Ts)
        np.random.seed(self.random_seed)
        return [system.get_noise() for _ in range(length)]
    
    def set_params_from_dict(self, **kwargs) -> None:
        self.Ts = kwargs.get('Ts', self.Ts)
        self.L = kwargs.get('L', self.L)
