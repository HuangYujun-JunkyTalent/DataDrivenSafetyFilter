from typing import Tuple, List, Callable, Optional, Any, Dict
from dataclasses import dataclass, field
from itertools import accumulate
from warnings import warn

import numpy as np
import numpy.linalg as npl
from scipy.linalg import block_diag
import cvxpy as cp
import pytope as pt

# from SafetyFilters.DDSF import DDSafetyFilter, SFParams
from IOData.IOData import IOData, HankelMatrix
from IOData.IODataWith_l import IODataWith_l
from System.LATI import LATI
from System.DynamicErrorModel import DynamicErrorModelVxyL

# define the new version with fewer slack variables
# Robust SF with fewer slack variables
@dataclass
class IndirectFixMuMaxLParams:
    # v_0: float # steady state velocity, for deciding size of l
    # Ts: float # sampling time, for deciding size of l

    L: int # prediction horizon
    lag: int # approximation of lag of system
    # R: np.matrix
    # lam_alph: float
    lam_sig: float
    # epsilon: float
    c: List[List[float]]
    t_new_data: float = 6.0

    W_xi: np.matrix = np.matrix(np.eye(1))
    min_dist: float = 0.1
    min_num_slices: int = 300
    min_portion_slices: float = 0.1
    f: Callable[[float], float] = lambda x: 1/x**2
    use_zero_l_initial: bool = False
    # steady_input: Optional[np.ndarray] = None
    # steady_output: Optional[np.ndarray] = None

    steps: int = 1 # number of steps forward
    solver: Optional[Any] = None
    solver_args: Dict = field(default_factory=dict)
    warm_start: bool = False
    verbose: bool = False
    # use_estimated_c_pe: bool = False


class IndirectFixMuMaxLController:
    """
    Data-driven controller with indirect approach, terminal constraint Fix Mu and other variables to be the same, and objective to maximize progress along track center l
    """
    params: IndirectFixMuMaxLParams
    @property
    def _lag(self):
        return self.params.lag
    v_0: float # steady state velocity, for deciding size of l
    Ts: float # sampling time, for deciding size of l
    _m: int
    _p: int
    _n: int # system parameters

    _y: cp.Variable # output variable
    # _alpha: cp.Variable # slack variables
    _sigma: cp.Variable # slack variables for compensating noise
    _io_data_list: List[IODataWith_l] # list of data trajectories
    # _epsilon: float # noise level
    # _R: np.matrix
    _c_sum: List[List[float]] # tightening constants

    y_max: np.ndarray # maximum output
    y_min: np.ndarray # minimum output
    u_max: np.ndarray # maximum input
    u_min: np.ndarray # minimum input

    _u_s: np.ndarray # steady state input
    _y_s: np.ndarray # steady state output, estimated from data

    _constraints: List # list of constraints that will not be changed
    _objective: cp.Minimize # objective function that will not be changed

    # _a1: List[float]
    # @property
    # def a1(self):
    #     return self._a1
    # _a2: List[float]
    # @property
    # def a2(self):
    #     return self._a2
    # _a3: List[float]
    # @property
    # def a3(self):
    #     return self._a3
    # _a4: List[float]
    # @property
    # def a4(self):
    #     return self._a4
    
    def __init__(self,
                 sys: DynamicErrorModelVxyL,
                 io_data: IOData,
                 params: IndirectFixMuMaxLParams,
                ) -> None:
        self.params = params
        self.v_0 = sys.v_0
        self.Ts = sys.Ts
        # steady_input = params.steady_input
        # steady_output = params.steady_output
        self._c_sum = [list(accumulate(list_i)) for list_i in params.c]
        self._io_data_list = [io_data]
        self._m = sys.m
        self._p = sys.p
        self._n_y = sys.p - 1 # number of constrainted outputs
        self._n = sys.n

        # parameters for estimation matrix Phi
        y_max_lateral = np.array(sys.b_y[:self._n_y]).flatten()
        y_min_lateral = -np.array(sys.b_y[self._n_y:self._n_y*2]).flatten()
        self.y_max = np.append(y_max_lateral, self.Ts*self.v_0*self.params.L)
        self.y_min = np.append(y_min_lateral, 0)

        self.u_max = np.array(sys.b_u[:sys.m]).flatten()
        self.u_min = -np.array(sys.b_u[sys.m:2*sys.m]).flatten()
        # self.min_num_slices = params.min_num_slices
        # self.min_portion_slices = params.min_portion_slices
        # self.min_dist = params.min_dist
        # self.f = params.f

        # self._t_new_data = params.t_new_data
        self._new_data_count = 0 # count the number of new data points added
        # assert self._L >= 2*self._n, f"Prediction horizon too short! System order is {self._n}, horizon given is {self._L}."
        # self._lag = lag
        # self._R = R
        # self._lam_alph = params.lam_alph
        # self._lam_sig = params.lam_sig
        # if params.sf_params.steps != self._n:
        #     warn(f"Given steps {params.sf_params.steps}!")
        # DDSafetyFilter.__init__(self, params.sf_params)

        # if steady_input is not None:
        #     self._u_s = steady_input
        #     self._y_s = steady_output
        # else:
        #     self._u_s, self._y_s = self.fit_steady_state(3,5, int(self._io_data.length/3), 8)
        #     print(f"Fited steady state input: {self._u_s}")
        #     print(f"Fited steady state output: {self._y_s}")
        # self._y_s = np.array(sys._C@self._y_s).flatten()
        # self._u_s = np.zeros((self._m,1))
        # self._y_s = np.zeros((self._p,1))
        # self._y_s[2] = -sys._v_0

        # self._io_data_list[0].update_depth(self._L+self._lag)

        # self._epsilon = params.epsilon

        # initialize cvxpy variables:
        # width_H = self._io_data_list[0].H_input.shape[1]
        self._u = cp.Variable(shape=(self._m*(self.params.L),))
        self._y = cp.Variable(shape=(self._p*(self.params.L),))
        # self._alpha = cp.Variable(shape=(width_H,))
        self._sigma = cp.Variable(shape=((self._p)*(self.params.L),)) # we use fewer slack variables
        # self._sigma = cp.Variable(shape=(self._p*(self._L-l_terminal),)) # we use fewer slack variables

        # initialize cvxpy parameters:
        # self._u_obj = cp.Parameter(shape=(self._m*self._steps,)) # we need m steps of objective inputs
        # self._xi_t = cp.Parameter(shape=(self.params.lag*(self._m+self._p),))
        # self._xi_t = cp.Parameter(shape=(self.params.lag*(self._m+self._p)-1,))

        # # calcualte constant tightenging constants a_1-a_4
        # c_pe: float = 0
        # if full_slack_sf_params.use_estimated_c_pe:
        #     c_pe = npl.norm(npl.pinv(self._io_data.H_u_xi_noised), 1)
        # else:
        #     c_pe = npl.norm(npl.pinv(self._io_data.H_u_xi), 1)
        # a1, a2, a3, a4 = self.constants_for_tightening(sys, c_pe)
        # self._a1, self._a2, self._a3, self._a4 = a1, a2, a3, a4
        # # print(np.max(a1), np.max(a2), np.max(a3), np.max(a4))

        # objective function
        # R_n = block_diag(*( (self._R,)*self._steps ))
        # self._objective = cp.quad_form(self._u[0:self._m*(self._steps)]-self._u_obj, R_n)  # error term 
        # obj = obj + self._epsilon*self._lam_alph * cp.norm(self._alpha, 2) # penalty term
        # obj = obj + self._lam_sig * cp.norm(self._sigma, 2) # penalty term
        objective_fuc = - cp.sum(self._y[self._p-1::self._p]) # maximize final progress along track center l
        # objective_fuc =  - cp.sum(self._y[2:self.params.steps*self._p:self._p])
        # objective_fuc =  - self._y[self._p*self.params.steps-1]
        objective_fuc = objective_fuc + self.params.lam_sig * cp.sum_squares(self._sigma) # penalty term
        # self._objective = self._objective + self._lam_sig * cp.norm_inf(self._sigma) # penalty term
        self._objective = cp.Minimize(objective_fuc)

        # constraints
        # constraints = [self._io_data.H_input @ self._alpha == self._u]
        # constraints.append(self._io_data.H_output_noised @ self._alpha == self._y - self._sigma) # dynamic constraints including slack varialbes

        #Computing alpha directly from inverse of H_uy_noised
        # H_uy_noised: np.matrix = np.vstack( (self._io_data.H_input, self._io_data.H_output_noised_part((0, self._lag)), np.ones((1, width_H))) )
        # H_uy_noised_inv = npl.pinv(H_uy_noised)
        # A = self._io_data.H_output_noised_part((self._lag, self._lag+self._L)) @ H_uy_noised_inv
        # constraints.append(self._alpha == npl.pinv(H_uy_noised) @ cp.hstack((self._u, self._y[:self._lag*self._p])))
        
        # constraints.append(self._u[:self._lag*self._m] == self._xi_t[:self._lag*self._m]) # initial input constraints
        # constraints.append(self._y[:self._p*self._lag] == self._xi_t[-self._p*self._lag:]) # initial output constraints
        # constraints.append(self._u[-self._m:] == self._u[-2*self._m:-self._m])
        # constraints.append(self._y[-self._p:] == self._y[-2*self._p:-self._p]) # -1==-2
        # for i in range(2,self._n+10): # -2 == -3, ..., -(n-1) == -n
        #     constraints.append(self._u[-i*self._m:-(i-1)*self._m] == self._u[-(i+1)*self._m:-i*self._m])
        #     constraints.append(self._y[-i*self._p:-(i-1)*self._p] == self._y[-(i+1)*self._p:-i*self._p])
        # terminal constraints, fix acceleration to zero, velocity to steady state value
        constraints = []
        l_terminal = self.params.steps
        if self._p == 4: # system with few output
            constraints.append(self._y[-self._p*l_terminal+2::self._p] == np.zeros((l_terminal))) # v
            constraints.append(self._u[-self._m*l_terminal::self._m] == np.zeros((l_terminal))) # throttle
            for i in range(1,l_terminal):
                constraints.append(self._y[-self._p*i+1] == self._y[-self._p*(i+1)+1]) # mu
                constraints.append(self._y[-self._p*i] == self._y[-self._p*(i+1)]) # e_lat 
                constraints.append(self._u[-self._m*i+1] == self._u[-self._m*(i+1)+1]) # delta
        elif self._p == 5: # system with more output
            constraints.append(self._y[-self._p*l_terminal+2::self._p] == np.zeros((l_terminal))) # v_x
            constraints.append(self._u[-self._m*l_terminal::self._m] == np.zeros((l_terminal))) # throttle
            for i in range(1,l_terminal):
                constraints.append(self._y[-self._p*i+1] == self._y[-self._p*(i+1)+1]) # mu
                constraints.append(self._y[-self._p*i] == self._y[-self._p*(i+1)]) # e_lat 
                constraints.append(self._y[-self._p*i+3] == self._y[-self._p*(i+1)+3]) # v_y
                constraints.append(self._u[-self._m*i+1] == self._u[-self._m*(i+1)+1]) # delta

        l_constrained = self.params.L
        if sys.A_y is not None:
            A_y_used = sys.A_y
            P_y_list: List[pt.Polytope] = [] # list of tightened polytopes
            P_y = pt.Polytope(A_y_used, np.array(sys.b_y))
            if (l_constrained) % self.params.steps == 0:
                J = int((l_constrained)/self.params.steps)
            else:
                J = int((l_constrained)/self.params.steps) + 1
            for j in range(J):
                bound = np.zeros(0)
                for i in range(self._p):
                    if i<len(self._c_sum):
                        bound_i_list = self._c_sum[i]
                    else:
                        bound_i_list = self._c_sum[-1]
                    if j<len(bound_i_list):
                        bound = np.append(bound, bound_i_list[j])
                    else:
                        bound = np.append(bound, bound_i_list[-1])
                Delta_j = pt.Polytope(lb=-bound, ub=bound)
                P_j = P_y-Delta_j
                P_j.minimize_H_rep()
                P_y_list.append(P_j)
        for k in range(l_constrained):
            i = k
            if sys.A_u is not None:
                constraints.append(sys.A_u @ self._u[i*self._m:(i+1)*self._m] <= np.array(sys.b_u.flat))
                # constraints.append(cp.norm_inf(self._y[i*self._p:(i+1)*self._p]) + a1[k]*cp.norm(self._u,1) + a2[k]*cp.norm(self._alpha,1) + a3[k]*cp.norm_inf(self._sigma) + a4[k] <= sys.b_u[0,0]) # tightended output constraint, only support the form ||y||_inf <= y_max here!
            if sys.A_y is not None:
                j = int(k/self.params.steps)
                constraints.append(P_y_list[j].A@(self._y[i*self._p:(i+1)*self._p]) <= P_y_list[j].b.flatten())

        constraints.append(self._sigma[self._p-1::self._p]==np.zeros((self.params.L,))) # initial progress along track center l
        self._constraints = constraints

        self._first_io_data_poped = False
        
        # A = self.get_estimation_matrix()
        
        # constraints.append(A[:,self._lag*self._m:-self._p*self._lag-1] @ self._u == \
        #                         self._y - self._sigma - \
        #                         A[:,:self._lag*self._m]@self._xi_t[:self._lag*self._m] - \
        #                         A[:,-self._p*self._lag-1:-1]@self._xi_t[-self._p*self._lag:] - \
        #                         np.array((A[:,-1:]@np.ones((1,)))).squeeze())

        # self._opt_prob = cp.Problem(self._objective, constraints=constraints)

    def filter(self, xi_t: np.matrix, u_obj: np.matrix, set_new_dataset: bool = False) \
        -> Tuple[np.matrix, str, float]:
        """Return tuple of (filtered control input, status, optimal value)
        set_new_dataset: if True, creat a new dataset for next trajectories
        
        Note: xi_t contains the progress along track center, but it will not be used in the filter"""
        if set_new_dataset and self._new_data_count*self.Ts < self.params.t_new_data:
            self._io_data_list.append(IODataWith_l(depth=self.params.L+self.params.lag, n=self._n, m=self._m, p=self._p)) # create a new dataset
        
        # add new data points, pop old data points
        u_history = xi_t[:self.params.lag*self._m]
        u_history = u_history[-self.params.steps*self._m:]
        y_history = xi_t[-self.params.lag*self._p:]
        y_history = y_history[-self.params.steps*self._p:]
        if self._new_data_count*self.Ts < self.params.t_new_data:
            for i in range(self.params.steps):
                u = u_history[i*self._m:(i+1)*self._m]
                y = y_history[i*self._p:(i+1)*self._p]
                n = np.matrix(np.zeros(y.shape))
                self._io_data_list[-1].add_point(u, y, n)
                # is_empty = False
                # if self._io_data_list[-1].length > self._L+self._lag:
                #     is_empty = not self._io_data_list[0].remove_last_point()
            # if self._io_data_list[0].length < self._L+self._lag: # if a dataset is empty, remove it
            #     self._io_data_list.pop(0)
            #     self._first_io_data_poped = True
            self._new_data_count += self.params.steps

        # set progress l to start from 0
        if self.params.use_zero_l_initial:
            l_0 = xi_t[self.params.lag*self._m+self._p-1]
            xi_t[self.params.lag*self._m+self._p-1::self._p] = xi_t[self.params.lag*self._m+self._p-1::self._p] - l_0
        # get estimation matrix from current dataset list
        Phi = self.get_estimation_matrix(xi_t)

        # dynamic constraint, remember to use proper _p to exclude progress along track center l
        if self.params.use_zero_l_initial:
            xi_t = np.delete(xi_t, self.params.lag*self._m+self._p-1, axis=0)
            self._constraints.append(Phi[:,self.params.lag*self._m:-self._p*self.params.lag] @ self._u + \
                                    np.array(Phi[:,:self.params.lag*self._m]@xi_t[:self.params.lag*self._m]).flatten() + \
                                    np.array(Phi[:,-self._p*self.params.lag:-1]@xi_t[-self._p*self.params.lag+1:]).flatten() + \
                                    np.array((Phi[:,-1:]@np.ones((1,)))).squeeze() == \
                                    self._y - self._sigma)
        else:
            self._constraints.append(Phi[:,self.params.lag*self._m:-self._p*self.params.lag-1] @ self._u + \
                                 np.array(Phi[:,:self.params.lag*self._m]@xi_t[:self.params.lag*self._m]).flatten() + \
                                 np.array(Phi[:,-self._p*self.params.lag-1:-1]@xi_t[-self._p*self.params.lag:]).flatten() + \
                                 np.array((Phi[:,-1:]@np.ones((1,)))).squeeze() == \
                                self._y - self._sigma)

        self._opt_prob = cp.Problem(objective=self._objective, constraints=self._constraints)

        # xi_t_used = np.delete(xi_t, slice(self._lag*self._m+self._p-1, None, self._p))
        # self._xi_t.value = np.array(xi_t.flat)
        # self._u_obj.value = np.array(u_obj.flat)
        if self.params.solver is None:
            self._opt_prob.solve(verbose=self.params.verbose, warm_start = self.params.warm_start, **self.params.solver_args)
        else:
            self._opt_prob.solve(verbose=self.params.verbose, solver=self.params.solver, warm_start = self.params.warm_start, **self.params.solver_args)
        if not (self._opt_prob.status is cp.OPTIMAL or self._opt_prob.status is cp.OPTIMAL_INACCURATE):
            raise Exception(f"Problem not optimal: status is {self._opt_prob.status}")
        
        # pop old dynamic constraint
        self._constraints.pop(-1)

        return (np.matrix(self._u.value[:self._m*self.params.steps]).transpose(), self._opt_prob.status, self._opt_prob.value)
    
    def get_estimation_matrix(self, xi_t: np.matrix) -> np.matrix:
        """
        Return estimation matrix Phi
        Remember to set progress l to start from 0 for every trajectory slice
        """

        # get dataset matrix without progress along track center l by deleting corresponding rows
        H_uy_noised: np.matrix = np.matrix(np.zeros(( self._p*self.params.lag+self._m*(self.params.L+self.params.lag),0 )))
        H_future_noised: np.matrix = np.matrix(np.zeros(( self._p*self.params.L,0 )))
        for io_data in self._io_data_list:
            if io_data.length >= self.params.L+self.params.lag: # only use data with enough length
                io_data.update_depth(self.params.L+self.params.lag)

                H_output_noised_initial = io_data.H_output_noised_part((0, self.params.lag))

                H_future_noised_single = io_data.H_output_noised_part((self.params.lag, self.params.lag+self.params.L))

                H_uy_noised_single = np.vstack( (io_data.H_input, H_output_noised_initial,) )
                H_uy_noised = np.hstack(( H_uy_noised, H_uy_noised_single ))
                H_future_noised = np.hstack(( H_future_noised, H_future_noised_single ))

        width_H = H_uy_noised.shape[1]
        H_uy_noised: np.matrix = np.vstack( (H_uy_noised, np.ones((1, width_H))) )

        # subtract progress along track center l from output if self.params.use_zero_l_initial is True
        if self.params.use_zero_l_initial:
            H_future_noised[self._p-1::self._p,:] = H_future_noised[self._p-1::self._p,:] - \
            H_uy_noised[self._m*(self.params.lag+self.params.L)+self._p-1,:]

            H_uy_noised[self._m*(self.params.lag+self.params.L)+self._p-1::self._p,:] = H_uy_noised[self._m*(self.params.lag+self.params.L)+self._p-1::self._p,:] - \
            H_uy_noised[self._m*(self.params.lag+self.params.L)+self._p-1,:]
        
        # calculate weights for each data segment
        size_y = self.y_max-self.y_min
        size_u = self.u_max-self.u_min
        # W_xi = np.matrix(np.eye(self._lag*self._m + self._lag*self._p)) # used for calculating weights of xi_t distance, can be a hyper-parameter to be learned!
        W_xi = np.matrix(np.diag(np.hstack((1/size_u**2,)*self.params.lag + (1/size_y**2,)*self.params.lag)))
        H_xi = np.vstack(( H_uy_noised[:self.params.lag*self._m,:],H_uy_noised[-self.params.lag*self._p-1:-1,:] ))
        delta_H_xi  = H_xi - xi_t
        d_array = np.zeros((width_H,))
        for i in range(width_H):
            d_array[i] = (delta_H_xi[:,i].T @ W_xi @ delta_H_xi[:,i])[0,0]
        num = max(self.params.min_num_slices, int(width_H*self.params.min_portion_slices))
        r_range = max(self.params.min_dist, np.partition(d_array, num)[num])
        d_inv_array = np.zeros((width_H,))
        for i in range(width_H):
            if d_array[i] < r_range:
                d_inv_array[i] = self.params.f(d_array[i])
            else:
                d_inv_array[i] = 0
        D_inv = np.diag(d_inv_array)

        # calculate the estimation matrix
        if self.params.use_zero_l_initial:
            H_uy_noised = np.delete(H_uy_noised, self._m*(self.params.lag+self.params.L)+self._p-1, axis=0)
        D_inv_Huy_T = D_inv @ H_uy_noised.T
        Phi = H_future_noised @ D_inv_Huy_T @ npl.pinv(H_uy_noised @ D_inv_Huy_T)

        return Phi
