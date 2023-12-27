import numpy as np
import numpy.linalg as npl
from scipy.linalg import block_diag
import cvxpy as cp
import pytope as pt

from typing import Tuple, List, Callable, Optional
from dataclasses import dataclass
from itertools import accumulate
from warnings import warn

from .DDSF import DDSafetyFilter, SFParams
from IOData.IOData import IOData, HankelMatrix
from IOData.IODataWith_l import IODataWith_l
from System.LATI import LATI
from System.ErrorKinematicAcceLATI import LinearizedErrorKinematicAcceModel

# define the new version with fewer slack variables
# Robust SF with fewer slack variables
@dataclass
class IndirectNominalFixMuWeightingAddDataWithLParams:
    L: int
    lag: int # approximation of lag of system
    R: np.matrix
    # lam_alph: float
    lam_sig: float
    epsilon: float
    c: List[List[float]]
    t_new_data: float = 6.0

    W_xi: np.matrix = np.matrix(np.eye(1))
    min_dist: float = 0.1
    min_num_slices: int = 300
    min_portion_slices: float = 0.1
    f: Callable[[float], float] = lambda x: 1/x**2
    # steady_input: Optional[np.ndarray] = None
    # steady_output: Optional[np.ndarray] = None
    sf_params: SFParams = SFParams()
    # use_estimated_c_pe: bool = False


class IndirectNominalFixMuWeightingAddDataWithLFilter(DDSafetyFilter):
    """This dataset fed-in/fed-out should include the progress along track center, but they will not be used in the filter"""
    #  use fewer slack variables to compensate noise, slack for initial outputs are omitted
    _y: cp.Variable # output variable
    # _alpha: cp.Variable # slack variables
    _sigma: cp.Variable # slack variables for compensating noise
    _lam_alph: float # regulizing term for alpha
    _lam_sig: float # regulizing term for sigma
    _io_data_list: List[IODataWith_l] # list of data trajectories
    _epsilon: float # noise level
    _R: np.matrix
    _c_sum: List[List[float]] # tightening constants

    y_max: np.ndarray # maximum output
    y_min: np.ndarray # minimum output
    u_max: np.ndarray # maximum input
    u_min: np.ndarray # minimum input

    _u_s: np.ndarray # steasy state input
    _y_s: np.ndarray # steasy state output, estimated from data

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
                 sys: LinearizedErrorKinematicAcceModel, io_data: IOData, params: IndirectNominalFixMuWeightingAddDataWithLParams,
                ) -> None:
        L, lag, R = params.L, params.lag, params.R
        # steady_input = params.steady_input
        # steady_output = params.steady_output
        self._c_sum = [list(accumulate(list_i)) for list_i in params.c]
        self._io_data_list = [io_data]
        self._m = sys.m
        self._p = sys.p
        self._p_used = self._p - 1 # exclude progress along track center l
        self._n = sys.n

        self.y_max = np.array(sys.b_y[:self._p_used]).flatten()
        self.y_min = -np.array(sys.b_y[self._p_used:self._p_used+self._p_used]).flatten()
        self.u_max = np.array(sys.b_u[:sys.m]).flatten()
        self.u_min = -np.array(sys.b_u[sys.m:2*sys.m]).flatten()
        self.min_num_slices = params.min_num_slices
        self.min_portion_slices = params.min_portion_slices
        self.min_dist = params.min_dist
        self.f = params.f

        self._L = L
        self._Ts = sys.Ts
        self._t_new_data = params.t_new_data
        self._new_data_count = 0
        # assert self._L >= 2*self._n, f"Prediction horizon too short! System order is {self._n}, horizon given is {self._L}."
        self._lag = lag
        self._R = R
        # self._lam_alph = params.lam_alph
        self._lam_sig = params.lam_sig
        # if params.sf_params.steps != self._n:
        #     warn(f"Given steps {params.sf_params.steps}!")
        DDSafetyFilter.__init__(self, params.sf_params)

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

        self._epsilon = params.epsilon

        # initialize cvxpy variables:
        # width_H = self._io_data_list[0].H_input.shape[1]
        self._u = cp.Variable(shape=(self._m*(self._L),))
        self._y = cp.Variable(shape=(self._p_used*(self._L),))
        # self._alpha = cp.Variable(shape=(width_H,))
        self._sigma = cp.Variable(shape=(self._p_used*(self._L),)) # we use fewer slack variables
        # self._sigma = cp.Variable(shape=(self._p*(self._L-l_terminal),)) # we use fewer slack variables

        # initialize cvxpy parameters:
        self._u_obj = cp.Parameter(shape=(self._m*self._steps,)) # we need m steps of objective inputs
        self._xi_t = cp.Parameter(shape=(self._lag*(self._m+self._p_used),))

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
        R_n = block_diag(*( (self._R,)*self._steps ))
        self._objective = cp.quad_form(self._u[0:self._m*(self._steps)]-self._u_obj, R_n)  # error term 
        # obj = obj + self._epsilon*self._lam_alph * cp.norm(self._alpha, 2) # penalty term
        # obj = obj + self._lam_sig * cp.norm(self._sigma, 2) # penalty term
        self._objective = self._objective + self._lam_sig * cp.sum_squares(self._sigma) # penalty term
        # self._objective = self._objective + self._lam_sig * cp.norm_inf(self._sigma) # penalty term
        self._objective = cp.Minimize(self._objective)

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
        l_terminal = self._steps
        if self._p_used == 3: # system with few output
            constraints.append(self._y[-self._p_used*l_terminal+2::self._p_used] == np.zeros((l_terminal))) # v
            constraints.append(self._u[-self._m*l_terminal::self._m] == np.zeros((l_terminal))) # a
            for i in range(1,l_terminal):
                constraints.append(self._y[-self._p_used*i+1] == self._y[-self._p_used*(i+1)+1]) # mu
                constraints.append(self._y[-self._p_used*i] == self._y[-self._p_used*(i+1)]) # e_lat 
                constraints.append(self._u[-self._m*i+1] == self._u[-self._m*(i+1)+1]) # delta
        elif self._p_used == 4: # system with more output
            constraints.append(self._y[-self._p_used*l_terminal+2::self._p_used] == np.zeros((l_terminal))) # v_x
            constraints.append(self._u[-self._m*l_terminal::self._m] == np.zeros((l_terminal))) # throttle
            for i in range(1,l_terminal):
                constraints.append(self._y[-self._p_used*i+1] == self._y[-self._p_used*(i+1)+1]) # mu
                constraints.append(self._y[-self._p_used*i] == self._y[-self._p_used*(i+1)]) # e_lat 
                constraints.append(self._y[-self._p_used*i+3] == self._y[-self._p_used*(i+1)+3]) # v_y
                constraints.append(self._u[-self._m*i+1] == self._u[-self._m*(i+1)+1]) # delta

        l_constrained = self._L
        if sys.A_y is not None:
            A_y_used = sys.A_y[:,:self._p_used]
            P_y_list: List[pt.Polytope] = [] # list of tightened polytopes
            P_y = pt.Polytope(A_y_used, np.array(sys.b_y))
            if (l_constrained) % self._steps == 0:
                J = int((l_constrained)/self._steps)
            else:
                J = int((l_constrained)/self._steps) + 1
            for j in range(J):
                bound = np.zeros(0)
                for i in range(self._p_used):
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
                j = int(k/self._steps)
                constraints.append(P_y_list[j].A@(self._y[i*self._p_used:(i+1)*self._p_used]) <= P_y_list[j].b.flatten())

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
        if set_new_dataset and self._new_data_count*self._Ts < self._t_new_data:
            self._io_data_list.append(IODataWith_l(depth=self._L+self._lag, n=self._n, m=self._m, p=self._p)) # create a new dataset
        
        # add new data points, pop old data points
        u_history = xi_t[:self._lag*self._m]
        u_history = u_history[-self._steps*self._m:]
        y_history = xi_t[-self._lag*self._p:]
        y_history = y_history[-self._steps*self._p:]
        if self._new_data_count*self._Ts < self._t_new_data:
            for i in range(self._steps):
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
            self._new_data_count += self._steps
        
        # get estimation matrix from current dataset list
        Phi = self.get_estimation_matrix(xi_t)

        # dynamic constraint, remember to use proper _p to exclude progress along track center l
        self._constraints.append(Phi[:,self._lag*self._m:-self._p_used*self._lag-1] @ self._u == \
                                self._y - self._sigma - \
                                Phi[:,:self._lag*self._m]@self._xi_t[:self._lag*self._m] - \
                                Phi[:,-self._p_used*self._lag-1:-1]@self._xi_t[-self._p_used*self._lag:] - \
                                np.array((Phi[:,-1:]@np.ones((1,)))).squeeze())

        self._opt_prob = cp.Problem(objective=self._objective, constraints=self._constraints)

        xi_t_used = np.delete(xi_t, slice(self._lag*self._m+self._p-1, None, self._p))
        self._xi_t.value = np.array(xi_t_used.flat)
        self._u_obj.value = np.array(u_obj.flat)
        if self._solver is None:
            self._opt_prob.solve(verbose=self._verbose, warm_start = self._warm_start, **self._solver_args)
        else:
            self._opt_prob.solve(verbose=self._verbose, solver=self._solver, warm_start = self._warm_start, **self._solver_args)
        if not (self._opt_prob.status is cp.OPTIMAL or self._opt_prob.status is cp.OPTIMAL_INACCURATE):
            raise Exception(f"Problem not optimal: status is {self._opt_prob.status}")
        
        # pop old dynamic constraint
        self._constraints.pop(-1)

        return (np.matrix(self._u.value[:self._m*self._steps]).transpose(), self._opt_prob.status, self._opt_prob.value)
    
    def get_estimation_matrix(self, xi_t: np.matrix) -> np.matrix:
        """Return estimation matrix Phi
        Remember to use proper _p to exclude progress along track center l, also exclude progress along track center l in xi_t and datasets"""
        # get xi_t without progress along track center l by deleting corresponding elements
        xi_t_used = np.delete(xi_t, slice(self._lag*self._m+self._p-1, None, self._p))

        # get dataset matrix without progress along track center l by deleting corresponding rows
        H_uy_noised: np.matrix = np.matrix(np.zeros(( self._p_used*self._lag+self._m*(self._L+self._lag),0 )))
        H_future_noised: np.matrix = np.matrix(np.zeros(( self._p_used*self._L,0 )))
        for io_data in self._io_data_list:
            if io_data.length >= self._L+self._lag: # only use data with enough length
                io_data.update_depth(self._L+self._lag)
                H_output_noised_initial_used = np.delete(io_data.H_output_noised_part((0, self._lag)), slice(self._p-1, None, self._p), axis=0)
                H_uy_noised_single = np.vstack( (io_data.H_input, H_output_noised_initial_used,) )
                H_uy_noised = np.hstack(( H_uy_noised, H_uy_noised_single ))
                H_future_noised_used = np.delete(io_data.H_output_noised_part((self._lag, self._lag+self._L)), slice(self._p-1, None, self._p), axis=0)
                H_future_noised = np.hstack(( H_future_noised, H_future_noised_used ))

        width_H = H_uy_noised.shape[1]
        H_uy_noised: np.matrix = np.vstack( (H_uy_noised, np.ones((1, width_H))) )
        
        # calculate weights for each data segment
        size_y = self.y_max-self.y_min
        size_u = self.u_max-self.u_min
        # W_xi = np.matrix(np.eye(self._lag*self._m + self._lag*self._p)) # used for calculating weights of xi_t distance, can be a hyper-parameter to be learned!
        W_xi = np.matrix(np.diag(np.hstack((1/size_u**2,)*self._lag + (1/size_y**2,)*self._lag)))
        H_xi = np.vstack(( H_uy_noised[:self._lag*self._m,:],H_uy_noised[-self._lag*self._p_used-1:-1,:] ))
        delta_H_xi  = H_xi - np.matrix(xi_t_used).T
        d_array = np.zeros((width_H,))
        for i in range(width_H):
            d_array[i] = (delta_H_xi[:,i].T @ W_xi @ delta_H_xi[:,i])[0,0]
        num = max(self.min_num_slices, int(width_H*self.min_portion_slices))
        r_range = max(self.min_dist, np.partition(d_array, num)[num])
        d_inv_array = np.zeros((width_H,))
        for i in range(width_H):
            if d_array[i] < r_range:
                d_inv_array[i] = self.f(d_array[i])
            else:
                d_inv_array[i] = 0
        D_inv = np.diag(d_inv_array)

        # calculate the estimation matrix
        D_inv_Huy_T = D_inv @ H_uy_noised.T
        Phi = H_future_noised @ D_inv_Huy_T @ npl.inv(H_uy_noised @ D_inv_Huy_T)

        return Phi
    
    # def get_datasets_hankel_matrix(self) -> Tuple[np.matrix, np.matrix]:
    #     """Return tuple of [U_p; U_f; \tilde{Y}_p] and [\tilde{Y}_f]"""
    #     H_uy_noised: np.matrix = np.matrix(np.zeros(( self._p*self._lag+self._m*(self._L+self._lag),0 )))
    #     H_future_noised: np.matrix = np.matrix(np.zeros(( self._p*self._L,0 )))
    #     for io_data in self._io_data_list:
    #         if io_data.length >= self._L+self._lag: # only use data with enough length
    #             io_data.update_depth(self._L+self._lag)
    #             H_uy_noised_single = np.vstack( (io_data.H_input, io_data.H_output_noised_part((0, self._lag)),) )
    #             H_uy_noised = np.hstack(( H_uy_noised, H_uy_noised_single ))
    #             H_future_noised = np.hstack(( H_future_noised, io_data.H_output_noised_part((self._lag, self._lag+self._L)) ))
    #     return H_uy_noised, H_future_noised

    # def fit_steady_state(self, l: int, L: int, N: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
    #     """Return tuple of (steady state input, steady state output)
    #        Only works for Kinematic Error Dynamics of Cronos"""
    #     io_data = self._io_data_list[0]
    #     depth = io_data.depth
    #     lag = self._lag+l
    #     Len = self._steps+L
    #     uy_s: List[np.matrix] = []
    #     if K  > 1:
    #         step = int((io_data.length-N)/(K-1))
    #     else:
    #         step = io_data.length
    #     for i in range(K):
    #         input_data = io_data._input_data[i*step:i*step+N]
    #         output_data = [y+n for y,n in zip(io_data._output_data[i*step:i*step+N], io_data._noise_data[i*step:i*step+N])]
    #         H_input = HankelMatrix(
    #             input_data,
    #             io_data.Hankel_matrix(lag+Len, input_data)
    #         )
    #         H_output = HankelMatrix(
    #             output_data,
    #             io_data.Hankel_matrix(lag+Len, output_data)
    #         )
    #         width_H = H_input._H.shape[1]
    #         H_uy1_noised: np.matrix = np.vstack( (H_input[:], H_output[:lag], np.ones((1, width_H))) )
    #         H_uy1_noised_inv = npl.pinv(H_uy1_noised)

    #         H_y = H_output[lag:]

    #         b = H_y @ H_uy1_noised_inv[:,-1] # RHS for final Linear equation!
    #         A_1 = np.hstack((
    #             np.matrix(np.zeros((Len*self._p,self._m))),
    #             np.vstack((np.matrix(np.eye(self._p)),)*Len)
    #         )) # first simple term of the linear equation
    #         A_2 = H_y @ H_uy1_noised_inv @ np.vstack((
    #             np.hstack((
    #                 np.vstack((np.matrix(np.eye(self._m)),)*(lag+Len)),
    #                 np.zeros((self._m*(Len+lag),self._p))
    #             )),
    #             np.hstack((
    #                 np.zeros((self._p*lag,self._m)),
    #                 np.vstack((np.matrix(np.eye(self._p)),)*lag)
    #             )),
    #             np.matrix(np.zeros((1,self._p+self._m)))
    #         ))
    #         A = A_1 - A_2
    #         # # add additional equations
    #         # A_real = np.vstack((
    #         #     A,
    #         #     np.hstack((
    #         #         np.matrix(np.eye(1)),
    #         #         np.matrix(np.zeros((1,self._p+self._m-1)))
    #         #     )),
    #         #     np.hstack((
    #         #         np.matrix(np.zeros((1,2))),
    #         #         np.matrix(np.eye(1)),
    #         #         np.matrix(np.zeros((1,self._p+self._m-3)))
    #         #     )),
    #         #     np.hstack((
    #         #         np.matrix(np.zeros((1,4))),
    #         #         np.matrix(np.eye(1)),
    #         #         np.matrix(np.zeros((1,self._p+self._m-5)))
    #         #     )),
    #         # ))
    #         # b_real = np.vstack((
    #         #     b,
    #         #     np.matrix(np.zeros((3,1))),
    #         # ))
    #         A_real = np.hstack((
    #             A[:,1],
    #             A[:,3]
    #         ))
    #         b_real = b
    #         uy_s.append(npl.pinv(A_real) @ b_real)
    #     uy_s_average = np.average(uy_s, axis=0)
    #     return (np.array([0, uy_s_average[0,0]]), np.array([0, uy_s_average[1,0], 0]))

    # # def constants_for_tightening(self, sys: LTI, c_pe: float) -> Tuple[List[float], List[float], List[float], List[float]]:
    # #     gamma = sys.gamma()
    # #     L, n, epsilon = self._L, self._n, self._epsilon
    # #     a1 = np.ndarray((L-n,))
    # #     a2 = np.ndarray((L-n,))
    # #     a3 = np.ndarray((L-n,))
    # #     a4 = np.ndarray((L-n,))
    # #     rho = np.ndarray((L+n,))
    # #     for i in range(L+n):
    # #         rho[i] = sys.rho(i)
    # #     rho_n = np.max(rho[n:2*n])
    # #     rho_L = np.max(rho[L:L+n])

    # #     for k in range(self._n):  # since L>=2*n, there will always be at least n variables to initialize
    # #         a1[k] = 0
    # #         a3[k] = 1 + rho_n
    # #         a2[k] = epsilon * a3[k]
    # #         a4[k] = epsilon * rho_n

    # #     l = self._n
    # #     xi_max = sys.xi_max(self._n)
    # #     while l + n <= L - n:
    # #         for k in range(l, l+n):
    # #             a1[k] = a1[k-n] + c_pe*( a2[k-n]+epsilon*a3[k-n] )
    # #             a3[k] = 1 + rho[k+n] + gamma*a1[k]*(1+rho_L)
    # #             a2[k] = epsilon * a3[k]
    # #             a4[k] = a4[k-n] + epsilon*rho[k+n] + epsilon*gamma*rho_L*a1[k] + epsilon*a3[k-n] + c_pe*xi_max*(a2[k-n] + epsilon*a3[k-n])
    # #         l = l + n
    # #     for k in range(l, L-n):
    # #         a1[k] = a1[k-n] + c_pe*( a2[k-n]+epsilon*a3[k-n] )
    # #         a3[k] = 1 + rho[k+n] + gamma*a1[k]*(1+rho_L)
    # #         a2[k] = epsilon * a3[k]
    # #         a4[k] = a4[k-n] + epsilon*rho[k+n] + epsilon*gamma*rho_L*a1[k] + epsilon*a3[k-n] + c_pe*xi_max*(a2[k-n] + epsilon*a3[k-n])

    # #     return a1, a2, a3, a4
    