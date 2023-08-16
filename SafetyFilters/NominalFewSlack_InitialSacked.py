import numpy as np
import numpy.linalg as npl
from scipy.linalg import block_diag
import cvxpy as cp

from typing import Tuple, List
from dataclasses import dataclass

from .DDSF import DDSafetyFilter, SFParams
from IOData import IOData
from System import LTI

# define the new version with fewer slack variables
# Robust SF with fewer slack variables
@dataclass
class NominalFewSlackSFParams:
    L: int
    lag: int # approximation of lag of system
    R: np.matrix
    # lam_alph: float
    lam_sig: float
    epsilon: float
    sf_params: SFParams = SFParams()
    # use_estimated_c_pe: bool = False


class NominalFewSlackDDSafetyFilter(DDSafetyFilter):
    #  use fewer slack variables to compensate noise, slack for initial outputs are omitted
    _y: cp.Variable # output variable
    _alpha: cp.Variable # slack variables
    _sigma: cp.Variable # slack variables for compensating noise
    _lam_alph: float # regulizing term for alpha
    _lam_sig: float # regulizing term for sigma
    _io_data: IOData
    _epsilon: float # noise level
    _R: np.matrix

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
    
    def __init__(self, sys: LTI, io_data: IOData, few_slack_sf_params: NominalFewSlackSFParams, lam_sig_1: float) -> None:
        L, lag, R = few_slack_sf_params.L, few_slack_sf_params.lag, few_slack_sf_params.R
        self._io_data = io_data
        self._m = sys.m
        self._p = sys.p
        self._n = sys.n
        self._L = L
        assert self._L >= 2*self._n, f"Prediction horizon too short! System order is {self._n}, horizon given is {self._L}."
        self._lag = lag
        self._R = R
        # self._lam_alph = few_slack_sf_params.lam_alph
        self._lam_sig = few_slack_sf_params.lam_sig
        assert few_slack_sf_params.sf_params.steps == self._n, f"Given steps {few_slack_sf_params.sf_params.steps}, not compatible with this SF!"
        DDSafetyFilter.__init__(self, few_slack_sf_params.sf_params)

        self._io_data.update_depth(self._L+self._lag)

        self._epsilon = few_slack_sf_params.epsilon

        # initialize cvxpy variables:
        width_H = self._io_data.H_input.shape[1]
        self._u = cp.Variable(shape=(self._m*(self._L+self._lag),))
        self._y = cp.Variable(shape=(self._p*(self._L+self._lag),))
        self._alpha = cp.Variable(shape=(width_H,))
        self._sigma = cp.Variable(shape=(self._p*self._L,)) # we use fewer slack variables, this is for y_[0,L-1]
        self._sigma_1 = cp.Variable(shape=(self._p*self._lag,)) # this is for 

        # initialize cvxpy parameters:
        self._u_obj = cp.Parameter(shape=(self._m*self._n,)) # we need n steps of objective inputs
        self._xi_t = cp.Parameter(shape=(self._lag*(self._m+self._p),))

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
        R_n = block_diag(*( (self._R,)*self._n ))
        obj = cp.quad_form(self._u[self._m*self._lag:self._m*(self._n+self._lag)]-self._u_obj, R_n)  # error term 
        # obj = obj + self._epsilon*self._lam_alph * cp.norm(self._alpha, 2) # penalty term
        obj = obj + self._lam_sig * cp.norm(self._sigma, 2) # penalty term
        obj = obj + lam_sig_1 * cp.norm(self._sigma_1, 2)

        # constraints
        # constraints = [self._io_data.H_input @ self._alpha == self._u]
        # constraints.append(self._io_data.H_output_noised @ self._alpha == self._y - self._sigma) # dynamic constraints including slack varialbes
        constraints = [self._io_data.H_output_noised_part((self._lag, self._lag+self._L)) @ self._alpha == self._y[self._lag:] - self._sigma]
        constraints.append(self._u[:self._lag*self._m] == self._xi_t[:self._lag*self._m]) # initial input constraints
        constraints.append(self._y[:self._p*self._lag] == self._xi_t[-self._p*self._lag:]) # initial output constraints
        constraints.append(self._u[-self._n*self._m:] == np.zeros((self._n*self._m,)))
        constraints.append(self._y[-self._n*self._p:] == np.zeros((self._n*self._p,))) # terminal constraints
        for k in range(self._L-self._n):
            i = k + self._lag
            constraints.append(sys.A_u @ self._u[i*self._m:(i+1)*self._m] <= np.array(sys.b_u.flat))
            # constraints.append(cp.norm_inf(self._y[i*self._p:(i+1)*self._p]) + a1[k]*cp.norm(self._u,1) + a2[k]*cp.norm(self._alpha,1) + a3[k]*cp.norm_inf(self._sigma) + a4[k] <= sys.b_u[0,0]) # tightended output constraint, only support the form ||y||_inf <= y_max here!
            constraints.append(cp.norm_inf(self._y[i*self._p:(i+1)*self._p]) + 0.2 <= sys.b_u[0,0])
        
        # constraint regarding alpha
        H_uy_noised: np.matrix = np.vstack( (self._io_data.H_input, self._io_data.H_output_noised_part((0, self._lag))) )
        # constraints.append(self._alpha == npl.pinv(H_uy_noised) @ cp.hstack((self._u, self._y[:self._lag*self._p])))
        constraints.append(self._alpha == npl.pinv(H_uy_noised) @ cp.hstack((self._u, self._y[:self._lag*self._p]+self._sigma_1)))

        self._opt_prob = cp.Problem(cp.Minimize(obj), constraints=constraints)

    def constants_for_tightening(self, sys: LTI, c_pe: float) -> Tuple[List[float], List[float], List[float], List[float]]:
        gamma = sys.gamma()
        L, n, epsilon = self._L, self._n, self._epsilon
        a1 = np.ndarray((L-n,))
        a2 = np.ndarray((L-n,))
        a3 = np.ndarray((L-n,))
        a4 = np.ndarray((L-n,))
        rho = np.ndarray((L+n,))
        for i in range(L+n):
            rho[i] = sys.rho(i)
        rho_n = np.max(rho[n:2*n])
        rho_L = np.max(rho[L:L+n])

        for k in range(self._n):  # since L>=2*n, there will always be at least n variables to initialize
            a1[k] = 0
            a3[k] = 1 + rho_n
            a2[k] = epsilon * a3[k]
            a4[k] = epsilon * rho_n

        l = self._n
        xi_max = sys.xi_max(self._n)
        while l + n <= L - n:
            for k in range(l, l+n):
                a1[k] = a1[k-n] + c_pe*( a2[k-n]+epsilon*a3[k-n] )
                a3[k] = 1 + rho[k+n] + gamma*a1[k]*(1+rho_L)
                a2[k] = epsilon * a3[k]
                a4[k] = a4[k-n] + epsilon*rho[k+n] + epsilon*gamma*rho_L*a1[k] + epsilon*a3[k-n] + c_pe*xi_max*(a2[k-n] + epsilon*a3[k-n])
            l = l + n
        for k in range(l, L-n):
            a1[k] = a1[k-n] + c_pe*( a2[k-n]+epsilon*a3[k-n] )
            a3[k] = 1 + rho[k+n] + gamma*a1[k]*(1+rho_L)
            a2[k] = epsilon * a3[k]
            a4[k] = a4[k-n] + epsilon*rho[k+n] + epsilon*gamma*rho_L*a1[k] + epsilon*a3[k-n] + c_pe*xi_max*(a2[k-n] + epsilon*a3[k-n])

        return a1, a2, a3, a4
    