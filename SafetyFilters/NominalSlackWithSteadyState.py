import numpy as np
import numpy.linalg as npl
from scipy.linalg import block_diag
import cvxpy as cp

from typing import Tuple, List
from dataclasses import dataclass

from System import LTI
from IOData import IOData
from .DDSF import DDSafetyFilter, SFParams

# Robust SF with full slack variables
@dataclass
class NominalSlackSFParams:
    L: int
    lag: int # approximation of lag of system
    R: np.matrix
    lam_alph: float
    lam_sig: float
    epsilon: float
    sf_params: SFParams = SFParams()
    # use_estimated_c_pe: bool = False


class NominalSlackDDSafetyFilter(DDSafetyFilter):
    #  use full-size slack variables to compensate noise
    _y: cp.Variable # output variable
    _alpha: cp.Variable # slack variables
    _sigma: cp.Variable # slack variables for compensating noise
    _lam_alph: float # regulizing term for alpha
    _lam_sig: float # regulizing term for sigma
    _io_data: IOData
    _epsilon: float # noise level
    _R: np.matrix
    # c_pe: float

    _a1: List[float]
    @property
    def a1(self):
        return self._a1
    _a2: List[float]
    @property
    def a2(self):
        return self._a2
    _a3: List[float]
    @property
    def a3(self):
        return self._a3
    _a4: List[float]
    @property
    def a4(self):
        return self._a4
    
    def __init__(self, sys: LTI, io_data: IOData, nominal_slack_sf_params: NominalSlackSFParams) -> None:
        L, lag, R = nominal_slack_sf_params.L, nominal_slack_sf_params.lag, nominal_slack_sf_params.R
        self._io_data = io_data
        self._m = sys.m
        self._p = sys.p
        self._n = sys.n
        self._L = L
        assert self._L >= 2*self._n, f"Prediction horizon too short! System order is {self._n}, horizon given is {self._L}."
        self._lag = lag
        self._R = R
        self._lam_alph = nominal_slack_sf_params.lam_alph
        self._lam_sig = nominal_slack_sf_params.lam_sig
        assert nominal_slack_sf_params.sf_params.steps == self._n, f"Given steps {nominal_slack_sf_params.sf_params.steps}, not compatible with this SF!"
        DDSafetyFilter.__init__(self, nominal_slack_sf_params.sf_params)

        self._io_data.update_depth(self._L+self._lag)

        self._epsilon = nominal_slack_sf_params.epsilon

        # initialize cvxpy variables:
        width_H = self._io_data.H_input.shape[1]
        self._u = cp.Variable(shape=(self._m*(self._L+self._lag),))
        self._y = cp.Variable(shape=(self._p*(self._L+self._lag),))
        self._alpha = cp.Variable(shape=(width_H,))
        self._sigma = cp.Variable(shape=self._y.shape)

        # initialize cvxpy parameters:
        self._u_obj = cp.Parameter(shape=(self._m*self._n,)) # we need n steps of objective inputs
        self._xi_t = cp.Parameter(shape=(self._lag*(self._m+self._p),))

        # # calcualte constant tightenging constants a_1-a_4
        # c_pe: float = 0
        # if nominal_slack_sf_params.use_estimated_c_pe:
        #     c_pe = npl.norm(npl.pinv(self._io_data.H_u_xi_noised), 1)
        # else:
        #     c_pe = npl.norm(npl.pinv(self._io_data.H_u_xi), 1)
        # self.c_pe = c_pe
        # a1, a2, a3, a4 = self.constants_for_tightening(sys, c_pe)
        # self._a1, self._a2, self._a3, self._a4 = a1, a2, a3, a4
        # # print(np.max(a1), np.max(a2), np.max(a3), np.max(a4))

        # objective function
        R_n = block_diag(*( (self._R,)*self._n ))
        obj = cp.quad_form(self._u[self._m*self._lag:self._m*(self._n+self._lag)]-self._u_obj, R_n)  # error term 
        obj = obj + self._epsilon*self._lam_alph * cp.norm(self._alpha, 2) # penalty term
        obj = obj + self._lam_sig * cp.norm(self._sigma, 2) # penalty term

        # constraints
        constraints = [self._io_data.H_input @ self._alpha == self._u]
        constraints.append(self._io_data.H_output_noised @ self._alpha == self._y - self._sigma) # dynamic constraints including slack varialbes
        constraints.append(self._u[:self._lag*self._m] == self._xi_t[:self._lag*self._m]) # initial input constraints
        constraints.append(self._y[:self._p*self._lag] == self._xi_t[-self._p*self._lag:]) # initial output constraints
        constraints.append(self._y[-self._n*self._p:] == 4.6 * self._u[-self._n*self._m:]) # terminal constraints
        constraints.append(self._u[-self._m:] == self._u[-2*self._m:-self._m])
        for i in range(2,self._n):
            constraints.append(self._u[-i*self._m:-(i-1)*self._m] == self._u[-(i+1)*self._m:-i*self._m])
        for k in range(self._L):
            i = k + self._lag
            constraints.append(sys.A_u @ self._u[i*self._m:(i+1)*self._m] <= np.array(sys.b_u.flat))
            constraints.append(sys.A_y @ self._y[i*self._p:(i+1)*self._p] <= np.array(sys.b_y.flat))

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
    