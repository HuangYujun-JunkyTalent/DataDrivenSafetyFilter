import numpy as np
import cvxpy as cp

from typing import List
from dataclasses import dataclass

from System import LTI
from IOData import IOData
from .DDSF import DDSafetyFilter, SFParams

# Define Nominal DDSF class, use vectors inside cvxpy

@dataclass
class NominalSFParams:
    L: int
    lag: int # approximation of lag of system
    R: np.matrix
    sf_params: SFParams = SFParams()


class NominalDDSafetyFilter(DDSafetyFilter):
    _y: cp.Variable # output variable
    _alpha: cp.Variable # slack variables
    _io_data: IOData
    _R: np.matrix

    def __init__(self, sys: LTI, io_data: IOData, nominal_sf_params: NominalSFParams) -> None:
        L, lag, R = nominal_sf_params.L, nominal_sf_params.lag, nominal_sf_params.R
        self._io_data = io_data
        self._m = sys.m
        self._p = sys.p
        self._n = sys.n
        self._L = L
        self._lag = lag
        self._R = R
        DDSafetyFilter.__init__(self, nominal_sf_params.sf_params)

        self._io_data.update_depth(self._L+self._lag)

        # initialize cvxpy variables:
        width_H = self._io_data.H_input.shape[1]
        self._u = cp.Variable(shape=(self._m*(self._L+self._lag),))
        self._y = cp.Variable(shape=(self._p*(self._L+self._lag),))
        self._alpha = cp.Variable(shape=(width_H,))

        # initialize cvxpy parameters:
        self._u_obj = cp.Parameter(shape=(self._m,))
        self._xi_t = cp.Parameter(shape=(self._lag*(self._m+self._p),))

        # objective:
        # self._u_diff: cp.Variable = self._u[self._m*self._lag:self._m*(self._lag+1)] - self._u_obj
        # obj = cp.quad_form(self._u_diff, self._R)
        obj = cp.quad_form(self._u[self._m*self._lag:self._m*(self._lag+1)] - self._u_obj, self._R)

        # constraints
        constraints: List[cp.Constraint] = [self._io_data.H_input @ self._alpha == self._u,
                       self._io_data.H_output @ self._alpha == self._y, # dynamic constraints
                       self._u[0:self._m*self._lag] == self._xi_t[0:self._m*self._lag], # initial input
                       self._y[0:self._p*self._lag] == self._xi_t[-self._p*self._lag:], # initial output
                       self._u[-self._m*self._n:] == np.zeros((self._m*self._n,)), # terminal input
                       self._y[-self._p*self._n:] == np.zeros((self._p*self._n,))] # terminal output
        if sys.A_u is not None:
            b_u_flat: np.ndarray = np.array(sys.b_u.flat)
            for i in range(self._lag, self._L + self._lag):
                constraints.append(sys.A_u @ self._u[self._m*i:self._m*(i+1)] <= b_u_flat)
        if sys.A_y is not None:
            b_y_flat: np.ndarray = np.array(sys.b_y.flat)
            for i in range(self._lag, self._L + self._lag):
                constraints.append(sys.A_y @ self._y[self._p*i:self._p*(i+1)] <= b_y_flat)
        
        self._opt_prob = cp.Problem(objective=cp.Minimize(obj), constraints=constraints)