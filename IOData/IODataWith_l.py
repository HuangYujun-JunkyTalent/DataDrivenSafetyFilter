import numpy as np
import numpy.linalg as npl
import numpy.random as npr
from pylfsr import LFSR

from enum import Enum
from typing import Tuple, List, Optional, Union
from warnings import warn
from math import sin, pi

from System.LTI import LTI
from System.LATI import LATI
from System.ErrorKinematicAcceLATI import LinearizedErrorKinematicAcceModel
from .IOData import IOData, InputRule, HankelMatrix

class IODataWith_l(IOData):
    '''
    IO data only for kinematic model of cronos.
    Aside from input and output, also includes a matrix for predicting trajectory of l:
    A_l @ [u[t-lag, t-1], u[t, t+L-1], l[t-lag, t-1] , 1] =(approximately) l[t, t+L-1]
    '''
    _l_data: List[np.matrix] # l data
    _A_l: List[np.matrix] # estimation matrix for l
    _H_l: List[HankelMatrix] # Hankel matrix for l
    _A_l_used: np.matrix
    @property
    def H_l(self) -> HankelMatrix:
        return self._H_l
    _n_l: float # noise level of observation of l
    _lag: int
    _L: int
    _N_l: int # number of data points used to estimate l
    _K_l: int # number of batches that used to estimate l
    
    noisy: bool = False # whether the data in y_data is noisy
    
    def __init__(self,  depth: int,
                 sys: Optional[LinearizedErrorKinematicAcceModel] = None, n: Optional[int] = None, m: Optional[int] = None, p: Optional[int] = None,
                 input_rule: Optional[InputRule] = None, A_u_d: Optional[np.matrix] = None, b_u_d: Optional[np.matrix] = None, 
                 d_u_max: Optional[np.matrix] = None, Ts: Optional[float] = 0.01, mean_input: Optional[np.matrix] = None,
                 length: Optional[int] = None,
                 n_l: float = 0.002, lag: int = 5, L: int = 50, N_l: int = 100, K_l: int = 5) -> None:
        '''
        IO data class only for kinematic error dynamics of cronos.
        n_l: noise level of observation of l.
        lag and L should be the same as defined for the safety filter!
        N_l must be <= length, represents the batch size of data points used to estimate l
        '''
        if sys is not None:
            assert length > 0, "Try to initialize with length {length} and input rule {input_rule}, which is invalid!"
            if input_rule in InputRule.methods_with_mean:
                assert mean_input is not None, f"When using {input_rule}, mean_input needs to be specified!"
                assert mean_input.shape[0] == sys.m, "mean_input.shape[0] != sys.m"
                assert mean_input.shape[1] == 1, "mean_input.shape[1] != 1"
                self._mean_input = mean_input
            if input_rule is InputRule.BOUNDED_RATE_WITH_MEAN:
                assert d_u_max is not None, f"When using {input_rule}, d_u_max needs to be specified!"
                self._d_u_max = d_u_max
            if input_rule is InputRule.PRBS_RATE_WITH_MEAN:
                assert d_u_max is not None, f"When using {input_rule}, d_u_max needs to be specified!"
                self._d_u_max = d_u_max
                init_state = [1,0,0,0,0,0,0]
                f_poly = [7,1]
                # update PRBS state every _i_block_prbs steps
                self._i_block_prbs = int(length/(2**len(init_state)-1))
                self.L = LFSR(initstate=init_state, fpoly=f_poly, counter_start_zero=False)
            if input_rule is InputRule.PRBS_WITH_MEAN:
                init_state = [1,0,0,0,0,0,0]
                f_poly = [7,1]
                self._i_block_prbs = int(length/(2**len(init_state)-1))
                self.L = LFSR(initstate=init_state, fpoly=f_poly, counter_start_zero=False)
            if input_rule is InputRule.PRBS_TIMES_RANDOM_MEAN:
                init_state = [1,0,0,0,0,0,0]
                f_poly = [7,1]
                self._i_block_prbs = int(length/(2**len(init_state)-1))
                self.L = LFSR(initstate=init_state, fpoly=f_poly, counter_start_zero=False)
            # if n is None:
            #     n = sys.n
            # self._n = n
            self._n = sys.n
            self._length = length
            self._input_data = []
            self._output_data = []
            self._noise_data = []
            self._u_xi_0 = []
            self._y_xi_0 = []
            self._y_n_xi_0 = []

            self._A_u_d = A_u_d
            self._b_u_d = b_u_d
            self._Ts = Ts

            self._n_l = n_l
            self._lag = lag
            self._L = L
            self._N_l = N_l
            self._K_l = K_l

            l_data: List[np.matrix] = []

            for i in range(-self._n, 0):
                u = self.get_input(i, sys, input_rule)
                y, n = sys.step(u)
                self._u_xi_0.append(u)
                self._y_xi_0.append(y)
                self._y_n_xi_0.append(y+n)
            for i in range(length):
                u = self.get_input(i, sys, input_rule)
                l_data.append(np.matrix( sys.state[-1] + self._n_l*(-1+2*npr.rand(1, 1)) ))
                y, n = sys.step(u)
                # append the 
                self._input_data.append(u)
                self._output_data.append(y)
                self._noise_data.append(n)
            self._m = sys.m
            self._p = sys.p

            self.update_depth(depth)

            # prepare and calculate matrix for estimating l
            self._l_data = l_data
            # H_l = self.Hankel_matrix(depth, l_data[:N_l])
            # self._H_l = HankelMatrix(l_data[:N_l], H_l)
            self._A_l_used = self.get_l_estimation_matrix()
        else: # initialize with no data, add data points later
            self._input_data = []
            self._output_data = []
            self._noise_data = []
            self._l_data = []
            self._n, self._m, self._p = n, m, p
            self._length = 0
            self.noisy = True # in this case the data will be noisy
    
    def add_error_dynamic_point(self, u: np.matrix, error_state: np.ndarray) -> None:
        """Add point to IO data.
        error_state: [e_lat, mu, v, l]
        """
        if self._m is None:
            self._m = u.shape[0]
        if self._p is None:
            self._p = 3
        self.add_point(u, np.matrix(error_state[0:3]).transpose(), np.matrix(np.zeros( (3,1) )))
        self.add_l_point(np.matrix(error_state[3]))
        # self._length += 1
    
    def remove_last_point(self) -> bool:
        """Remove the oldest datapoint
        return true if successful, false if failed (no data)
        """
        if self._length == 0:
            return False
        self._input_data.pop(0)
        self._output_data.pop(0)
        self._noise_data.pop(0)
        try:
            self._l_data.pop(0)
        except IndexError:
            print("This Dataset has no data for l, not poping anything.")
        self._length -= 1
        return True
    
    def update_depth(self, depth: int) -> None: # override, not dealing with xi_0 here
        self._H_input = self.Hankel_matrix(depth, self._input_data)
        # if npl.matrix_rank(self._H_input) != self._H_input.shape[0]:
        #     warn("Hankel matrix for input is not full rank!")
        self._H_output = self.Hankel_matrix(depth, self._output_data)
        self._H_noise = self.Hankel_matrix(depth, self._noise_data)
        
        self._depth = depth

    def add_l_point(self, l: np.matrix) -> None:
        self._l_data.append(l)

    def update_l_estimation_matrix(self, L: int, lag: int) -> None:
        self._L = L
        self._lag = lag
        self._A_l_used = self.get_l_estimation_matrix()
            
    def get_l_estimation_matrix(self) -> np.matrix:
        '''
        calculate the estimation matrix for l.
        '''
        self._H_l: List[HankelMatrix] = []
        self._A_l: List[np.matrix] = []
        if self._K_l > 1:
            step = int((self._length-self._N_l)/(self._K_l-1))
        else:
            step = self._length
        for i in range(self._K_l):
            input_data = self._input_data[i*step:i*step+self._N_l]
            l_data = self._l_data[i*step:i*step+self._N_l]
            l_max = np.max(l_data)
            l_min = np.min(l_data)
            l_data = l_data - (l_max+l_min)/2

            H_input = self.Hankel_matrix(self._lag+self._L, input_data)
            H_l = self.Hankel_matrix(self._lag+self._L, l_data)
            H_l_indexable = HankelMatrix(l_data, H_l)
            self._H_l.append(H_l_indexable)

            width_H = H_input.shape[1]
            H_ul1 = np.vstack((
                H_input,
                H_l_indexable[:self._lag],
                np.ones((1, width_H))
            ))
            self._A_l.append(H_l_indexable[self._lag:] @ npl.pinv(H_ul1))
        return np.average(self._A_l, axis=0)

    def get_l_estimation(self, u_obs: np.matrix, u_candidate: np.matrix, l_obs: np.matrix):
        x = np.vstack((u_obs, u_candidate, l_obs, np.matrix('1')))
        l_estimated = self._A_l_used @ x
        l_estimated = np.array(l_estimated).squeeze()
        return l_estimated - l_estimated[0]
            