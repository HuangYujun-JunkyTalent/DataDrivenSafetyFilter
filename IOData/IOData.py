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


# Define data class


class InputRule(Enum):
    RANDOM = 1
    RANDOM_2 = 2
    RANDOM_2_WITH_MEAN = 3
    SINE = 4
    MIX_WITH_MEAN = 5
    BOUNDED_RATE_WITH_MEAN = 6
    PRBS_RATE_WITH_MEAN = 7
    PRBS_WITH_MEAN = 8

    @classmethod
    @property
    def methods_with_mean(self):
        return (
            InputRule.RANDOM_2_WITH_MEAN,
            InputRule.MIX_WITH_MEAN,
            InputRule.BOUNDED_RATE_WITH_MEAN,
            InputRule.PRBS_RATE_WITH_MEAN,
            InputRule.PRBS_WITH_MEAN
        )


class HankelMatrix:
    '''
    Class for hankel matrix, supports subindex
    '''
    _data: List[np.matrix]
    _i: int # length of each datapoint
    _H: np.matrix # Hankel matrix
    _depth: int # depth of hankel matrix
    def __init__(self, data: List[np.matrix], H: np.matrix) -> None:
        self._data = data
        self._H = H
        self._i = data[0].shape[0]
        self._depth = int(H.shape[0] / self._i)

    def __getitem__(self, index: Union[int, slice]) -> np.matrix:
        if isinstance(index, int):
            return self._H[index*self._i:(index+1)*self._i]
        elif isinstance(index, slice):
            if index.step is not None:
                step = index.step
                warn("non-continuous index of hankel matrix not implemented yet!")
            else:
                step  = None
            if index.start is not None:
                start = index.start * self._i
            else:
                start = None
            if index.stop is not None:
                stop = index.stop * self._i
            else:
                stop = None
            return self._H[start:stop:step]
        else:
            raise TypeError("Index must be int or slice")
    
    # def __getitem__(self, index: Union[int, slice]) -> np.matrix:
    #     if isinstance(index, int):
    #         return self._H[index*self._i:(index+1)*self._i]
    #     elif isinstance(index, slice):
    #         if index.stop is None:
    #             stop = self._depth
    #         else:
    #             stop = index.stop
    #         return np.vstack([
    #             self._H[i*self._i:(i+1)*self._i] for i in range(slice.start,stop,slice.step)
    #         ])
    #     else:
    #         raise TypeError("Index must be int or slice")
        

class IOData:
    _input_data: List[np.matrix]
    _output_data: List[np.matrix] # without noise!
    _noise_data: List[np.matrix]
    _u_xi_0: List[np.matrix] # input for initial xi
    _y_xi_0: List[np.matrix] # output for initial xi
    _n_xi_0: List[np.matrix] # noise for initial xi
    _m: int # size of input vector
    _p: int # size of output vector
    _n: int # system order
    # Hankel matrix of given depth self._depth
    _H_input: np.matrix
    _H_output: np.matrix
    _H_noise: np.matrix
    _H_u_xi: np.matrix
    _H_u_xi_noised: np.matrix
    # properties
    _depth: int
    @property
    def depth(self):
        return self._depth
    _length: int
    @property
    def length(self):
        return self._length
    # input constraint for generating io_data only
    _A_u_d: np.matrix
    _b_u_d: np.matrix
    _d_u_max: np.matrix # maximum change rate of input (Optional)
    _Ts: float

    def __init__(self, depth: int,
                 n: Optional[int]=None,
                 input_data: Optional[List[np.matrix]] = None, output_data: Optional[List[np.matrix]] = None, noise_data: Optional[np.matrix] = None,
                 sys: Optional[Union[LTI,LATI]] = None, input_rule: Optional[InputRule] = None, A_u_d: Optional[np.matrix] = None, b_u_d: Optional[np.matrix] = None, Ts: Optional[float] = 0.01, mean_input: Optional[np.matrix] = None,
                 length: Optional[int] = None) -> None:
        """Attention: first n pairs of input-output data will be used to construct xi_0"""
        if input_data is not None:
            assert len(input_data) == len(output_data), f"Input data length {len(input_data)}, output data length: {len(output_data)}, not maching!"
            if n is None:
                raise Exception("When giving data directly, system order needs to be specified!")
            self._n = n
            self._input_data = input_data[n:]
            self._output_data = output_data[n:]
            self._noise_data = noise_data[n:]
            self._u_xi_0 = input_data[:n]
            self._y_xi_0 = output_data[:n]
            self._y_n_xi_0 = [output_data[i]+noise_data[i] for i in range(n)]
            self._m = input_data[0].shape[0]
            self._p = output_data[0].shape[0]
            self.update_depth(depth)
            self._length = len(self._input_data)
            return
        
        if sys is not None:
            assert input_rule is not None and length > 0, "Try to initialize with length {length} and input rule {input_rule}, which is invalid!"
            if input_rule is InputRule.RANDOM_2_WITH_MEAN or input_rule is InputRule.MIX_WITH_MEAN:
                assert mean_input is not None, f"When using {input_rule}, mean_input needs to be specified!"
                assert mean_input.shape[0] == sys.m, "mean_input.shape[0] != sys.m"
                assert mean_input.shape[1] == 1, "mean_input.shape[1] != 1"
                self._mean_input = mean_input
            if n is None:
                n = sys.n
            self._n = n
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

            for i in range(-self._n, 0):
                u = self.get_input(i, sys, input_rule)
                y, n = sys.step(u)
                self._u_xi_0.append(u)
                self._y_xi_0.append(y)
                self._y_n_xi_0.append(y+n)
            for i in range(length):
                u = self.get_input(i, sys, input_rule)
                y, n = sys.step(u)
                self._input_data.append(u)
                self._output_data.append(y)
                self._noise_data.append(n)
            self._m = sys.m
            self._p = sys.p
            self.update_depth(depth)
            return

        raise Exception("Not a valid way of initializing IOData instance!")

    def get_input(self, i: int, sys: LTI, input_rule: InputRule) -> np.matrix:
        if input_rule is InputRule.RANDOM:
            # For now only support ||u||_inf <= u_max!
            return np.matrix(2 * sys.b_u[0,0] * (npr.rand(1,1)-0.5))
        if input_rule is InputRule.RANDOM_2:
            return np.matrix(np.vstack( (2 * self._b_u_d[0,0] * (npr.rand(1,1)-0.5),
                                         2 * self._b_u_d[1,0] * (npr.rand(1,1)-0.5)) ))
        if input_rule is InputRule.RANDOM_2_WITH_MEAN:
            return np.matrix(np.vstack( (2 * self._b_u_d[0,0] * (npr.rand(1,1)-0.5),
                                         2 * self._b_u_d[1,0] * (npr.rand(1,1)-0.5)) )) + self._mean_input
        if input_rule is InputRule.SINE:
            # separate state into four parts: ++, +-, -+, --
            #to maintain the traj around zero
            #only works approximately
            t = (i*self._Ts) % 2
            if i<=2:
                u_1 = 0.5*self._b_u_d[0,0] * (sin(t*pi*12) + sin(t*pi*13))
                u_2 = 0.5*self._b_u_d[1,0] * (sin(t*pi*8) + sin(t*pi*13))
            elif i<=4:
                u_1 = 0.5*self._b_u_d[0,0] * (sin(t*pi*12) + sin(t*pi*13))
                u_2 = -0.5*self._b_u_d[1,0] * (sin(t*pi*8) + sin(t*pi*13))
            elif i<=6:
                u_1 = -0.5*self._b_u_d[0,0] * (sin(t*pi*12) + sin(t*pi*13))
                u_2 = 0.5*self._b_u_d[1,0] * (sin(t*pi*8) + sin(t*pi*13))
            else:
                u_1 = -0.5*self._b_u_d[0,0] * (sin(t*pi*12) + sin(t*pi*13))
                u_2 = -0.5*self._b_u_d[1,0] * (sin(t*pi*8) + sin(t*pi*13))
            return np.matrix([[u_1],[u_2]])
        if input_rule is InputRule.MIX_WITH_MEAN:
            # separate state into four parts: ++, +-, -+, --
            #to maintain the traj around zero
            #only works approximately
            t = i/self.length * 9
            if t<=2:
                u_1 = 0.5*self._b_u_d[0,0] * (sin(t*pi*9.2) + np.sign(sin(t*pi*13.5)))
                u_2 = 0.5*self._b_u_d[1,0] * (sin(t*pi*2.1) + np.sign(sin(t*pi*3.7)))
            elif t<=4:
                u_1 = 0.5*self._b_u_d[0,0] * (sin(t*pi*9.2) + np.sign(sin(t*pi*13.5)))
                u_2 = -0.5*self._b_u_d[1,0] * (sin(t*pi*2.1) + np.sign(sin(t*pi*3.7)))
            elif t<=6:
                u_1 = -0.5*self._b_u_d[0,0] * (sin(t*pi*9.2) + np.sign(sin(t*pi*13.5)))
                u_2 = 0.5*self._b_u_d[1,0] * (sin(t*pi*2.1) + np.sign(sin(t*pi*3.7)))
            else:
                u_1 = -0.5*self._b_u_d[0,0] * (sin(t*pi*9.2) + np.sign(sin(t*pi*13.5)))
                u_2 = -0.5*self._b_u_d[1,0] * (sin(t*pi*2.1) + np.sign(sin(t*pi*3.7)))
            return np.matrix([[u_1],[u_2]]) + self._mean_input
        if input_rule is InputRule.BOUNDED_RATE_WITH_MEAN:
            if len(self._input_data) == 0:
                last_delta_input = np.matrix('0;0')
            else:
                last_delta_input = self._input_data[-1] - self._mean_input
            d_u = sys.Ts*np.matrix(np.vstack( (2 * self._d_u_max[0,0] * (npr.rand(1,1)-0.5),
                                               2 * self._d_u_max[1,0] * (npr.rand(1,1)-0.5)) ))
            u = d_u + last_delta_input
            # saturate if out of bound
            if u[0,0] > self._b_u_d[0,0]:
                u[0,0] = self._b_u_d[0,0]
            elif u[0,0] < -self._b_u_d[0,0]:
                u[0,0] = -self._b_u_d[0,0]
            if u[1,0] > self._b_u_d[1,0]:
                u[1,0] = self._b_u_d[1,0]
            elif u[1,0] < -self._b_u_d[1,0]:
                u[1,0] = -self._b_u_d[1,0]
            return u + self._mean_input
        if input_rule is InputRule.PRBS_RATE_WITH_MEAN:
            bit = self.L.outbit
            if i % self._i_block_prbs == 0:
                self.L.next()
            if bit == 1:
                d_u = sys.Ts*self._d_u_max
            else:
                d_u = -sys.Ts*self._d_u_max
            if len(self._input_data) == 0:
                last_delta_input = np.matrix('0;0')
            else:
                last_delta_input = self._input_data[-1] - self._mean_input
            u = d_u + last_delta_input
            # saturate if out of bound
            if u[0,0] > self._b_u_d[0,0]:
                u[0,0] = self._b_u_d[0,0]
            elif u[0,0] < -self._b_u_d[0,0]:
                u[0,0] = -self._b_u_d[0,0]
            if u[1,0] > self._b_u_d[1,0]:
                u[1,0] = self._b_u_d[1,0]
            elif u[1,0] < -self._b_u_d[1,0]:
                u[1,0] = -self._b_u_d[1,0]
            return u + self._mean_input
        if input_rule is InputRule.PRBS_WITH_MEAN:
            bit = self.L.outbit
            if i % self._i_block_prbs == 0:
                self.L.next()
            if bit == 1:
                u = self._b_u_d[:2]
            else:
                u = -self._b_u_d[0:2]
            return u + self._mean_input
    
    def add_point(self, u: np.matrix, y: np.matrix, n: np.matrix) -> None:
        self._input_data.append(u)
        self._output_data.append(y)
        self._noise_data.append(n)
        self._length += 1

    def update_depth(self, depth: int) -> None:
        self._H_input = self.Hankel_matrix(depth, self._input_data)
        if npl.matrix_rank(self._H_input) != self._H_input.shape[0]:
            warn("Hankel matrix for input is not full rank!")
        self._H_output = self.Hankel_matrix(depth, self._output_data)
        self._H_noise = self.Hankel_matrix(depth, self._noise_data)
        self._depth = depth

        width = self._H_input.shape[1]
        H_u_xi_u = self.Hankel_matrix(self._n, self._u_xi_0+self._input_data[:width-1])
        H_u_xi_y = self.Hankel_matrix(self._n, self._y_xi_0+self._output_data[:width-1])
        H_u_xi_y_n = self.Hankel_matrix(self._n, self._y_n_xi_0+[self._output_data[i]+self._noise_data[i] for i in range(width-1)])
        self._H_u_xi = np.vstack((self.H_input, H_u_xi_u, H_u_xi_y))
        self._H_u_xi_noised = np.vstack((self.H_input, H_u_xi_u, H_u_xi_y_n))

    @staticmethod
    def Hankel_matrix(depth: int, data: List[np.matrix]) -> np.matrix:
        assert len(data) >= depth, f"Length of data is {len(data)}, smaller then given depth {depth}!"
        return np.hstack(tuple(
            [np.vstack(tuple(data[i:depth+i])) for i in range( len(data)-depth+1 )]
        ))
    
    @property
    def H_input(self) -> np.matrix:
        return self._H_input
    @property
    def H_output(self) -> np.matrix:
        return self._H_output
    @property
    def H_noise(self) -> np.matrix:
        return self._H_noise
    @property
    def H_output_noised(self) -> np.matrix:
        return self._H_output + self._H_noise
    @property
    def H_u_xi(self) -> np.matrix:
        return self._H_u_xi
    @property
    def H_u_xi_noised(self) -> np.matrix:
        return self._H_u_xi_noised
    
    def H_output_part(self, data_range: Tuple[int, int]) -> np.matrix:
        """Return cartain sub rows of output Hankel matrix: 
        Out put index row: from data_range[0] to data_range[1]-1"""
        start = data_range[0]*self._p
        end = data_range[1]*self._p
        return self._H_output[start:end,:]
    
    def H_output_noised_part(self, data_range: Tuple[int, int]) -> np.matrix:
        """Return cartain sub rows of noised output Hankel matrix: 
        Out put index row: from data_range[0] to data_range[1]-1"""
        start = data_range[0]*self._p
        end = data_range[1]*self._p
        return self._H_output[start:end,:] + self._H_noise[start:end,:]

    def H_noise_part(self, data_range: Tuple[int, int]) -> np.matrix:
        """Return cartain sub rows of noise Hankel matrix: 
        Out put index row: from data_range[0] to data_range[1]-1"""
        start = data_range[0]*self._p
        end = data_range[1]*self._p
        return self._H_noise[start:end,:]
    
    # @property
    # def H_u_xi(self):
