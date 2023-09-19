from typing import List, Tuple, Callable, Iterable
from timeit import default_timer as timer

import numpy as np

from IOData.IODataWith_l import IODataWith_l 


def get_datasets_hankel_matrix(io_data_list: List[IODataWith_l], lag: int, L: int) -> Tuple[np.matrix, np.matrix]:
    """Return tuple of [U_p; U_f; \tilde{Y}_p] and [\tilde{Y}_f]"""
    p = io_data_list[0]._output_data[0].shape[0]
    m = io_data_list[0]._input_data[0].shape[0]
    H_uy_noised: np.matrix = np.matrix(np.zeros(( p*lag+m*(L+lag),0 )))
    H_future_noised: np.matrix = np.matrix(np.zeros(( p*L,0 )))
    for io_data in io_data_list:
        if io_data.length >= L+lag: # only use data with enough length
            io_data.update_depth(L+lag)
            H_uy_noised_single = np.vstack( (io_data.H_input, io_data.H_output_noised_part((0, lag)),) )
            H_uy_noised = np.hstack(( H_uy_noised, H_uy_noised_single ))
            H_future_noised = np.hstack(( H_future_noised, io_data.H_output_noised_part((lag, lag+L)) ))
    return H_uy_noised, H_future_noised

def weighting_xi_in_datasets(W_xi: np.matrix, f: Callable[[float], float], io_data_list: List[IODataWith_l], lag: int, L: int, saturation: float, xi: np.ndarray)-> float:
    """Return the sum of f(||xi-y||^2_{W_xi}) for all y in datasets"""
    p = io_data_list[0]._output_data[0].shape[0]
    m = io_data_list[0]._input_data[0].shape[0]
    H_uy_noised, H_future_noised = get_datasets_hankel_matrix(io_data_list, lag, L)
    delta_H_xi = np.vstack(( H_uy_noised[:m*lag,:], H_uy_noised[-p*lag:,:] )) - np.matrix(xi).T
    width_H = delta_H_xi.shape[1]
    weight_list = np.zeros((width_H,))
    for i in range(width_H):
        weight_list[i] = f((delta_H_xi[:,i].T @ W_xi @ delta_H_xi[:,i])[0,0])
    return np.min([saturation, np.sum(weight_list)])

def weighting_u_pf_y_p_in_datasets(W: np.matrix, f: Callable[[float], float], io_data_list: List[IODataWith_l], lag: int, L: int, saturation: float, u_pf_y_p: np.ndarray)-> float:
    """Return the sum of f(||xi_u-y||^2_{W}) for all y in datasets. This includes both extended state and proposed input"""
    """Return the sum of f(||xi-y||^2_{W_xi}) for all y in datasets"""
    p = io_data_list[0]._output_data[0].shape[0]
    m = io_data_list[0]._input_data[0].shape[0]
    H_uy_noised, H_future_noised = get_datasets_hankel_matrix(io_data_list, lag, L)
    delta_H_xi = H_uy_noised - np.matrix(u_pf_y_p).T
    width_H = delta_H_xi.shape[1]
    weight_list = np.zeros((width_H,))
    for i in range(width_H):
        weight_list[i] = f((delta_H_xi[:,i].T @ W @ delta_H_xi[:,i])[0,0])
    return np.min([saturation, np.sum(weight_list)])

class Sampler:
    lag: int = 1
    L: int = 1
    a_min, a_max = -1.1, 5.5
    delta_max = 0.4
    v_0 = 1.0
    delta_v_x_min, delta_v_x_max = -0.95, 2.2
    v_y_min, v_y_max = -2.0, 2.0
    mu_min, mu_max = -0.5*np.pi, 0.5*np.pi
    track_width = 0.5

    def xi_iterator(self) -> Iterable[np.ndarray]:
        u_min = np.array([self.a_min, -self.delta_max])
        y_min = np.array([-self.track_width, self.mu_min, self.delta_v_x_min, self.v_y_min])
        x_min = np.hstack((u_min,)*self.lag + (y_min,)*self.lag)

        u_max = np.array([self.a_max, self.delta_max])
        y_max = np.array([self.track_width, self.mu_max, self.delta_v_x_max, self.v_y_max])
        x_max = np.hstack((u_max,)*self.lag + (y_max,)*self.lag)

        while True:
            yield(np.random.uniform(x_min, x_max))

    def u_pf_y_p_iterator(self) -> Iterable[np.ndarray]:
        u_min = np.array([self.a_min, -self.delta_max])
        y_min = np.array([-self.track_width, self.mu_min, self.delta_v_x_min, self.v_y_min])
        x_min = np.hstack((u_min,)*(self.lag+self.L) + (y_min,)*self.lag)

        u_max = np.array([self.a_max, self.delta_max])
        y_max = np.array([self.track_width, self.mu_max, self.delta_v_x_max, self.v_y_max])
        x_max = np.hstack((u_max,)*(self.lag+self.L) + (y_max,)*self.lag)

        while True:
            yield(np.random.uniform(x_min, x_max))
