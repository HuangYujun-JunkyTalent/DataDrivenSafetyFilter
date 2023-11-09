from typing import List, Dict, Tuple, Callable
import itertools
from itertools import islice
import math
import os
import pickle
from copy import deepcopy

import casadi as cas
import numpy as np
from scipy import signal
from dataclasses import dataclass
import matplotlib.pyplot as plt

from IOData.IOData import IOData, InputRule
from IOData.IODataWith_l import IODataWith_l
from System.DynamicErrorModel import DynamicErrorModelVxy
from tools.simualtion_results import Results, PlotStyle
from tools.dataset_analyse import get_datasets_hankel_matrix, Sampler


class DynamicModelPredictor:
    dynamic_error_model: DynamicErrorModelVxy
    sampler: Sampler

    L: int # prediction horizon
    lag: int # number of steps for extended initial condition
    weight_xi: np.matrix # weight matrix for xi, used for weighting trajectory slices
    f: Callable[[float], float] # function for evaluating weight of trajectory slice, w = f(d)
    d_range: float # ommit all trajectory slices with d > d_range
    min_num_slices: int # minimum number of slices to be used for prediction
    portion_slices: float # portion of slices to be used for prediction

    io_data_list: List[IOData]

    weight_y: np.matrix # weight matrix for y, used for calculating prediction error

    def __init__(self, dynamic_error_model: DynamicErrorModelVxy, sampler: Sampler,
                    L: int, lag: int,
                    weight_xi: np.matrix, f: Callable[[float], float],
                    d_range: float, min_num_slices: int, portion_slices: float,
                    io_data_list: List[IOData],
                    weight_y: np.matrix):
        self.dynamic_error_model = dynamic_error_model
        self.sampler = sampler
    
        self.L = L
        self.lag = lag

        self.weight_xi = weight_xi
        self.f = f
        self.d_range = d_range
        self.min_num_slices = min_num_slices
        self.portion_slices = portion_slices
        self.io_data_list = io_data_list

        self.weight_y = weight_y

        self.sampled_states: List[np.ndarray] = []
        self.sampled_initial_inputs: List[List[np.matrix]] = []
        self.sampled_future_inputs: List[List[np.matrix]] = []

    def get_estimation_matrix(self, xi_t: np.matrix) -> np.matrix:
        """
        Get estimation matrix for given xi_t, which is the extended state of system
        xi: extended initial condition, of shape (lag * (m + p), 1)
        """
        m = self.io_data_list[0]._input_data[0].shape[0]
        p = self.io_data_list[0]._output_data[0].shape[0]
        Huy, H_future = get_datasets_hankel_matrix(self.io_data_list, self.lag, self.L)
        Up = Huy[:m*self.lag,:]
        Uf = Huy[m*self.lag:m*(self.lag+self.L),:]
        Yp = Huy[m*(self.lag+self.L):,:]
        Yf = H_future
        width_H = Up.shape[1]

        delta_H_x_t  = np.vstack((Up, Yp)) - xi_t
        d_array = np.zeros((width_H,))
        for i in range(width_H):
            d_array[i] = np.sqrt((delta_H_x_t[:,i].T @ self.weight_xi @ delta_H_x_t[:,i])[0,0])

        # find a proper diatance range
        num_slices = max(self.min_num_slices, int(self.portion_slices*width_H))
        d_range = max(self.d_range, np.partition(d_array, num_slices)[num_slices])
        # print("d_range: ", d_range)

        d_inv_array = np.zeros((width_H,))
        for i in range(width_H-1, -1, -1):
            if d_array[i] < d_range:
                d_inv_array[i] = self.f(d_array[i])
            else:
                # np.delete(d_array, i)
                # np.delete(H_x_past, i, axis=1)
                # np.delete(H_u_future, i, axis=1)
                # np.delete(H_x_future, i, axis=1)
                d_inv_array[i] = 0
        D_inv = np.diag(d_inv_array)

        # calculate the estimation matrix
        H_u_y_1 = np.vstack(( Up, Uf, Yp, np.matrix(np.ones((1, width_H))) ))
        D_inv_Huy_T = D_inv @ H_u_y_1.T
        Phi = Yf @ D_inv_Huy_T @ np.linalg.pinv(H_u_y_1 @ D_inv_Huy_T)

        return Phi
    
    def predict(self, xi_t: np.matrix, u_f: List[np.matrix]) -> List[np.matrix]:
        """
        Predict future states of the system
        Args:
            xi: extended initial condition, of shape (lag * (m + p), 1)
            u_future: list of future states to be applied
        Return:
            y_future: list of future states to be applied
        """
        m = self.io_data_list[0]._input_data[0].shape[0]
        p = self.io_data_list[0]._output_data[0].shape[0]
        Phi = self.get_estimation_matrix(xi_t)
        u_future_matrix = np.vstack(u_f)

        y_future_matrix = Phi[:,:m*self.lag] @ xi_t[:m*self.lag] + Phi[:,m*self.lag:m*(self.lag+self.L)] @ u_future_matrix + Phi[:,m*(self.lag+self.L):-1] @ xi_t[m*self.lag:] + Phi[:,-1]

        return np.split(y_future_matrix, self.L)
    
    def propogate_real_system(self, x_t: np.ndarray, u_future: List[np.matrix]) -> List[np.matrix]:
        self.dynamic_error_model.set_error_state(x_t)
        y_future = []
        for u in u_future:
            y, _ = self.dynamic_error_model.step(u)
            y_future.append(y)
        
        return y_future
    
    def prediction_error(self, x_t: np.ndarray, u_initial: List[np.matrix], u_future: List[np.matrix]) -> np.ndarray:
        """Start from a state x_t, first get extended initial using u_initial, then predict using u_future
        Calculate the prediction error
        return 1-d array, each element is the prediction error for one step"""
        self.dynamic_error_model.set_error_state(x_t)
        u_init_list = []
        y_init_list = []
        try:
            for u in u_initial:
                y, _ = self.dynamic_error_model.step(u)
                u_init_list.append(u)
                y_init_list.append(y)
        except Exception as e:
                print(e)
                print("\nError occured when propogating real system:")
                print("state: ", self.dynamic_error_model.state)
                print("input: ", u)
                return np.array([np.nan]*self.L)
        xi_t = np.vstack(u_init_list + y_init_list)

        y_pred_list = self.predict(xi_t, u_future)
        y_real_list: List[np.matrix] = []

        for u in u_future:
            try:
                y, _ = self.dynamic_error_model.step(u)
                y_real_list.append(y)
            except Exception as e:
                print(e)
                print("\nError occured when propogating real system:")
                print("state: ", self.dynamic_error_model.state)
                print("input: ", u)
                break
                
        error_list = np.empty(shape=(len(y_pred_list),))
        error_list.fill(np.nan)
        for i in range(len(y_pred_list)):
            y_pred = y_pred_list[i]
            y_real = y_real_list[i]
            y_diff = y_pred - y_real
            error_list[i] = (y_diff.T @ self.weight_y @ y_diff)[0,0]
        
        return error_list

    def sample_state_and_input(self, n_states: int) -> Tuple[List[np.ndarray], List[List[np.matrix]], List[List[np.matrix]]]:
        """Use sampler to sample state and input, return list of states and inputs"""
        state_list: List[np.ndarray] = []
        u_intial_list_list: List[List[np.matrix]] = []
        u_future_list_list: List[List[np.matrix]] = []
        for state in islice(self.sampler.state_iterator(), n_states):
            state_list.append(state)
            u_intial_list: List[np.matrix] = []
            u_future_list: List[np.matrix] = []
            for u in islice(self.sampler.input_iterator(), self.lag):
                u_intial_list.append(np.matrix(u).T)
            for u in islice(self.sampler.input_iterator(), self.L):
                u_future_list.append(np.matrix(u).T)
            u_intial_list_list.append(u_intial_list)
            u_future_list_list.append(u_future_list)
        self.sampled_states = state_list
        self.sampled_initial_inputs = u_intial_list_list
        self.sampled_future_inputs = u_future_list_list
        return state_list, u_intial_list_list, u_intial_list_list

    def get_prediction_error(self) -> List[List[float]]:
        """Use sampler to sample state and input, then calculate prediction error
        return list of prediction error"""
        error_list: List[List[float]] = []
        for state, u_initial_list, u_future_list in zip(self.sampled_states, self.sampled_initial_inputs, self.sampled_future_inputs):
            error_for_state = self.prediction_error(state, u_initial_list, u_future_list)
            error_list.append(error_for_state)
        return error_list
