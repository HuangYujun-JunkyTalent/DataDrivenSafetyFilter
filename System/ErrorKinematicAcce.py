import casadi as cas
import numpy as np
import numpy.linalg as npl
from scipy import signal
from dataclasses import dataclass
from typing import Optional, Tuple

import math

from System.LTI import LTI, LTIParams

@dataclass
class KinematicAcceModelParams:
    l_r: float
    l_f: float
    Ts: float

class KinematicAcceModel:
    '''
    Input [a, delta] as np.ndarray; state [x_p_t, y_p_t, Psi_t, v_t] as np.ndarray
    '''
    _l_r: float
    _l_f: float # kinematic parameters
    _Ts: float # sampling time
    @property
    def Ts(self):
        return self._Ts

    _state: np.ndarray # actual state of the system, [x_p, y_p, Psi, v]
    @property
    def state(self):
        return self._state
    _z: cas.SX # state variable, [x_p, y_p, Psi, v]
    _u: cas.SX # input varialbe, [a, delta]
    _beta: cas.SX # "transformed steering input"
    _f_c: cas.SX # kinematic ode, input is [a, delta]
    _F_c: cas.Function # integrator of kinematic system

    def __init__(self, params: KinematicAcceModelParams, state0: np.ndarray) -> None:
        self._l_r = params.l_r
        self._l_f = params.l_f
        self._Ts = params.Ts
        self._state = state0

        self._z = cas.SX.sym('_z', 4)
        self._u = cas.SX.sym('_u', 2)
        self._beta = cas.SX.sym('_beta', 1)
        self._f_c = cas.SX.sym('_f_c', 4)
        self._beta = cas.atan(self._l_r/(self._l_r+self._l_f)*cas.tan(self._u[1]))
        self._f_c[0] = self._z[3]*cas.cos(self._z[2]+self._beta)
        self._f_c[1] = self._z[3]*cas.sin(self._z[2]+self._beta)
        self._f_c[2] = self._z[3]*cas.sin(self._beta)/self._l_r
        self._f_c[3] = self._u[0]
        # Pay attention to the name here: x is state of ode solver!
        ode = {'x':self._z, 'u':self._u, 'ode':self._f_c}
        self._F_c = cas.integrator('F', 'cvodes', ode, 0, self._Ts)
    
    def step(self, input: np.ndarray) -> np.ndarray:
        '''update and output system state'''
        state_t = self._state

        state_tp = self._F_c(x0=state_t, u=input)
        self._state = np.array(state_tp['xf']).flatten() # type and dimension conversion
        return state_t
    
    def set_state(self, state: np.ndarray) -> None:
        self._state = state


@dataclass
class LinearizedErrorKinematicAcceModelParams:
    kinematic_acceleration_params: KinematicAcceModelParams
    # state0: np.ndarray # initial system state [x_p_0, y_p_0, Psi_0, v_0]

    c: float # curvature of track center line, >=0
    state_s0: np.ndarray # position and orientation of starting point of track, [x_p_0, y_p_0, Psi_0]
    v_0: float # reference velocity

    A_y: Optional[np.matrix] = None
    b_y: Optional[np.matrix] = None
    A_u: Optional[np.matrix] = None
    b_u: Optional[np.matrix] = None
    # Noise properties
    A_n: Optional[np.matrix] = None
    b_n: Optional[np.matrix] = None


class LinearizedErrorKinematicAcceModel(LTI):
    '''
    Kinematic Model of Chronos, includes a Linearized version of error dynamics
    use [a, delta] as input
    '''
    kinematic_model: KinematicAcceModel

    _c: float # curvature of track center line, >=0
    _state_s0: np.ndarray # position and orientation of starting point of track, [x_p_0, y_p_0, Psi_0]
    _v_0: float # reference velocity
    _p_Center: Optional[np.ndarray] # position of certer of circle, only valid if self._c > 0
    _R: Optional[float] # radius of circle, only valid if self._c > 0

    _state: np.ndarray # error state, [e_lat, mu, v]
    @property
    def state(self):
        return self._state
    _zero_state: np.ndarray # zero state for linearized error dynamics, [e_lat = 0, mu = 0, v = self._v_0]
    _zero_input: np.ndarray # zero input for linearized error dynamics, [a = 0, delta = delta_0]

    # varialbes for casadi
    _x_r: cas.SX # state variable, [e_lat, mu, v]
    _u: cas.SX # input varialbe, [a, delta]
    _beta: cas.SX # "transformed steering input"
    _f_c_r: cas.SX # kinematic error ode, input is [a, delta]
    _F_c_r: cas.Function # integrator of kinematic error system

    def __init__(self, params: LinearizedErrorKinematicAcceModelParams, state_0: np.ndarray) -> None:
        '''
        state_0: initial global state of system, [x_p_0, y_p_0, Psi_0, v_0]
        '''
        self.kinematic_model = KinematicAcceModel(params.kinematic_acceleration_params, state_0)

        self._c = params.c
        self._state_s0 = params.state_s0
        self._v_0 = params.v_0
        if self._c > 0: # circle
            self._R =  1/self._c
            unit_vec = np.array([-math.sin(self._state_s0[2]), math.cos(self._state_s0[2])])
            self._p_Center = self._state_s0[0:2] + self._R * unit_vec
        self._zero_state = self.get_zero_state()
        self._zero_input = self.get_zero_input()
        self._state = self.get_error_state()
        assert abs(self._state[1]) <= math.pi/2, 'mu is not in [-pi/2, pi/2]'

        # setup non-linear error dynamics
        self._x_r = cas.SX.sym('_x_r', 3)
        self._u = cas.SX.sym('_u', 2)
        self._beta = cas.SX.sym('_beta', 1)
        self._f_c_r = cas.SX.sym('_f_c_r', 3)
        self._beta = cas.atan( self.kinematic_model._l_r/(self.kinematic_model._l_r+self.kinematic_model._l_f)*cas.tan(self._u[1]) )
        # ode function for error dynamics
        self._f_c_r[0] = self._x_r[2]*cas.sin(self._x_r[1] + self._beta)
        self._f_c_r[1] = self._x_r[2]/self.kinematic_model._l_r*cas.sin(self._beta) - self._c*self._x_r[2]*cas.cos(self._x_r[1] + self._beta)/(1-self._c*self._x_r[0])
        self._f_c_r[2] = self._u[0]
        # integrator for error dynamics
        ode = {'x':self._x_r, 'u':self._u, 'ode':self._f_c_r}
        self._F_c_r = cas.integrator('F_r', 'cvodes', ode, 0, self.kinematic_model.Ts)

        # get linearized error dynamics
        A, B, C, D = self.get_lti()
        lti_params = LTIParams(
            np.matrix(A), np.matrix(B), np.matrix(C), np.matrix(D),
            np.matrix(self._state-self._zero_state).transpose(),
            params.A_y, params.b_y, params.A_u, params.b_u, params.A_n, params.b_n
        )
        LTI.__init__(self, lti_params)
        
    def step(self, input: np.matrix) -> Tuple[np.matrix, np.matrix]:
        """ Update system state and give output
        Note that noise is due to both linearization and real noise

        Input:
            system input
        Returns:
            (unnoised system output, output noise)
        Rasies:
            Exception if input shape not correct
        """
        # get differrent inputs
        matrix_input = input
        delta_input = np.array(matrix_input).flatten()
        # remember to add zero-state input!
        input_to_kinematic = delta_input + self._zero_input

        # get noise and output
        y: np.matrix = np.matmul(self._C, self._x) + np.matmul(self._D, matrix_input)
        # e_lin = self._C@np.matrix(self._state-self._zero_state).transpose() - y
        e_lin = self._C@np.matrix(self.get_error_state()-self._zero_state).transpose() - y
        # If noise matrix is not None, get and add noise
        if self._A_n is not None:
            e = e_lin + self.get_noise()
        else:
            e = e_lin

        # update states of systems
        self.kinematic_model.step(input_to_kinematic) # global kinematic model
        new_state_dict = self._F_c_r(x0=self._state, u=input_to_kinematic)
        self._state = np.array(new_state_dict['xf']).flatten() # error dynamics
        self._x = np.matmul(self._A, self._x) + np.matmul(self._B, matrix_input)

        return y, e

    def get_lti(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        return LTI system for error dynamics
        '''
        # calculate jacobians
        J_x_r = cas.jacobian(self._f_c_r, self._x_r)
        J_u_r = cas.jacobian(self._f_c_r, self._u)
        # get functions of jacobians
        A_c_r_fun = cas.Function('A_c_r_fun', [self._x_r, self._u], [J_x_r])
        B_c_r_fun = cas.Function('B_c_r_fun', [self._x_r, self._u], [J_u_r])
        # get dynamic matrices
        A_c_r = A_c_r_fun(self._zero_state, self._zero_input)
        B_c_r = B_c_r_fun(self._zero_state, self._zero_input)
        
        # ues exact discretization
        o = 3 # number of variables observable
        A, B, C, D , _ = signal.cont2discrete((A_c_r, B_c_r, np.hstack(( np.eye(o),np.zeros((o,3-o)) )), np.zeros((o,2))), self.kinematic_model.Ts)
        return A, B, C, D

    def get_zero_state(self) -> np.ndarray:
        '''
        return zero state for error dynamics, [0, mu_0, v_0]
        '''
        if self._c == 0:
            return np.array([0, 0, 0])
        else:
            beta_0 = math.asin(self.kinematic_model._l_r/self._R)
            return np.array([0, -beta_0, self._v_0])

    def get_zero_input(self) -> np.ndarray:
        '''
        return zero input for error dynamics, [0, delta_0]
        '''
        if self._c == 0:
            return np.array([0, 0])
        else:
            beta_0 = math.asin(self.kinematic_model._l_r/self._R)
            delta_0 = math.atan((self.kinematic_model._l_r+self.kinematic_model._l_f)/self.kinematic_model._l_r*math.tan(beta_0))
            return np.array([0, delta_0])
        
    def get_following_input(self, v: float) -> np.ndarray:
        '''
        Return path-following input, [a, delta], for given speed v.
        '''
        if self._c == 0:
            return np.array([0, 0])
        else:
            beta_0 = math.asin(self.kinematic_model._l_r/self._R)
            delta_0 = math.atan((self.kinematic_model._l_r+self.kinematic_model._l_f)/self.kinematic_model._l_r*math.tan(beta_0))
            return np.array([0, delta_0])
    
    def get_error_state(self) -> np.ndarray:
        '''
        return error state [e_lat, mu, v], based on current global state
        Error mu ranges in [-pi, pi]
        Ideally the state will stay in [-pi/2, pi/2]
        '''
        # get reference position on center line
        s_t = self.get_reference_center_line_pos()
        # get error state
        if self._c == 0:
            unit_vec = np.array([-math.sin(s_t[2]), math.cos(s_t[2])])
            e_lat = np.inner(unit_vec, self.kinematic_model.state[0:2] - s_t[0:2])
        else:
            e_lat = self._R - npl.norm(self.kinematic_model.state[0:2] - self._p_Center)
        mu = self.kinematic_model.state[2] - s_t[2]
        mu = np.mod(mu + np.pi, 2*np.pi) - np.pi # range in [-pi, pi]
        v = self.kinematic_model.state[3]
        return np.array([e_lat, mu, v])
        
        
    def get_reference_center_line_pos(self) -> np.ndarray:
        '''
        return reference position on center line [x_t, y_t, Psi_t]
        Psi_t ranges in [-pi, pi]
        '''
        if self._c == 0: # reference is straight line
            unit_vec = np.array([math.cos(self._state_s0[2]), math.sin(self._state_s0[2])])
            Delta_vec = self.kinematic_model.state[0:2] - self._state_s0[0:2]
            p_t = self._state_s0[0:2] + np.inner(Delta_vec, unit_vec) * unit_vec
            return np.hstack((p_t, self._state_s0[2]))
        else: # reference is a circle
            unit_vec = self.kinematic_model.state[0:2] - self._p_Center
            assert npl.norm(unit_vec) != 0, "Mass center of vehicle at center of circle!"
            unit_vec = unit_vec / npl.norm(unit_vec)
            p_xy = self._p_Center + self._R * unit_vec
            unit_x = self._state_s0[0:2]-self._p_Center
            unit_x = unit_x / npl.norm(unit_x)
            unit_y = np.array([-unit_x[1], unit_x[0]])
            Phi_t = math.atan2(np.inner(unit_y, unit_vec), np.inner(unit_x, unit_vec)) + self._state_s0[2]
            Phi_t = np.mod(Phi_t + np.pi, 2*np.pi) - np.pi # range in [-pi, pi]
            return np.hstack((p_xy, Phi_t))
