import casadi as cas
import numpy as np
import numpy.linalg as npl
from scipy import signal
from dataclasses import dataclass
from typing import Optional, Tuple

import math
from warnings import warn

from System.LATI import LATI, LATIParams

@dataclass
class KinematicAcceModelParams:
    l_r: float
    l_f: float
    m: float
    Ts: float

class KinematicAcceModel:
    '''
    Input [a, delta] as np.ndarray; state [x_p_t, y_p_t, Psi_t, v_t] as np.ndarray
    '''
    _l_r: float
    _l_f: float # kinematic parameters
    _m: float # mass of vehicle
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
        self._m = params.m
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
        self._f_c[3] = self._u[0]/self._m
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
    """
    state_s0: position and direction of starting point, [x_p_0, y_p_0, Psi_0]
    """
    kinematic_acceleration_params: KinematicAcceModelParams

    cur: float # curvature of track center line, 0: straight, >0: left turn, <0: right turn
    state_s0: np.ndarray # position and tangent direction of starting point of track, [x_p_0, y_p_0, Psi_0]
    v_0: float # reference velocity

    A_y: Optional[np.matrix] = None
    b_y: Optional[np.matrix] = None
    A_u: Optional[np.matrix] = None
    b_u: Optional[np.matrix] = None
    # Noise properties
    A_n: Optional[np.matrix] = None
    b_n: Optional[np.matrix] = None


class LinearizedErrorKinematicAcceModel(LATI):
    '''
    Kinenmatic Error dynamic model, with state: x_r = [e_lat, mu, v, l], input u = [a, delta]
    also calculates l, which represents the length of trajectory-arc that has been traveled.

    The model is linearized around x_r_lin, u_r_lin, resulting into a linear affine time invariant system.
    output is [e_lat, mu, v-v_0]
    '''
    kinematic_model: KinematicAcceModel
    @property
    def Ts(self):
        return self.kinematic_model.Ts

    _cur: float # curvature of track center line, >=0
    @property
    def cur(self):
        return self._cur
    _state_s0: np.ndarray # position and orientation of starting point of track, [x_p_0, y_p_0, Psi_0]
    @property
    def segment_start(self):
        return self._state_s0
    _v_0: float # reference velocity
    _p_Center: Optional[np.ndarray] # position of certer of circle, only valid if self._cur > 0
    _R: Optional[float] # radius of circle, only valid if self._cur > 0

    _state: np.ndarray # error state, [e_lat, mu, v, l]
    _l_0: float # setpoint for l, original value is 0
    @property
    def state(self):
        return self._state
    _zero_state: np.ndarray # zero state for linearized error dynamics, [e_lat = 0, mu = mu_0, v = self._v_0, l = 0]
    _zero_input: np.ndarray # zero input for linearized error dynamics, [a = 0, delta = delta_0]

    _x_r_lin: np.ndarray
    _u_r_lin: np.ndarray # state and input around which the system is linearized

    # varialbes for casadi
    _x_r: cas.SX # state variable, [e_lat, mu, v]
    _u: cas.SX # input varialbe, [a, delta]
    _beta: cas.SX # "transformed steering input"
    _f_c_r: cas.SX # kinematic error ode, input is [a, delta]
    _f_c_fun: cas.Function # kinematic ode function
    _F_c_r: cas.Function # integrator of kinematic error system

    @property
    def a_max(self):
        return self.b_u[0,0]/self.kinematic_model._m

    def __init__(self, params: LinearizedErrorKinematicAcceModelParams, state_0: np.ndarray) -> None:
        '''
        state_0: initial global state of system, [x_p_0, y_p_0, Psi_0, v_0]
        '''
        self.kinematic_model = KinematicAcceModel(params.kinematic_acceleration_params, state_0)

        self._cur = params.cur
        self._state_s0 = params.state_s0
        self._v_0 = params.v_0
        if self._cur != 0: # circle, attention: >0: left, <0: right!
            self._R =  1/np.abs(self._cur)
            unit_vec = np.sign(self._cur)*np.array([-math.sin(self._state_s0[2]), math.cos(self._state_s0[2])])
            self._p_Center = self._state_s0[0:2] + self._R * unit_vec
        else:
            self._R = None
            self._p_Center = None
        self._l_0 = 0
        self._zero_state = self.get_zero_state()
        self._zero_input = self.get_zero_input()
        self._state = self.get_error_state()
        if abs(self._state[1]) > math.pi/2:
            warn('mu is not in [-pi/2, pi/2]')
        # define linearized points
        self._x_r_lin = self._zero_state # [0, mu_0, self._v_0, 0]
        self._u_r_lin = self._zero_input # [0, delta_0]
        # self._x_r_lin = np.array([0,0,self._v_0,0]) # [0, 0, self._v_0, 0]
        # self._u_r_lin = np.array([0,0]) # [0, 0]

        # setup non-linear error dynamics
        self._x_r = cas.SX.sym('_x_r', 4)
        self._u = cas.SX.sym('_u', 2)
        self._beta = cas.SX.sym('_beta', 1)
        self._f_c_r = cas.SX.sym('_f_c_r', 4)
        self._beta = cas.atan( self.kinematic_model._l_r/(self.kinematic_model._l_r+self.kinematic_model._l_f)*cas.tan(self._u[1]) )
        # ode function for error dynamics
        self._f_c_r[0] = self._x_r[2]*cas.sin(self._x_r[1] + self._beta)
        self._f_c_r[1] = self._x_r[2]/self.kinematic_model._l_r*cas.sin(self._beta) - self._cur*self._x_r[2]*cas.cos(self._x_r[1] + self._beta)/(1-self._cur*self._x_r[0])
        self._f_c_r[2] = self._u[0]/self.kinematic_model._m
        self._f_c_r[3] = self._x_r[2]*cas.cos(self._x_r[1] + self._beta)/(1-self._cur*self._x_r[0])
        # function to get value of ode function
        self._f_c_fun = cas.Function('f_c', [self._x_r, self._u], [self._f_c_r])
        # integrator for error dynamics
        ode = {'x':self._x_r, 'u':self._u, 'ode':self._f_c_r}
        self._F_c_r = cas.integrator('F_r', 'cvodes', ode, 0, self.kinematic_model.Ts)

        # get linearized error dynamics
        A, B, C, D, c, r = self.get_lati()
        lati_params = LATIParams(
            np.matrix(A), np.matrix(B), np.matrix(C), np.matrix(D), np.matrix(c), np.matrix(r),
            params.A_y, params.b_y, params.A_u, params.b_u, params.A_n, params.b_n
        )
        LATI.__init__(self, lati_params, np.matrix(self._state-self._x_r_lin).transpose())
        
    def step_lin(self, input: np.matrix) -> Tuple[np.matrix, np.matrix, np.matrix]:
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
        input_to_kinematic = np.array(matrix_input).flatten()
        # input_to_kinematic = np.array(matrix_input).flatten() + self._zero_input

        # get noise and output
        y: np.matrix = np.matmul(self._C, self._x) + np.matmul(self._D, matrix_input) + self._r
        e_lin = self._C@np.matrix(self._state-np.array([0, 0, self._v_0, 0])).transpose() - y
        # e_lin = self._C@np.matrix(self.get_error_state()).transpose() - y
        # If noise matrix is not None, get the additional noise
        if self._A_n is not None:
            noise = self.get_noise()
        else:
            noise = np.matrix(np.zeros(y.shape))

        # update states of systems
        self.kinematic_model.step(input_to_kinematic) # global kinematic model
        new_state_dict = self._F_c_r(x0=self._state, u=input_to_kinematic)
        self._state = np.array(new_state_dict['xf']).flatten() # error dynamics
        self._x = np.matmul(self._A, self._x) + np.matmul(self._B, matrix_input) + self._c

        return y, e_lin, noise
    
    def step(self, input: np.matrix) -> Tuple[np.matrix, np.matrix]:
        """Compatible with previous convention. Combine e_lin and e
        """
        y, e_lin, noise = self.step_lin(input)
        e = e_lin + noise
        return y, e

    def get_lati(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        A_c_r = A_c_r_fun(self._x_r_lin, self._u_r_lin)
        B_c_r = B_c_r_fun(self._x_r_lin, self._u_r_lin)
        c_c = np.array(self._f_c_fun(self._x_r_lin, self._u_r_lin)).reshape((-1,1)) - np.array(B_c_r@self._u_r_lin).reshape((-1,1))
        B_c_r_affine = np.hstack((B_c_r, c_c)) # includs the affine term to be discretized

        # ues exact discretization
        o = 3 # number of variables observable
        A, B, C, D , _ = signal.cont2discrete((A_c_r, B_c_r_affine, np.hstack(( np.eye(o),np.zeros((o,4-o)) )), np.zeros((o,2))), self.kinematic_model.Ts)
        # affine term
        c = B[:,-1]
        c = c.reshape((-1, 1))
        r = C@(self._x_r_lin - np.array([0,0,self._v_0,0]))
        # c = self._zero_state - A@self._zero_state
        r = r.reshape((-1, 1))
        return A, B[:,:-1], C, D, c, r # remember to remove the affine term from B

    def get_zero_state(self) -> np.ndarray:
        '''
        return zero state for error dynamics, [0, mu_0, v_0, 0]
        '''
        if self._cur == 0: # straight
            return np.array([0, 0, self._v_0, 0])
        elif self._cur>0: # left
            beta_0 = math.asin(self.kinematic_model._l_r/self._R)
            return np.array([0, -beta_0, self._v_0, 0])
        else: #right
            beta_0 = -math.asin(self.kinematic_model._l_r/self._R)
            return np.array([0, -beta_0, self._v_0, 0])

    def get_zero_input(self) -> np.ndarray:
        '''
        return zero input for error dynamics, [0, delta_0]
        '''
        if self._cur == 0: #straight
            return np.array([0, 0])
        elif self._cur>0: # left turn
            beta_0 = math.asin(self.kinematic_model._l_r/self._R)
            delta_0 = math.atan((self.kinematic_model._l_r+self.kinematic_model._l_f)/self.kinematic_model._l_r*math.tan(beta_0))
            return np.array([0, delta_0])
        else: # right turn
            beta_0 = -math.asin(self.kinematic_model._l_r/self._R)
            delta_0 = math.atan((self.kinematic_model._l_r+self.kinematic_model._l_f)/self.kinematic_model._l_r*math.tan(beta_0))
            return np.array([0, delta_0])
        
    def get_following_input(self, v: float) -> np.ndarray:
        '''
        Return path-following input, [a, delta], for given speed v.
        '''
        if self._cur == 0:
            return np.array([0, 0])
        elif self._cur>0: # left turn
            beta_0 = math.asin(self.kinematic_model._l_r/self._R)
            delta_0 = math.atan((self.kinematic_model._l_r+self.kinematic_model._l_f)/self.kinematic_model._l_r*math.tan(beta_0))
            return np.array([0, delta_0])
        else: # right turn
            beta_0 = -math.asin(self.kinematic_model._l_r/self._R)
            delta_0 = math.atan((self.kinematic_model._l_r+self.kinematic_model._l_f)/self.kinematic_model._l_r*math.tan(beta_0))
            return np.array([0, delta_0])
    
    def get_error_state(self, round: float = 0.0) -> np.ndarray:
        '''
        return error state [e_lat, mu, v, l], based on current global state
        Error mu ranges in [-pi, pi]
        Ideally the state mu will stay in [-pi/2, pi/2]
        l takes the reference point l_0 into consideration. if reference is circle, l reanges in [round, round+2*pi*R]
        '''
        # get reference position on center line
        s_t = self.get_reference_center_line_pos()
        # get error state
        if self._cur == 0:
            unit_vec = np.array([-math.sin(s_t[2]), math.cos(s_t[2])])
            e_lat = np.inner(unit_vec, self.kinematic_model.state[0:2] - s_t[0:2])
            unit_x = np.array([math.cos(s_t[2]), math.sin(s_t[2])])
            l = np.inner(unit_x, self.kinematic_model.state[0:2] - self._state_s0[0:2]) - self._l_0
            mu = self.kinematic_model.state[2] - s_t[2]
            mu = np.mod(mu + np.pi, 2*np.pi) - np.pi # range in [-pi, pi]
        elif self._cur>0:
            e_lat = self._R - npl.norm(self.kinematic_model.state[0:2] - self._p_Center)
            l = self._R * (s_t[2] - self._state_s0[2]) - self._l_0
            l = np.mod(l-round, 2*np.pi*self._R)+round # range in [round, round+2*pi*R]
            mu = self.kinematic_model.state[2] - s_t[2]
            mu = np.mod(mu + np.pi, 2*np.pi) - np.pi # range in [-pi, pi]
        else:
            e_lat = -self._R + npl.norm(self.kinematic_model.state[0:2] - self._p_Center)
            l = self._R * (-s_t[2] + self._state_s0[2]) - self._l_0
            l = np.mod(l-round, 2*np.pi*self._R)+round # range in [round, round+2*pi*R]
            mu = self.kinematic_model.state[2] - s_t[2]
            mu = np.mod(mu + np.pi, 2*np.pi) - np.pi # range in [-pi, pi]
        v = self.kinematic_model.state[3]
        return np.array([e_lat, mu, v, l])
    
    
    def get_reference_center_line_pos(self) -> np.ndarray:
        '''
        return reference position on center line [x_t, y_t, Psi_t]
        Psi_t ranges in [-pi, pi]
        '''
        if self._cur == 0: # reference is straight line
            unit_vec = np.array([math.cos(self._state_s0[2]), math.sin(self._state_s0[2])])
            Delta_vec = self.kinematic_model.state[0:2] - self._state_s0[0:2]
            p_t = self._state_s0[0:2] + np.inner(Delta_vec, unit_vec) * unit_vec
            return np.hstack((p_t, self._state_s0[2]))
        elif self._cur>0: # reference is a left turn
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
        else: # reference is a right turn
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
    
    def set_kinematic_model_state(self, kinematic_state: np.ndarray, round: float = 0.0) -> None:
        '''
        Set state for kinematic model. Also reset the error state! \\
        kinematic_state: [x, y, psi, v] in global reference frame \\
        round: if segment is circle, return l ranging in [round, round+2*pi*R]
        '''
        self.kinematic_model.set_state(kinematic_state)
        error_state_0 = self.get_error_state(round)
        self._state = error_state_0
        self._x = np.matrix(self.state-self._x_r_lin).transpose()

    def set_error_state(self, error_state: np.ndarray) -> None:
        '''
        Set error state [e_lat, mu, v, l]. l takes the reference point l_0 into consideration
        Also reset the kinematic state
        '''
        self._state = error_state
        self._x = np.matrix(self.state-self._x_r_lin).transpose()
        kinematic_state = np.array([0.0,0.0,0.0,0.0])
        if self._cur == 0: # straight line
            unit_x_local = np.array([math.cos(self._state_s0[2]), math.sin(self._state_s0[2])])
            unit_y_local = np.array([-unit_x_local[1], unit_x_local[0]])
            kinematic_state[0:2] = self._state_s0[0:2] + \
                (self._state[3]+self._l_0)*unit_x_local + self._state[0]*unit_y_local # x,y position
            kinematic_state[2] = self._state[1] + self._state_s0[2] # psi = mu + psi_0
            kinematic_state[3] = self._state[2] # velocity
        elif self._cur>0: # left turn
            l_traveled = self._state[3] + self._l_0
            theta_traveled = l_traveled/self._R
            psi_ref = theta_traveled + self._state_s0[2]
            theta_from_R = psi_ref - 0.5*math.pi
            kinematic_state[0:2] = self._p_Center + (self._R-self._state[0])*np.array([math.cos(theta_from_R), math.sin(theta_from_R)])
            heading_angle = psi_ref + self._state[1] # heading angle
            kinematic_state[2] = np.mod(heading_angle + np.pi, 2*np.pi) - np.pi # range in [-pi, pi]
            kinematic_state[3] = self._state[2] # velocity
        else: # right turn
            l_traveled = self._state[3] + self._l_0
            theta_traveled = l_traveled/self._R
            psi_ref = self._state_s0[2] - theta_traveled
            theta_from_R = 0.5*math.pi + psi_ref
            kinematic_state[0:2] = self._p_Center + (self._R+self._state[0])*np.array([math.cos(theta_from_R), math.sin(theta_from_R)])
            heading_angle = psi_ref + self._state[1] # heading angle
            kinematic_state[2] = np.mod(heading_angle + np.pi, 2*np.pi) - np.pi # range in [-pi, pi]
            kinematic_state[3] = self._state[2] # velocity
        self.kinematic_model.set_state(kinematic_state)

    def set_l_0(self, l_0: float) -> None:
        '''
        Set the reference point l_0.
        State of the kinematic model not affected
        '''
        self._state[3] = self._state[3] + self._l_0 - l_0
        self._l_0 = l_0
        self._x = np.matrix(self.state-self._x_r_lin).transpose()
