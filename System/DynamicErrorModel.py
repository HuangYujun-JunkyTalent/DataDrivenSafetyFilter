import casadi as cas
import numpy as np
import numpy.linalg as npl
from scipy import signal
from dataclasses import dataclass
from typing import Optional, Tuple, Optional

import math

from System.DynamicModel import DynamicModel, DynamicModelParams, DynamicModelFewOutput


@dataclass
class DynamicErrorModelParams:
    dynamic_params: DynamicModelParams

    cur: float # curvature of track center line, 0: straight, >0: left turn, <0: right turn
    segment_start: np.ndarray # position and tangent direction of starting point of the segment, [x_p_0, y_p_0, Psi_0]
    v_0: float # reference velocity

    # constraint for states
    A_y: Optional[np.matrix] = None
    b_y: Optional[np.matrix] = None
    A_u: Optional[np.matrix] = None
    b_u: Optional[np.matrix] = None
    # Noise properties
    A_n: Optional[np.matrix] = None
    b_n: Optional[np.matrix] = None


class DynamicErrorModel:
    """
    Dynamic model expressed in error dynmaics.
    Implemented in a way that is compatible with LATI class
    """
    _n: int
    _m: int
    _p: int
    @property
    def n(self) -> int:
        return self._n
    @property
    def m(self) -> int:
        return self._m
    @property
    def p(self) -> int:
        return self._p
    params: DynamicErrorModelParams
    @property
    def cur(self) -> float:
        return self.params.cur
    @property
    def segment_start(self):
        return self.params.segment_start
    @property
    def v_0(self) -> float:
        return self.params.v_0
    @property
    def A_y(self) -> Optional[np.matrix]:
        return self.params.A_y
    @property
    def b_y(self) -> Optional[np.matrix]:  
        return self.params.b_y
    @property
    def A_u(self) -> Optional[np.matrix]:
        return self.params.A_u
    @property
    def b_u(self) -> Optional[np.matrix]:
        return self.params.b_u
    @property
    def A_n(self) -> Optional[np.matrix]:
        return self.params.A_n
    @property
    def b_n(self) -> Optional[np.matrix]:
        return self.params.b_n

    dynamic_model: DynamicModel
    @property
    def Ts(self) -> float:
        return self.params.dynamic_params.Ts
    @property
    def kinematic_model(self):
        return self.dynamic_model

    _p_Center: Optional[np.ndarray] # position of certer of circle, only valid if cur > 0
    _R: Optional[float] # radius of circle, only valid if cur > 0

    _state: np.ndarray # error state, [e_lat, mu, v_x, v_y, yaw_rate, l]
    _l_0: float # setpoint for l, original value is 0
    @property
    def state(self):
        return self._state
    _zero_state: np.ndarray # zero state for linearized error dynamics, [e_lat = 0, mu = mu_0, v_x = self._v_0, v_y = v_y_0, r = r0, l = 0], l not useful here
    _zero_input: np.ndarray # zero input for linearized error dynamics, [tau = tau_0, delta = delta_0]

    # varialbes for casadi
    _z_r: cas.SX # state variable, [e_lat, mu, v_x, v_y, yaw_rate, l]
    _u: cas.SX # input varialbe, [tau, delta]
    _beta: cas.SX # "transformed steering input"
    _f_c_r: cas.SX # kinematic error ode, input is [tau, delta]
    _f_c_fun: cas.Function # kinematic ode function
    _F_c_r: cas.Function # integrator of kinematic error system

    _g: cas.SX # output variables
    _g_fun: cas.Function # output function

    @property
    def a_max(self) -> float:
        return self.params.b_u[0,0]/self.params.dynamic_params.m

    def __init__(self, params: DynamicErrorModelParams, initial_state: np.ndarray) -> None:
        """
        initial_state: initial state for dynamic model, [x_p, ]
        """
        # to be comparable with LATI class, we need to set up system constants
        self._n = 5
        self._m = 2
        self._p = 3
        self.params = params
        self.dynamic_model = DynamicModel(params.dynamic_params, initial_state)

        if params.cur != 0: # circle, attention: >0: left, <0: right!
            self._R =  1/np.abs(params.cur)
            unit_vec = np.sign(params.cur)*np.array([-math.sin(params.segment_start[2]), math.cos(params.segment_start[2])])
            self._p_Center = params.segment_start[0:2] + self._R * unit_vec
        else:
            self._R = None
            self._p_Center = None
        self._l_0 = 0

        # setup dynamic error model
        self._z_r = cas.SX.sym('_z_r', 6)
        self._u = cas.SX.sym('_u', 2)
        self._f_c_r = cas.SX.sym('_f_c_r', 6)

        # for readability, get variable names
        e_lat = self._z_r[0]
        mu = self._z_r[1]
        v_x, v_y = self._z_r[2], self._z_r[3]
        yaw_rate = self._z_r[4]
        l = self._z_r[5]

        throttle = self._u[0]
        steer = self._u[1]

        # intermediate variables
        Fx = (params.dynamic_params.Cm1 - params.dynamic_params.Cm2 * v_x) * throttle - params.dynamic_params.Cd * v_x * v_x - params.dynamic_params.Croll
        ar = - cas.atan2(v_y-params.dynamic_params.l_r*yaw_rate, v_x) # rear slip angle
        af = steer - cas.atan2(v_y+params.dynamic_params.l_f*yaw_rate, v_x) # front slip angle
        Fr = params.dynamic_params.Dr * cas.sin(params.dynamic_params.Cr * cas.atan(params.dynamic_params.Br * ar))
        Ff = params.dynamic_params.Df * cas.sin(params.dynamic_params.Cf * cas.atan(params.dynamic_params.Bf * af))

        # dynamic equations
        e_lat_dot = v_x * cas.sin(mu) + v_y * cas.cos(mu)
        mu_dot = yaw_rate - params.cur * (v_x * cas.cos(mu) - v_y * cas.sin(mu)) / (1 - params.cur * e_lat)
        v_x_dot = 1.0 / params.dynamic_params.m * (Fx - Ff * cas.sin(steer) + params.dynamic_params.m * v_y * yaw_rate)
        v_y_dot = 1.0 / params.dynamic_params.m * (Fr + Ff * cas.cos(steer) - params.dynamic_params.m * v_x * yaw_rate)
        yaw_rate_dot = 1.0 / params.dynamic_params.Iz * (Ff * params.dynamic_params.l_f * cas.cos(steer) - Fr * params.dynamic_params.l_r)
        l_dot = (v_x * cas.cos(mu) - v_y * cas.sin(mu)) / (1 - params.cur * e_lat)

        self._f_c_r = cas.vertcat(e_lat_dot, mu_dot, v_x_dot, v_y_dot, yaw_rate_dot, l_dot)
        self._f_c_fun = cas.Function('f_c', [self._z_r, self._u], [self._f_c_r])

        # output equations
        v = cas.sqrt(v_x**2 + v_y**2)
        self._g = cas.vertcat(e_lat, mu, v_x, v_y)
        self._g_fun = cas.Function('g', [self._z_r], [self._g])

        # Set up ODE solver, pay attention to the name here: x is state of ode solver!
        ode = {'x':self._z_r, 'u':self._u, 'ode':self._f_c_r}
        self._F_c_r = cas.integrator('F', 'cvodes', ode, 0, params.dynamic_params.Ts)

        self._zero_input, self._zero_state = self.calculate_zero_input_state()


    def step(self, input: np.ndarray) -> np.ndarray:
        """
        input: [tau, delta]
        """
        state_t = self._state

        state_tp = self._F_c_r(x0=state_t, u=input)
        self._state = np.array(state_tp['xf']).flatten() # type and dimension conversion
        self.dynamic_model.step(input)
 
        return state_t

    def set_error_state(self, error_state: np.ndarray) -> None:
        self._state = error_state
        dynamic_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dynamic_state[3:6] = error_state[2:5] # v_x, v_y, yaw_rate
        cur = self.params.cur
        track_start = self.params.segment_start
        if cur == 0: # straight line
            unit_x_local = np.array([math.cos(track_start[2]), math.sin(track_start[2])])
            unit_y_local = np.array([-unit_x_local[1], unit_x_local[0]])
            dynamic_state[0:2] = track_start[0:2] + \
                (self._state[-1]+self._l_0)*unit_x_local + self._state[0]*unit_y_local # x,y position
            dynamic_state[2] = self._state[1] + track_start[2] # psi = mu + psi_0
        elif cur>0: # left turn
            l_traveled = self._state[-1] + self._l_0
            theta_traveled = l_traveled/self._R
            psi_ref = theta_traveled + track_start[2]
            theta_from_R = psi_ref - 0.5*math.pi
            dynamic_state[0:2] = self._p_Center + (self._R-self._state[0])*np.array([math.cos(theta_from_R), math.sin(theta_from_R)])
            heading_angle = psi_ref + self._state[1] # heading angle
            dynamic_state[2] = np.mod(heading_angle + np.pi, 2*np.pi) - np.pi # range in [-pi, pi]
        else: # right turn
            l_traveled = self._state[-1] + self._l_0
            theta_traveled = l_traveled/self._R
            psi_ref = track_start[2] - theta_traveled
            theta_from_R = 0.5*math.pi + psi_ref
            dynamic_state[0:2] = self._p_Center + (self._R+self._state[0])*np.array([math.cos(theta_from_R), math.sin(theta_from_R)])
            heading_angle = psi_ref + self._state[1] # heading angle
            dynamic_state[2] = np.mod(heading_angle + np.pi, 2*np.pi) - np.pi # range in [-pi, pi]
        self.dynamic_model.set_state(dynamic_state)

    def set_kinematic_model_state(self, state: np.ndarray, round: float = 0.0) -> None:
        """Set global state of dynamic model. Name is misleading, but for compatibility with Kinematic Model"""
        self.dynamic_model.set_state(state)
        error_state_0 = self.get_error_state(round)
        self._state = error_state_0

    def get_error_state(self, round: float = 0.0) -> np.ndarray:
        '''
        return error state [e_lat, mu, v_x, v_y, yaw_rate, l], based on current global state
        Error mu ranges in [-pi, pi]
        Ideally the state mu will stay in [-pi/2, pi/2]
        l takes the reference point l_0 into consideration. if reference is circle, l reanges in [round, round+2*pi*R]
        '''
        # get reference position on center line
        s_t = self.get_reference_center_line_pos()
        cur = self.params.cur
        xy_p = self.dynamic_model.state[0:2]
        yaw = self.dynamic_model.state[2]
        track_start = self.params.segment_start
        # get error state
        if cur == 0:
            unit_vec = np.array([-math.sin(s_t[2]), math.cos(s_t[2])])
            e_lat = np.inner(unit_vec, xy_p - s_t[0:2])
            unit_x = np.array([math.cos(s_t[2]), math.sin(s_t[2])])
            l = np.inner(unit_x, xy_p - track_start[0:2]) - self._l_0
            mu = yaw - s_t[2]
            mu = np.mod(mu + np.pi, 2*np.pi) - np.pi # range in [-pi, pi]
        elif cur>0:
            e_lat = self._R - npl.norm(xy_p - self._p_Center)
            l = self._R * (s_t[2] - track_start[2]) - self._l_0
            l = np.mod(l-round, 2*np.pi*self._R)+round # range in [round, round+2*pi*R]
            mu = yaw - s_t[2]
            mu = np.mod(mu + np.pi, 2*np.pi) - np.pi # range in [-pi, pi]
        else:
            e_lat = npl.norm(xy_p - self._p_Center) - self._R
            l = self._R * (-s_t[2] + track_start[2]) - self._l_0
            l = np.mod(l-round, 2*np.pi*self._R)+round # range in [round, round+2*pi*R]
            mu = yaw - s_t[2]
            mu = np.mod(mu + np.pi, 2*np.pi) - np.pi # range in [-pi, pi]
        v_x, v_y = self.dynamic_model._state[3], self.dynamic_model._state[4]
        yaw_rate = self.dynamic_model._state[5]
        return np.array([e_lat, mu, v_x, v_y, yaw_rate, l])

    def set_l_0(self, l_0: float) -> None:
        '''
        Set the reference point l_0.
        State of the dynamic model not affected
        '''
        self._state[-1] = self._state[-1] + self._l_0 - l_0
        self._l_0 = l_0

    def get_reference_center_line_pos(self) -> np.ndarray:
        '''
        return reference position on center line [x_t, y_t, Psi_t]
        Psi_t ranges in [-pi, pi]
        '''
        cur = self.params.cur
        track_start = self.params.segment_start
        xy_p = self.dynamic_model.state[0:2]
        if cur == 0: # reference is straight line
            unit_vec = np.array([math.cos(track_start[2]), math.sin(track_start[2])])
            Delta_vec = xy_p - track_start[0:2]
            p_t = track_start[0:2] + np.inner(Delta_vec, unit_vec) * unit_vec
            return np.hstack((p_t, track_start[2]))
        elif cur>0: # reference is a left turn
            unit_vec = xy_p - self._p_Center
            assert npl.norm(unit_vec) != 0, "Mass center of vehicle at center of circle!"
            unit_vec = unit_vec / npl.norm(unit_vec)
            p_xy = self._p_Center + self._R * unit_vec
            unit_x = track_start[0:2]-self._p_Center
            unit_x = unit_x / npl.norm(unit_x)
            unit_y = np.array([-unit_x[1], unit_x[0]])
            Phi_t = math.atan2(np.inner(unit_y, unit_vec), np.inner(unit_x, unit_vec)) + track_start[2]
            Phi_t = np.mod(Phi_t + np.pi, 2*np.pi) - np.pi # range in [-pi, pi]
            return np.hstack((p_xy, Phi_t))
        else: # reference is a right turn
            unit_vec = xy_p - self._p_Center
            assert npl.norm(unit_vec) != 0, "Mass center of vehicle at center of circle!"
            unit_vec = unit_vec / npl.norm(unit_vec)
            p_xy = self._p_Center + self._R * unit_vec
            unit_x = track_start[0:2]-self._p_Center
            unit_x = unit_x / npl.norm(unit_x)
            unit_y = np.array([-unit_x[1], unit_x[0]])
            Phi_t = math.atan2(np.inner(unit_y, unit_vec), np.inner(unit_x, unit_vec)) + track_start[2]
            Phi_t = np.mod(Phi_t + np.pi, 2*np.pi) - np.pi # range in [-pi, pi]
            return np.hstack((p_xy, Phi_t))

    def calculate_zero_input_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (zero_input, zero_state)"""
        # for readability, get variable names
        e_lat = self._z_r[0]
        mu = self._z_r[1]
        v_x, v_y = self._z_r[2], self._z_r[3]
        yaw_rate = self._z_r[4]
        l = self._z_r[5]

        throttle = self._u[0]
        steer = self._u[1]

        e_lat_dot = self._f_c_r[0]
        mu_dot = self._f_c_r[1]
        v_x_dot, v_y_dot = self._f_c_r[2], self._f_c_r[3]
        yaw_rate_dot = self._f_c_r[4]
        l_dot = self._f_c_r[5]

        # construct unkonwn varialbes
        z = cas.vertcat(mu, v_y, yaw_rate, throttle, steer)
        # construct input varialbes
        x = cas.vertcat(e_lat, v_x)
        # construct equations
        g0 = cas.vertcat(e_lat_dot, mu_dot, v_x_dot, v_y_dot, yaw_rate_dot)
        
        # construct rootfinder in casadi
        g = cas.Function('g', [z, x], [g0,])
        rootfinder_function = cas.rootfinder('rootfinder', 'newton', g)

        # solve the non_linear equaiton system
        v_0 = self.params.v_0
        Cd, Croll, Cm1, Cm2 = self.params.dynamic_params.Cd, self.params.dynamic_params.Croll, self.params.dynamic_params.Cm1, self.params.dynamic_params.Cm2

        tau_0 = (Cd * v_0 * v_0 + Croll)/(Cm1 - Cm2 * self.params.v_0)
        if self.params.cur == 0:
            z_guess = np.array([0,0,0,tau_0,0])
        else:
            l_f, l_r = self.params.dynamic_params.l_f, self.params.dynamic_params.l_r
            yaw_rate_0 = v_0 * self.params.cur
            # use kinematic model to get initial guess
            beta_0 = math.asin(l_r * self.params.cur)
            steer_0 = math.atan((l_r+l_f)/l_r*math.tan(beta_0))
            z_guess = np.array([-beta_0,0,yaw_rate_0,tau_0,steer_0])
        z_sol = rootfinder_function(z_guess, np.array([0, v_0]))
        z_sol = np.array(z_sol).flatten()

        # return the calculated zero input and zero state
        return np.array([z_sol[3], z_sol[4]]), np.array([0, z_sol[0], v_0, z_sol[1], z_sol[2], 0])

    def get_zero_input(self) -> np.ndarray:
        return self._zero_input

    def get_zero_state(self) -> np.ndarray:
        return self._zero_state

    def get_noise(self) -> np.matrix:
        if self.params.A_n is None:
            return np.matrix(np.zeros((self.p, 1)))
        else:
            n_array = []
            for i in range(self._p):
                n_array.append(2 * self.params.b_n[i,0] * (np.random.rand(1)-0.5))
            return np.matrix(n_array)


class DynamicErrorModelVxyL(DynamicErrorModel):
    """Dynamic Error Model that outputs [e_lat, mu, v_x, v_y, l] as the state"""
    """Dynamic Error Model that outputs [e_lat, mu, v_x, v_y] as the state"""
    _p: int # override _p in DynamicErrorModel
    @property
    def p(self) -> int:
        return self._p
    @property
    def state(self):
        return self._state
    @property
    def _v_0(self) -> float:
        return self.params.v_0
    @property
    def v_0(self) -> float:
        return self._v_0
    
    def __init__(self, params: DynamicErrorModelParams, initial_state: np.ndarray) -> None:
        DynamicErrorModel.__init__(self, params, initial_state)
        self._p = 5
    
    def step_lin(self, input: np.matrix) -> Tuple[np.matrix, np.matrix, np.matrix]:
        """
        Step the dynamic system. output state BEFORE applying the input
        input: [throttle, steering]
        output: ([[e_lat], [mu], [v_x], [v_y], [l]], zeros, noise)
        """
        state_t = self._state
        input = np.array(input).flatten()

        state_tp = self._F_c_r(x0=state_t, u=input)
        self._state = np.array(state_tp['xf']).flatten()
        self.dynamic_model.step(input)

        y = np.matrix(state_t[0:4]).T
        y = np.vstack((y, np.matrix(state_t[-1]))) # append l to y
        y[2,0] = y[2,0] - self._v_0
        n_lin = np.matrix(np.zeros((self._p, 1)))
        n = self.get_noise()
        
        return y, n_lin, n
    
    def step(self, input: np.matrix) -> Tuple[np.matrix, np.matrix]:
        """
        Only outputs ([e_lat, mu, v_x, v_y, l], noise)
        """
        y, n_lin, n = self.step_lin(input)
        return y+n_lin, n


class DynamicErrorModelVxy(DynamicErrorModel):
    """Dynamic Error Model that outputs [e_lat, mu, v_x, v_y] as the state"""
    _p: int # override _p in DynamicErrorModel
    @property
    def p(self) -> int:
        return self._p
    @property
    def state(self):
        return self._state
    @property
    def _v_0(self) -> float:
        return self.params.v_0
    @property
    def v_0(self) -> float:
        return self._v_0
    
    def __init__(self, params: DynamicErrorModelParams, initial_state: np.ndarray) -> None:
        DynamicErrorModel.__init__(self, params, initial_state)
        self._p = 4
    
    def step_lin(self, input: np.matrix) -> Tuple[np.matrix, np.matrix, np.matrix]:
        """
        Step the dynamic system. output state BEFORE applying the input
        input: [throttle, steering]
        output: ([[e_lat], [mu], [v_x], [v_y]], zeros, noise)
        """
        state_t = self._state
        input = np.array(input).flatten()

        state_tp = self._F_c_r(x0=state_t, u=input)
        self._state = np.array(state_tp['xf']).flatten()
        self.dynamic_model.step(input)

        y = np.matrix(state_t[0:4]).T
        y[2,0] = y[2,0] - self._v_0
        n_lin = np.matrix(np.zeros((self._p, 1)))
        n = self.get_noise()
        
        return y, n_lin, n
    
    def step(self, input: np.matrix) -> Tuple[np.matrix, np.matrix]:
        """
        Only outputs ([e_lat, mu, v_x, v_y], noise)
        """
        y, n_lin, n = self.step_lin(input)
        return y+n_lin, n


class DynamicErrorModelFewOutput(DynamicErrorModel):
    """Dynamic Model with only few outputs, similar to kinematic model"""
    @property
    def state(self):
        v = math.sqrt(self._state[2]**2 + self._state[3]**2)
        return np.hstack(( self._state[0:2], np.array([v]), self._state[-1] ))

    def __init__(self, params: DynamicErrorModelParams, initial_state: np.ndarray) -> None:
        DynamicErrorModel.__init__(self, params, initial_state)
    
    def step_lin(self, input: np.matrix) -> Tuple[np.matrix, np.matrix, np.matrix]:
        """
        Step the dynamic system. output state BEFORE applying the input
        input: [throttle, steering]
        output: ([[e_lat], [mu], [v]], zeros, noise)
        To be compatible with Kinematic Model based on LATI class
        """
        state_t = self._state
        input = np.array(input).flatten()

        state_tp = self._F_c_r(x0=state_t, u=input)
        self._state = np.array(state_tp['xf']).flatten()
        self.dynamic_model.step(input)

        v = math.sqrt(state_t[2]**2 + state_t[3]**2)
        state_few = np.hstack(( state_t[0:2], np.array([v-self.params.v_0]) ))
        y = np.matrix(state_few).T
        n_lin = np.matrix(np.zeros((self.p, 1)))
        n = self.get_noise()
        
        return y, n_lin, n
    
    def step(self, input: np.matrix) -> Tuple[np.matrix, np.matrix]:
        """
        Only outputs ([e_lat, mu, v], noise)
        """
        y, n_lin, n = self.step_lin(input)
        return y+n_lin, n
