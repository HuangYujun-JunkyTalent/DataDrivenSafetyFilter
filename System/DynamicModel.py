import casadi as cas
import numpy as np
from dataclasses import dataclass


@dataclass
class DynamicModelParams:
    Ts: float

    l_f: float
    l_r: float

    m: float
    Iz: float

    Bf: float
    Cf: float
    Df: float

    Br: float
    Cr: float
    Dr: float

    Croll: float
    Cm1: float
    Cm2: float
    Cd: float


class DynamicModel:
    _params: DynamicModelParams
    @property
    def Ts(self):
        return self._params.Ts

    # actual state of the system
    _state: np.ndarray
    @property
    def state(self):
        return self._state
    _z: cas.SX
    _u: cas.SX
    _f_c: cas.SX # ODE function
    _F_c: cas.Function

    # compatibality with linearized model
    m = 2
    n = 6
    p = 4

    def __init__(self, params: DynamicModelParams, initial_state: np.ndarray) -> None:
        """
        initial_state: [x_p0, y_p0, Psi_0, v_x0, v_y0, omega_0]
        """
        self._params = params
        self._state = initial_state
        
        self._z = cas.SX.sym('_z', 6) # state for dynamic model
        self._u = cas.SX.sym('_u', 2)
        self._f_c = cas.SX.sym('_f_c', 6) # dynamic equation

        # for readibility, set name of variables
        throttle = self._u[0]
        steer = self._u[1]
        xp, yp = self._z[0], self._z[1]
        yaw = self._z[2]
        v_x, v_y = self._z[3], self._z[4]
        yaw_rate = self._z[5]

        # intermediate varialbes
        Fx = (params.Cm1 - params.Cm2 * v_x) * throttle - params.Cd * v_x * v_x - params.Croll
        ar = - cas.atan2(v_y-params.l_r*yaw_rate, v_x) # rear slip angle
        af = steer - cas.atan2(v_y+params.l_f*yaw_rate, v_x) # front slip angle
        Fr = params.Dr * cas.sin(params.Cr * cas.atan(params.Br * ar))
        Ff = params.Df * cas.sin(params.Cf * cas.atan(params.Bf * af))

        # dynamic equations
        xp_dot = v_x * cas.cos(yaw) - v_y * cas.sin(yaw)
        yp_dot = v_x * cas.sin(yaw) + v_y * cas.cos(yaw)
        yaw_dot = yaw_rate
        v_x_dot = 1.0 / params.m * (Fx - Ff * cas.sin(steer) + params.m * v_y * yaw_rate)
        v_y_dot = 1.0 / params.m * (Fr + Ff * cas.cos(steer) - params.m * v_x * yaw_rate)
        yaw_rate_dot = 1.0 / params.Iz * (Ff * params.l_f * cas.cos(steer) - Fr * params.l_r)

        self._f_c = cas.vertcat(xp_dot, yp_dot, yaw_dot, v_x_dot, v_y_dot, yaw_rate_dot)

        # Set up ODE solver, pay attention to the name here: x is state of ode solver!
        ode = {'x':self._z, 'u':self._u, 'ode':self._f_c}
        self._F_c = cas.integrator('F', 'cvodes', ode, 0, params.Ts)

    def step(self, input: np.ndarray) -> np.ndarray:
        """
        Step the dynamic system. output state BEFORE applying the input
        input: [throttle, steering]
        """
        state_t = self._state

        state_tp = self._F_c(x0=state_t, u=input)
        self._state = np.array(state_tp['xf']).flatten() # type and dimension conversion
        
        return state_t
    
    def set_state(self, state: np.ndarray) -> None:
        """
        state: [x_p, y_p, yaw, v_x, v_y, yaw_rate]
        """
        self._state = state


class DynamicModelFewOutput:
    """
    Dynamic model of chronos, but only ouput [xp, yp, yaw, v], similar to that of kinematic model
    """
    dynamic_model: DynamicModel

    # compatibality with linearized model
    m = 2
    n = 6
    p = 3

    @property
    def _state(self):
        return self.dynamic_model._state

    @property
    def state(self):
        full_dynamic_state = self.dynamic_model.state
        return self.get_short_state(full_dynamic_state)

    def __init__(self, params: DynamicModelParams, initial_state: np.ndarray) -> None:
        self.dynamic_model = DynamicModel(params, initial_state)
    
    def set_state(self, state: np.ndarray) -> None:
        self.dynamic_model.set_state(state)
    
    def step(self, input: np.ndarray) -> np.ndarray:
        """
        Return only [xp, yp, yaw, v]
        """
        full_dynamic_state = self.dynamic_model.step(input)

        return self.get_short_state(full_dynamic_state)
    
    @staticmethod
    def get_short_state(full_state: np.ndarray) -> np.ndarray:
        xp, yp = full_state[0], full_state[1]
        yaw = full_state[2]
        v_x, v_y = full_state[3], full_state[4]
        v = np.sqrt(v_x**2 + v_y**2)
        return np.array([xp, yp, yaw, v])
