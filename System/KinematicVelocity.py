import casadi as cas
import numpy as np
import numpy.linalg as npl
from scipy import signal
from dataclasses import dataclass
from typing import Optional, Tuple

from System.LTI import LTI, LTIParams

@dataclass
class KinematicModelParams:
    l_r: float
    l_f: float
    Ts: float
    # state0: np.ndarray # initial system state [x_p_0, y_p_0, Phi_0]

class KinematicModel:
    '''
    Input [v, beta] as np.ndarray; output [x_p_t, y_p_t, Phi_t] as np.ndarray
    '''
    _l_r: float
    _l_f: float # kinematic parameters
    _Ts: float # sampling time
    @property
    def Ts(self):
        return self._Ts

    _state: np.ndarray # actual state of the system
    _z: cas.SX # state variable, [x_p, y_p, Phi]
    _u: cas.SX # input varialbe, [v, beta]
    _f_c: cas.SX # kinematic ode, input is [v, beta]
    _F_c: cas.Function # integrator of kinematic system

    def __init__(self, params: KinematicModelParams, state0: np.ndarray) -> None:
        self._l_r = params.l_r
        self._l_f = params.l_f
        self._Ts = params.Ts
        self._state = state0

        self._z = cas.SX.sym('_z', 3)
        self._u = cas.SX.sym('_u', 2)
        self._f_c = cas.SX.sym('_f_c', 3)
        self._f_c[0] = self._u[0]*cas.cos(self._z[2]+self._u[1])
        self._f_c[1] = self._u[0]*cas.sin(self._z[2]+self._u[1])
        self._f_c[2] = self._u[0]*cas.sin(self._u[1])/self._l_r
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
class LinearizedKinematicModelParams:
    kinematic_params: KinematicModelParams
    # state0: np.ndarray # initial system state [x_p_0, y_p_0, Phi_0]

    state0s: np.ndarray # initial condition for steady trajectory
    input_s: np.ndarray # control input for steady trajectory
    ts: float = 0 # time has passed in the steady state trajectory

    A_y: Optional[np.matrix] = None
    b_y: Optional[np.matrix] = None
    A_u: Optional[np.matrix] = None
    b_u: Optional[np.matrix] = None
    # Noise properties
    A_n: Optional[np.matrix] = None
    b_n: Optional[np.matrix] = None


class LinearizedKinematicModel(LTI):
    '''
    Kinematic Model of Chronos, includes a Linearization around a steady trajectory
    '''
    kinematic_model: KinematicModel

    _t: cas.SX # time varialbe for steady trajectory
    _us: cas.SX # control input varialbe for steady trajectory
    _z0s: cas.SX # initial condition varialbe for steady trajectory
    _ts: float # time since initialization in steady trajectory
    _input_s: np.ndarray # control input for steady trajectory
    _state0s: np.ndarray # initial condition for steady trajectory
    _zs: cas.SX # state varialbe for steady trajectory
    _zs_fun: cas.Function # steady state function, [time, input, initial] |-> [state] of steady trajectory

    _U: cas.SX # variable for coordinate tranformation from global to steady
    _U_fun: cas.Function # coordinate transformation function, [time, input, initial] |-> [transformation matrix]
    _A_tilde_fun: cas.Function # function for continuous time LTI, [time, input, initial] |-> [A matrix]
    _B_tilde_fun: cas.Function # function for continuous time LTI, [time, input, initial] |-> [B matrix]

    @property
    def U(self) -> np.matrix:
        '''
        Coordinate transformation from global frame to current steady state transform
        '''
        return np.matrix(self._U_fun(self._ts, self._input_s, self._state0s))

    def __init__(self, params: LinearizedKinematicModelParams, state0: np.ndarray) -> None:
        
        self.kinematic_model = KinematicModel(params.kinematic_params, state0)
        self._input_s = params.input_s
        self._state0s = params.state0s
        self._ts = params.ts

        # variables and functions for steady state trajectory
        self._t = cas.SX.sym('_t', 1)
        self._us = cas.SX.sym('_us', 2)
        self._z0s = cas.SX.sym('_z0s', 3)
        self._zs = cas.SX.sym('_zs', 3)
        self._zs[2] = self._z0s[2] + self._us[0] * cas.sin(self._us[1]) / self.kinematic_model._l_r * self._t
        self._zs[0] = self._z0s[0] + self.kinematic_model._l_r/cas.sin(self._us[1])*\
            ( cas.sin(self._zs[2]+self._us[1])-cas.sin(self._z0s[2]+self._us[1]) )
        self._zs[1] = self._z0s[1] - self.kinematic_model._l_r/cas.sin(self._us[1])*\
            ( cas.cos(self._zs[2]+self._us[1])-cas.cos(self._z0s[2]+self._us[1]) )
        self._zs_fun = cas.Function('zs_fun', [self._t, self._us, self._z0s], [self._zs])

        # variables and functions for coordinate transformation
        self._U = cas.SX.zeros(3,3)
        self._U[2,2] = 1
        self._U[0,0] = cas.cos(self._zs[2])
        self._U[0,1] = cas.sin(self._zs[2])
        self._U[1,0] = -cas.sin(self._zs[2])
        self._U[1,1] = cas.cos(self._zs[2])
        self._U_fun = cas.Function('U_fun', [self._t, self._us, self._z0s], [self._U])

        A, B, C, D = self.get_lti()

        # define LTI params and initialize
        delta_z_global = state0-self.get_zs()
        lti_params = LTIParams(
            np.matrix(A), np.matrix(B), np.matrix(C), np.matrix(D),
            np.matrix(self._U_fun( self._ts, self._input_s, self._state0s ))@\
                np.transpose(np.matrix(delta_z_global)),
            params.A_y, params.b_y, params.A_u, params.b_u, params.A_n, params.b_n
        )
        LTI.__init__(self, lti_params)

    def step(self, input: np.matrix) -> Tuple[np.matrix, np.matrix]:
        """ Update system state and give output
        Note that noise is due to both linearization and real noise

        Input:
            system input
        Returns:
            (un noised system output, output noise)
        Rasies:
            Exception if input shape not correct
        
        """
        # LTI system state update, it is in steady state frame!
        matrix_input = input
        input = np.array(matrix_input.transpose())
        y: np.matrix = np.matmul(self._C, self._x) + np.matmul(self._D, matrix_input)
        self._x = np.matmul(self._A, self._x) + np.matmul(self._B, matrix_input)

        # Update exact system state. Remember to add steady state input!
        u_kinematic = input + self._input_s
        z_global = self.kinematic_model.step(u_kinematic) # it is in global frame!

        # Get linearization error, remember to do frame transforamtion!
        zs_global = self.get_zs()
        U_t = self.U
        e_lin = U_t@(z_global-zs_global)
        e_lin = np.transpose(e_lin)
        # If noise matrix is not None, get and add noise
        if self._A_n is not None:
            e = e_lin + self.get_noise()
        else:
            e = e_lin

        # Update steady state time
        #Do this after getting frame transformation at this time step!
        self._ts = self._ts + self.kinematic_model.Ts

        # return np.array(y).flatten(), e_lin
        return np.matrix(y), np.matrix(e)

    def get_lti(self) -> Tuple[np.matrix, np.matrix, np.matrix, np.matrix]:
        '''
        Update LTI system based on current time, initial state and input for steady trajectory
        '''
        # get continus-time LTI matrices
        # jacobian using exact variables
        J_z = cas.jacobian(self.kinematic_model._f_c, self.kinematic_model._z)
        J_u = cas.jacobian(self.kinematic_model._f_c, self.kinematic_model._u)
        # linearized system around steady state traj, still LTV
        A_c = cas.substitute(J_z, self.kinematic_model._z, self._zs)
        A_c = cas.substitute(A_c, self.kinematic_model._u, self._us)
        B_c = cas.substitute(J_u, self.kinematic_model._z, self._zs)
        B_c = cas.substitute(B_c, self.kinematic_model._u, self._us)
        # change coordinate system. now they should be LTI (independent of self._t)
        # but due to the way CasADi simplify expressions, it still depend on t on graph
        B_tilde = self._U@B_c
        dU = cas.jacobian(self._U,self._t)
        dU = cas.reshape(dU, 3, 3)
        U_inv = cas.solve(self._U, cas.SX.eye(self._U.size1()))
        A_tilde = dU@U_inv + self._U@A_c@U_inv
        self._A_tilde_fun = cas.Function('A_tilde_fun', [self._t, self._us, self._z0s], [A_tilde])
        self._B_tilde_fun = cas.Function('B_tilde_fun', [self._t, self._us, self._z0s], [B_tilde])
        # get matrix from function. the first argument corresponds to t, should not affect result
        A_c_LTI = np.array(self._A_tilde_fun(0, self._input_s, self._state0s))
        B_c_LTI = np.array(self._B_tilde_fun(0, self._input_s, self._state0s))
        # ues exact discretization
        A, B, C, D , _ = signal.cont2discrete((A_c_LTI, B_c_LTI, np.eye(3), np.zeros((3,2))), self.kinematic_model.Ts)

        return A, B, C, D
    
    def get_zs(self) -> np.ndarray:
        ''''
        Function to get state of steady state in global frame
        This extra function is needed to avoid sigularity when steering input is 0
        '''
        if self._input_s[1] == 0: # beta_s equals 0
            x_s = self._state0s[0] + self._ts*cas.cos(self._state0s[2])*self._input_s[0]
            y_s = self._state0s[1] + self._ts*cas.sin(self._state0s[2])*self._input_s[0]
            Phi_s = self._state0s[2]
            return np.array([x_s, y_s, Phi_s])
        else:
            state_s = self._zs_fun(self._ts, self._input_s, self._state0s)
            return np.array(state_s).flatten()
    
    def set_steady_state(self, state0s: np.ndarray, input_s: np.ndarray, ts: float) -> None:
        ''' This only sets the steady state, does not modify both of the system states! '''
        self._state0s = state0s
        self._input_s = input_s
        self._ts = ts

        # update the LTI parameters after reset the steady state trajectory!
        A, B, C, D = self.get_lti()
        self._A, self._B, self._C, self._D = np.matrix(A), np.matrix(B), np.matrix(C), np.matrix(D)

        self._Phi, self._l = self.observibility_const()

        Phi_n = np.matrix(np.ndarray((0,self._n)))
        C_n = np.matrix(np.ndarray(shape=(self.n,0)))
        new_block = self._C
        new_block_C = self._B
        for _ in range(self._n):
            Phi_n = np.vstack( (Phi_n, new_block) )
            C_n = np.hstack( (new_block_C,C_n) )
            new_block = np.matmul(new_block, self._A)
            new_block_C = np.matmul(self._A, new_block_C)
        self._Phi_n, self._C_n = Phi_n, C_n

    def set_local_exact_state(self, state: np.ndarray) -> None:
        ''' This only sets the exact system state, according to the given local frame state '''
        z_s = self.get_zs()
        U = self.U # transformation from GLOBAL to LOCAL!
        U_inv = npl.inv(U)

        self.kinematic_model.set_state(z_s + np.array(U_inv@state))

    def set_global_exact_state(self, state_global: np.ndarray) -> None:
        ''' This only sets the exact system state, as in the global frame '''
        self.kinematic_model.set_state(state_global)

    def set_local_lin_state(self, state: np.ndarray) -> None:
        ''' This only sets the linearized system state, as in local frame '''
        self._x = np.transpose(np.matrix(state))
    
    def set_global_lin_state(self, state_global: np.ndarray) -> None:
        ''' This only sets the linearized system state, as in global frame '''
        z_s = self.get_zs()
        U = self.U # transformation from GLOBAL to LOCAL!

        state_local = U@(state_global-z_s)
        self._x = np.transpose(np.matrix(state_local))
