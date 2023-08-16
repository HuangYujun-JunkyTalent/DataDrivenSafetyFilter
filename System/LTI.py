import numpy as np
import numpy.linalg as npl
import numpy.random as npr
from scipy.linalg import block_diag
import cvxpy as cp
from pytope import Polytope

from typing import Tuple, Optional
from dataclasses import dataclass
from warnings import warn

# Define LTI system calss

@dataclass
class LTIParams:
    # Dynamics
    A: np.matrix
    B: np.matrix
    C: np.matrix
    D: np.matrix
    # Initial state
    x0: np.matrix
    # Constraints
    A_y: Optional[np.matrix] = None
    b_y: Optional[np.matrix] = None
    A_u: Optional[np.matrix] = None
    b_u: Optional[np.matrix] = None
    # Noise properties
    A_n: Optional[np.matrix] = None
    b_n: Optional[np.matrix] = None


class LTI:
    # State
    _x: np.matrix
    # Dynamic matrics
    _A: np.matrix
    _B: np.matrix
    _C: np.matrix
    _D: np.matrix
    # System properties
    _n: int
    _m: int
    _p: int
    _l: int
    @property
    def n(self) -> int:
        return self._n
    @property
    def m(self) -> int:
        return self._m
    @property
    def p(self) -> int:
        return self._p
    @property
    def l(self) -> int:
        return self._l
    # Observibiily matrix
    _Phi: np.matrix
    # Constraints
    _A_y: np.matrix
    _b_y: np.matrix
    _A_u: np.matrix
    _b_u: np.matrix
    @property
    def A_y(self):
        return self._A_y
    @property
    def b_y(self):
        return self._b_y
    @property
    def A_u(self):
        return self._A_u
    @property
    def b_u(self):
        return self._b_u
    # noise
    _A_n: np.matrix
    _b_n: np.matrix

    _Phi_n: np.matrix
    _C_n: np.matrix
    
    def __init__(self, lti_params: LTIParams):
        # Controllability and Minimal Represention are not checked!
        A, B, C, D, x0 = lti_params.A, lti_params.B, lti_params.C, lti_params.D, lti_params.x0
        assert A.shape[0] == A.shape[1], f"Wrong shape of matrix A: {A.shape}"
        self._n = A.shape[0]
        assert self._n == B.shape[0], f"Wrong shape of Matrix B: {B.shape}, system n is {self._n}"
        self._m = B.shape[1]
        assert self._n == C.shape[1], f"Wrong shape of Matrix C: {C.shape}, system n is {self._n}"
        self._p = C.shape[0]
        assert D.shape[0] == self._p and D.shape[1] == self._m ,f"Wrong shape of Matrix D: {D.shape}, system p is {self._p}, system m is {self._m}"
        assert x0.shape[0] == self._n and x0.shape[1] == 1, f"Wrong shape of x0: {x0.shape}, system n is {self._n}"
        self._A, self._B, self._C, self._D, self._x = A, B, C, D, x0

        if lti_params.A_y is not None:
            assert lti_params.A_y.shape[1] == self._p and lti_params.b_y.shape == (lti_params.A_y.shape[0], 1), f"Wrong output constraint shape, shape of A_y: {lti_params.A_y.shape}, shape of b_y: {lti_params.b_y.shape}, system p: {self._p}"
        if lti_params.A_u is not None:
            assert lti_params.A_u.shape[1] == self._m and lti_params.b_u.shape == (lti_params.A_u.shape[0], 1), f"Wrong input constraint shape, shape of A_u: {lti_params.A_u.shape}, shape of b_u: {lti_params.b_u.shape}, system m: {self._m}"

        self._A_y, self._A_u, self._b_y, self._b_u = lti_params.A_y, lti_params.A_u, lti_params.b_y, lti_params.b_u

        if lti_params.A_n is not None:
            assert lti_params.A_n.shape[1] == self._p and lti_params.b_n.shape == (lti_params.A_n.shape[0], 1), f"Wrong noise constraint shape, shape of A_n: {lti_params.A_n.shape}, shape of b_n: {lti_params.b_u.shape}, system p: {self._p}"
        self._A_n, self._b_n = lti_params.A_n, lti_params.b_n

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
            
    def observibility_const(self) -> Tuple[np.matrix, int]:
        """
        Returns:
            (Observability Matrix Phi, system lag l)
        Raises:
            Exception if system is not observable
        """
        Phi = np.matrix(np.ndarray((0,self._n)))
        new_block = self._C
        for i in range(self._n):
            Phi = np.vstack( (Phi, new_block) )
            if npl.matrix_rank(Phi) == self._n:
                return Phi, i+1
            new_block = np.matmul(new_block, self._A)
        raise Exception("System not observable!")
    
    def step(self, u: np.matrix) -> Tuple[np.matrix, np.matrix]:
        """ Update system state and give output

        Input:
            system input
        Returns:
            (un noised system output, output noise)
        Rasies:
            Exception if input shape not correct
        """
        assert u.shape[0] == self._m and u.shape[1] == 1, f"u of wrong shape: {u.shape}, system m is {self._m}"
        if not all(self.A_u @ u <= self.b_u):
            warn("Input constraint not satisfied!")
        
        y: np.matrix = np.matmul(self._C, self._x) + np.matmul(self._D, u)
        self._x = np.matmul(self._A, self._x) + np.matmul(self._B, u)
        if self._A_n is not None:
            # for now this is only true for constraints like ||n||_inf <= n_max
            noise = self.get_noise()
        else:
            noise = 0
        return y, noise
    
    def set_state(self, x0: np.matrix) -> None:
        assert x0.shape == (self.n, 1), f"State matrix of wrong size: {x0.shape}, should be of size: {(self.n, 1)}"
        self._x = x0

    def get_noise(self) -> np.matrix:
        '''For now only works for box-shaped noise!'''
        n_array = []
        for i in range(self._p):
            n_array.append(2 * self._b_n[i,0] * (npr.rand(1)-0.5))
        return np.matrix(n_array)

    def rho(self, k: int) -> float:
        """return the constant rho_k for the LTI system
        Defined as: ||C @ A^k @ pinv(Phi_n)||_inf"""
        Phi_n_inv = npl.pinv(self._Phi_n)

        A_k = npl.matrix_power(self._A, k)

        return npl.norm(self._C @ A_k @ Phi_n_inv, np.inf)
    
    def gamma(self) -> float:
        """return the constant \Gamma of the LTI system
        it is a constant relating extended state and input needed to drive system to zero"""
        P = Polytope(np.array(np.vstack((self._Phi_n, -self._Phi_n))), np.append(np.ones(self.n), [np.ones(self.n)]))
        P.minimize_V_rep()
        V = P.V
        
        gamma_array = []
        u = cp.Variable(shape=(self.n*self.m,))
        x_0 = cp.Parameter(shape=(self.n,))
        obj = cp.norm1(u)
        constraints = [self._C_n @ u == -npl.matrix_power(self._A, self.n) @ x_0]
        opt_prob = cp.Problem(objective=cp.Minimize(obj), constraints=constraints)

        for v in V:
            x_0.value = v
            opt_prob.solve()
            if opt_prob.status is not cp.OPTIMAL:
                warn("LP solver not optimal when calculating Gamma!")
            gamma_array.append(opt_prob.value)
        
        return np.max(gamma_array)
    
    def xi_max(self, l: int) -> float:
        """return the maximum 1-norm of extended state xi"""
        # since maximizing convex function is not convex, we enumerating over all vertices of the polytope
        A = np.array(block_diag(*( (self.A_u,)*l + (self.A_y,)*l )))
        b = np.vstack((self.b_u,)*l + (self.b_y,)*l)
        b = np.array(b.flat)
        P = Polytope(A, b)
        P.minimize_V_rep()

        V = P.V
        K = len(V)
        l = np.ndarray((K,))
        for i in range(K):
            l[i] = npl.norm(V[i], 1)
        
        return np.max(l)