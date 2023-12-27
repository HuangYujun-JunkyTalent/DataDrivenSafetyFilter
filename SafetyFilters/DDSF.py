import numpy as np
import cvxpy as cp

from typing import Tuple, Optional, Any, Dict
from dataclasses import dataclass, field
from abc import ABC

@dataclass
class SFParams:
    steps: int = 1 # number of steps forward
    solver: Optional[Any] = None
    solver_args: Dict = field(default_factory=dict)
    warm_start: bool = False
    verbose: bool = False


# Define Safety Filter class
class DDSafetyFilter(ABC):
    """Interfaces are all in np.matrix form, with proper shape"""
    _opt_prob: cp.Problem
    _xi_t: cp.Parameter # extended state
    _u_obj: cp.Parameter # objective input
    _u: cp.Variable # variable for control input

    # following parameters need to be defined in subclass
    _m: int
    _p: int
    _n: int
    _L: int # Prediction Horizen
    _lag: int # lag of system used for initial condition, might be an approximation (e.g. n)

    _steps: int # number of steps to consider
    _solver: Optional[Any]
    _solver_args: Dict
    _verbose: bool
    _warm_start: bool

    def __init__(self, sf_params: SFParams) -> None:
        self._steps = sf_params.steps
        self._solver = sf_params.solver
        self._solver_args = sf_params.solver_args
        self._verbose = sf_params.verbose
        self._warm_start = sf_params.warm_start

    def filter(self, xi_t: np.matrix, u_obj: np.matrix) -> Tuple[np.matrix, str, float]:
        """Return tuple of (filtered control input, status, optimal value)"""
        self._xi_t.value = np.array(xi_t.flat)
        self._u_obj.value = np.array(u_obj.flat)
        if self._solver is None:
            self._opt_prob.solve(verbose=self._verbose, warm_start = self._warm_start, **self._solver_args)
        else:
            self._opt_prob.solve(verbose=self._verbose, solver=self._solver, warm_start = self._warm_start, **self._solver_args)
        if not (self._opt_prob.status is cp.OPTIMAL or self._opt_prob.status is cp.OPTIMAL_INACCURATE):
            raise Exception(f"Problem not optimal: status is {self._opt_prob.status}")
        return (np.matrix(self._u.value[self._m*self._lag:self._m*(self._lag+self._steps)]).transpose(), self._opt_prob.status, self._opt_prob.value)
    