from typing import Callable
from scipy.integrate import quad


__all__ = [
    "compute_1d_integral",
]


def compute_1d_integral(
    func: Callable,
    start: float,
    end: float,
    absolute_error: float,
    relative_error: float,
)-> float:
    
    integral_result = quad(
        func = func, 
        a = start,
        b = end,
        epsabs = absolute_error, 
        epsrel = relative_error,
    )[0]
    
    return integral_result