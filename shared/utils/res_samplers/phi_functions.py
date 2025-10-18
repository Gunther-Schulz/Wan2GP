"""
Φ-functions (phi functions) for exponential integrators in RES samplers.
Adapted from RES4LYF: https://github.com/ClownsharkBatwing/RES4LYF

These implement the mathematical φ-functions used in exponential integrators:
φⱼ(-h) = 1/h^j * ∫₀ʰ e^(τ-h) * (τ^(j-1))/(j-1)! dτ

References:
- Original implementation by ClownsharkBatwing
- Analytic solution by Clybius: https://github.com/Clybius/ComfyUI-Extra-Samplers
- Theory: https://arxiv.org/abs/2308.02157 (Lemma 1)
"""

import torch
import math
from typing import Optional
from mpmath import mp, mpf, factorial, exp, gamma, gammainc


# Set precision for mpmath (80 decimal digits ~ float256)
mp.dps = 80


def calculate_gamma(c2, c3):
    """Calculate gamma coefficient for 3-stage RK methods."""
    return (3*(c3**3) - 2*c3) / (c2*(2 - 3*c2))


def _gamma(n: int) -> int:
    """
    Gamma function for positive integers.
    https://en.wikipedia.org/wiki/Gamma_function
    For every positive integer n: Γ(n) = (n-1)!
    """
    return math.factorial(n-1)


def _incomplete_gamma(s: int, x: float, gamma_s: Optional[int] = None) -> float:
    """
    Incomplete gamma function for positive integer s.
    https://en.wikipedia.org/wiki/Incomplete_gamma_function#Special_values
    If s is a positive integer: Γ(s, x) = (s-1)! * ∑_{k=0..s-1}(x^k/k!)
    """
    if gamma_s is None:
        gamma_s = _gamma(s)

    sum_: float = 0
    for k in range(s):
        numerator: float = x**k
        denom: int = math.factorial(k)
        quotient: float = numerator / denom
        sum_ += quotient
    
    incomplete_gamma_: float = sum_ * math.exp(-x) * gamma_s
    return incomplete_gamma_


def phi(j: int, neg_h: float) -> float:
    """
    Standard precision φ-function implementation.
    
    φⱼ(-h) using gamma functions:
    = e^(-h) * (-h)^(-j) * (1 - Γ(j,-h)/Γ(j))
    
    Args:
        j: Order of the phi function (must be > 0)
        neg_h: The value -h where h is the step size
        
    Returns:
        φⱼ(-h) value
    """
    assert j > 0
    gamma_: float = _gamma(j)
    incomp_gamma_: float = _incomplete_gamma(j, neg_h, gamma_s=gamma_)
    phi_: float = math.exp(neg_h) * neg_h**-j * (1 - incomp_gamma_ / gamma_)
    return phi_


def phi_mpmath_series(j: int, neg_h: float) -> float:
    """
    High-precision φ-function using mpmath for arbitrary precision.
    Uses the remainder-series definition with 80 decimal digits precision.
    
    φⱼ(-h) = (e^z - ∑_{k=0..j-1} z^k/k!) / z^j  where z = -h
    
    Args:
        j: Order of the phi function
        neg_h: The value -h where h is the step size
        
    Returns:
        High-precision φⱼ(-h) value as float
    """
    j = int(j)
    z = mpf(float(neg_h))
    
    # Calculate sum: S = ∑_{k=0..j-1} z^k / k!
    S = mp.mpf('0')
    for k in range(j):
        S += (z**k) / factorial(k)
    
    # φⱼ(z) = (e^z - S) / z^j
    phi_val = (exp(z) - S) / (z**j)
    return float(phi_val)


def superphi(j: int, neg_h: float) -> float:
    """
    Alternative high-precision φ-function using mpmath gamma functions.
    
    Args:
        j: Order of the phi function
        neg_h: The value -h where h is the step size
        
    Returns:
        High-precision φⱼ(-h) value as float
    """
    gamma_: float = gamma(j)
    incomp_gamma_: float = gamma_ - gammainc(j, 0, float(neg_h))
    phi_: float = float(math.exp(float(neg_h)) * neg_h**-j) * (1 - incomp_gamma_ / gamma_)
    return float(phi_)


class Phi:
    """
    Cached φ-function evaluator for efficient coefficient calculation.
    
    This class provides a callable interface to φ-functions with caching
    to avoid redundant calculations. It can use either standard precision
    or high-precision (mpmath) implementations.
    
    Args:
        h: Step size
        c: Array of c_i coefficients for RK method
        analytic_solution: If True, use high-precision mpmath implementation
    """
    
    def __init__(self, h, c, analytic_solution=False):
        self.h = h
        self.c = c
        self.cache = {}
        
        if analytic_solution:
            # Use high-precision mpmath version
            self.phi_f = phi_mpmath_series
            self.h = mpf(float(h))
            self.c = [mpf(c_val) for c_val in c]
        else:
            # Use standard precision version
            self.phi_f = phi
    
    def __call__(self, j, i=-1):
        """
        Evaluate φⱼ(-h*cᵢ) with caching.
        
        Args:
            j: Order of phi function
            i: Index of c coefficient (-1 means c=1)
            
        Returns:
            φⱼ(-h*cᵢ) value
        """
        # Check cache first
        if (j, i) in self.cache:
            return self.cache[(j, i)]
        
        # Determine coefficient
        if i < 0:
            c = 1
        else:
            c = self.c[i - 1]
            if c == 0:
                self.cache[(j, i)] = 0
                return 0
        
        # Calculate φ function
        if j == 0 and type(c) in {float, torch.Tensor}:
            result = math.exp(float(-self.h * c))
        else:
            result = self.phi_f(j, -self.h * c)
        
        # Cache and return
        self.cache[(j, i)] = result
        return result

