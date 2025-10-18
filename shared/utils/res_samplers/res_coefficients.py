"""
RES (Refined Exponential Solver) coefficient generation.
Adapted from RES4LYF: https://github.com/ClownsharkBatwing/RES4LYF

Implements coefficient calculation for res_2s and res_3s samplers using
exponential integrators with φ-functions.
"""

import numpy as np
from .phi_functions import Phi, calculate_gamma


def get_res_2s_coefficients(h, c2=0.5, use_analytic=True):
    """
    Calculate Runge-Kutta coefficients for RES_2S (2-stage exponential integrator).
    
    This is a 2nd-order exponential integrator that uses φ-functions for better
    accuracy with stiff ODEs. It performs 2 model evaluations per step.
    
    Args:
        h: Step size (sigma difference)
        c2: Stage coefficient (default 0.5 for optimal convergence)
        use_analytic: Use high-precision mpmath calculations
        
    Returns:
        tuple: (a, b) where:
            - a: Stage coefficients matrix [[0,0], [a21, 0]]
            - b: Output coefficients [b1, b2]
            
    Mathematical formulation:
        a21 = c2 * φ₁(-h*c2)
        b2 = φ₂(-h) / c2
        b1 = φ₁(-h) - b2
    """
    ci = [0, c2]
    φ = Phi(h, ci, use_analytic)
    
    # Calculate coefficients using φ-functions
    a2_1 = c2 * φ(1, 2)     # a21 coefficient
    b2 = φ(2) / c2           # b2 output weight
    b1 = φ(1) - b2           # b1 output weight (from FSAL property)
    
    # Build coefficient matrices
    a = np.array([
        [0, 0],
        [a2_1, 0],
    ])
    b = np.array([b1, b2])
    
    return a, b


def get_res_3s_coefficients(h, c2=0.5, c3=1.0, use_analytic=True):
    """
    Calculate Runge-Kutta coefficients for RES_3S (3-stage exponential integrator).
    
    This is a 3rd-order exponential integrator that uses φ-functions. It performs
    3 model evaluations per step for higher accuracy.
    
    Args:
        h: Step size (sigma difference)
        c2: Second stage coefficient (default 0.5)
        c3: Third stage coefficient (default 1.0)
        use_analytic: Use high-precision mpmath calculations
        
    Returns:
        tuple: (a, b) where:
            - a: Stage coefficients matrix (3x3)
            - b: Output coefficients [b1, b2, b3]
            
    Mathematical formulation:
        γ = (3c3³ - 2c3) / (c2(2 - 3c2))
        a31 = c3 * φ₁(-h*c3)
        a32 = γ*c2*φ₂(-h*c2) + (c3²/c2)*φ₂(-h*c3)
        b3 = φ₂(-h) / (γ*c2 + c3)
        b2 = γ * b3
        b1 = φ₁(-h) - (c2*b2 + c3*b3)
    """
    ci = [0, c2, c3]
    φ = Phi(h, ci, use_analytic)
    
    # Calculate gamma coefficient for 3-stage method
    gamma = calculate_gamma(c2, c3)
    
    # Calculate stage coefficients
    # First column (using FSAL - first same as last)
    a2_1 = c2 * φ(1, 2)
    a3_1 = c3 * φ(1, 3)
    
    # Second column
    a3_2 = gamma * c2 * φ(2, 2) + (c3**2 / c2) * φ(2, 3)
    
    # Output weights
    b3 = (1 / (gamma * c2 + c3)) * φ(2)
    b2 = gamma * b3
    b1 = φ(1) - (c2 * b2 + c3 * b3)
    
    # Build coefficient matrices
    a = np.array([
        [0, 0, 0],
        [a2_1, 0, 0],
        [a3_1, a3_2, 0],
    ])
    b = np.array([b1, b2, b3])
    
    return a, b


def test_coefficients():
    """Test coefficient generation with known values."""
    print("Testing RES coefficient generation:\n")
    
    # Test res_2s
    h = 0.1
    a, b = get_res_2s_coefficients(h, c2=0.5, use_analytic=False)
    print("RES_2S coefficients (h=0.1, c2=0.5):")
    print(f"  a matrix:\n{a}")
    print(f"  b weights: {b}")
    print(f"  b sum (should ≈ φ₁(-h)): {b.sum():.10f}\n")
    
    # Test res_3s
    a, b = get_res_3s_coefficients(h, c2=0.5, c3=1.0, use_analytic=False)
    print("RES_3S coefficients (h=0.1, c2=0.5, c3=1.0):")
    print(f"  a matrix:\n{a}")
    print(f"  b weights: {b}")
    print(f"  b sum (should ≈ φ₁(-h)): {b.sum():.10f}\n")


if __name__ == "__main__":
    test_coefficients()

