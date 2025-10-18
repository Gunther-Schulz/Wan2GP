"""
True RES (Refined Exponential Solver) implementation with exponential integrators.
Adapted from RES4LYF for Wan2GP's flow matching framework.

This implements actual exponential Runge-Kutta methods using φ-functions,
not just step multipliers.
"""

import torch
import numpy as np
from typing import Callable, Optional, Tuple
from .phi_functions import Phi
from .res_coefficients import get_res_2s_coefficients, get_res_3s_coefficients


class RESExponentialIntegrator:
    """
    Exponential integrator for flow matching diffusion models.
    
    Implements true RES sampling using φ-functions and optimal RK coefficients.
    This is different from just running more steps - it uses exponential integrators
    that are mathematically optimal for stiff ODEs.
    """
    
    def __init__(self, rk_type: str = 'res_2s', use_analytic: bool = True):
        """
        Initialize RES exponential integrator.
        
        Args:
            rk_type: Either 'res_2s' (2-stage) or 'res_3s' (3-stage)
            use_analytic: Use high-precision mpmath calculations (recommended)
        """
        self.rk_type = rk_type
        self.use_analytic = use_analytic
        self.num_stages = 2 if rk_type == 'res_2s' else 3
        
        # Coefficients will be calculated per-step
        self.a = None  # Stage coefficients
        self.b = None  # Output weights
        self.c = None  # Stage nodes
    
    def calculate_coefficients(self, h: float):
        """
        Calculate RK coefficients for current step size h.
        
        The φ-functions and coefficients depend on the step size,
        so we recalculate them for each step.
        
        Args:
            h: Step size (sigma difference)
        """
        if self.rk_type == 'res_2s':
            self.a, self.b = get_res_2s_coefficients(h, c2=0.5, use_analytic=self.use_analytic)
            self.c = np.array([0, 0.5])
        elif self.rk_type == 'res_3s':
            self.a, self.b = get_res_3s_coefficients(h, c2=0.5, c3=1.0, use_analytic=self.use_analytic)
            self.c = np.array([0, 0.5, 1.0])
        else:
            raise ValueError(f"Unknown rk_type: {self.rk_type}")
    
    def step(self,
             model_fn: Callable,
             x: torch.Tensor,
             sigma: float,
             sigma_next: float,
             **model_kwargs) -> torch.Tensor:
        """
        Perform one RES step using exponential RK integration.
        
        This is the core stepping function that uses φ-functions and
        optimal RK coefficients to compute the next state.
        
        Args:
            model_fn: Model function that takes (x, sigma) and returns denoised output
            x: Current latent state
            sigma: Current noise level
            sigma_next: Next noise level
            **model_kwargs: Additional arguments for model (conditioning, etc.)
            
        Returns:
            Next latent state x_next
        """
        # Step size (negative because we're going from high to low sigma)
        h = sigma_next - sigma
        
        # Calculate coefficients for this step size
        self.calculate_coefficients(float(h))
        
        # Convert coefficients to tensors
        device = x.device
        dtype = x.dtype
        a = torch.tensor(self.a, device=device, dtype=torch.float32)
        b = torch.tensor(self.b, device=device, dtype=torch.float32)
        c = torch.tensor(self.c, device=device, dtype=torch.float32)
        
        # Storage for stage derivatives (epsilon values)
        k = []
        
        # Stage 1 (always at current point)
        with torch.no_grad():
            # Get model prediction at current state
            denoised = model_fn(x, sigma, **model_kwargs)
            # Calculate epsilon (velocity field in flow matching)
            eps_1 = (x - denoised) / sigma
            k.append(eps_1)
        
        # Stage 2 (and 3 for res_3s)
        for i in range(1, self.num_stages):
            # Calculate intermediate state using previous stages
            x_intermediate = x.clone()
            
            # Add contribution from all previous stages
            for j in range(i):
                # a[i,j] is the coefficient for stage j in computing stage i
                a_ij = a[i, j]
                if abs(a_ij) > 1e-10:  # Skip if coefficient is essentially zero
                    x_intermediate = x_intermediate + a_ij * k[j]
            
            # Intermediate sigma value
            sigma_intermediate = sigma + c[i] * h
            
            with torch.no_grad():
                # Get model prediction at intermediate state
                denoised = model_fn(x_intermediate, sigma_intermediate, **model_kwargs)
                # Calculate epsilon
                eps_i = (x_intermediate - denoised) / sigma_intermediate
                k.append(eps_i)
        
        # Final update: combine all stages using output weights b
        x_next = x.clone()
        for i in range(self.num_stages):
            x_next = x_next + b[i] * k[i]
        
        return x_next


class RES2SSampler:
    """
    RES 2S sampler wrapper for diffusers integration.
    Uses 2-stage exponential integrator with φ-functions.
    """
    
    def __init__(self):
        self.integrator = RESExponentialIntegrator(rk_type='res_2s', use_analytic=True)
    
    def step(self, model_fn, x, sigma, sigma_next, **model_kwargs):
        """Perform one RES 2S step."""
        return self.integrator.step(model_fn, x, sigma, sigma_next, **model_kwargs)


class RES3SSampler:
    """
    RES 3S sampler wrapper for diffusers integration.
    Uses 3-stage exponential integrator with φ-functions.
    """
    
    def __init__(self):
        self.integrator = RESExponentialIntegrator(rk_type='res_3s', use_analytic=True)
    
    def step(self, model_fn, x, sigma, sigma_next, **model_kwargs):
        """Perform one RES 3S step."""
        return self.integrator.step(model_fn, x, sigma, sigma_next, **model_kwargs)


def test_res_integrator():
    """Test RES integrator coefficient calculation."""
    print("Testing RES Exponential Integrator:\n")
    
    # Test coefficient calculation
    integrator_2s = RESExponentialIntegrator('res_2s')
    integrator_3s = RESExponentialIntegrator('res_3s')
    
    h = -0.1  # Step size
    
    print(f"Step size h = {h}\n")
    
    # RES 2S
    integrator_2s.calculate_coefficients(h)
    print("RES 2S Coefficients:")
    print(f"  a (stage matrix):\n{integrator_2s.a}")
    print(f"  b (output weights): {integrator_2s.b}")
    print(f"  c (stage nodes): {integrator_2s.c}")
    print(f"  b sum: {integrator_2s.b.sum():.10f}\n")
    
    # RES 3S
    integrator_3s.calculate_coefficients(h)
    print("RES 3S Coefficients:")
    print(f"  a (stage matrix):\n{integrator_3s.a}")
    print(f"  b (output weights): {integrator_3s.b}")
    print(f"  c (stage nodes): {integrator_3s.c}")
    print(f"  b sum: {integrator_3s.b.sum():.10f}\n")


if __name__ == "__main__":
    test_res_integrator()

