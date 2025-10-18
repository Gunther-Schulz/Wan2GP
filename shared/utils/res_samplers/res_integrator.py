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
        # Store x_0 as anchor point for this step (RES4LYF anchoring)
        x_0 = x.clone()
        main_sigma = sigma
        
        # Safety check: exponential methods can't handle sigma_next = 0
        if sigma_next == 0.0:
            raise ValueError(
                "RES exponential integrator cannot handle sigma_next=0 "
                "(would result in h = -log(0) = infinity). "
                "Use Euler or another method for the final step to sigma=0."
            )
        
        # Step size for RK integration in LOG SPACE (exponential methods)
        # RES uses: h = -log(sigma_next/sigma) 
        # This is POSITIVE when sigma decreases (log gets more negative)
        h = -torch.log(torch.tensor(sigma_next / sigma)).item()
        
        # Calculate coefficients for this step size
        self.calculate_coefficients(float(h))
        
        # Debug: Print coefficients on first call
        if not hasattr(self, '_coeff_printed'):
            print(f"   ⚙️  RES-{self.num_stages}S (EXPONENTIAL) Coefficients:")
            print(f"      h = -log(σ_next/σ) = {float(h):.4f} (positive in log space)")
            print(f"      Stage nodes c: {self.c}")
            print(f"      Output weights b: {self.b}")
            print(f"      Stage matrix a:")
            for row_idx, row in enumerate(self.a):
                print(f"         a[{row_idx}]: {row}")
            print(f"      Formula: x_next = x + h * sum(b[i] * k[i]) where k = velocity")
            print(f"      Exponential σ_intermediate = σ * exp(-h * c[i])")
            self._coeff_printed = True
        
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
            # Get velocity from model with anchoring
            # Pass x_0 and main_sigma for RES4LYF-style anchoring
            velocity = model_fn(x, sigma, x_0=x_0, main_sigma=main_sigma, **model_kwargs)
            k.append(velocity)
            
            # Debug first stage
            if not hasattr(self, '_stage_debug_count'):
                self._stage_debug_count = 0
            if self._stage_debug_count < 1:
                self._stage_debug_count += 1
                print(f"   🔬 Stage 1 (at σ={sigma:.4f}):")
                print(f"      Input x: [{x.min():.3f}, {x.max():.3f}] mean={x.mean():.3f}")
                print(f"      Velocity: [{velocity.min():.3f}, {velocity.max():.3f}] mean={velocity.mean():.3f}")
                print(f"      k[0] = velocity")
        
        # Stage 2 (and 3 for res_3s)
        for i in range(1, self.num_stages):
            # Calculate intermediate state using previous stages
            # RK formula: x_intermediate = x + h * sum(a[i,j] * k[j])
            x_intermediate = x.clone()
            
            # Add contribution from all previous stages
            for j in range(i):
                # a[i,j] is the coefficient for stage j in computing stage i
                a_ij = a[i, j]
                if abs(a_ij) > 1e-10:  # Skip if coefficient is essentially zero
                    x_intermediate = x_intermediate + h * a_ij * k[j]  # RK: x + h * a * velocity
            
            # Intermediate sigma value for EXPONENTIAL methods (RES uses log space)
            # Formula: sigma_intermediate = sigma * exp(-h * c[i])
            # Where h = -log(sigma_next/sigma) is positive in log space
            sigma_intermediate = sigma * torch.exp(-h * c[i])
            
            with torch.no_grad():
                # Get velocity at intermediate state with anchoring
                velocity_i = model_fn(x_intermediate, sigma_intermediate, x_0=x_0, main_sigma=main_sigma, **model_kwargs)
                k.append(velocity_i)
                
                # Debug intermediate stages
                if hasattr(self, '_stage_debug_count') and self._stage_debug_count == 1:
                    print(f"   🔬 Stage {i+1} (at σ={sigma_intermediate:.4f}):")
                    print(f"      x_intermediate: [{x_intermediate.min():.3f}, {x_intermediate.max():.3f}] mean={x_intermediate.mean():.3f}")
                    print(f"      Velocity: [{velocity_i.min():.3f}, {velocity_i.max():.3f}] mean={velocity_i.mean():.3f}")
                    print(f"      k[{i}] = velocity")
        
        # Final update: combine all stages using output weights b
        # Exponential RK formula: x_next = x + h * sum(b[i] * k[i])
        # Where h > 0 (positive in log space), k[i] = velocity = -σ*noise_pred (negative, toward clean)
        # Result: x + (positive h) * (negative velocity) = x - something = denoising!
        x_next = x.clone()
        
        # Debug combination in detail
        if hasattr(self, '_stage_debug_count') and self._stage_debug_count == 1:
            print(f"   🎯 Final Combination (Exponential RK):")
            print(f"      x_next = x + h * sum(b[i] * k[i]) where h={h:.4f} > 0")
            print(f"      Starting x: mean={x.mean():.3f}")
            
        for i in range(self.num_stages):
            contribution = h * b[i] * k[i]  # Multiply by step size h
            x_next = x_next + contribution
            
            if hasattr(self, '_stage_debug_count') and self._stage_debug_count == 1:
                print(f"      + h*b[{i}]={h:.4f}*{b[i]:.6f} * k[{i}] (mean={k[i].mean():.3f}) = {contribution.mean():.3f}")
        
        # Debug final result
        if hasattr(self, '_stage_debug_count') and self._stage_debug_count == 1:
            print(f"      Final x_next: [{x_next.min():.3f}, {x_next.max():.3f}] mean={x_next.mean():.3f}")
            self._stage_debug_count = 2  # Only show detailed debug for first step
        
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

