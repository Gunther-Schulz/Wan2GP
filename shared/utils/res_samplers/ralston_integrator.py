"""
Ralston Runge-Kutta integrator for the final step to sigma=0.

Ralston is a 2nd-order linear RK method that can handle sigma=0 
(unlike exponential methods which require log operations).

This is what RES4LYF uses as fallback for the final step when using res_2s/res_3s.
"""

import torch
import numpy as np
from typing import Callable, Optional


class RalstonIntegrator:
    """
    Ralston's 2nd/3rd-order Runge-Kutta method for linear ODE integration.
    
    Unlike exponential methods, this uses simple linear interpolation:
    - h = sigma_next - sigma (not log-space)
    - Standard RK formulas without phi functions
    - Can handle sigma_next = 0 without issues
    """
    
    def __init__(self, num_stages: int = 2):
        """
        Initialize Ralston integrator.
        
        Args:
            num_stages: 2 for Ralston-2S or 3 for Ralston-3S
        """
        self.num_stages = num_stages
        
        # Ralston coefficients (fixed, don't depend on step size)
        if num_stages == 2:
            # Ralston-2S: 2nd-order method
            self.a = np.array([
                [0.0, 0.0],
                [2.0/3.0, 0.0],
            ])
            self.b = np.array([1.0/4.0, 3.0/4.0])
            self.c = np.array([0.0, 2.0/3.0])
        elif num_stages == 3:
            # Ralston-3S: 3rd-order method
            self.a = np.array([
                [0.0, 0.0, 0.0],
                [1.0/2.0, 0.0, 0.0],
                [0.0, 3.0/4.0, 0.0],
            ])
            self.b = np.array([2.0/9.0, 1.0/3.0, 4.0/9.0])
            self.c = np.array([0.0, 1.0/2.0, 3.0/4.0])
        else:
            raise ValueError(f"Ralston only supports 2 or 3 stages, got {num_stages}")
    
    def step(self, model_fn: Callable, x: torch.Tensor, sigma: float, 
             sigma_next: float, **model_kwargs) -> torch.Tensor:
        """
        Perform one Ralston RK step.
        
        Args:
            model_fn: Function that takes (x, sigma) and returns velocity
            x: Current latent state
            sigma: Current noise level
            sigma_next: Next noise level
            **model_kwargs: Additional arguments for model
            
        Returns:
            Next latent state x_next
        """
        # Step size (linear, not log-space!)
        h = sigma_next - sigma  # Negative when denoising
        
        # Debug: Print info on first call
        if not hasattr(self, '_coeff_printed'):
            print(f"   🎯 Ralston-{self.num_stages}S (LINEAR) for final step:")
            print(f"      h = σ_next - σ = {h:.4f} (negative = denoising)")
            print(f"      Stage nodes c: {self.c}")
            print(f"      Output weights b: {self.b}")
            print(f"      Stage matrix a:")
            for row_idx, row in enumerate(self.a):
                print(f"         a[{row_idx}]: {row}")
            self._coeff_printed = True
        
        # Convert coefficients to tensors
        device = x.device
        dtype = x.dtype
        a = torch.tensor(self.a, device=device, dtype=torch.float32)
        b = torch.tensor(self.b, device=device, dtype=torch.float32)
        c = torch.tensor(self.c, device=device, dtype=torch.float32)
        
        # Storage for stage derivatives
        k = []
        
        # Stage 1 (always at current point)
        with torch.no_grad():
            # For LINEAR methods, model_fn returns epsilon = noise_pred
            epsilon = model_fn(x, sigma, **model_kwargs)
            k.append(epsilon)
            
            if not hasattr(self, '_stage_debug'):
                # Compute what denoised would be using flow matching formula
                denoised_stage1 = x - sigma * epsilon
                print(f"   🔬 Ralston Stage 1 (at σ={sigma:.4f}):")
                print(f"      Input x: [{x.min():.3f}, {x.max():.3f}] mean={x.mean():.6f} std={x.std():.6f}")
                print(f"      epsilon (noise_pred): [{epsilon.min():.3f}, {epsilon.max():.3f}] mean={epsilon.mean():.6f}")
                print(f"      Implied denoised = x - σ*ε: [{denoised_stage1.min():.3f}, {denoised_stage1.max():.3f}] std={denoised_stage1.std():.6f}")
                if denoised_stage1.std() < x.std():
                    print(f"      ✅ Denoised is cleaner ({denoised_stage1.std():.6f} < {x.std():.6f})")
                else:
                    print(f"      ⚠️  Denoised is NOT cleaner ({denoised_stage1.std():.6f} >= {x.std():.6f})")
                self._stage_debug = True
        
        # Subsequent stages (2 or 3 depending on num_stages)
        for i in range(1, self.num_stages):
            # Calculate intermediate state
            # LINEAR RK: x_intermediate = x + h * sum(a[i,j] * k[j])
            # Where h < 0 (denoising direction) and k[j] = epsilon = noise_pred
            x_intermediate = x.clone()
            
            for j in range(i):
                a_ij = a[i, j]
                if abs(a_ij) > 1e-10:
                    # h is negative, epsilon is noise_pred
                    # x_intermediate = x + (negative h) * (positive noise_pred) = x - something
                    x_intermediate = x_intermediate + h * a_ij * k[j]
            
            # Intermediate sigma (linear interpolation)
            sigma_intermediate = sigma + c[i] * h
            
            with torch.no_grad():
                epsilon_i = model_fn(x_intermediate, sigma_intermediate, **model_kwargs)
                k.append(epsilon_i)
                
                if hasattr(self, '_stage_debug') and self._stage_debug:
                    denoised_stagei = x_intermediate - sigma_intermediate * epsilon_i
                    print(f"   🔬 Ralston Stage {i+1} (at σ={sigma_intermediate:.4f}):")
                    print(f"      x_intermediate: [{x_intermediate.min():.3f}, {x_intermediate.max():.3f}] std={x_intermediate.std():.6f}")
                    print(f"      epsilon (noise_pred): [{epsilon_i.min():.3f}, {epsilon_i.max():.3f}] mean={epsilon_i.mean():.6f}")
                    print(f"      Implied denoised = x_int - σ*ε: std={denoised_stagei.std():.6f}")
        
        # Final update: x_next = x + h * sum(b[i] * k[i])
        # Where h < 0, k[i] = epsilon = noise_pred
        # So: x_next = x - |h| * sum(b[i] * noise_pred) = denoising
        x_next = x.clone()
        
        if hasattr(self, '_stage_debug') and self._stage_debug:
            print(f"   🎯 Ralston Final Combination (LINEAR):")
            print(f"      Formula: x_next = x + h * sum(b[i] * k[i])")
            print(f"      h = {h:.6f} (should be negative for denoising)")
            print(f"      k[i] = epsilon = noise_pred (velocity toward noise)")
            print(f"      Expected: x + (negative h) * (positive noise_pred) = x - something = denoising")
            self._stage_debug = False  # Only debug once
        
        # Track total contribution
        total_contribution = torch.zeros_like(x)
        
        for i in range(self.num_stages):
            contribution = h * b[i] * k[i]
            total_contribution = total_contribution + contribution
            x_next = x_next + contribution
            
            if not hasattr(self, '_combination_done'):
                print(f"      Stage {i}: h*b[{i}] = {h:.6f}*{b[i]:.6f} = {h*b[i]:.6f}")
                print(f"                k[{i}] mean={k[i].mean():.6f}, contribution mean={contribution.mean():.6f}")
        
        if not hasattr(self, '_combination_done'):
            print(f"      Total contribution: mean={total_contribution.mean():.6f}, std_change={total_contribution.std():.6f}")
            print(f"      Expected: contribution mean should be NEGATIVE (to denoise)")
            if total_contribution.mean() > 0:
                print(f"      ⚠️ WARNING: Positive contribution = ADDING NOISE!")
            
            # Compare with simple Euler for sanity check
            euler_step = h * k[0]  # Simple Euler: x_next = x + h * k[0]
            x_euler = x + euler_step
            print(f"   ")
            print(f"   📊 Comparison with Euler:")
            print(f"      Euler step = h * k[0] = {h:.6f} * {k[0].mean():.6f} = {euler_step.mean():.6f}")
            print(f"      Euler x_next std: {x_euler.std():.6f} (vs input {x.std():.6f})")
            print(f"      Ralston x_next std: {x_next.std():.6f} (vs input {x.std():.6f})")
            if x_euler.std() < x.std():
                print(f"      Euler would denoise ✅")
            else:
                print(f"      Euler would NOT denoise ⚠️ - model predictions may be wrong!")
            self._combination_done = True
        
        return x_next

