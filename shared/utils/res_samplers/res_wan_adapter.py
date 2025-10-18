"""
Adapter for integrating TRUE RES samplers with Wan2GP's sampling loop.

This adapts the RES exponential integrators to work with Wan's model calling
convention and flow matching architecture.
"""

import torch
from typing import Callable, Dict, Any
from .res_integrator import RES2SSampler, RES3SSampler


class RESWanAdapter:
    """
    Adapts RES exponential integrators for Wan's sampling loop.
    
    Wraps the model calling convention and provides step() interface
    compatible with Wan's sampling architecture.
    """
    
    def __init__(self, model, rk_type: str = 'res_2s'):
        """
        Initialize RES adapter.
        
        Args:
            model: Wan transformer model
            rk_type: Either 'res_2s' or 'res_3s'
        """
        self.model = model
        self.rk_type = rk_type
        
        # Create appropriate RES sampler
        if rk_type == 'res_2s':
            self.res_sampler = RES2SSampler()
        elif rk_type == 'res_3s':
            self.res_sampler = RES3SSampler()
        else:
            raise ValueError(f"Unknown rk_type: {rk_type}")
    
    def model_fn(self, x: torch.Tensor, sigma: float, gen_args: Dict[str, Any], kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Model function wrapper that RES integrator can call.
        
        This adapts Wan's model calling convention for the RES integrator.
        The RES integrator needs a function that takes (x, sigma) and returns denoised output.
        
        Args:
            x: Current latent state
            sigma: Current noise level (0-1 range for flow matching)
            gen_args: Conditioning arguments (context, etc.)
            kwargs: Additional model kwargs
            
        Returns:
            Denoised prediction
        """
        # Convert sigma (0-1) to timestep (0-1000)
        timestep_value = sigma * 1000.0
        timestep = torch.tensor([timestep_value], device=x.device, dtype=torch.float32)
        
        # Update kwargs with current timestep
        kwargs_copy = kwargs.copy()
        kwargs_copy.update({"t": timestep})
        
        # Call model to get noise prediction
        # Model returns noise, we need to convert to denoised
        with torch.no_grad():
            noise_pred = self.model(**gen_args, **kwargs_copy)
        
        # Flow matching: x = signal + sigma * noise
        # So: signal = x - sigma * noise
        # For proper flow matching, we use timestep/1000 as sigma
        denoised = x - (sigma * noise_pred)
        
        return denoised
    
    def step(self, latents: torch.Tensor, current_timestep: torch.Tensor, next_timestep: torch.Tensor,
             gen_args: Dict[str, Any], kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Perform one RES step.
        
        This is called from Wan's sampling loop and uses the RES exponential integrator
        to compute the next latent state.
        
        Args:
            latents: Current latent state
            current_timestep: Current timestep value (0-1000)
            next_timestep: Next timestep value (0-1000)
            gen_args: Model conditioning arguments
            kwargs: Additional model kwargs
            
        Returns:
            Next latent state
        """
        # Convert timesteps to sigmas (0-1 range)
        sigma = current_timestep.item() / 1000.0
        sigma_next = next_timestep.item() / 1000.0
        
        # Create model function that RES can call
        def model_fn_wrapper(x, sig):
            return self.model_fn(x, sig, gen_args, kwargs)
        
        # Use RES exponential integrator to compute next state
        latents_next = self.res_sampler.step(
            model_fn=model_fn_wrapper,
            x=latents,
            sigma=sigma,
            sigma_next=sigma_next
        )
        
        return latents_next


def create_res_adapter(model, rk_type: str) -> RESWanAdapter:
    """
    Factory function to create RES adapter.
    
    Args:
        model: Wan transformer model
        rk_type: Either 'res_2s' or 'res_3s'
        
    Returns:
        RESWanAdapter instance
    """
    return RESWanAdapter(model, rk_type)

