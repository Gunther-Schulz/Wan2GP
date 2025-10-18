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
    
    def __init__(self, model, rk_type: str = 'res_2s', guide_scale: float = 1.0):
        """
        Initialize RES adapter.
        
        Args:
            model: Wan transformer model
            rk_type: Either 'res_2s' or 'res_3s'
            guide_scale: CFG guidance scale
        """
        self.model = model
        self.rk_type = rk_type
        self.guide_scale = guide_scale
        
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
        
        # CRITICAL: Update gen_args with current latent state
        # gen_args contains 'x' which is a list of latents for CFG
        # We need to update all of them with the current x value
        gen_args_copy = gen_args.copy()
        if 'x' in gen_args_copy:
            # Replace all x values in the list with current x
            x_list = gen_args_copy['x']
            if isinstance(x_list, list):
                gen_args_copy['x'] = [x for _ in range(len(x_list))]
            else:
                gen_args_copy['x'] = x
        
        # Call model to get noise prediction
        # Model returns noise, we need to convert to denoised
        with torch.no_grad():
            ret_values = self.model(**gen_args_copy, **kwargs_copy)
        
        # Apply CFG if guide_scale > 1
        if self.guide_scale > 1.0 and isinstance(ret_values, (list, tuple)) and len(ret_values) >= 2:
            # Standard CFG: noise = uncond + guide_scale * (cond - uncond)
            noise_pred_cond, noise_pred_uncond = ret_values[0], ret_values[1]
            noise_pred = noise_pred_uncond + self.guide_scale * (noise_pred_cond - noise_pred_uncond)
        elif isinstance(ret_values, (list, tuple)):
            noise_pred = ret_values[0]
        else:
            noise_pred = ret_values
        
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


def create_res_adapter(model, rk_type: str, guide_scale: float = 1.0) -> RESWanAdapter:
    """
    Factory function to create RES adapter.
    
    Args:
        model: Wan transformer model
        rk_type: Either 'res_2s' or 'res_3s'
        guide_scale: CFG guidance scale
        
    Returns:
        RESWanAdapter instance
    """
    return RESWanAdapter(model, rk_type, guide_scale)

