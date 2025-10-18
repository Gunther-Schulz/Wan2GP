"""
Adapter for integrating TRUE RES samplers with Wan2GP's sampling loop.

This adapts the RES exponential integrators to work with Wan's model calling
convention and flow matching architecture.
"""

import torch
from typing import Callable, Dict, Any
from .res_integrator import RES2SSampler, RES3SSampler
from .ralston_integrator import RalstonIntegrator


class RESWanAdapter:
    """
    Adapts RES exponential integrators for Wan's sampling loop.
    
    Wraps the model calling convention and provides step() interface
    compatible with Wan's sampling architecture.
    """
    
    def __init__(self, model, rk_type: str = 'res_2s', guide_scale: float = 1.0, joint_pass: bool = False):
        """
        Initialize RES adapter.
        
        Args:
            model: Wan transformer model
            rk_type: Either 'res_2s' or 'res_3s'
            guide_scale: CFG guidance scale
            joint_pass: Whether model uses joint pass for CFG
        """
        self.model = model
        self.rk_type = rk_type
        self.guide_scale = guide_scale
        self.joint_pass = joint_pass
        
        # Create appropriate RES sampler
        if rk_type == 'res_2s':
            self.res_sampler = RES2SSampler()
            self.ralston_sampler = RalstonIntegrator(num_stages=2)
        elif rk_type == 'res_3s':
            self.res_sampler = RES3SSampler()
            self.ralston_sampler = RalstonIntegrator(num_stages=3)
        else:
            raise ValueError(f"Unknown rk_type: {rk_type}")
    
    def model_fn(self, x: torch.Tensor, sigma: float, gen_args: Dict[str, Any], kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Model function wrapper that RES integrator can call.
        
        This adapts Wan's model calling convention for the RES integrator.
        The RES integrator needs a function that takes (x, sigma) and returns the VELOCITY (noise_pred).
        
        Args:
            x: Current latent state
            sigma: Current noise level (0-1 range for flow matching)
            gen_args: Conditioning arguments (context, etc.)
            kwargs: Additional model kwargs
            
        Returns:
            Velocity prediction (noise_pred) - NOT denoised!
        """
        # Convert sigma (0-1) to timestep (0-1000)
        timestep_value = sigma * 1000.0
        timestep = torch.tensor([timestep_value], device=x.device, dtype=torch.float32)
        
        # Update kwargs with current timestep
        kwargs_copy = kwargs.copy()
        kwargs_copy.update({"t": timestep})
        
        # CRITICAL: Handle both joint_pass and non-joint_pass modes
        # Update gen_args with current latent state
        # Check if context has None values (happens when CFG is disabled but gen_args still has 2 branches)
        context_has_none = False
        if 'context' in gen_args and isinstance(gen_args['context'], list):
            context_has_none = any(v is None for v in gen_args['context'])
            if context_has_none and self.guide_scale > 1.0 and not hasattr(self, '_warned_cfg_disabled'):
                print(f"⚠️  RES Sampler: CFG disabled (guide_scale={self.guide_scale} ignored) - context_null is None")
                print("   This can happen if guidance was disabled when the model was initialized.")
                self._warned_cfg_disabled = True
        
        gen_args_copy = {}
        for key, value in gen_args.items():
            if key == 'x':
                # Replace all x values in the list with current x
                if isinstance(value, list):
                    if context_has_none:
                        # If context has None, only use first branch (conditional)
                        gen_args_copy['x'] = [x]
                    else:
                        gen_args_copy['x'] = [x for _ in range(len(value))]
                else:
                    gen_args_copy['x'] = x
            elif key == 'context':
                # Handle context with None values
                if isinstance(value, list) and context_has_none:
                    # Only keep non-None values (fallback to conditional only)
                    gen_args_copy[key] = [v for v in value if v is not None]
                else:
                    gen_args_copy[key] = value
            else:
                # Keep other parameters as-is (audio, etc.)
                # But also filter them if context has None to match the reduced batch size
                if isinstance(value, list) and context_has_none:
                    # Only keep first element to match reduced batch
                    gen_args_copy[key] = [value[0]] if len(value) > 0 else value
                else:
                    gen_args_copy[key] = value
        
        # Call model based on joint_pass mode
        with torch.no_grad():
            if self.joint_pass:
                # Joint pass: Call model once with all CFG branches
                ret_values = self.model(**gen_args_copy, **kwargs_copy)
            else:
                # Non-joint pass: Call model separately for each CFG branch
                # EXACTLY like the main sampling loop does it
                # Handle case where CFG is disabled (guide_scale == 1)
                size = len(gen_args_copy['x']) if self.guide_scale != 1.0 else 1
                ret_values = []
                for x_id in range(size):
                    # Create sub_gen_args EXACTLY like main loop: {k : [v[x_id]] for...}
                    sub_gen_args = {k: [v[x_id]] for k, v in gen_args_copy.items()}
                    result = self.model(**sub_gen_args, x_id=x_id, **kwargs_copy)
                    ret_values.append(result[0])
        
        # Apply CFG if guide_scale > 1
        # If context had None values, we only have conditional branch, so skip CFG
        if context_has_none:
            # No CFG possible, use conditional only
            noise_pred = ret_values[0] if isinstance(ret_values, (list, tuple)) else ret_values
        elif self.guide_scale > 1.0 and isinstance(ret_values, (list, tuple)) and len(ret_values) >= 2:
            # Standard CFG: noise = uncond + guide_scale * (cond - uncond)
            noise_pred_cond, noise_pred_uncond = ret_values[0], ret_values[1]
            noise_pred = noise_pred_uncond + self.guide_scale * (noise_pred_cond - noise_pred_uncond)
        elif isinstance(ret_values, (list, tuple)):
            noise_pred = ret_values[0]
        else:
            noise_pred = ret_values
        
        # Convert to exponential parametrization velocity
        # RES4LYF exponential expects: velocity = denoised - x (points toward clean)
        # From flow matching: denoised = x - sigma * noise_pred
        # Therefore: velocity = (x - sigma * noise_pred) - x = -sigma * noise_pred
        # 
        # Actually checking RES4LYF line 959: epsilon = denoised - x_0
        # So velocity should be: denoised - x, which equals -sigma * noise_pred
        # But wait - let me recalculate:
        #   denoised = x - sigma * noise_pred
        #   velocity = denoised - x = (x - sigma * noise_pred) - x = -sigma * noise_pred ✓
        velocity_exponential = -sigma * noise_pred
        
        # Compute denoised for debugging
        denoised = x - sigma * noise_pred
        
        # Debug: Always print model call info
        if not hasattr(self, '_model_call_count'):
            self._model_call_count = 0
        self._model_call_count += 1
        
        print(f"      🤖 Model call #{self._model_call_count}: σ={sigma:.4f}")
        print(f"         Input x: [{x.min():.3f}, {x.max():.3f}] mean={x.mean():.3f}")
        print(f"         noise_pred: [{noise_pred.min():.3f}, {noise_pred.max():.3f}] mean={noise_pred.mean():.3f}")
        print(f"         denoised = x - σ*noise_pred: [{denoised.min():.3f}, {denoised.max():.3f}] mean={denoised.mean():.3f}")
        print(f"         velocity = denoised - x: [{velocity_exponential.min():.3f}, {velocity_exponential.max():.3f}] mean={velocity_exponential.mean():.3f}")
        print(f"         (Velocity should be NEGATIVE to denoise)")
        print(f"         (RK: x_next = x + h * velocity where h > 0 in log space)")
        
        return velocity_exponential
    
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
        
        # Special case: Final step to sigma=0
        # Exponential methods can't handle log(0), so use Ralston instead
        if sigma_next == 0.0:
            print(f"\n⚠️  Final step to σ=0 detected - using Ralston instead of RES")
            print(f"   (Exponential methods can't handle h = -log(0/σ) = infinity)")
            
            # Create model function wrapper for Ralston that returns LINEAR epsilon
            # LINEAR methods need: epsilon = (x - denoised) / sigma (i.e., noise_pred)
            # NOT the exponential velocity = denoised - x
            def model_fn_linear(x, sig):
                # Get raw noise_pred before exponential conversion
                timestep_value = sig * 1000.0
                timestep = torch.tensor([timestep_value], device=x.device, dtype=torch.float32)
                
                kwargs_copy = kwargs.copy()
                kwargs_copy.update({"t": timestep})
                
                # Handle context with None values
                context_has_none = False
                if 'context' in gen_args and isinstance(gen_args['context'], list):
                    context_has_none = any(v is None for v in gen_args['context'])
                
                gen_args_copy = {}
                for key, value in gen_args.items():
                    if key == 'x':
                        if isinstance(value, list):
                            if context_has_none:
                                gen_args_copy['x'] = [x]
                            else:
                                gen_args_copy['x'] = [x for _ in range(len(value))]
                        else:
                            gen_args_copy['x'] = x
                    elif key == 'context':
                        if isinstance(value, list) and context_has_none:
                            gen_args_copy[key] = [v for v in value if v is not None]
                        else:
                            gen_args_copy[key] = value
                    else:
                        if isinstance(value, list) and context_has_none:
                            gen_args_copy[key] = [value[0]] if len(value) > 0 else value
                        else:
                            gen_args_copy[key] = value
                
                # Call model
                with torch.no_grad():
                    if self.joint_pass:
                        ret_values = self.model(**gen_args_copy, **kwargs_copy)
                    else:
                        size = len(gen_args_copy['x']) if self.guide_scale != 1.0 else 1
                        ret_values = []
                        for x_id in range(size):
                            sub_gen_args = {k: [v[x_id]] for k, v in gen_args_copy.items()}
                            result = self.model(**sub_gen_args, x_id=x_id, **kwargs_copy)
                            ret_values.append(result[0])
                    
                    # Apply CFG
                    if context_has_none:
                        noise_pred = ret_values[0] if isinstance(ret_values, (list, tuple)) else ret_values
                    elif self.guide_scale > 1.0 and isinstance(ret_values, (list, tuple)) and len(ret_values) >= 2:
                        noise_pred_cond, noise_pred_uncond = ret_values[0], ret_values[1]
                        noise_pred = noise_pred_uncond + self.guide_scale * (noise_pred_cond - noise_pred_uncond)
                    elif isinstance(ret_values, (list, tuple)):
                        noise_pred = ret_values[0]
                    else:
                        noise_pred = ret_values
                    
                    # For LINEAR methods, return (x - denoised) / sigma = noise_pred
                    # This is what RES4LYF's RK_Method_Linear returns as epsilon
                    return noise_pred
            
            # Use Ralston integrator (2nd or 3rd order linear RK)
            latents_next = self.ralston_sampler.step(
                model_fn=model_fn_linear,
                x=latents,
                sigma=sigma,
                sigma_next=sigma_next
            )
            
            print(f"   Ralston step complete")
            print(f"   Input:  [{latents.min():.3f}, {latents.max():.3f}] std={latents.std():.3f}")
            print(f"   Output: [{latents_next.min():.3f}, {latents_next.max():.3f}] std={latents_next.std():.3f}")
            
            return latents_next
        
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
        
        # Debug: Print step info
        if not hasattr(self, '_step_count'):
            self._step_count = 0
        self._step_count += 1
        
        print(f"\n{'='*80}")
        print(f"📊 RES Step {self._step_count}: σ={sigma:.4f} → σ_next={sigma_next:.4f}")
        print(f"   Step size h = σ - σ_next = {sigma:.4f} - {sigma_next:.4f} = {sigma-sigma_next:.4f}")
        print(f"   Input latents:  [{latents.min():.3f}, {latents.max():.3f}] mean={latents.mean():.3f} std={latents.std():.3f}")
        print(f"   Output latents: [{latents_next.min():.3f}, {latents_next.max():.3f}] mean={latents_next.mean():.3f} std={latents_next.std():.3f}")
        
        # Check if we're making progress toward clean signal
        if latents_next.std() < latents.std():
            print(f"   ✅ GOOD: std decreased ({latents.std():.3f} → {latents_next.std():.3f})")
        else:
            print(f"   ⚠️  WARNING: std increased ({latents.std():.3f} → {latents_next.std():.3f}) - going toward noise!")
        
        print(f"{'='*80}")
        
        return latents_next


def create_res_adapter(model, rk_type: str, guide_scale: float = 1.0, joint_pass: bool = False) -> RESWanAdapter:
    """
    Factory function to create RES adapter.
    
    Args:
        model: Wan transformer model
        rk_type: Either 'res_2s' or 'res_3s'
        guide_scale: CFG guidance scale
        joint_pass: Whether model uses joint pass for CFG
        
    Returns:
        RESWanAdapter instance
    """
    return RESWanAdapter(model, rk_type, guide_scale, joint_pass)

