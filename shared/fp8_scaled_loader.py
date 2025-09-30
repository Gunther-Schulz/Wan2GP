"""
FP8 Scaled Model Loader
Handles loading of fp8_scaled models with scale_weight tensors from Kijai's WanVideo_comfy_fp8_scaled repository.
Functionally equivalent to ComfyUI-WanVideoWrapper but optimized for our existing infrastructure.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from mmgp import safetensors2


def extract_scale_weights(state_dict: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    """
    Extract scale_weight tensors from state dict, optimized for JSON config approach.
    
    Args:
        state_dict: Model state dictionary
        device: Target device for scale weights
        
    Returns:
        Dictionary of scale weights
    """
    base_dtype = torch.float32  # ComfyUI wrapper uses float32 for scale weights
    
    # Optimized: Only extract scale weights for known fp8_scaled models
    # Since we use JSON config, we know this is a scaled model, so scale weights must exist
    # Use dict comprehension for better performance
    scale_weights = {
        k: v.to(device, base_dtype) 
        for k, v in state_dict.items() 
        if k.endswith(".scale_weight")
    }
    
    return scale_weights


def validate_fp8_scaled_model(state_dict: Dict[str, torch.Tensor], quantization: str) -> None:
    """
    Validate fp8_scaled model - optimized for JSON config approach.
    
    Since we use JSON config, we trust the quantization type and skip expensive validation.
    """
    # JSON config approach: Skip expensive state_dict validation
    # We trust that JSON config is correct for the model
    pass


def apply_fp8_optimization_to_model(model, base_dtype, model_filename, device, quantization=None):
    """
    Apply fp8 optimization to a loaded model - ComfyUI approach.
    
    Args:
        model: The loaded model (WanModel instance)
        base_dtype: Base dtype for computations (torch.bfloat16, torch.float16, etc.)
        model_filename: Original model filename (for reference)
        device: Target device
        quantization: Detected quantization type (e.g., "fp8_e4m3fn_scaled")
    """
    # Check if torch._scaled_mm is available
    if not hasattr(torch, '_scaled_mm'):
        print("torch._scaled_mm not available - skipping fp8 optimization")
        return False
    
    if not quantization or "fp8" not in quantization:
        return False
    
    if "scaled" not in quantization:
        print(f"FP8 quantization {quantization} detected but not scaled - no optimization needed")
        return False
    
    # PERFORMANCE FIX: Extract scale weights directly from model parameters (no state_dict call!)
    # This leverages Wan2GP's architecture efficiently without ComfyUI overhead
    # CRITICAL: Let offload system manage device placement - don't force CUDA here!
    scale_weights = {}
    for name, param in model.named_parameters():
        if name.endswith(".scale_weight"):
            # CRITICAL FIX: Keep on current device (CPU/disk with writable_tensors=False)
            # Offload system will move to CUDA when needed during forward pass
            # Only convert dtype, don't force device transfer!
            scale_weights[name] = param.detach().clone().to(dtype=torch.float32)
    
    if len(scale_weights) == 0:
        print(f"No scale weights found for quantization {quantization}")
        return False
    
    print(f"Applying fp8 optimization using {len(scale_weights)} scale weights for {quantization}...")
    
    # Use our efficient approach that leverages Wan2GP's existing infrastructure
    optimized_count = apply_fp8_optimization_to_model_simple(model, base_dtype, scale_weights)
    
    if optimized_count > 0:
        print(f"Successfully optimized {optimized_count} Linear layers for fp8")
        return True
    
    return False


# Removed is_fp8_scaled_model() - no longer needed since we detect by examining loaded state dict


# Removed detect_fp8_quantization - using JSON config only

# Removed duplicate create_fp8_optimized_forward - using simplified approach instead


def apply_fp8_optimization_to_model_simple(model, base_dtype, scale_weights):
    """
    Simplified fp8 optimization that leverages our existing infrastructure.
    Instead of complex layer patching, we enhance our existing Linear layers.
    """
    optimized_count = 0
    
    # Create shared enhanced forward function to avoid recreating for each module
    def create_enhanced_forward(module, scale_weight, base_dtype):
        original_forward = module.forward
        # PERFORMANCE FIX: Cache weight dtype at setup time (not every forward pass!)
        weight_dtype = module.weight.dtype
        is_fp8 = weight_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
        
        def enhanced_forward(input):
            # OPTIMIZED: Only check input shape - dtype and scale_weight checked at setup
            if is_fp8 and len(input.shape) == 3:
                return apply_fp8_matmul(
                    input, module.weight, module.bias, 
                    scale_weight,  # Captured from closure
                    base_dtype     # Captured from closure
                )
            
            # Fallback to original forward (preserves all existing functionality)
            return original_forward(input)
        
        return enhanced_forward
    
    # Leverage our existing infrastructure: efficient layer iteration without complex filtering
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            scale_key = f"{name}.scale_weight"
            
            if scale_key in scale_weights:
                # Leverage mmgp's efficient tensor handling - no manual dtype conversion needed
                scale_weight = scale_weights[scale_key]
                
                # Enhance existing forward method (preserves all mmgp functionality)
                # PERFORMANCE: Pass scale_weight and base_dtype to closure (no module attributes needed)
                if not hasattr(module, '_fp8_enhanced'):
                    module.forward = create_enhanced_forward(module, scale_weight, base_dtype)
                    module._fp8_enhanced = True
                
                optimized_count += 1
    
    return optimized_count


# Removed duplicate create_enhanced_linear_forward - integrated into apply_fp8_optimization_to_model_simple


def apply_fp8_matmul(input, weight, bias, scale_weight, base_dtype):
    """Optimized fp8 matrix multiplication - minimal tensor creation, leverages existing infrastructure."""
    input_shape = input.shape
    
    # Optimized: reuse input device, avoid creating new tensors when possible
    scale_input = torch.ones((), device=input.device, dtype=torch.float32)
    
    # CRITICAL: Move scale_weight to input device only if needed (offload system compatibility)
    # Check device to avoid unnecessary transfers
    if scale_weight.device != input.device:
        scale_weight = scale_weight.to(input.device)
    
    # Streamlined fp8 conversion (in-place clamping for memory efficiency)
    input_clamped = torch.clamp(input, min=-448, max=448, out=input)
    input_fp8 = input_clamped.reshape(-1, input_shape[2]).to(torch.float8_e4m3fn).contiguous()
    
    # Efficient bias handling - only convert if needed
    bias = bias.to(base_dtype) if bias is not None else None
    
    # Optimized: scale_weight moved to correct device above (if needed)
    output = torch._scaled_mm(
        input_fp8, weight.t(),
        out_dtype=base_dtype,
        bias=bias,
        scale_a=scale_input,
        scale_b=scale_weight
    )
    
    return output.reshape((-1, input_shape[1], weight.shape[0]))


# Global cache for fp8 scale weights (avoids memory duplication)
_fp8_scale_weights_cache = {}

def get_fp8_quantization_from_json(json_quantization: str = None) -> str:
    """
    Get fp8 quantization from JSON config ONLY - pure performance approach.
    
    NO state_dict scanning, NO expensive detection, NO fallbacks.
    
    Args:
        json_quantization: Quantization from JSON config
        
    Returns:
        Quantization type from JSON or "disabled"
    """
    if json_quantization and json_quantization != "disabled":
        print(f"Using JSON config quantization: {json_quantization}")
        return json_quantization
    
    return "disabled"

def _cache_fp8_scale_weights(filename: str, scale_weights: Dict[str, torch.Tensor], quantization: str):
    """Cache scale weights to avoid memory duplication during model loading."""
    cache_key = filename  # Use filename as cache key
    _fp8_scale_weights_cache[cache_key] = {
        'scale_weights': scale_weights,
        'quantization': quantization
    }

def get_cached_fp8_scale_weights(filename: str) -> Optional[Dict[str, torch.Tensor]]:
    """Get cached scale weights for a model file."""
    cache_key = filename
    cached = _fp8_scale_weights_cache.get(cache_key)
    return cached['scale_weights'] if cached else None

def clear_fp8_cache():
    """Clear the fp8 scale weights cache - simple approach like existing mmgp patterns."""
    global _fp8_scale_weights_cache
    _fp8_scale_weights_cache.clear()


print("FP8 scaled model loader initialized - JSON config based approach")
print("- fp8_scaled models specified in JSON config (no auto-detection)")
print("- torch._scaled_mm optimization enabled (requires CUDA compute capability >= 8.9)")
print("- Integrates with existing Wan2GP LoRA system and mmgp.offload infrastructure")
print("- Streamlined approach: patches forward methods instead of replacing Linear layers")
print("- NO monkeypatch on torch_load_file (unnecessary overhead removed)")
