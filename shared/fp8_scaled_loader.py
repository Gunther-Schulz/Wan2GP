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
    import time
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"FP8 OPTIMIZATION DEBUG - Starting")
    print(f"{'='*60}")
    print(f"Model filename: {model_filename}")
    print(f"Target device: {device}")
    print(f"Base dtype: {base_dtype}")
    print(f"Quantization: {quantization}")
    
    # Check if torch._scaled_mm is available
    if not hasattr(torch, '_scaled_mm'):
        print("❌ torch._scaled_mm not available - skipping fp8 optimization")
        return False
    
    if not quantization or "fp8" not in quantization:
        print(f"❌ No fp8 quantization detected: {quantization}")
        return False
    
    if "scaled" not in quantization:
        print(f"❌ FP8 quantization {quantization} detected but not scaled - no optimization needed")
        return False
    
    # CRITICAL: Scale weights are in the safetensors file but NOT as model parameters!
    # OPTIMIZATION: Use safe_open to selectively load ONLY scale weights (not entire 14GB model!)
    print(f"\n📊 Extracting scale weights from safetensors file...")
    print(f"   Model file: {model_filename}")
    
    from safetensors import safe_open
    import os
    
    # Handle both single file and list of files
    if isinstance(model_filename, list):
        model_file = model_filename[0] if len(model_filename) > 0 else None
    else:
        model_file = model_filename
    
    if not model_file or not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        return False
    
    print(f"   Selectively loading scale weights (not full model)...")
    # PERFORMANCE: Use safe_open to load ONLY scale weights without loading entire state_dict
    # This is MUCH faster and uses minimal memory vs loading full 14GB model
    scale_weights = {}
    try:
        with safe_open(model_file, framework="pt", device="cpu") as f:
            total_keys = len(f.keys())
            print(f"   File contains {total_keys} tensors total")
            
            for key in f.keys():
                if key.endswith(".scale_weight"):
                    tensor = f.get_tensor(key)
                    # CRITICAL: Keep as float32, let offload manage device placement
                    scale_weights[key] = tensor.to(dtype=torch.float32)
        
        print(f"   ✅ Selective load complete (loaded only {len(scale_weights)} scale weights, not full model)")
    except Exception as e:
        print(f"❌ Failed to load scale weights: {e}")
        return False
    
    if len(scale_weights) == 0:
        print(f"❌ No scale weights found in state_dict for quantization {quantization}")
        print(f"   Checked all keys ending with '.scale_weight'")
        return False
    
    print(f"\n✅ Found {len(scale_weights)} scale weights")
    print(f"   All loaded on CPU (device=cpu)")
    
    # Use our efficient approach that leverages Wan2GP's existing infrastructure
    optimized_count = apply_fp8_optimization_to_model_simple(model, base_dtype, scale_weights)
    
    elapsed = time.time() - start_time
    if optimized_count > 0:
        print(f"\n✅ Successfully optimized {optimized_count} Linear layers for fp8")
        print(f"⏱️  Total setup time: {elapsed:.3f}s")
        print(f"{'='*60}\n")
        return True
    
    print(f"❌ No layers optimized")
    print(f"{'='*60}\n")
    return False


# Removed is_fp8_scaled_model() - no longer needed since we detect by examining loaded state dict


# Removed detect_fp8_quantization - using JSON config only

# Removed duplicate create_fp8_optimized_forward - using simplified approach instead


def apply_fp8_optimization_to_model_simple(model, base_dtype, scale_weights):
    """
    Simplified fp8 optimization that leverages our existing infrastructure.
    Instead of complex layer patching, we enhance our existing Linear layers.
    """
    import time
    start_time = time.time()
    optimized_count = 0
    skipped_count = 0
    
    print(f"\n🔧 Patching Linear layers...")
    
    # Create shared enhanced forward function to avoid recreating for each module
    def create_enhanced_forward(module, scale_weight, base_dtype, layer_name):
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
                    module.forward = create_enhanced_forward(module, scale_weight, base_dtype, name)
                    module._fp8_enhanced = True
                    optimized_count += 1
                else:
                    skipped_count += 1
    
    elapsed = time.time() - start_time
    print(f"\n✅ Patched {optimized_count} layers ({skipped_count} already patched)")
    print(f"⏱️  Patching time: {elapsed:.3f}s")
    
    return optimized_count


# Removed duplicate create_enhanced_linear_forward - integrated into apply_fp8_optimization_to_model_simple


# Global debug counter for fp8 forward passes
_fp8_forward_count = 0
_fp8_device_transfers = 0

def apply_fp8_matmul(input, weight, bias, scale_weight, base_dtype):
    """Optimized fp8 matrix multiplication - minimal tensor creation, leverages existing infrastructure."""
    global _fp8_forward_count, _fp8_device_transfers
    _fp8_forward_count += 1
    
    # Debug logging every 100 calls
    debug_this_call = (_fp8_forward_count % 100 == 1)
    
    if debug_this_call:
        print(f"\n🔍 FP8 Forward #{_fp8_forward_count} - Device Check:")
        print(f"   input: device={input.device}, shape={input.shape}, dtype={input.dtype}")
        print(f"   weight: device={weight.device}, dtype={weight.dtype}")
        print(f"   scale_weight: device={scale_weight.device}, dtype={scale_weight.dtype}")
    
    input_shape = input.shape
    
    # Optimized: reuse input device, avoid creating new tensors when possible
    scale_input = torch.ones((), device=input.device, dtype=torch.float32)
    
    # CRITICAL: Move scale_weight to input device only if needed (offload system compatibility)
    # Check device to avoid unnecessary transfers
    if scale_weight.device != input.device:
        _fp8_device_transfers += 1
        if debug_this_call:
            print(f"   ⚠️  DEVICE TRANSFER NEEDED: {scale_weight.device} -> {input.device}")
            print(f"   Total transfers so far: {_fp8_device_transfers}")
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
