"""
FP8 Scaled Model Loader - FIXED VERSION
All performance fixes and compatibility patches applied.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

# Global counters for debugging
_fp8_forward_count = 0
_fp8_device_transfers = 0


def apply_fp8_optimization_to_model(model, base_dtype, model_filename, device, quantization=None):
    """
    Apply FP8 optimization with ALL fixes applied.
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
    
    if not hasattr(torch, '_scaled_mm'):
        print("❌ torch._scaled_mm not available - skipping fp8 optimization")
        return False
    
    if not quantization or "fp8" not in quantization:
        print(f"❌ No fp8 quantization detected: {quantization}")
        return False
    
    if "scaled" not in quantization:
        print(f"❌ FP8 quantization {quantization} detected but not scaled - no optimization needed")
        return False
    
    # CRITICAL FIX: Scale weights are in safetensors file, NOT model parameters!
    # Use safe_open to selectively load ONLY scale weights (not entire 14GB model!)
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
    scale_weights = {}
    try:
        with safe_open(model_file, framework="pt", device="cpu") as f:
            total_keys = len(f.keys())
            print(f"   File contains {total_keys} tensors total")
            
            for key in f.keys():
                if key.endswith(".scale_weight"):
                    tensor = f.get_tensor(key)
                    # Keep as float32 on CPU, let offload manage device placement
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


def apply_fp8_optimization_to_model_simple(model, base_dtype, scale_weights):
    """
    Apply FP8 optimization with ALL fixes.
    """
    import time
    
    start_time = time.time()
    optimized_count = 0
    skipped_count = 0
    
    print(f"\n🔧 Patching Linear layers...")
    
    def create_enhanced_forward(module, scale_weight, base_dtype, layer_name):
        original_forward = module.forward
        weight_dtype = module.weight.dtype
        is_fp8 = weight_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
        
        # CRITICAL: Cache GPU scale_weight after first transfer (mutable dict)
        scale_cache = {'weight': scale_weight, 'device': scale_weight.device}
        
        def enhanced_forward(input):
            # CRITICAL: Handle ALL input shapes for FP8 weights
            if is_fp8:
                # Use cached GPU version if available
                cached_weight = scale_cache['weight']
                
                if len(input.shape) == 3:
                    # Standard 3D case (attention layers)
                    return apply_fp8_matmul(
                        input, module.weight, module.bias, 
                        cached_weight, base_dtype, scale_cache
                    )
                elif len(input.shape) == 2:
                    # 2D case: reshape to 3D, apply FP8 matmul, reshape back
                    input_3d = input.unsqueeze(1)  # [B, D] -> [B, 1, D]
                    output_3d = apply_fp8_matmul(
                        input_3d, module.weight, module.bias,
                        cached_weight, base_dtype, scale_cache
                    )
                    return output_3d.squeeze(1)  # [B, 1, D] -> [B, D]
                else:
                    # Other shapes: flatten to 2D, process, reshape back
                    original_shape = input.shape
                    input_2d = input.reshape(-1, input.shape[-1])
                    input_3d = input_2d.unsqueeze(1)
                    output_3d = apply_fp8_matmul(
                        input_3d, module.weight, module.bias,
                        cached_weight, base_dtype, scale_cache
                    )
                    output_2d = output_3d.squeeze(1)
                    return output_2d.reshape(*original_shape[:-1], -1)
            
            # Non-FP8 fallback
            return original_forward(input)
        
        return enhanced_forward
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            scale_key = f"{name}.scale_weight"
            
            if scale_key in scale_weights:
                scale_weight = scale_weights[scale_key]
                
                # CRITICAL: Always patch if called after offload (which replaces forward methods)
                needs_patching = not hasattr(module, '_fp8_enhanced') or \
                                not hasattr(module.forward, '__name__') or \
                                'enhanced_forward' not in module.forward.__name__
                
                if needs_patching:
                    module.forward = create_enhanced_forward(module, scale_weight, base_dtype, name)
                    module._fp8_enhanced = True
                    # CRITICAL: Mark module so mmgp offload skips dtype assertions
                    module._lock_dtype = True
                    optimized_count += 1
                else:
                    skipped_count += 1
    
    elapsed = time.time() - start_time
    print(f"\n✅ Patched {optimized_count} layers ({skipped_count} already patched)")
    print(f"⏱️  Patching time: {elapsed:.3f}s")
    
    return optimized_count


def apply_fp8_matmul(input, weight, bias, scale_weight, base_dtype, scale_cache):
    """
    FP8 matmul with ALL fixes: device caching, fp8 input handling, debug.
    """
    global _fp8_forward_count, _fp8_device_transfers
    _fp8_forward_count += 1
    
    debug_this_call = (_fp8_forward_count % 100 == 1)
    
    if debug_this_call:
        print(f"\n🔍 FP8 Forward #{_fp8_forward_count} - Device Check:")
        print(f"   input: device={input.device}, shape={input.shape}, dtype={input.dtype}")
        print(f"   weight: device={weight.device}, dtype={weight.dtype}")
        print(f"   scale_weight: device={scale_weight.device}, dtype={scale_weight.dtype}")
    
    input_shape = input.shape
    scale_input = torch.ones((), device=input.device, dtype=torch.float32)
    
    # CRITICAL: Cache GPU scale_weight to avoid repeated transfers
    if scale_weight.device != input.device:
        _fp8_device_transfers += 1
        if debug_this_call:
            print(f"   ⚠️  DEVICE TRANSFER NEEDED: {scale_weight.device} -> {input.device}")
            print(f"   Total transfers so far: {_fp8_device_transfers}")
        scale_weight = scale_weight.to(input.device)
        # Update cache with GPU version
        scale_cache['weight'] = scale_weight
        scale_cache['device'] = input.device
    
    # CRITICAL: Check if input is already fp8 (skip clamp - NotImplementedError!)
    if input.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        if debug_this_call:
            print(f"   ℹ️  Input already fp8, skipping clamp/convert")
        input_fp8 = input.reshape(-1, input_shape[2]).contiguous()
    else:
        # Standard path: clamp then convert
        input_clamped = torch.clamp(input, min=-448, max=448, out=input)
        input_fp8 = input_clamped.reshape(-1, input_shape[2]).to(torch.float8_e4m3fn).contiguous()
    
    bias = bias.to(base_dtype) if bias is not None else None
    
    output = torch._scaled_mm(
        input_fp8, weight.t(),
        out_dtype=base_dtype,
        bias=bias,
        scale_a=scale_input,
        scale_b=scale_weight
    )
    
    return output.reshape((-1, input_shape[1], weight.shape[0]))


def get_fp8_quantization_from_json(json_quantization: str = None) -> str:
    """Get fp8 quantization from JSON config."""
    if json_quantization and json_quantization != "disabled":
        print(f"Using JSON config quantization: {json_quantization}")
        return json_quantization
    return "disabled"


print("FP8 scaled model loader initialized - FIXED VERSION")
print("- Selective scale weight loading (safetensors.safe_open)")
print("- All input shape support (2D, 3D, ND)")
print("- Device transfer caching (no repeated CPU→GPU)")
print("- FP8 input handling (skip clamp NotImplementedError)")
print("- mmgp.offload compatibility (_lock_dtype)")
print("- Re-patching support (after offload hooks)")