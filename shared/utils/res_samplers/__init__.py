"""
RES (Refined Exponential Solver) Samplers
Adapted from RES4LYF: https://github.com/ClownsharkBatwing/RES4LYF

These samplers use exponential integrators with φ-functions for better
quality when using few steps (2-8 steps), ideal for Lightning LoRAs.
"""

from .res_scheduler import get_res_2s_sigmas, get_res_3s_sigmas
from .res_integrator import RES2SSampler, RES3SSampler, RESExponentialIntegrator
from .res_wan_adapter import RESWanAdapter, create_res_adapter

__all__ = [
    'get_res_2s_sigmas',
    'get_res_3s_sigmas', 
    'RES2SSampler',
    'RES3SSampler',
    'RESExponentialIntegrator',
    'RESWanAdapter',
    'create_res_adapter'
]

