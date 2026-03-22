"""Alternative diffusion steppers: Euler Ancestral and DPM++ SDE helpers.

EulerAncestralStep implements the DiffusionStepProtocol and can be used as a
drop-in replacement for EulerDiffusionStep.

DPM++ SDE requires two model evaluations per step and therefore cannot follow
the single-evaluation DiffusionStepProtocol.  Instead, the stepping logic for
DPM++ SDE is implemented as helper functions consumed by a dedicated denoising
loop (``dpmpp_sde_denoising_loop`` in helpers.py).
"""

import math

import torch


# ---------------------------------------------------------------------------
# Shared helpers (ported from k-diffusion)
# ---------------------------------------------------------------------------

def _get_ancestral_step(sigma_from: float, sigma_to: float, eta: float = 1.0):
    """Compute sigma_down and sigma_up for an ancestral sampling step."""
    if not eta or sigma_to == 0:
        return sigma_to, 0.0
    sigma_up = min(
        sigma_to,
        eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5,
    )
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def _to_d(x: torch.Tensor, sigma: float, denoised: torch.Tensor) -> torch.Tensor:
    """Convert a denoised sample to a derivative (dx/dsigma)."""
    return (x - denoised) / sigma


def _sigma_fn(t: float) -> float:
    return math.exp(-t)


def _t_fn(sigma: float) -> float:
    return -math.log(sigma)


# ---------------------------------------------------------------------------
# Euler Ancestral (single-evaluation, fits DiffusionStepProtocol)
# ---------------------------------------------------------------------------

class EulerAncestralStep:
    """Euler method with ancestral (stochastic) noise injection.

    Compatible with ``DiffusionStepProtocol``.
    """

    def __init__(self, eta: float = 1.0, s_noise: float = 1.0, generator: torch.Generator | None = None):
        self.eta = eta
        self.s_noise = s_noise
        self.generator = generator

    def step(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        sigma = float(sigmas[step_index])
        sigma_next = float(sigmas[step_index + 1])

        sigma_down, sigma_up = _get_ancestral_step(sigma, sigma_next, self.eta)

        # Euler step toward sigma_down
        d = _to_d(sample, sigma, denoised_sample)
        dt = sigma_down - sigma
        x = (sample.to(torch.float32) + d.to(torch.float32) * dt).to(sample.dtype)

        # Add ancestral noise
        if sigma_up > 0 and sigma_next > 0:
            noise = torch.randn(
                sample.shape,
                dtype=sample.dtype,
                device=sample.device,
                generator=self.generator,
            )
            x = x + noise * self.s_noise * sigma_up

        return x


# ---------------------------------------------------------------------------
# DPM++ SDE helpers (two-evaluation, used by dpmpp_sde_denoising_loop)
# ---------------------------------------------------------------------------

def dpmpp_sde_step_1(
    x: torch.Tensor,
    denoised: torch.Tensor,
    sigma: float,
    sigma_next: float,
    eta: float = 1.0,
    s_noise: float = 1.0,
    r: float = 0.5,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, float]:
    """First half of a DPM++ SDE step.

    Returns ``(x_mid, sigma_mid)`` — the intermediate sample and sigma.
    The caller must evaluate the model at ``sigma_mid`` to obtain
    ``denoised_mid``, then call :func:`dpmpp_sde_step_2` to complete the step.

    If ``sigma_next == 0`` (final step), returns the Euler result and sigma 0.
    """
    if sigma_next == 0:
        d = _to_d(x, sigma, denoised)
        dt = sigma_next - sigma
        return (x.to(torch.float32) + d.to(torch.float32) * dt).to(x.dtype), 0.0

    t = _t_fn(sigma)
    t_next = _t_fn(sigma_next)
    h = t_next - t
    s = t + h * r

    sigma_s = _sigma_fn(s)
    sd, su = _get_ancestral_step(sigma, sigma_s, eta)
    s_ = _t_fn(sd)

    # DPM-Solver++ first half: x_2 = (sigma(s_)/sigma(t)) * x - expm1(t - s_) * denoised
    x_mid = (_sigma_fn(s_) / sigma) * x.to(torch.float32) - math.expm1(t - s_) * denoised.to(torch.float32)
    x_mid = x_mid.to(x.dtype)

    if su > 0:
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        x_mid = x_mid + noise * s_noise * su

    return x_mid, float(sigma_s)


def dpmpp_sde_step_2(
    x: torch.Tensor,
    denoised: torch.Tensor,
    denoised_mid: torch.Tensor,
    sigma: float,
    sigma_next: float,
    eta: float = 1.0,
    s_noise: float = 1.0,
    r: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Second half of a DPM++ SDE step.

    *denoised* is the prediction from the first evaluation (at ``sigma``),
    *denoised_mid* is from the second evaluation (at the midpoint sigma).
    """
    if sigma_next == 0:
        d = _to_d(x, sigma, denoised)
        dt = sigma_next - sigma
        return (x.to(torch.float32) + d.to(torch.float32) * dt).to(x.dtype)

    t = _t_fn(sigma)
    t_next = _t_fn(sigma_next)
    fac = 1.0 / (2.0 * r)

    sd, su = _get_ancestral_step(sigma, sigma_next, eta)
    t_next_ = _t_fn(sd)

    # DPM-Solver++ second half: blend denoised predictions and step
    denoised_d = (1.0 - fac) * denoised.to(torch.float32) + fac * denoised_mid.to(torch.float32)
    x_next = (_sigma_fn(t_next_) / sigma) * x.to(torch.float32) - math.expm1(t - t_next_) * denoised_d
    x_next = x_next.to(x.dtype)

    if su > 0:
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        x_next = x_next + noise * s_noise * su

    return x_next
