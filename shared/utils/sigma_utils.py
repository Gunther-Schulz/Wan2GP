"""Utilities for custom sigma schedules and easing curves."""

import math


def parse_custom_sigmas(raw: "str | list[float] | None") -> "list[float] | None":
    """Parse a comma-separated sigma string into a list of floats.

    Returns *None* if *raw* is empty or None.
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        return [float(v) for v in raw] if raw else None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts:
            return None
        return [float(p) for p in parts]
    return None


def apply_sigma_easing(
    sigmas: list[float],
    easing: str,
    strength: float = 1.0,
) -> list[float]:
    """Apply an easing curve to a sigma schedule.

    The easing remaps the *interpolation parameter* (linear position between
    first and last sigma) through a curve, then reconstructs sigma values.
    *strength* blends between the original schedule (0.0) and the fully eased
    schedule (1.0).

    Supported easing types: ``"linear"``, ``"cubic"``, ``"cubic_in_out"``.
    """
    if not sigmas or len(sigmas) < 2 or not easing:
        return sigmas

    n = len(sigmas)
    first, last = sigmas[0], sigmas[-1]
    span = first - last
    if abs(span) < 1e-12:
        return sigmas

    result = list(sigmas)
    for i in range(1, n - 1):
        t = i / (n - 1)  # linear position 0..1
        eased_t = _ease(t, easing)
        blended_t = t + strength * (eased_t - t)
        result[i] = first - blended_t * span

    return result


def _ease(t: float, easing: str) -> float:
    if easing == "linear":
        return t
    if easing == "cubic":
        return t * t * t
    if easing == "cubic_in_out":
        if t < 0.5:
            return 4.0 * t * t * t
        p = 2.0 * t - 2.0
        return 0.5 * p * p * p + 1.0
    return t
