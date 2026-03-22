"""Per-step schedule for denoising loop parameters (CFG, normalization, etc.)."""


class StepSchedule:
    """Returns a value for each denoising step.

    Accepts a single float (constant), a list of floats, or a comma-separated
    string.  When the step index exceeds the schedule length the last value is
    repeated.
    """

    def __init__(self, values: "float | list[float]"):
        if isinstance(values, (int, float)):
            self._values = [float(values)]
        elif isinstance(values, list):
            self._values = [float(v) for v in values]
        else:
            raise TypeError(f"Expected float or list, got {type(values)}")

    def at(self, step: int) -> float:
        if step < len(self._values):
            return self._values[step]
        return self._values[-1]

    def __len__(self) -> int:
        return len(self._values)

    @staticmethod
    def parse(raw: "str | float | list | None") -> "StepSchedule | None":
        """Parse from various input formats.  Returns *None* when disabled."""
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return StepSchedule(float(raw))
        if isinstance(raw, list):
            if not raw:
                return None
            return StepSchedule(raw)
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw:
                return None
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if not parts:
                return None
            return StepSchedule([float(p) for p in parts])
        return None
