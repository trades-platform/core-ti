# Auto-register all builtin indicators on package import
from . import moving_averages, oscillators, volatility, trend, scalars

__all__ = ["moving_averages", "oscillators", "volatility", "trend", "scalars"]
