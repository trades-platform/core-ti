# Auto-register all builtin indicators on package import
from . import moving_averages, oscillators, volatility, trend, scalars, ma_angle, ma_cross

__all__ = ["moving_averages", "oscillators", "volatility", "trend", "scalars", "ma_angle", "ma_cross"]
