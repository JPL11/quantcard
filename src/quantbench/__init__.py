"""quantbench: quantization report cards for PyTorch models headed to the edge."""

from .bench import BenchResult, load_entrypoint, run_config
from .configs import CONFIGS

__version__ = "0.1.0a1"
__all__ = ["BenchResult", "CONFIGS", "load_entrypoint", "run_config", "__version__"]
