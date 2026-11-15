"""speech_sieve: modular speech dataset quality profiling, repetition detection, and alignment auditing."""

from .config import ConfigError, load_preset, validate_config
from .pipeline import audit_batch, audit_record, evaluate_gates

__all__ = [
    "__version__",
    "ConfigError",
    "load_preset",
    "validate_config",
    "audit_record",
    "audit_batch",
    "evaluate_gates",
]

__version__ = "0.1.0"
