# ============================================================
# Backends Module - Quantum Backend Abstractions
# Systolic Quantum Memory Research Project
# ============================================================
"""
Backends module providing abstract interface and concrete implementations
for quantum backends (local simulation and real IBM Quantum hardware).
"""

from src.backends.backend_interface import BackendInterface
from src.backends.aer_simulator_backend import AerSimulatorBackend
from src.backends.ibm_hardware_backend import IBMHardwareBackend

__all__ = [
    "BackendInterface",
    "AerSimulatorBackend",
    "IBMHardwareBackend",
]
