# ============================================================
# Backend Interface - Abstract Interface
# Systolic Quantum Memory Research Project
# Role: Defines the contract for quantum backends
# ============================================================

from typing import Any, Dict
from abc import ABC, abstractmethod

from qiskit import QuantumCircuit


class BackendInterface(ABC):
    """
    Abstract interface for quantum backends (local simulation or real hardware).
    Enables flexible backend switching without modifying compiler logic.
    
    Attributes
    ----------
    time_idle_ns : float
        Duration of one idle unit in nanoseconds (represents passive decoherence time)
    use_native_delay : bool
        Flag indicating whether backend uses native delay() instruction (for hardware)
        or fallback id() gates (for simulation)
    """

    @property
    @abstractmethod
    def time_idle_ns(self) -> float:
        """
        Duration of one idle unit in nanoseconds.
        
        Returns
        -------
        float
            Idle period duration in ns
        """
        pass

    @property
    @abstractmethod
    def use_native_delay(self) -> bool:
        """
        Whether this backend uses native delay() instructions.
        
        Returns
        -------
        bool
            True for hardware backends using qc.delay(), False for simulators using qc.id()
        """
        pass

    @abstractmethod
    def run(self, qc_transpiled: QuantumCircuit, shots: int, seed: int = 42) -> Any:
        """
        Execute a transpiled quantum circuit on the backend.

        Parameters
        ----------
        qc_transpiled : QuantumCircuit
            Transpiled circuit ready for execution
        shots : int
            Number of circuit executions
        seed : int, optional
            Random seed for reproducibility (default: 42)

        Returns
        -------
        Any
            Job result object (AerSimulator result for now)
        """
        pass

    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        """
        Retrieve backend configuration information.

        Returns
        -------
        Dict[str, Any]
            Dictionary with backend details (name, num_qubits, noise model info)
        """
        pass

    @abstractmethod
    def get_backend_device(self) -> Any:
        """
        Get the backend device (e.g., FakeKyiv).
        
        Returns
        -------
        Any
            Backend device object for qubit mapping and topology info
        """
        pass
