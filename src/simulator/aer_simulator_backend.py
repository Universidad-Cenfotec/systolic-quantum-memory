# ============================================================
# AerSimulator Backend Manager
# Systolic Quantum Memory Research Project
# Role: Encapsulates all local simulation and noise configuration
# ============================================================

from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeKyiv
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error


class BackendInterface(ABC):
    """
    Abstract interface for quantum backends (local simulation or real hardware).
    Enables flexible backend switching without modifying compiler logic.
    
    Attributes
    ----------
    time_idle_ns : float
        Duration of one idle unit in nanoseconds (represents passive decoherence time)
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


class AerSimulatorBackend(BackendInterface):
    """
    Local AerSimulator backend manager with thermal relaxation noise model.
    
    Encapsulates all noise configuration, thermal relaxation, and simulator
    initialization. Provides a clean interface for circuit execution.
    
    Attributes
    ----------
    backend : FakeKyiv
        Base backend device (IBM Kyiv fake provider)
    noise_model : NoiseModel
        Thermal relaxation noise model for decoherence simulation
    simulator : AerSimulator
        Qiskit Aer simulator with MPS method
    time_idle_ns : float
        Duration of one idle unit in nanoseconds (7000 ns = 7 μs per identity gate)
    t1_ns : float
        T1 relaxation time in nanoseconds
    t2_ns : float
        T2 dephasing time in nanoseconds
    """

    # Thermal relaxation parameters (worst-case 10th percentile for FakeKyiv)
    DEFAULT_T1_NS = 149_149  # 149.149 μs
    DEFAULT_T2_NS = 38_194   # 38.194 μs
    DEFAULT_IDLE_TIME_NS = 7000  # 7 μs per identity gate unit

    def __init__(
        self,
        backend_device: Any = None,
        t1_ns: Optional[float] = None,
        t2_ns: Optional[float] = None,
        idle_time_ns: Optional[float] = None,
        method: str = "matrix_product_state",
    ):
        """
        Initialize AerSimulator backend with thermal relaxation noise.

        Parameters
        ----------
        backend_device : Any, optional
            Base backend device (default: FakeKyiv)
        t1_ns : float, optional
            T1 relaxation time in nanoseconds (default: 149_149 ns)
        t2_ns : float, optional
            T2 dephasing time in nanoseconds (default: 38_194 ns)
        idle_time_ns : float, optional
            Duration of idle period in nanoseconds (default: 7000 ns)
        method : str, optional
            Simulator method (default: "matrix_product_state")

        Notes
        -----
        The thermal relaxation error is configured on the 'id' (identity) gate,
        representing passive qubit decoherence during idle periods.
        """

        # Use provided backend or default to FakeKyiv
        self.backend = backend_device if backend_device is not None else FakeKyiv()

        # Use provided thermal parameters or defaults
        self.t1_ns = t1_ns if t1_ns is not None else self.DEFAULT_T1_NS
        self.t2_ns = t2_ns if t2_ns is not None else self.DEFAULT_T2_NS
        self._time_idle_ns = idle_time_ns if idle_time_ns is not None else self.DEFAULT_IDLE_TIME_NS

        # ----------------------------------------------------------
        # Initialize Noise Model with Thermal Relaxation
        # ----------------------------------------------------------

        # Start with backend's native noise model
        self.noise_model = NoiseModel.from_backend(self.backend)

        # Create thermal relaxation error for idle periods
        idle_error = thermal_relaxation_error(self.t1_ns, self.t2_ns, self._time_idle_ns)

        # Inject into 'id' gate on all physical qubits
        num_physical_qubits = self.backend.configuration().n_qubits
        for q in range(num_physical_qubits):
            self.noise_model.add_quantum_error(idle_error, 'id', [q], warnings=False)

        # Add backend's native gate errors (x, z, h, cx, measure, reset)
        backend_noise = NoiseModel.from_backend(self.backend)
        for gate in ['x', 'z', 'h', 'cx', 'measure', 'reset']:
            if gate in backend_noise._default_quantum_errors:
                self.noise_model.add_all_qubit_quantum_error(
                    backend_noise._default_quantum_errors[gate], gate
                )

        # Initialize AerSimulator with configured noise
        self.simulator = AerSimulator(
            noise_model=self.noise_model,
            method=method,
            seed_simulator=42,
        )

        print(f"[AerSimulatorBackend] Initialized with FakeKyiv device")
        print(f"[AerSimulatorBackend] Thermal relaxation: T1={self.t1_ns/1000:.1f}μs, T2={self.t2_ns/1000:.1f}μs")
        print(f"[AerSimulatorBackend] Idle period: {self._time_idle_ns} ns per unit")
        print(f"[AerSimulatorBackend] Simulator method: {method} with {num_physical_qubits} qubits")

    @property
    def time_idle_ns(self) -> float:
        """Return the idle period duration in nanoseconds."""
        return self._time_idle_ns

    def run(
        self,
        qc_transpiled: QuantumCircuit,
        shots: int = 1024,
        seed: int = 42,
    ) -> Any:
        """
        Execute a transpiled quantum circuit on AerSimulator.

        Parameters
        ----------
        qc_transpiled : QuantumCircuit
            Transpiled circuit ready for execution
        shots : int, optional
            Number of shots (default: 1024)
        seed : int, optional
            Random seed for reproducibility (default: 42)

        Returns
        -------
        Any
            Result object from AerSimulator job execution
        """

        print(f"[Execution] Sending circuit to AerSimulator backend...")
        print(f"[Execution] Circuit specs: {qc_transpiled.num_qubits} qubits, "
              f"{qc_transpiled.num_clbits} clbits, depth={qc_transpiled.depth()}")

        job = self.simulator.run(qc_transpiled, shots=shots, seed=seed)
        result = job.result()

        print(f"[Execution] Backend execution completed")

        return result

    def get_backend_info(self) -> Dict[str, Any]:
        """
        Retrieve backend configuration and noise model information.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing backend details
        """

        return {
            "backend_name": self.backend.__class__.__name__,
            "num_qubits": self.backend.configuration().n_qubits,
            "t1_ns": self.t1_ns,
            "t2_ns": self.t2_ns,
            "idle_time_ns": self.time_idle_ns,
            "noise_model_gates": list(self.noise_model._default_quantum_errors.keys()),
        }

    def get_backend_device(self) -> Any:
        """
        Get the backend device (e.g., FakeKyiv) for qubit mapping and topology.
        
        Returns
        -------
        Any
            Backend device object
        """
        return self.backend
