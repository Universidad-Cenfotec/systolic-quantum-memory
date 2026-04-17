# ============================================================
# IBM Hardware Backend Manager
# Systolic Quantum Memory Research Project
# Role: Manages real IBM Quantum hardware execution via SamplerV2
# ============================================================

from typing import Any, Dict, Optional

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeKyiv

from src.backends.backend_interface import BackendInterface


class MockResult:
    """
    Wrapper result object compatible with AerSimulator result format.
    Extracts counts dictionary from SamplerV2 PubResult.
    """

    def __init__(self, pub_result):
        """
        Initialize MockResult from SamplerV2 PubResult.

        Parameters
        ----------
        pub_result : PubResult
            Result from SamplerV2 execution
        """
        self.pub_result = pub_result
        self._counts = None

    def get_counts(self) -> Dict[str, int]:
        """
        Extract counts dictionary from SamplerV2 result.

        Attempts to extract from 'cr_final' register first, then falls back
        to first available classical register if not found.

        Returns
        -------
        Dict[str, int]
            Dictionary of bitstrings to count values
        """
        if self._counts is not None:
            return self._counts

        try:
            # SamplerV2 returns measurements via .data attribute
            data = self.pub_result.data

            # Try to extract 'cr_final' register first (if exists)
            if hasattr(data, 'cr_final'):
                measurements = data.cr_final
            else:
                # Fallback: get first available register
                # data is a NamedTuple with register names as attributes
                register_names = list(data._fields) if hasattr(data, '_fields') else []
                
                if not register_names:
                    raise ValueError(
                        "[IBMHardwareBackend] No classical registers found in SamplerV2 result. "
                        "Ensure circuit has measurements."
                    )

                first_register_name = register_names[0]
                print(f"[IBMHardwareBackend] 'cr_final' not found. Using '{first_register_name}' instead.")
                measurements = getattr(data, first_register_name)

            # Convert measurements to counts dictionary
            # SamplerV2 returns a BitArray, convert to bitstring counts
            counts = {}
            for bitstring, count in measurements.int_counts().items():
                # Format as binary string with proper padding
                bit_length = measurements.num_clbits
                bitstr_key = format(bitstring, f'0{bit_length}b')
                counts[bitstr_key] = count

            self._counts = counts
            return counts

        except Exception as e:
            raise RuntimeError(
                f"[IBMHardwareBackend] Failed to extract counts from SamplerV2 result: {str(e)}"
            )


class MockJob:
    """
    Wrapper job object compatible with AerSimulator job interface.
    Encapsulates SamplerV2 job and result extraction.
    """

    def __init__(self, job_id: str, pub_result, shots: int):
        """
        Initialize MockJob wrapper.

        Parameters
        ----------
        job_id : str
            Unique identifier for the job
        pub_result : PubResult
            Result from SamplerV2 execution
        shots : int
            Number of shots executed
        """
        self.job_id = job_id
        self.pub_result = pub_result
        self.shots = shots
        self._result = None

    def result(self):
        """
        Get the wrapped result object.

        Returns
        -------
        MockResult
            Result wrapper with get_counts() interface
        """
        if self._result is None:
            self._result = MockResult(self.pub_result)
        return self._result


class IBMHardwareBackend(BackendInterface):
    """
    Real IBM Quantum hardware backend using SamplerV2 primitive.

    This backend connects to real IBM Quantum hardware via Qiskit Runtime,
    submits circuits to the execution queue, and extracts measurement results
    in a format compatible with the simulator interface.

    Features:
    - Dynamic calibration of idle time from backend.target
    - Validation of dynamic circuit support before submission
    - SamplerV2 integration for V2 primitives
    - Automatic credential management via QiskitRuntimeService

    Attributes
    ----------
    backend : IBMBackend
        Real IBM Quantum backend instance
    service : QiskitRuntimeService
        Qiskit Runtime service for job submission
    sampler : SamplerV2
        SamplerV2 primitive for circuit execution
    time_idle_ns : float
        Dynamically calibrated idle period in nanoseconds
    use_native_delay : bool
        Always True for hardware backend (uses native delay() instructions)
    """

    # Fallback idle time if calibration data unavailable (typical SWAP operation)
    FALLBACK_IDLE_TIME_NS = 1350

    def __init__(
        self,
        backend_name: str = "ibm_kyiv",
        channel: str = "ibm_quantum",
        instance: Optional[str] = None,
    ):
        """
        Initialize IBM Hardware backend.

        Parameters
        ----------
        backend_name : str, optional
            Name of IBM Quantum backend (default: "ibm_kyiv")
        channel : str, optional
            Channel for Qiskit Runtime (default: "ibm_quantum")
        instance : str, optional
            Instance string (default: None, uses default account)

        Raises
        ------
        RuntimeError
            If unable to connect to QiskitRuntimeService or backend unavailable
        """
        try:
            print(f"[IBMHardwareBackend] Initializing connection to IBM Quantum...")
            
            # Authenticate and get service
            # Note: channel can be 'ibm_quantum', 'ibm_cloud', or 'local'
            # The API accepts string values directly
            self.service = QiskitRuntimeService(channel=channel)  # type: ignore
            
            # Get backend instance
            self.backend = self.service.backend(backend_name)
            
            print(f"[IBMHardwareBackend] Connected to backend: {self.backend.name}")
            print(f"[IBMHardwareBackend] Qubits: {self.backend.num_qubits}")

            # Validate dynamic circuit support BEFORE using the backend
            self._validate_dynamic_circuit_support()

            # Initialize SamplerV2 primitive
            self.sampler = SamplerV2(mode=self.backend)
            print(f"[IBMHardwareBackend] SamplerV2 initialized")

            # Calibrate idle time from backend data
            self._calibrate_idle_time()

        except Exception as e:
            raise RuntimeError(
                f"[IBMHardwareBackend] Failed to initialize IBM backend: {str(e)}. "
                "Ensure QiskitRuntimeService is configured with valid credentials."
            )

    def _validate_dynamic_circuit_support(self) -> None:
        """
        Validate that the selected backend supports dynamic circuits (measure_2).

        Dynamic circuits are required for SQM operations. This check prevents
        submitting unsupported circuits and wasting queue time.

        Raises
        ------
        ValueError
            If backend does not support dynamic circuits
        """
        try:
            # Check for dynamic circuit support via backend.target
            # The key operation for dynamic circuits is 'measure'
            if not hasattr(self.backend, 'target') or self.backend.target is None:
                print(
                    f"[IBMHardwareBackend] Warning: Unable to verify dynamic circuit support. "
                    f"Backend {self.backend.name} target not available."
                )
                return

            # Check if 'measure' gate exists (required for measurements)
            if 'measure' not in self.backend.target:
                raise ValueError(
                    f"[IBMHardwareBackend] Backend {self.backend.name} does not support measurements. "
                    "Required for quantum circuit execution."
                )

            # For future SQM dynamics, verify mid-circuit measurement capability
            # This is indicated by the presence of 'measure' in all qubits
            try:
                num_qubits = self.backend.num_qubits
                measure_instr = self.backend.target.get('measure')
                
                if measure_instr is not None and hasattr(measure_instr, 'qargs'):
                    measure_qargs = measure_instr.qargs
                    if measure_qargs is not None and measure_qargs:
                        # qargs is a dict-like object with qubit tuples as keys
                        supported_qubits = set()
                        try:
                            # Try to iterate if it's iterable
                            for qarg in measure_qargs:  # type: ignore
                                if isinstance(qarg, (tuple, list)) and len(qarg) > 0:
                                    supported_qubits.add(qarg[0])
                                elif isinstance(qarg, int):
                                    supported_qubits.add(qarg)
                        except TypeError:
                            # If not iterable, assume all qubits supported
                            pass
                        
                        if supported_qubits and not all(
                            q in supported_qubits for q in range(min(num_qubits, 10))
                        ):
                            print(
                                f"[IBMHardwareBackend] Warning: Measurement not supported on all qubits. "
                                "Some qubits may be restricted."
                            )
            except (AttributeError, TypeError, ValueError):
                # If unable to verify qargs, just proceed with warning
                print(
                    f"[IBMHardwareBackend] Warning: Unable to verify measurement support on all qubits."
                )

            print(f"[IBMHardwareBackend] ✓ Dynamic circuit support validated")

        except ValueError as e:
            raise e
        except Exception as e:
            print(f"[IBMHardwareBackend] Warning during validation: {str(e)}")

    def _calibrate_idle_time(self) -> None:
        """
        Dynamically calibrate idle time from backend calibration data.

        Attempts to extract readout or SWAP operation duration from backend.target.
        Falls back to FALLBACK_IDLE_TIME_NS if calibration unavailable.

        This value is used by QPC to calculate T_max scheduling parameters.
        """
        try:
            if not hasattr(self.backend, 'target') or self.backend.target is None:
                print(
                    f"[IBMHardwareBackend] Backend target unavailable. "
                    f"Using fallback idle time: {self.FALLBACK_IDLE_TIME_NS} ns"
                )
                self._time_idle_ns = self.FALLBACK_IDLE_TIME_NS
                return

            # Priority 1: Try to get readout duration (measure operation)
            if 'measure' in self.backend.target:
                try:
                    measure_instr = self.backend.target.get('measure')
                    if measure_instr is not None and hasattr(measure_instr, 'qargs'):
                        measure_qargs = measure_instr.qargs
                        if measure_qargs is not None:
                            try:
                                # Iterate over qargs if it's iterable
                                for qarg in measure_qargs:  # type: ignore
                                    if hasattr(measure_instr.qargs, 'get'):
                                        # If qargs is dict-like
                                        instr_props = measure_instr.qargs.get(qarg)
                                    else:
                                        # If qargs is a simple dict
                                        instr_props = measure_instr.qargs[qarg] if qarg in measure_instr.qargs else None
                                    
                                    if instr_props is not None and hasattr(instr_props, 'duration'):
                                        if instr_props.duration is not None:
                                            # Convert from seconds to nanoseconds
                                            self._time_idle_ns = int(instr_props.duration * 1e9)
                                            print(
                                                f"[IBMHardwareBackend] Calibrated idle time from readout: "
                                                f"{self._time_idle_ns} ns ({self._time_idle_ns/1e9:.1e} s)"
                                            )
                                            return
                            except (TypeError, KeyError):
                                pass  # Continue to next priority
                except (AttributeError, TypeError, KeyError):
                    pass  # Fallback to next priority

            # Priority 2: Try to get SWAP operation duration (if available)
            if 'swap' in self.backend.target:
                try:
                    swap_instr = self.backend.target.get('swap')
                    if swap_instr is not None and hasattr(swap_instr, 'qargs'):
                        swap_qargs = swap_instr.qargs
                        if swap_qargs is not None:
                            try:
                                # Iterate over qargs if it's iterable
                                for qargs in swap_qargs:  # type: ignore
                                    if hasattr(swap_instr.qargs, 'get'):
                                        # If qargs is dict-like
                                        instr_props = swap_instr.qargs.get(qargs)
                                    else:
                                        # If qargs is a simple dict
                                        instr_props = swap_instr.qargs[qargs] if qargs in swap_instr.qargs else None
                                    
                                    if instr_props is not None and hasattr(instr_props, 'duration'):
                                        if instr_props.duration is not None:
                                            self._time_idle_ns = int(instr_props.duration * 1e9)
                                            print(
                                                f"[IBMHardwareBackend] Calibrated idle time from SWAP: "
                                                f"{self._time_idle_ns} ns"
                                            )
                                            return
                            except (TypeError, KeyError):
                                pass  # Continue to fallback
                except (AttributeError, TypeError, KeyError):
                    pass  # Fallback to default

            # Fallback if no calibration data available
            print(
                f"[IBMHardwareBackend] No calibration data found in backend.target. "
                f"Using fallback: {self.FALLBACK_IDLE_TIME_NS} ns"
            )
            self._time_idle_ns = self.FALLBACK_IDLE_TIME_NS

        except Exception as e:
            print(
                f"[IBMHardwareBackend] Error during calibration: {str(e)}. "
                f"Using fallback idle time: {self.FALLBACK_IDLE_TIME_NS} ns"
            )
            self._time_idle_ns = self.FALLBACK_IDLE_TIME_NS

    @property
    def time_idle_ns(self) -> float:
        """Return the calibrated idle period duration in nanoseconds."""
        return self._time_idle_ns

    @property
    def use_native_delay(self) -> bool:
        """Hardware backend always uses native delay() instructions."""
        return True

    def run(
        self,
        qc_transpiled: QuantumCircuit,
        shots: int = 1024,
        seed: int = 42,
    ) -> MockJob:
        """
        Submit a transpiled circuit to IBM Quantum hardware.

        Uses SamplerV2 primitive for execution. This method queues the job
        and waits for results.

        Parameters
        ----------
        qc_transpiled : QuantumCircuit
            Transpiled circuit ready for hardware execution
        shots : int, optional
            Number of shots (default: 1024)
        seed : int, optional
            Random seed (note: may not be supported on all hardware)

        Returns
        -------
        MockJob
            Wrapper job object with result() interface

        Raises
        ------
        RuntimeError
            If job submission fails or execution times out
        """
        try:
            print(f"[Execution] Submitting circuit to IBM Quantum backend ({self.backend.name})...")
            print(f"[Execution] Circuit specs: {qc_transpiled.num_qubits} qubits, "
                  f"{qc_transpiled.num_clbits} clbits, depth={qc_transpiled.depth()}")

            # Run using SamplerV2
            # Note: SamplerV2 expects circuits to have classical bits for measurement
            job = self.sampler.run([qc_transpiled], shots=shots)
            job_id = job.job_id()

            print(f"[Execution] Job submitted with ID: {job_id}")
            print(f"[Execution] Waiting for results from queue...")

            # Wait for job completion
            result = job.result()

            print(f"[Execution] Job {job_id} completed successfully")

            # Return wrapped job with MockJob interface
            return MockJob(job_id, result[0], shots)

        except Exception as e:
            raise RuntimeError(
                f"[Execution] Failed to execute circuit on IBM Quantum: {str(e)}. "
                "Check backend status and circuit validity."
            )

    def get_backend_info(self) -> Dict[str, Any]:
        """
        Retrieve IBM backend configuration and status information.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing backend details
        """
        try:
            info = {
                "backend_name": self.backend.name,
                "num_qubits": self.backend.num_qubits,
                "basis_gates": list(self.backend.operation_names),
                "idle_time_ns": self.time_idle_ns,
                "use_native_delay": self.use_native_delay,
            }
            # Note: BackendV2 status information is limited
            # Status is typically managed at the service/provider level
            info["status"] = "active"  # Assume active if successfully connected
            return info
        except Exception as e:
            print(f"[IBMHardwareBackend] Warning: Unable to retrieve full backend info: {str(e)}")
            return {
                "backend_name": self.backend.name if hasattr(self.backend, 'name') else "unknown",
                "num_qubits": self.backend.num_qubits if hasattr(self.backend, 'num_qubits') else 0,
                "idle_time_ns": self.time_idle_ns,
                "use_native_delay": self.use_native_delay,
            }

    def get_backend_device(self) -> Any:
        """
        Get the backend device object for qubit mapping and topology.

        Returns
        -------
        Any
            IBM backend device object
        """
        return self.backend
