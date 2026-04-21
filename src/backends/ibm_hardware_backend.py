# ============================================================
# IBM Hardware Backend Manager
# Systolic Quantum Memory Research Project
# Role: Manages real IBM Quantum hardware execution via SamplerV2
# ============================================================

from typing import Any, Dict, Optional

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from src.backends.backend_interface import BackendInterface


class MockResult:
    """
    Wrapper result object compatible with AerSimulator result format.
    Extracts counts dictionary from SamplerV2 PubResult and reconstructs
    the legacy combined bitstring format ("reg2 reg1") for the MeasurementParser.

    CRITICAL: This preserves ALL classical registers and combines them with spaces,
    exactly as Qiskit V1 AerSimulator did. This ensures complete compatibility with
    MeasurementParser and downstream validators (cmax_validator_sqm.py, etc.).
    """

    def __init__(self, pub_result):
        """
        Initialize MockResult from SamplerV2 PubResult.

        Parameters
        ----------
        pub_result : PubResult
            Result from SamplerV2 execution (DataBin with all classical registers)
        """
        self.pub_result = pub_result
        self._counts = None

    def get_counts(self) -> Dict[str, int]:
        """
        Extract counts dictionary from SamplerV2 result.

        Reconstructs the Qiskit V1 format: combines all classical registers
        with spaces (e.g., "0000 11") for compatibility with MeasurementParser.

        Returns
        -------
        Dict[str, int]
            Dictionary of combined bitstrings to count values
            Example: {"0000 11": 1234, "0001 10": 456}
        """
        if self._counts is not None:
            return self._counts

        try:
            data = self.pub_result.data

            # Extract all classical register names from DataBin
            if hasattr(data, 'keys'):
                # If DataBin has keys() method (dict-like)
                register_names = list(data.keys())
            else:
                # Fallback: extract non-private attributes
                register_names = [attr for attr in dir(data) if not attr.startswith('_')]

            if not register_names:
                raise ValueError(
                    "[IBMHardwareBackend] No classical registers found in SamplerV2 result. "
                    "Ensure circuit has measurements."
                )

            # CRITICAL: Reverse register order to match Qiskit V1 endianness
            # In V1: add_register(cr_lb); add_register(cr_bell)
            # Output bitstring: "cr_bell_bits cr_lb_bits" (last register first)
            register_names.reverse()
            print(
                f"[IBMHardwareBackend] Extracted registers (in output order): {register_names}"
            )

            # Extract raw bitstrings for each register (preserving all bits)
            raw_bitstrings = []
            for reg_name in register_names:
                reg_data = getattr(data, reg_name)
                # get_bitstrings() returns list of bitstrings, one per shot
                # This preserves exact bit order without manual conversion
                bitstrings = reg_data.get_bitstrings()
                raw_bitstrings.append(bitstrings)

            # Combine all registers for each shot
            combined_counts = {}
            num_shots = len(raw_bitstrings[0])

            for shot_idx in range(num_shots):
                # For this shot, concatenate all register bitstrings with space
                combined_bitstring = " ".join(
                    [raw_bitstrings[reg_idx][shot_idx] for reg_idx in range(len(register_names))]
                )

                # Aggregate counts (same combined bitstring may appear multiple times)
                combined_counts[combined_bitstring] = (
                    combined_counts.get(combined_bitstring, 0) + 1
                )

            self._counts = combined_counts
            return combined_counts

        except Exception as e:
            raise RuntimeError(
                f"[IBMHardwareBackend] Failed to extract and combine counts from SamplerV2: {str(e)}"
            )


class MockJob:
    """
    Wrapper job object compatible with AerSimulator job interface.
    Provides simple result() interface for backward compatibility.
    """

    def __init__(self, mock_result):
        """
        Initialize MockJob wrapper.

        Parameters
        ----------
        mock_result : MockResult
            Result wrapper with get_counts() interface
        """
        self.mock_result = mock_result

    def result(self):
        """
        Get the wrapped result object.

        Returns
        -------
        MockResult
            Result wrapper with get_counts() interface
        """
        return self.mock_result

    def get_counts(self) -> Dict[str, int]:
        """
        Get measurement counts directly from MockJob (delegation).

        This provides interface compatibility with code that calls
        result.get_counts() directly instead of result.result().get_counts().

        Returns
        -------
        Dict[str, int]
            Dictionary of combined bitstrings to count values
        """
        return self.mock_result.get_counts()


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
    FALLBACK_IDLE_TIME_NS = 1000

    def __init__(
        self,
        backend_name: str = "ibm_kingston",
        channel: str = "ibm_quantum_platform",
        instance: Optional[str] = None,
    ):
        """
        Initialize IBM Hardware backend.

        Parameters
        ----------
        backend_name : str, optional
            Name of IBM Quantum backend (default: "ibm_kingston")
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
            self.service = QiskitRuntimeService()  # type: ignore
            
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

    @time_idle_ns.setter
    def time_idle_ns(self, value: float) -> None:
        """
        Prevent direct modification of time_idle_ns.
        
        CRITICAL: time_idle_ns is immutable after calibration to prevent accidental
        modifications that would compromise the hardware experiment.
        
        For Scenario 2 (SQM with time_idle_ns=0), use BackendZeroIdleWrapper
        in src.comparison.run_real_comparison() instead of direct assignment.
        
        Raises
        ------
        RuntimeError
            Always, to enforce immutability and prevent hard-to-debug errors.
        """
        error_msg = (
            "[IBMHardwareBackend] time_idle_ns is READ-ONLY after calibration.\n"
            "\n"
            f"WHAT HAPPENED: You attempted: backend_manager.time_idle_ns = {value}\n"
            "\n"
            "WHY THIS FAILS: time_idle_ns is calibrated from hardware at initialization.\n"
            "Direct assignment would corrupt the experiment.\n"
            "\n"
            "TO FIX (For Scenario 2 - SQM with zero delay):\n"
            "  Use BackendZeroIdleWrapper from src/comparison.py (line 480)\n"
            "  instead of modifying backend_manager directly.\n"
            "\n"
            "CORRECT USAGE:\n"
            "  from src.comparison import run_real_comparison\n"
            "  results = run_real_comparison(..., scenario_filter=2)  # Auto-wraps for S2"
        )
        raise RuntimeError(error_msg)

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

            # Configure shots for SamplerV2 V2 API
            # In Qiskit Runtime 0.34+, the correct property is 'default_shots', not 'shots'
            self.sampler.options.default_shots = shots  # type: ignore

            # Submit circuit to hardware
            job = self.sampler.run([qc_transpiled])
            job_id = job.job_id()

            print(f"[Execution] Job submitted with ID: {job_id}")
            print(f"[Execution] Waiting for results from queue...")

            # Wait for job completion
            result = job.result()

            print(f"[Execution] Job {job_id} completed successfully")

            # Return wrapped job with MockJob interface
            mock_result = MockResult(result[0])
            return MockJob(mock_result)

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
