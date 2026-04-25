# ============================================================
# IBM Backend Helper for Time Calculation Validators
# Systolic Quantum Memory Research Project
# Role: Shared utilities for connecting to real IBM Quantum
#       hardware and running circuits via SamplerV2.
# ============================================================

from typing import Any, Dict, Optional

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


# =============================================================================
# Result wrappers (compatible with AerSimulator result format)
# =============================================================================

class MockResult:
    """
    Wrapper result object compatible with AerSimulator result format.
    Extracts counts dictionary from SamplerV2 PubResult and reconstructs
    the legacy combined bitstring format ("reg2 reg1") for downstream parsers.
    """

    def __init__(self, pub_result):
        self.pub_result = pub_result
        self._counts = None

    def get_counts(self) -> Dict[str, int]:
        """
        Extract counts dictionary from SamplerV2 result.

        Reconstructs the Qiskit V1 format: combines all classical registers
        with spaces (e.g., "0000 11") for compatibility with MeasurementParser.
        """
        if self._counts is not None:
            return self._counts

        try:
            data = self.pub_result.data

            # Extract all classical register names from DataBin
            if hasattr(data, 'keys'):
                register_names = list(data.keys())
            else:
                register_names = [attr for attr in dir(data) if not attr.startswith('_')]

            if not register_names:
                raise ValueError(
                    "[IBMBackendHelper] No classical registers found in SamplerV2 result."
                )

            # CRITICAL: Reverse register order to match Qiskit V1 endianness
            register_names.reverse()
            print(f"[IBMBackendHelper] Extracted registers (in output order): {register_names}")

            # Extract raw bitstrings for each register
            raw_bitstrings = []
            for reg_name in register_names:
                reg_data = getattr(data, reg_name)
                bitstrings = reg_data.get_bitstrings()
                raw_bitstrings.append(bitstrings)

            # Combine all registers for each shot
            combined_counts = {}
            num_shots = len(raw_bitstrings[0])

            for shot_idx in range(num_shots):
                combined_bitstring = " ".join(
                    [raw_bitstrings[reg_idx][shot_idx] for reg_idx in range(len(register_names))]
                )
                combined_counts[combined_bitstring] = (
                    combined_counts.get(combined_bitstring, 0) + 1
                )

            self._counts = combined_counts
            return combined_counts

        except Exception as e:
            raise RuntimeError(
                f"[IBMBackendHelper] Failed to extract counts from SamplerV2: {str(e)}"
            )


class MockJob:
    """
    Wrapper job object compatible with AerSimulator job interface.
    """

    def __init__(self, mock_result):
        self.mock_result = mock_result

    def result(self):
        return self.mock_result

    def get_counts(self) -> Dict[str, int]:
        return self.mock_result.get_counts()


# =============================================================================
# Connection and execution helpers
# =============================================================================

def get_ibm_backend(backend_name: str = "ibm_kingston"):
    """
    Connect to IBM Quantum and return the real backend object.

    Parameters
    ----------
    backend_name : str
        Name of the IBM Quantum backend (default: "ibm_kingston").

    Returns
    -------
    backend : IBMBackend
        Real IBM Quantum backend instance.
    """
    print(f"[IBMBackendHelper] Connecting to IBM Quantum...")
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    print(f"[IBMBackendHelper] Connected to: {backend.name} ({backend.num_qubits} qubits)")
    return backend


def run_on_ibm(
    qc_transpiled: QuantumCircuit,
    backend: Any,
    shots: int = 4000,
) -> Dict[str, int]:
    """
    Execute a transpiled circuit on real IBM hardware via SamplerV2
    and return counts in dict[str, int] format (same as AerSimulator).

    Parameters
    ----------
    qc_transpiled : QuantumCircuit
        Circuit already transpiled for the target backend.
    backend : IBMBackend
        Real IBM Quantum backend instance.
    shots : int
        Number of shots.

    Returns
    -------
    Dict[str, int]
        Measurement counts dictionary.
    """
    print(f"[IBMBackendHelper] Submitting circuit to {backend.name}...")
    print(f"[IBMBackendHelper] Circuit: {qc_transpiled.num_qubits} qubits, "
          f"{qc_transpiled.num_clbits} clbits, depth={qc_transpiled.depth()}")

    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = shots

    job = sampler.run([qc_transpiled])
    job_id = job.job_id()
    print(f"[IBMBackendHelper] Job submitted: {job_id}")
    print(f"[IBMBackendHelper] Waiting for results...")

    result = job.result()
    print(f"[IBMBackendHelper] Job {job_id} completed successfully")

    mock_result = MockResult(result[0])
    return mock_result.get_counts()
