# ============================================================
# IBM Backend Helper for Experiments
# Systolic Quantum Memory Research Project
# Role: Compatibility shim — delegates to src.backends.ibm_hardware_backend.
#       Do NOT duplicate logic here; add it to IBMHardwareBackend instead.
# ============================================================

from typing import Any, Dict

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

# Single source of truth: MockResult and MockJob live only in ibm_hardware_backend.
# Re-exported here so existing experiment imports keep working unchanged.
from src.backends.ibm_hardware_backend import MockResult, MockJob  # noqa: F401


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
    qc_transpiled: QuantumCircuit | list[QuantumCircuit],
    backend: Any,
    shots: int = 4000,
) -> Dict[str, int] | list[Dict[str, int]]:
    """
    Execute transpiled circuit(s) on real IBM hardware via SamplerV2
    and return counts in dict[str, int] format (same as AerSimulator).

    Delegates result parsing to MockResult from src.backends.ibm_hardware_backend
    — the single source of truth for SamplerV2 result extraction.

    Parameters
    ----------
    qc_transpiled : QuantumCircuit | list[QuantumCircuit]
        Circuit or list of circuits already transpiled for the target backend.
    backend : IBMBackend
        Real IBM Quantum backend instance.
    shots : int
        Number of shots.

    Returns
    -------
    Dict[str, int] | list[Dict[str, int]]
        Measurement counts dictionary, or list of dictionaries if multiple circuits.
    """
    is_single = isinstance(qc_transpiled, QuantumCircuit)
    circuits_to_run = [qc_transpiled] if is_single else qc_transpiled

    num_circuits = len(circuits_to_run)
    print(f"[IBMBackendHelper] Submitting {num_circuits} circuit(s) to {backend.name}...")

    # If submitting a single circuit, print its details
    if is_single:
        print(f"[IBMBackendHelper] Circuit: {qc_transpiled.num_qubits} qubits, "
              f"{qc_transpiled.num_clbits} clbits, depth={qc_transpiled.depth()}")

    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = shots

    job = sampler.run(circuits_to_run)
    job_id = job.job_id()
    print(f"[IBMBackendHelper] Job submitted: {job_id}")
    print(f"[IBMBackendHelper] Waiting for results...")

    result = job.result()
    print(f"[IBMBackendHelper] Job {job_id} completed successfully")

    counts_list = []
    for pub_res in result:
        mock_result = MockResult(pub_res)
        counts_list.append(mock_result.get_counts())

    if is_single:
        return counts_list[0]
    return counts_list
