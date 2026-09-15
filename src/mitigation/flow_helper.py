"""Helper for executing mitigation flow in simulators."""

from typing import Any, Dict, List

from qiskit import QuantumCircuit
from src.mitigation.readout_mitigator import ReadoutMitigator
from src.mitigation.mitigation_executor import MitigationExecutor, MAX_HARDWARE_SHOTS
from src.mitigation.zne_extrapolator import ZNEExtrapolator
from src.mitigation.zne_folding import ZNEFolder


def run_mitigation_flow(
    compiler,
    qc_transpiled: QuantumCircuit,
    shots: int,
    target_state: str,
    qr_work: List[Any],
) -> Dict[str, Any]:
    """Execute the circuit with the configured mitigation strategies."""
    
    mitigation_config = compiler.mitigation_config
    
    # Global switch
    if not mitigation_config.get('enabled', True):
        return None
    
    # Architecture filter
    target_arch = mitigation_config.get('target_architecture', 'both').lower()
    compiler_name = compiler.__class__.__name__.lower()
    is_sqm = 'sqm' in compiler_name
    is_swap = 'swap' in compiler_name
    
    if (target_arch == 'sqm' and not is_sqm) or (target_arch == 'swap' and not is_swap):
        print(f"  [Mitigation] Skipped for {compiler.__class__.__name__} (target_architecture={target_arch})")
        return None
    
    zne_enabled = mitigation_config.get('zne', {}).get('enabled', False)
    rem_enabled = mitigation_config.get('rem', {}).get('enabled', False)
    
    if not zne_enabled and not rem_enabled:
        return None

    readout_mitigator = None
    if rem_enabled:
        # Target physical qubits used by operation register
        target_physical_qubits = []
        for i in range(compiler.n):
            target_physical_qubits.append(compiler.logical_to_physical_map[qr_work[i]])
        readout_mitigator = ReadoutMitigator.from_backend_properties(compiler.backend, target_physical_qubits)

    # Determine max_shots
    max_shots_cfg = mitigation_config.get('hardware', {}).get('max_shots_per_execution', MAX_HARDWARE_SHOTS)
    # Ensure simulator runs can use their configured shots without ValueError
    executor_max_shots = max(max_shots_cfg, shots)

    # Initialize executor
    executor = MitigationExecutor(
        backend_manager=compiler.backend_manager,
        folder=ZNEFolder(),
        extrapolator=ZNEExtrapolator(),
        readout_mitigator=readout_mitigator,
        max_shots=executor_max_shots
    )

    # ZNE Seed
    zne_seed = mitigation_config.get('zne', {}).get('zne_seed')
    if zne_seed is None:
        zne_seed = 42

    mitigation_results = executor.execute(
        circuit=qc_transpiled,
        target_state=target_state,
        shots=shots,
        noise_factors=mitigation_config.get('zne', {}).get('noise_factors', [1, 3]),
        zne_enabled=zne_enabled,
        rem_enabled=rem_enabled,
        seed=zne_seed,
        extrapolator=mitigation_config.get('zne', {}).get('extrapolator', 'linear'),
    )

    # Expand the mitigation results into the expected format
    fidelities = {}
    f_rem = None
    f_zne = None
    if zne_enabled:
        if rem_enabled:
            f_zne = mitigation_results['zne_rem_fidelity']
            f_zne_raw = mitigation_results['zne_rem_fidelity_raw']
            fidelities = mitigation_results['rem_fidelities']
        else:
            f_zne = mitigation_results['zne_fidelity']
            f_zne_raw = mitigation_results['zne_fidelity_raw']
            fidelities = mitigation_results['raw_fidelities']
        
        fidelity = f_zne
        counts = mitigation_results['raw_counts'][1]
        f_rem = mitigation_results.get('rem_fidelities', {}).get(1, fidelity)
        print(f"  Fidelity FLOW (Mitigated ZNE): {fidelity:.4f}")

    else: # Only REM
        f_rem = mitigation_results['rem_fidelities'][1]
        fidelity = f_rem
        counts = mitigation_results['rem_counts'][1]
        print(f"  Fidelity FLOW (Mitigated REM): {fidelity:.4f}")

    # Prepare the return dictionary
    ret = {
        "fidelity": fidelity,
        "counts": counts,
        "total_shots": shots,
        "f_rem": f_rem,
        "f_zne": f_zne,
        "f_raw": mitigation_results['raw_fidelities'].get(1, 0.0),
        "zne_per_factor": fidelities,
        "mitigation_results": mitigation_results  # Include all executor metadata
    }
    
    return ret
