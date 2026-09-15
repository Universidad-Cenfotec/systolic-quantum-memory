"""Execution orchestration for RAW, REM, ZNE and ZNE+REM."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from qiskit import QuantumCircuit

from .readout_mitigator import ReadoutMitigator
from .zne_extrapolator import ZNEExtrapolator
from .zne_folding import ZNEFolder


MAX_HARDWARE_SHOTS = 1024


class MitigationExecutor:
    """Run folded circuits once per factor and derive all scalar metrics."""

    def __init__(
        self,
        backend_manager: Any,
        folder: ZNEFolder | None = None,
        extrapolator: ZNEExtrapolator | None = None,
        readout_mitigator: ReadoutMitigator | None = None,
        max_shots: int = MAX_HARDWARE_SHOTS,
    ) -> None:
        self.backend_manager = backend_manager
        self.folder = folder or ZNEFolder()
        self.extrapolator = extrapolator or ZNEExtrapolator()
        self.readout_mitigator = readout_mitigator
        if max_shots < 1:
            raise ValueError("max_shots must be positive")
        self.max_shots = max_shots

    @staticmethod
    def _fidelity(counts: Dict[str, int | float], target_state: str) -> float:
        total = float(sum(counts.values()))
        if total <= 0:
            return 0.0
        success = 0.0
        for outcome, count in counts.items():
            final_bits = outcome.strip().split()[0]
            if final_bits == target_state:
                success += float(count)
        return success / total

    def execute(
        self,
        circuit: QuantumCircuit,
        target_state: str,
        shots: int,
        noise_factors: Sequence[int] = (1, 3),
        zne_enabled: bool = False,
        rem_enabled: bool = False,
        seed: int = 42,
        extrapolator: str = "linear",
        **metadata: Any,
    ) -> Dict[str, Any]:
        """Execute the requested factors and derive mitigation metrics.

        ``circuit`` must already be transpiled. The executor never changes it.
        REM is applied to final-register counts only through the configured
        ``ReadoutMitigator``.

        ZNE extrapolation now produces both ``raw`` (mathematical) and
        ``bounded`` ([0,1]-clipped) fidelity values.
        """
        if shots < 1 or shots > self.max_shots:
            raise ValueError(
                f"shots must be between 1 and {self.max_shots} per execution"
            )
        if not target_state or any(bit not in "01" for bit in target_state):
            raise ValueError("target_state must be a non-empty bitstring")
        if rem_enabled and self.readout_mitigator is None:
            raise ValueError("REM is enabled but no ReadoutMitigator was provided")

        factors = tuple(int(factor) for factor in (noise_factors if zne_enabled else (1,)))
        if not factors or any(factor < 1 or factor % 2 == 0 for factor in factors):
            raise ValueError("noise_factors must contain positive odd integers")
        if len(set(factors)) != len(factors):
            raise ValueError("noise_factors must be unique")
        if zne_enabled and 1 not in factors:
            raise ValueError("ZNE noise_factors must include factor 1")

        raw_fidelities: Dict[int, float] = {}
        rem_fidelities: Dict[int, float] = {}
        raw_counts: Dict[int, Dict[str, int]] = {}
        rem_counts: Dict[int, Dict[str, float]] = {}

        for factor in factors:
            folded = self.folder.fold_circuit(circuit, factor)
            run_result = self.backend_manager.run(folded, shots=shots, seed=seed + factor)
            counts = run_result.get_counts()
            raw_counts[factor] = counts
            raw_fidelities[factor] = self._fidelity(counts, target_state)

            if rem_enabled:
                corrected = self.readout_mitigator.apply(counts)
                rem_counts[factor] = corrected
                rem_fidelities[factor] = self._fidelity(corrected, target_state)

        result: Dict[str, Any] = {
            "raw_counts": raw_counts,
            "rem_counts": rem_counts,
            "raw_fidelities": raw_fidelities,
            "rem_fidelities": rem_fidelities,
            "raw_fidelity": raw_fidelities[1],
            "rem_fidelity": rem_fidelities.get(1),
            # ZNE results: bounded (legacy-compatible) and raw (mathematical)
            "zne_fidelity": None,
            "zne_fidelity_raw": None,
            "zne_rem_fidelity": None,
            "zne_rem_fidelity_raw": None,
            "noise_factors": list(factors),
            "shots": shots,
            "execution_count": len(factors),
            "total_circuit_shots": len(factors) * shots,
            "seed": seed,
        }
        result.update(metadata)
        if zne_enabled:
            # Asymptote for maximum entropy state of n qubits
            asymptote = 1.0 / (2 ** len(target_state))
            zne_result = self.extrapolator.extrapolate(
                factors, [raw_fidelities[f] for f in factors], extrapolator, asymptote=asymptote
            )
            result["zne_fidelity"] = zne_result.bounded
            result["zne_fidelity_raw"] = zne_result.raw
            if rem_enabled:
                zne_rem_result = self.extrapolator.extrapolate(
                    factors, [rem_fidelities[f] for f in factors], extrapolator, asymptote=asymptote
                )
                result["zne_rem_fidelity"] = zne_rem_result.bounded
                result["zne_rem_fidelity_raw"] = zne_rem_result.raw
        return result
