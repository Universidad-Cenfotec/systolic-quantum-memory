"""Readout mitigation restricted to the final measurement register."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class ReadoutMitigator:
    """Apply an explicitly supplied readout matrix to final-register counts.

    Counts may contain several classical registers. Only ``final_position`` is
    extracted; mid-circuit registers such as ``cr_bell`` are never corrected.
    """

    matrix: np.ndarray
    final_position: int = 0
    source: str = "explicit"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("readout matrix must be square")
        if matrix.shape[0] < 2:
            raise ValueError("readout matrix must contain at least one qubit")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("readout matrix must be finite")
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def from_assignment_errors(
        cls,
        prob_meas1_prep0: Sequence[float],
        prob_meas0_prep1: Sequence[float],
        final_position: int = 0,
        source: str = "properties",
    ) -> "ReadoutMitigator":
        """Build an independent-bit assignment matrix from backend data."""
        prep0 = np.asarray(prob_meas1_prep0, dtype=float)
        prep1 = np.asarray(prob_meas0_prep1, dtype=float)
        if prep0.shape != prep1.shape or prep0.ndim != 1:
            raise ValueError("assignment error arrays must have equal one-dimensional shape")
        if np.any((prep0 < 0) | (prep0 > 1) | (prep1 < 0) | (prep1 > 1)):
            raise ValueError("assignment errors must be in [0, 1]")

        dimension = 2 ** len(prep0)
        matrix = np.zeros((dimension, dimension), dtype=float)
        for prepared in range(dimension):
            prepared_bits = format(prepared, f"0{len(prep0)}b")
            for measured in range(dimension):
                measured_bits = format(measured, f"0{len(prep0)}b")
                probability = 1.0
                for true_bit, observed_bit, error0, error1 in zip(
                    prepared_bits, measured_bits, prep0, prep1
                ):
                    if true_bit == "0":
                        probability *= 1.0 - error0 if observed_bit == "0" else error0
                    else:
                        probability *= error1 if observed_bit == "0" else 1.0 - error1
                matrix[measured, prepared] = probability
        return cls(matrix, final_position=final_position, source=source)

    @classmethod
    def from_backend_properties(
        cls, backend: Any, qubits: Sequence[int], final_position: int = 0
    ) -> "ReadoutMitigator":
        """Build a matrix when backend properties expose readout parameters."""
        properties = backend.properties()
        if properties is None or not hasattr(properties, "qubits"):
            raise ValueError("backend does not expose qubit readout properties")

        prep0: list[float] = []
        prep1: list[float] = []
        for qubit in qubits:
            parameters = {parameter.name: parameter.value for parameter in properties.qubits[qubit]}
            try:
                prep0.append(float(parameters["prob_meas1_prep0"]))
                prep1.append(float(parameters["prob_meas0_prep1"]))
            except KeyError as error:
                raise ValueError(
                    f"readout parameter {error.args[0]!r} is unavailable for qubit {qubit}"
                ) from error
        return cls.from_assignment_errors(prep0, prep1, source="properties")

    @staticmethod
    def _final_bits(outcome: str, final_position: int) -> str:
        registers = outcome.strip().split()
        if not registers:
            raise ValueError("empty count outcome")
        try:
            return registers[final_position]
        except IndexError as error:
            raise ValueError("final measurement register is missing from outcome") from error

    def apply(self, raw_counts: Dict[str, int | float]) -> Dict[str, float]:
        """Return corrected counts for ``final_meas`` only.

        The returned dictionary contains final-register bitstrings and floating
        counts. Other classical registers are deliberately omitted rather than
        silently being corrected or reconstructed.
        """
        if not raw_counts:
            return {}
        dimension = self.matrix.shape[0]
        n_qubits = int(np.log2(dimension))
        if 2**n_qubits != dimension:
            raise ValueError("readout matrix dimension must be a power of two")

        observed = np.zeros(dimension, dtype=float)
        total = 0.0
        for outcome, count in raw_counts.items():
            bits = self._final_bits(outcome, self.final_position)
            if len(bits) != n_qubits or any(bit not in "01" for bit in bits):
                raise ValueError("final measurement outcome does not match matrix size")
            observed[int(bits, 2)] += float(count)
            total += float(count)
        if total <= 0:
            return {}

        probabilities = observed / total
        corrected = np.linalg.lstsq(self.matrix, probabilities, rcond=None)[0]
        corrected = np.clip(corrected, 0.0, None)
        normalization = corrected.sum()
        if normalization > 0:
            corrected /= normalization
        return {
            format(index, f"0{n_qubits}b"): float(probability * total)
            for index, probability in enumerate(corrected)
            if probability > 0
        }
