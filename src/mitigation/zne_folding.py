"""Conservative gate folding for dynamic SQM circuits."""

from __future__ import annotations

from typing import List

from qiskit import QuantumCircuit


UNFOLDABLE_INSTRUCTIONS = {
    "measure",
    "reset",
    "barrier",
    "delay",
    "id",
    "if_else",
    "switch_case",
    "for_loop",
    "while_loop",
}

FOLDABLE_INSTRUCTIONS = {"cx", "ecr", "cz", "sx", "x", "rz", "h", "rzx"}


class ZNEFolder:
    """Fold only top-level physical unitary gates in a transpiled circuit."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    @staticmethod
    def get_foldable_gates(circuit: QuantumCircuit) -> List[str]:
        """Return names eligible for folding in the supplied circuit."""
        return [
            instruction.operation.name
            for instruction in circuit.data
            if ZNEFolder._is_foldable(instruction)
        ]

    @staticmethod
    def _is_foldable(instruction) -> bool:
        operation = instruction.operation
        return (
            operation.name in FOLDABLE_INSTRUCTIONS
            and operation.name not in UNFOLDABLE_INSTRUCTIONS
            and not instruction.clbits
            and getattr(instruction, "condition", None) is None
            and not getattr(operation, "blocks", None)
        )

    def fold_circuit(self, circuit: QuantumCircuit, noise_factor: int) -> QuantumCircuit:
        """Return an independent circuit with odd integer folding factor.

        A factor of 1 returns a structural copy. Factors 3 and 5 repeat an
        eligible operation as U U-dagger U ... while preserving all dynamic
        instructions exactly once and in their original order.
        """
        if noise_factor < 1 or noise_factor % 2 == 0:
            raise ValueError("noise_factor must be a positive odd integer")
        if noise_factor == 1:
            return circuit.copy()

        folded = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=circuit.name)
        folded.global_phase = circuit.global_phase
        if circuit.metadata:
            folded.metadata = circuit.metadata.copy()

        for instruction in circuit.data:
            operation = instruction.operation
            if not self._is_foldable(instruction):
                folded.append(operation, instruction.qubits, instruction.clbits)
                continue

            inverse = operation.inverse()
            for repetition in range(noise_factor):
                folded.append(
                    operation if repetition % 2 == 0 else inverse,
                    instruction.qubits,
                    instruction.clbits,
                )

        return folded
