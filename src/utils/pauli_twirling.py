"""Pauli twirling helpers for Clifford circuits."""

import random
from typing import Any, Optional

from qiskit import QuantumCircuit


class PauliTwirler:
    """Apply logically neutral Pauli frames around common Clifford gates."""

    _PAULIS = ("I", "X", "Y", "Z")

    def __init__(self, enabled: bool = False, seed: Optional[int] = None) -> None:
        self.enabled = enabled
        self._rng = random.Random(seed)

    @staticmethod
    def _apply_pauli(qc: QuantumCircuit, qubit: Any, pauli: str) -> None:
        if pauli == "X":
            qc.x(qubit)
        elif pauli == "Y":
            qc.y(qubit)
        elif pauli == "Z":
            qc.z(qubit)

    @staticmethod
    def _h_image(pauli: str) -> str:
        return {"I": "I", "X": "Z", "Y": "Y", "Z": "X"}[pauli]

    @staticmethod
    def _cx_image(position: str, pauli: str) -> tuple[str, str]:
        return {
            ("control", "I"): ("I", "I"),
            ("control", "X"): ("X", "X"),
            ("control", "Y"): ("Y", "X"),
            ("control", "Z"): ("Z", "I"),
            ("target", "I"): ("I", "I"),
            ("target", "X"): ("I", "X"),
            ("target", "Y"): ("Z", "Y"),
            ("target", "Z"): ("Z", "Z"),
        }[(position, pauli)]

    def h(self, qc: QuantumCircuit, qubit: Any) -> None:
        """Append H with a Pauli frame when enabled."""
        if not self.enabled:
            qc.h(qubit)
            return
        pauli = self._rng.choice(self._PAULIS)
        self._apply_pauli(qc, qubit, pauli)
        qc.h(qubit)
        self._apply_pauli(qc, qubit, self._h_image(pauli))

    def cx(self, qc: QuantumCircuit, control: Any, target: Any) -> None:
        """Append CX with a two-qubit Pauli frame when enabled."""
        if not self.enabled:
            qc.cx(control, target)
            return
        control_pauli = self._rng.choice(self._PAULIS)
        target_pauli = self._rng.choice(self._PAULIS)
        self._apply_pauli(qc, control, control_pauli)
        self._apply_pauli(qc, target, target_pauli)
        qc.cx(control, target)
        control_image = self._cx_image("control", control_pauli)
        target_image = self._cx_image("target", target_pauli)
        self._apply_pauli(qc, control, control_image[0])
        self._apply_pauli(qc, target, control_image[1])
        self._apply_pauli(qc, control, target_image[0])
        self._apply_pauli(qc, target, target_image[1])

    def identity(self, qc: QuantumCircuit, qubit: Any) -> None:
        """Append an identity with a self-inverse Pauli frame when enabled."""
        if not self.enabled:
            qc.id(qubit)
            return
        pauli = self._rng.choice(self._PAULIS)
        self._apply_pauli(qc, qubit, pauli)
        qc.id(qubit)
        self._apply_pauli(qc, qubit, pauli)
