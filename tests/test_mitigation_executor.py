import pytest

from qiskit import QuantumCircuit

from src.mitigation.mitigation_executor import MitigationExecutor
from src.mitigation.readout_mitigator import ReadoutMitigator


class Result:
    def get_counts(self):
        return {"0": 90, "1": 10}


class CountingBackend:
    def __init__(self):
        self.calls = []

    def run(self, circuit, shots, seed):
        self.calls.append((circuit, shots, seed))
        return Result()


def test_zne_and_rem_share_two_executions():
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    backend = CountingBackend()
    executor = MitigationExecutor(
        backend,
        readout_mitigator=ReadoutMitigator.from_assignment_errors([0.05], [0.05]),
    )

    result = executor.execute(
        circuit,
        target_state="0",
        shots=256,
        noise_factors=[1, 3],
        zne_enabled=True,
        rem_enabled=True,
    )

    assert len(backend.calls) == 2
    assert result["execution_count"] == 2
    assert result["zne_fidelity"] is not None
    assert result["zne_rem_fidelity"] is not None


def test_zne_returns_raw_and_bounded():
    """ZNE results include both raw (mathematical) and bounded values."""
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    backend = CountingBackend()
    executor = MitigationExecutor(backend)

    result = executor.execute(
        circuit,
        target_state="0",
        shots=256,
        noise_factors=[1, 3],
        zne_enabled=True,
    )

    assert result["zne_fidelity"] is not None
    assert result["zne_fidelity_raw"] is not None
    assert 0.0 <= result["zne_fidelity"] <= 1.0


def test_shot_limit_is_enforced():
    executor = MitigationExecutor(CountingBackend())
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)

    with pytest.raises(ValueError):
        executor.execute(circuit, target_state="0", shots=1025)


def test_mitigation_disabled_single_execution():
    """With ZNE disabled, only one execution at factor 1."""
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    backend = CountingBackend()
    executor = MitigationExecutor(backend)

    result = executor.execute(
        circuit,
        target_state="0",
        shots=256,
        zne_enabled=False,
        rem_enabled=False,
    )

    assert len(backend.calls) == 1
    assert result["execution_count"] == 1
    assert result["zne_fidelity"] is None
    assert result["zne_fidelity_raw"] is None
    assert result["raw_fidelity"] == pytest.approx(0.9)


def test_rem_only_without_zne():
    """REM can operate independently of ZNE."""
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    backend = CountingBackend()
    executor = MitigationExecutor(
        backend,
        readout_mitigator=ReadoutMitigator.from_assignment_errors([0.05], [0.05]),
    )

    result = executor.execute(
        circuit,
        target_state="0",
        shots=256,
        zne_enabled=False,
        rem_enabled=True,
    )

    assert len(backend.calls) == 1
    assert result["rem_fidelity"] is not None
    assert result["zne_fidelity"] is None


def test_rem_enabled_without_mitigator_raises():
    """Enabling REM without providing a ReadoutMitigator raises ValueError."""
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    executor = MitigationExecutor(CountingBackend())

    with pytest.raises(ValueError, match="no ReadoutMitigator"):
        executor.execute(
            circuit, target_state="0", shots=256, rem_enabled=True
        )
