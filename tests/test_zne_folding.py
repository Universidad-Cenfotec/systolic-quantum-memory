import pytest

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from src.mitigation.zne_folding import ZNEFolder


def dynamic_circuit():
    qubits = QuantumRegister(2, "q")
    bell = ClassicalRegister(1, "cr_bell")
    final = ClassicalRegister(1, "final_meas")
    circuit = QuantumCircuit(qubits, bell, final)
    circuit.h(qubits[0])
    circuit.cx(qubits[0], qubits[1])
    circuit.measure(qubits[0], bell[0])
    circuit.reset(qubits[1])
    circuit.delay(10, qubits[0], unit="ns")
    circuit.measure(qubits[1], final[0])
    with circuit.if_test((bell[0], 1)):
        circuit.x(qubits[1])
    return circuit


def test_folding_preserves_dynamic_operations():
    circuit = dynamic_circuit()
    original = [instruction.operation.name for instruction in circuit.data]
    folded = ZNEFolder().fold_circuit(circuit, 3)
    names = [instruction.operation.name for instruction in folded.data]

    assert names.count("h") == 3
    assert names.count("cx") == 3
    for name in ("measure", "reset", "delay", "if_else"):
        assert names.count(name) == original.count(name)
    assert len(circuit.data) == len(original)


def test_even_noise_factor_is_rejected():
    with pytest.raises(ValueError):
        ZNEFolder().fold_circuit(dynamic_circuit(), 2)

def test_folding_immutability_and_factor_one():
    circuit = dynamic_circuit()
    original_data = list(circuit.data)
    
    folded = ZNEFolder().fold_circuit(circuit, 1)
    
    # Must be a new object, but data must be identical
    assert folded is not circuit
    assert len(folded.data) == len(original_data)
    for f_inst, o_inst in zip(folded.data, original_data):
        assert f_inst.operation.name == o_inst.operation.name

def test_folding_factor_five():
    circuit = dynamic_circuit()
    folded = ZNEFolder().fold_circuit(circuit, 5)
    names = [instruction.operation.name for instruction in folded.data]
    
    assert names.count("h") == 5
    assert names.count("cx") == 5

def test_folding_ignores_inner_blocks():
    circuit = dynamic_circuit()
    folded = ZNEFolder().fold_circuit(circuit, 3)
    
    # Find the if_else block in folded
    if_else_inst = next(inst for inst in folded.data if inst.operation.name == "if_else")
    
    # The true body of if_else should only have 1 "x" gate, not 3.
    true_body = if_else_inst.operation.blocks[0]
    names_in_block = [inst.operation.name for inst in true_body.data]
    assert names_in_block.count("x") == 1
