import pytest

from src.mitigation.readout_mitigator import ReadoutMitigator


def test_rem_only_extracts_final_register():
    mitigator = ReadoutMitigator.from_assignment_errors([0.1], [0.1])
    corrected = mitigator.apply({"1 0": 80, "1 1": 20})
    assert sum(corrected.values()) == pytest.approx(100.0)
    assert set(corrected) <= {"0", "1"}


def test_invalid_assignment_error_is_rejected():
    with pytest.raises(ValueError):
        ReadoutMitigator.from_assignment_errors([1.2], [0.1])


def test_cr_bell_bits_never_enter_rem():
    """REM must only correct final_meas, never cr_bell.

    Bitstring format: ``"cr_bell final_meas"`` (split()[0]=cr_bell, [1]=final_meas).
    Using ``final_position=1`` targets the final measurement register.
    """
    mitigator = ReadoutMitigator.from_assignment_errors(
        [0.05], [0.05], final_position=1
    )
    raw_counts = {"00 0": 700, "01 0": 150, "10 1": 100, "11 1": 50}
    corrected = mitigator.apply(raw_counts)
    # Result should only contain final_meas bitstrings ("0" and "1")
    for key in corrected:
        assert " " not in key, f"cr_bell leaked into REM output: {key}"
    assert set(corrected) <= {"0", "1"}


def test_empty_counts_returns_empty():
    mitigator = ReadoutMitigator.from_assignment_errors([0.1], [0.1])
    assert mitigator.apply({}) == {}


def test_perfect_readout_no_correction():
    """With zero readout error, corrected ≈ raw probabilities."""
    mitigator = ReadoutMitigator.from_assignment_errors([0.0], [0.0])
    corrected = mitigator.apply({"0": 900, "1": 100})
    total = sum(corrected.values())
    assert corrected.get("0", 0) / total == pytest.approx(0.9, abs=0.01)


def test_two_qubit_final_register():
    """REM works correctly for 2-qubit final measurement."""
    mitigator = ReadoutMitigator.from_assignment_errors([0.05, 0.05], [0.05, 0.05])
    corrected = mitigator.apply({"00": 800, "01": 100, "10": 80, "11": 20})
    assert sum(corrected.values()) == pytest.approx(1000.0, abs=1.0)
    assert all(len(k) == 2 for k in corrected)


def test_source_metadata_preserved():
    mitigator = ReadoutMitigator.from_assignment_errors(
        [0.1], [0.1], source="calibration"
    )
    assert mitigator.source == "calibration"
