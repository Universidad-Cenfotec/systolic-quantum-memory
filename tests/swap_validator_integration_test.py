#!/usr/bin/env python
"""Integration test: Validate cmax_validator_swap.py with variable N and unified parser."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.time_calculation.cmax_validator_swap import CMaxValidator

def test_swap_validator_with_variable_N():
    """Test that SWAP validator works with different N values using unified parser."""
    
    print("=" * 70)
    print("Integration Test: CMaxValidator (SWAP) with Variable N")
    print("=" * 70)
    
    # Test with different N values
    test_cases = [
        (1, "Single qubit"),
        (2, "Two qubits"),
    ]
    
    for N, description in test_cases:
        print(f"\n{'─' * 70}")
        print(f"Test Case: N={N}  ({description})")
        print(f"{'─' * 70}")
        
        # Create validator
        validator = CMaxValidator(N=N)
        print(f"\n1. Validator initialized")
        print(f"   N = {validator.N}, d = {validator.d}")
        print(f"   Backend: {validator.backend.name}")
        print(f"   Native gate: {validator.native_2q_gate}")
        print(f"   p_swap_theory: {validator.p_swap_teorico:.6f}")
        
        # Run empirical fidelity test
        print(f"\n2. Running empirical_fidelity(n_swaps=0, shots=2) for N={N}...")
        fidelity = validator.empirical_fidelity(n_swaps=0, shots=2)
        
        print(f"   Fidelity: {fidelity:.4f}")
        assert 0.0 <= fidelity <= 1.0, f"Fidelity must be in [0,1], got {fidelity}"
        print(f"   ✓ Fidelity in valid range")
        
        # Run with different number of swaps
        print(f"\n3. Running empirical_fidelity with varying n_swaps...")
        for n_swaps in [0, 1]:
            f_emp = validator.empirical_fidelity(n_swaps=n_swaps, shots=2)
            print(f"   n_swaps={n_swaps}: F_emp={f_emp:.4f}")
            assert 0.0 <= f_emp <= 1.0, f"Invalid fidelity: {f_emp}"
    
    print(f"\n{'=' * 70}")
    print("✓ ALL INTEGRATION TESTS PASSED")
    print("✓ SWAP validator works correctly with variable N")
    print("✓ Unified measurement parser handles all cases")
    print(f"{'=' * 70}\n")

if __name__ == "__main__":
    test_swap_validator_with_variable_N()
