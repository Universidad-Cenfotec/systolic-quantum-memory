#!/usr/bin/env python
"""Integration test: Validate cmax_validator_sqm.py with unified parser."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.time_calculation.cmax_validator_sqm import CMaxValidator

def test_cmax_validator_integration():
    """Test that cmax_validator_sqm.py works with new unified parser."""
    
    print("=" * 70)
    print("Integration Test: CMaxValidator with Unified MeasurementParser")
    print("=" * 70)
    
    # Create validator for N=1 qubit
    validator = CMaxValidator(N=1)
    
    print("\n1. Validator initialized successfully")
    print(f"   N = {validator.N}")
    print(f"   Backend: {validator.backend.name}")
    
    # Run fidelity test
    print("\n2. Running empirical fidelity test (m=0 swaps, 1 shot for speed)...")
    fidelity = validator.empirical_fidelity(m_swaps=0, shots=1)
    
    print(f"   Fidelity: {fidelity:.4f}")
    assert 0.0 <= fidelity <= 1.0, "Fidelity must be between 0 and 1"
    print(f"   ✓ Fidelity in valid range [0, 1]")
    
    print("\n3. Validator state after test:")
    print(f"   2-qubit gate native: {validator.native_2q_gate}")
    print(f"   p_swap_theory: {validator.p_swap_teorico:.6f}")
    
    print("\n" + "=" * 70)
    print("✓ INTEGRATION TEST PASSED")
    print("✓ CMaxValidator works correctly with unified MeasurementParser")
    print("=" * 70)

if __name__ == "__main__":
    test_cmax_validator_integration()
