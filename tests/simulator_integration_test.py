#!/usr/bin/env python
"""Integration test: Validate simulators use unified parser correctly."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulator.sqm_simulator import SQMCompiler

def test_simulator_integration():
    """Test that simulators work with unified measurement parser."""
    
    print("=" * 70)
    print("Integration Test: SQMCompiler with Unified MeasurementParser")
    print("=" * 70)
    
    # Create compiler
    compiler = SQMCompiler(R=2, n=1, c_max=5, t_max_ns=10000.0)
    
    print("\n1. Compiler initialized successfully")
    print(f"   R={compiler.R}, n={compiler.n}")
    print(f"   c_max={compiler.c_max}, t_max={compiler.t_max_ns} ns")
    
    # Compile a simple workload
    print("\n2. Compiling workload: ['READ_0', 'READ_1']")
    workload = ["READ_0", "READ_1"]
    circuit = compiler.compile_workload(workload)
    
    print(f"   Circuit generated with {circuit.num_qubits} qubits")
    print(f"   ✓ Workload compiled successfully")
    
    # Test parser method
    print("\n3. Testing _parse_measurement_outcome() method:")
    test_cases = [
        ("1010", "1010"),
        ("1010 0011", "1010"),
        ("1010 0011 11", "1010"),
        ("0000  1111", "0000"),
    ]
    
    for outcome, expected in test_cases:
        result = compiler._parse_measurement_outcome(outcome)
        assert result == expected, f"Expected {expected}, got {result}"
        print(f"   ✓ '{outcome}' -> '{result}'")
    
    print("\n4. Check compiler state uses QPC:")
    state = compiler.get_compiler_state()
    assert "qpc_costs" in state, "State should have qpc_costs"
    assert "qpc_idle_times" in state, "State should have qpc_idle_times"
    assert "current_c" not in state, "Should NOT have redundant current_c"
    assert "current_t" not in state, "Should NOT have redundant current_t"
    print(f"   ✓ State correctly uses QPC data")
    print(f"   ✓ No redundant current_c or current_t")
    
    print("\n" + "=" * 70)
    print("✓ INTEGRATION TEST PASSED")
    print("✓ Simulators work correctly with unified MeasurementParser")
    print("=" * 70)

if __name__ == "__main__":
    test_simulator_integration()
