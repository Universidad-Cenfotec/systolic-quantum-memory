# ============================================================
# SQTM Research Project — Main Entry Point
# ============================================================

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.comparison import run_full_comparison

# ══════════════════════════════════════════════════════════════════════════════
# Configuration Parameters
# ══════════════════════════════════════════════════════════════════════════════

# Compiler Configuration
R = 2           # Number of memory registers
n = 1          # Qubits per register (quantum word width)
c_max = 2       # Gate cost threshold
t_max_ns = 50000 # Time threshold (nanoseconds)

# Simulation Configuration
shots = 4000    # Number of simulation shots

# Quantum State Configuration
# Set initial_state = 0 for |0⟩ target, or 1 for |1⟩ target
initial_state = 1  # 0 = |0⟩ state, 1 = |1⟩ state

# Test Workloads
workload1 = [
    "READ_01",

]

workload2 = [
   "READ_01", "IDLE_1", 
 

]

workload3 = [
  "READ_01", "IDLE_1", "READ_01",
 
]

workload4 = [
  "READ_0", "IDLE_1", "READ_0",
   "READ_0", 
]

workload5 = [
     "READ_0", "IDLE_1", "READ_0",
   "READ_0", "IDLE_1"
]

workload6 = [
     "READ_0", "IDLE_1", "READ_0",
   "READ_0", "IDLE_1", "READ_0",

]

workload7 = [
     "READ_0", "IDLE_1", "READ_0",
   "READ_0", "IDLE_1", "READ_0",
    "READ_0", ]

workload8 = [
        "READ_0", "IDLE_1", "READ_0",
   "READ_0", "IDLE_1", "READ_0",
    "READ_0", "IDLE_1",

]

# ══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main entry point - Execute comparative analysis with defined parameters.
    """
    print("\n" + "=" * 70)
    print("SQTM Research Project - Comparative Analysis")
    print("=" * 70)
    print(f"\n[Configuration]")
    print(f"  Memory Registers (R): {R}")
    print(f"  Qubits per Register (n): {n}")
    print(f"  Shots: {shots}")
    print(f"  Gate Cost Threshold: {c_max}")
    print(f"  Time Threshold: {t_max_ns} ns")
    state_label = "|1⟩" if initial_state == 1 else "|0⟩"
    print(f"  Target Quantum State: {state_label}")

    # Run comparative analysis with all workloads
    workloads = [
        (f"Workload 1 {len(workload1)} instructions", workload1),
        (f"Workload 2 {len(workload2)} instructions", workload2),
        (f"Workload 3 {len(workload3)} instructions", workload3),
        (f"Workload 4 {len(workload4)} instructions", workload4),
        (f"Workload 5 {len(workload5)} instructions", workload5),
        (f"Workload 6 {len(workload6)} instructions", workload6),
        (f"Workload 7 {len(workload7)} instructions", workload7),
    ]

    print("\n[Workloads]")
    for name, _ in workloads:
        print(f"  • {name}")

    print("\n" + "=" * 70)
    print("Starting Comparative Analysis...")
    print("=" * 70)

    run_full_comparison(
        R=R,
        n=n,
        c_max=c_max,
        t_max_ns=t_max_ns,
        shots=shots,
        workloads=workloads,
        initial_state=initial_state
    )


if __name__ == "__main__":
    main()

