# ============================================================
# SQM Research Project — Main Entry Point
# ============================================================

import sys
import os

# Configure matplotlib to use non-GUI backend before any other imports
import matplotlib
matplotlib.use('Agg')

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.comparison import run_full_comparison
from src.backends.aer_simulator_backend import AerSimulatorBackend

# ══════════════════════════════════════════════════════════════════════════════
# Configuration Parameters
# ══════════════════════════════════════════════════════════════════════════════

# Compiler Configuration
R = 1          # Number of memory registers
n = 1          # Qubits per register (quantum word width)
c_max = 10     # Gate cost threshold
t_max_ns = 1350*1 # Time threshold (nanoseconds)

# Simulation Configuration
shots = 4000         # Number of simulation shots
flow=1 # 0 = flow, 1 = memory
# Backend Thermal Parameters (Customize here for Opción 2)
backend_mode = "custom"  # "default" (auto-created) or "custom" (uses hardcoded thermal params)
t1_ns = 149149       # T1 relaxation time (ns) - default: 149.1 μs
t2_ns = 38194        # T2 dephasing time (ns) - default: 38.2 μs
idle_time_ns = 1000  # Idle period duration (ns) - default: 7.0 μs

# Quantum State Configuration
# initial_state = 0: |0> target (default)
# initial_state = 1: |1> target (X gate)
# initial_state = 2: |+> superposition (H gate, fidelity target = |0>)
# initial_state = 3: |-> superposition (X+H gates, fidelity target = |1>)
initial_state = 3  # 0 = |0>, 1 = |1>, 2 = |+> (H), 3 = |-> (XH)

# Test Workloads

workload1 = ["WRITE_0", "IDLE_10", "READ_0"]
workload2 = ["WRITE_0", "IDLE_20", "READ_0"]
workload3 = ["WRITE_0", "IDLE_40", "READ_0"]
workload4 = ["WRITE_0", "IDLE_80", "READ_0"]
workload5 = ["WRITE_0", "IDLE_100", "READ_0",]
workload6 = ["WRITE_0", "IDLE_150", "READ_0"]
workload7 = ["WRITE_0", "IDLE_200", "READ_0"]
workload8 = ["WRITE_0", "IDLE_300", "READ_0"]
workload9 = ["WRITE_0", "IDLE_400", "READ_0"]
workload10 = ["WRITE_0", "IDLE_500", "READ_0"]
workload11 = ["WRITE_0", "IDLE_600", "READ_0"]
workload12 = ["WRITE_0", "IDLE_700", "READ_0"]
workload13 = ["WRITE_0", "IDLE_800", "READ_0"]
workload14 = ["WRITE_0", "IDLE_900", "READ_0"]

# ══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════
def main():
    """
    Main entry point  - Execute comparative analysis with defined parameters.
    All configuration is in the Configuration Parameters section at the top of this file.
    """
    
    # ──────────────────────────────────────────────────────────
    # Backend Manager Setup (Dependency Injection)
    # ──────────────────────────────────────────────────────────
    # Backend is ALWAYS created in main.py (either custom or default)
    
    if backend_mode == "custom":
        # Opción 2: Create custom backend with hardcoded thermal parameters
        print("\n[Backend Configuration] Using CUSTOM backend (Opción 2)")
        
        backend_manager = AerSimulatorBackend(
            t1_ns=t1_ns,
            t2_ns=t2_ns,
            idle_time_ns=idle_time_ns
        )
        
        print(f"  T1 relaxation: {t1_ns} ns ({t1_ns/1000:.1f} μs)")
        print(f"  T2 dephasing: {t2_ns} ns ({t2_ns/1000:.1f} μs)")
        print(f"  Idle period: {idle_time_ns} ns ({idle_time_ns/1000:.1f} μs)")
    else:
        # Opción 1: Default backend (uses standard thermal parameters)
        backend_manager = AerSimulatorBackend()
        print("\n[Backend Configuration] Using DEFAULT backend (Opción 1)")
        print("  T1: 149149 ns (149.1 μs)")
        print("  T2: 38194 ns (38.2 μs)")
        print("  Idle period: 7000 ns (7.0 μs)")
    
    print("\n" + "=" * 70)
    print("SQM Research Project - Comparative Analysis")
    print("=" * 70)
    print(f"\n[Compiler Configuration]")
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
        (f"Workload 8 {len(workload8)} instructions", workload8),
        (f"Workload 9 {len(workload9)} instructions", workload9),
        (f"Workload 10 {len(workload10)} instructions", workload10),
        (f"Workload 11 {len(workload11)} instructions", workload11),
        (f"Workload 12 {len(workload12)} instructions", workload12),
        (f"Workload 13 {len(workload13)} instructions", workload13),
        (f"Workload 14 {len(workload14)} instructions", workload14)






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
        initial_state=initial_state,
        backend_manager=backend_manager,
        flow=flow
    )


if __name__ == "__main__":
    main()

