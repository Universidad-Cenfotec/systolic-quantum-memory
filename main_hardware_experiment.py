# ============================================================
# SQM Research Project — Main Entry Point (HARDWARE EXPERIMENT)
# 3-Scenario Experiment on IBM Quantum
# ============================================================

import sys
import os

# Configure matplotlib to use non-GUI backend before any other imports
import matplotlib
matplotlib.use('Agg')

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.comparison import run_real_comparison
from src.backends.ibm_hardware_backend import IBMHardwareBackend

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION: Multi-Workload Hardware Experiment
# ══════════════════════════════════════════════════════════════════════════════

# Compiler Configuration (Minimal for hardware - preserve quota)
R = 1          # Number of memory registers (single register = minimal)
n = 1          # Qubits per register (single qubit = fastest test)
c_max = 2      # Gate cost threshold (aggressive refresh)
t_max_ns = 20000000 # Time threshold (nanoseconds)

# Hardware Execution Configuration
shots = 1000         # CRITICAL: Keep low to preserve IBM quota (10 min/month)

# Quantum State Configuration
initial_state = 1   # 0 = |0⟩ target, 1 = |1⟩ target

# Test Workloads (Multiple workloads for comparison)
workload1 = ["READ_0","IDLE_100","WRITE_0","IDLE_100","READ_0"]
workload2 = ["READ_0","IDLE_150","WRITE_0","IDLE_150","READ_0"]
workload3 = ["READ_0","IDLE_200","WRITE_0","IDLE_200","READ_0"]

workload4 = ["READ_0","IDLE_80","WRITE_0","IDLE_160","READ_0"]
workload5 = ["READ_0","IDLE_100","WRITE_0","IDLE_200","READ_0"]
workload6 = ["READ_0","IDLE_150","WRITE_0","IDLE_300","READ_0"]


# Set to int (1, 2, 3), list ([1,3], etc.), or None (all scenarios)
# Examples:
#   scenario = [1 ]         -> Run only Scenario 1
#   scenario = [1, 3]     -> Run Scenarios 1 and 3 (generates comparison graph)
#   scenario = [1, 2, 3]  -> Run all 3 scenarios (generates comparison graph)
scenario = [1, 3]  # Run scenarios 1 and 3 for comparison

# ══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main entry point - Execute multi-workload experiment on real IBM Quantum hardware.
    
    Executes multiple workloads with specified scenarios (1, 2, or 3).
    Generates combined CSV and comparison graph across all workloads.
    """
    
    # Display header
    if isinstance(scenario, list):
        scenario_str = ", ".join(map(str, scenario))
        header = f"SQM RESEARCH: MULTI-WORKLOAD EXPERIMENT (Scenarios {scenario_str})"
    else:
        header = "SQM RESEARCH: MULTI-WORKLOAD EXPERIMENT (IBM QUANTUM HARDWARE)"
    
    print("╔" + "═" * 78 + "╗")
    print("║" + header.center(78) + "║")
    print("║" + "  Systolic Quantum Teleportation Memory".center(78) + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    # ──────────────────────────────────────────────────────────
    # PHASE 1: IBM QUANTUM BACKEND INITIALIZATION
    # ──────────────────────────────────────────────────────────
    
    print("PHASE 1: IBM QUANTUM BACKEND INITIALIZATION")
    print("=" * 80)
    
    try:
        print("\n[Connection] Initializing QiskitRuntimeService...")
        print("  → Loading saved IBM Quantum credentials from local cache")
        print("  → Scanning for available backends with dynamic circuit support")
        
        backend_manager = IBMHardwareBackend(
            backend_name='ibm_kingston',
            channel='ibm_quantum_platform'
        )
        
        backend_info = backend_manager.get_backend_info()
        
        print(f"\n✓ Backend Connected Successfully!")
        print(f"  Backend: {backend_info['backend_name']}")
        print(f"  Total Qubits: {backend_info['num_qubits']}")
        print(f"  Basis Gates: {backend_info['basis_gates']}")
        print(f"  Use Native Delay: {backend_info['use_native_delay']}")
        print(f"  Idle Timing (ns): {backend_info['idle_time_ns']}")
        
    except Exception as e:
        print(f"\n✗ Backend connection failed!")
        print(f"  Error: {str(e)}")
        print(f"\n  [TROUBLESHOOTING]")
        print(f"  1. Check IBM Quantum credentials:")
        print(f"     - Run: python -c \"from qiskit_ibm_runtime import QiskitRuntimeService\"")
        print(f"     - Run: QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')")
        print(f"  2. Ensure internet connection to IBM cloud")
        print(f"  3. Check account has active quota (minimum 10 min/month)")
        import traceback
        traceback.print_exc()
        return
    
    # ──────────────────────────────────────────────────────────
    # PHASE 2: MULTI-WORKLOAD CONFIGURATION
    # ──────────────────────────────────────────────────────────
    
    print("\n\nPHASE 2: MULTI-WORKLOAD CONFIGURATION")
    print("=" * 80)
    
    # Define workloads with descriptive names
    workloads = [
        (f"Workload 1 ({len(workload1)} instr)", workload1),
        (f"Workload 2 ({len(workload2)} instr)", workload2),
        (f"Workload 3 ({len(workload3)} instr)", workload3),
        (f"Workload 4 ({len(workload4)} instr)", workload4),
        (f"Workload 5 ({len(workload5)} instr)", workload5),
        (f"Workload 6 ({len(workload6)} instr)", workload6)

    ]
    
    print(f"\n[Workloads to Execute]")
    for idx, (name, wl) in enumerate(workloads, 1):
        print(f"  {idx}. {name}: {wl}")
    
    print(f"\n[Compiler Parameters]")
    print(f"  R={R}, n={n}, c_max={c_max}, t_max={t_max_ns} ns, shots={shots}")
    state_label = "|1⟩" if initial_state == 1 else "|0⟩"
    print(f"  Target state: {state_label}")
    
    print(f"\n✓ Configuration validated")
    print(f"  Qubits required: ~{n * (R + 2)} qubits")
    print(f"  Qubits available: {backend_info['num_qubits']}")
    print(f"  Total workloads: {len(workloads)}")
    
    if isinstance(scenario, list):
        scenario_str = ", ".join(map(str, scenario))
        print(f"  Scenario mode: MULTIPLE (Scenarios {scenario_str})")
    else:
        print(f"  Scenario mode: ALL (1, 2, and 3)")
    
    # ──────────────────────────────────────────────────────────
    # PHASE 3: EXECUTE ALL WORKLOADS
    # ──────────────────────────────────────────────────────────
    
    print(f"\n\nPHASE 3: EXECUTE MULTI-WORKLOAD EXPERIMENT")
    print("=" * 80)
    
    # Accumulate all workload results
    all_workload_results = []
    
    try:
        for workload_idx, (workload_name, workload_data) in enumerate(workloads, 1):
            print(f"\n\n{'─' * 80}")
            print(f"WORKLOAD {workload_idx}/{len(workloads)}: {workload_name}")
            print(f"{'─' * 80}")
            
            # Execute current workload with specified scenarios
            workload_results = run_real_comparison(
                R=R,
                n=n,
                c_max=c_max,
                t_max_ns=t_max_ns,
                shots=shots,
                workload=workload_data,
                backend_manager=backend_manager,
                initial_state=initial_state,
                scenarios=scenario  # Pass scenario filter
            )
            
            # Add workload name to results for tracking
            if workload_results:
                workload_results['workload_name'] = workload_name
                workload_results['workload_data'] = workload_data
                all_workload_results.append(workload_results)
                print(f"\n✓ Workload {workload_idx} completed and saved")
            else:
                print(f"\n✗ Workload {workload_idx} failed")
        
        # ──────────────────────────────────────────────────────────
        # PHASE 4: GENERATE COMBINED RESULTS
        # ──────────────────────────────────────────────────────────
        
        print(f"\n\nPHASE 4: GENERATE COMBINED RESULTS")
        print("=" * 80)
        
        if all_workload_results:
            # Call processor with all workloads
            from src.utils.hardware_results_processor import save_hardware_multi_workload_results
            save_hardware_multi_workload_results(
                all_workload_results=all_workload_results,
                backend_info=backend_info,
                params={
                    'R': R,
                    'n': n,
                    'c_max': c_max,
                    't_max_ns': t_max_ns,
                    'shots': shots,
                    'initial_state': initial_state
                }
            )
            
            print("\n✓ MULTI-WORKLOAD EXPERIMENT COMPLETE")
            print("  All results saved to data/ and results/")
        else:
            print("\n✗ No workload results available for saving")
        
    except Exception as e:
        print(f"\n✗ Multi-workload experiment failed!")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
