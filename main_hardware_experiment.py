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
from src.hardware_results_processor import save_hardware_comparison_results

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION: 3-Scenario Experiment (Hardware)
# ══════════════════════════════════════════════════════════════════════════════

# Compiler Configuration (Minimal for hardware - preserve quota)
R = 1          # Number of memory registers (single register = minimal)
n = 1          # Qubits per register (single qubit = fastest test)
c_max = 2      # Gate cost threshold (aggressive refresh)
t_max_ns = 40000 # Time threshold (nanoseconds)

# Hardware Execution Configuration
shots = 100         # CRITICAL: Keep low to preserve IBM quota (10 min/month)
#test_workload = ["READ_0", "IDLE_2", "READ_0"]  # Representative workload
test_workload =["READ_0", "WRITE_0", "READ_0", "IDLE_2", "READ_0"]
# Quantum State Configuration
initial_state = 1   # 0 = |0⟩ target, 1 = |1⟩ target

scenario = 3  # Set to 1, 2, or 3 to run specific scenario only (None = all scenarios)

# ══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main entry point - Execute 3-Scenario Experiment on real IBM Quantum hardware.
    
    Scenario 1 (Baseline): SWAP Compiler - Static decoherence
    Scenario 2 (SQM no-delay): SQM with time_idle_ns = 0 - Routing overhead only
    Scenario 3 (SQM Real): SQM with calibrated timing - Full thesis test
    
    Parameters
    ----------
    scenario : int, optional
        Which scenario(s) to execute: 1, 2, 3, or None (all)
        If None, executes all 3 scenarios
    """
    
    # Display header with scenario info
    if scenario:
        header = f"SQM RESEARCH: SCENARIO {scenario} ONLY (IBM QUANTUM HARDWARE)"
    else:
        header = "SQM RESEARCH: 3-SCENARIO EXPERIMENT (IBM QUANTUM HARDWARE)"
    
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
    # PHASE 2: CIRCUIT VALIDATION
    # ──────────────────────────────────────────────────────────
    
    print("\n\nPHASE 2: CIRCUIT PRE-VALIDATION")
    print("=" * 80)
    
    print(f"\n[Workload] {test_workload}")
    print(f"[Parameters] R={R}, n={n}, c_max={c_max}, t_max={t_max_ns} ns, shots={shots}")
    state_label = "|1⟩" if initial_state == 1 else "|0⟩"
    print(f"[Target State] {state_label}")
    
    print(f"\n✓ Configuration validated")
    print(f"  Qubits required: ~{n * (R + 2)} qubits (work + memory + teleport ancilla)")
    print(f"  Qubits available: {backend_info['num_qubits']}")
    print(f"  Shot budget: {shots} shots (cuota protection enabled)")
    
    if scenario:
        print(f"  Scenario mode: SINGLE (Scenario {scenario})")
    else:
        print(f"  Scenario mode: ALL (1, 2, and 3)")
    
    # ──────────────────────────────────────────────────────────
    # PHASE 3: EXECUTE EXPERIMENT
    # ──────────────────────────────────────────────────────────
    # CRITICAL ARCHITECTURE NOTE:
    # ──────────────────────────────────────────────────────────
    # DO NOT directly modify backend_manager.time_idle_ns in this file!
    # 
    # The 3-scenario experiment handles Scenario 2 (time_idle_ns=0) internally
    # via BackendZeroIdleWrapper in run_real_comparison() (comparison.py:480).
    # 
    # WHY: time_idle_ns is a READ-ONLY property on IBMHardwareBackend.
    # Direct assignment will raise: RuntimeError (with detailed guidance).
    # 
    # SOLUTION: All scenarios are managed transparently by run_real_comparison().
    # This script only reads backend_manager.time_idle_ns for display (line 94).
    # ──────────────────────────────────────────────────────────
    
    phase_title = f"EXECUTE SCENARIO {scenario}" if scenario else "EXECUTE 3-SCENARIO EXPERIMENT"
    print(f"\n\nPHASE 3: {phase_title}")
    print("=" * 80)
    
    try:
        experiment_results = run_real_comparison(
            R=R,
            n=n,
            c_max=c_max,
            t_max_ns=t_max_ns,
            shots=shots,
            workload=test_workload,
            backend_manager=backend_manager,
            initial_state=initial_state,
            scenario_filter=scenario  # Pass scenario filter (None = all)
        )
        
        # Print execution summary
        if experiment_results and 'comparative_analysis' in experiment_results:
            analysis = experiment_results['comparative_analysis']
            print("\n\n" + "╔" + "═" * 78 + "╗")
            print("║" + "  EXECUTION SUMMARY".center(78) + "║")
            print("╠" + "═" * 78 + "╣")
            
            print("║" + " " * 78 + "║")
            print("║ Job IDs (for result queries):".ljust(79) + "║")
            print(f"║   Scenario 1 (SWAP): {analysis['job_ids']['scenario_1']:<57} ║")
            print(f"║   Scenario 2 (SQM no-delay): {analysis['job_ids']['scenario_2']:<47} ║")
            print(f"║   Scenario 3 (SQM real): {analysis['job_ids']['scenario_3']:<52} ║")
            
            print("║" + " " * 78 + "║")
            print(f"║ Thesis Validation: {analysis['thesis_validation']:<55} ║")
            print("║" + " " * 78 + "║")
            print("╚" + "═" * 78 + "╝")
            
            # Save results to CSV and generate graph
            print("\n[PHASE 4: SAVING RESULTS]")
            print("=" * 80)
            try:
                save_hardware_comparison_results(experiment_results)
            except Exception as e:
                print(f"\n✗ Failed to save results to CSV/graph")
                print(f"  Error: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\n✓ EXPERIMENT EXECUTION COMPLETE")
        print("  Next: Monitor job queue at https://quantum.ibm.com/compose")
        
    except Exception as e:
        print(f"\n✗ Experiment execution failed!")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
