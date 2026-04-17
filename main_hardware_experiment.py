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
# CONFIGURATION: 3-Scenario Experiment (Hardware)
# ══════════════════════════════════════════════════════════════════════════════

# Compiler Configuration (Minimal for hardware - preserve quota)
R = 1          # Number of memory registers (single register = minimal)
n = 1          # Qubits per register (single qubit = fastest test)
c_max = 3      # Gate cost threshold (aggressive refresh)
t_max_ns = 40000 # Time threshold (nanoseconds)

# Hardware Execution Configuration
shots = 100         # CRITICAL: Keep low to preserve IBM quota (10 min/month)
test_workload = ["READ_0", "IDLE_2", "READ_0"]  # Representative workload

# Quantum State Configuration
initial_state = 1   # 0 = |0⟩ target, 1 = |1⟩ target

# ══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main entry point - Execute 3-Scenario Experiment on real IBM Quantum hardware.
    
    Scenario 1 (Baseline): SWAP Compiler - Static decoherence
    Scenario 2 (SQM no-delay): SQM with time_idle_ns = 0 - Routing overhead only
    Scenario 3 (SQM Real): SQM with calibrated timing - Full thesis test
    """
    
    print("╔" + "═" * 78 + "╗")
    print("║" + "  SQM RESEARCH: 3-SCENARIO EXPERIMENT (IBM QUANTUM HARDWARE)".center(78) + "║")
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
            backend_name='ibm_kyiv',
            channel='ibm_quantum'
        )
        
        backend_info = backend_manager.get_backend_info()
        
        print(f"\n✓ Backend Connected Successfully!")
        print(f"  Backend: {backend_info['name']}")
        print(f"  Total Qubits: {backend_info['num_qubits']}")
        print(f"  Basis Gates: {backend_info['basis_gates']}")
        print(f"  Calibration Time: {backend_info.get('calibration_date', 'N/A')}")
        print(f"  Use Native Delay: {backend_manager.use_native_delay}")
        print(f"  Idle Timing (ns): {backend_manager.time_idle_ns}")
        
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
    
    # ──────────────────────────────────────────────────────────
    # PHASE 3: EXECUTE 3-SCENARIO EXPERIMENT
    # ──────────────────────────────────────────────────────────
    
    print("\n\nPHASE 3: EXECUTE 3-SCENARIO EXPERIMENT")
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
            initial_state=initial_state
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
        
        print("\n✓ EXPERIMENT EXECUTION COMPLETE")
        print("  Next: Monitor job queue at https://quantum.ibm.com/compose")
        
    except Exception as e:
        print(f"\n✗ Experiment execution failed!")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
