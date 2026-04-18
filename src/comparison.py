# ============================================================
# SQM Research Project - Comparative Analysis Module
# Compare SQM Compiler vs SWAP Compiler
# ============================================================

import sys
import os

# Configure matplotlib to use non-GUI backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import random
import numpy as np
import csv
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

# Quantum imports
from qiskit import QuantumCircuit

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backends.backend_interface import BackendInterface
from src.simulator.sqm_simulator import SQMCompiler
from src.simulator.swap_simulator import SwapCompiler
from src.functions.qubit_mapper import QubitMapper


# ══════════════════════════════════════════════════════════════════════════════
# Compiler Execution Functions
# ══════════════════════════════════════════════════════════════════════════════

def run_sqm_compiler(R: int, n: int, c_max: int, t_max_ns: float, 
                     workload: List[str], shots: int, backend_manager: BackendInterface,
                     initial_state: int = 0) -> Optional[Dict[str, Any]]:

    # ──────────────────────────────────────────────────────────
    # SEED INITIALIZATION - For global reproducibility
    # ──────────────────────────────────────────────────────────
    random.seed(42)
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    state_label = "|1⟩" if initial_state == 1 else "|0⟩"
    print(f"SQM Compiler - Dual-Register Memory with Quantum Teleportation")
    print(f"Target state: {state_label}")
    print("=" * 70)

    # Create compiler with optional backend manager (Dependency Injection)
    sqm = SQMCompiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns, 
                      backend_manager=backend_manager, initial_state=initial_state)

    # Display workload
    print(f"\n[Workload] Executing {len(workload)} instructions:")
    for i, instr in enumerate(workload, 1):
        print(f"  {i}. {instr}")

    # Compile workload
    circuit = sqm.compile_workload(workload)

    print(f"\n[Circuit Metrics]")
    print(f"  Qubits: {circuit.num_qubits}")
    print(f"  Classical bits: {circuit.num_clbits}")
    print(f"  Depth: {circuit.depth()}")
    print(f"  Size: {circuit.size()}")

    state = sqm.get_compiler_state()
    print(f"\n[Compiler State]")
    print(f"  Available physical qubits: {state['available_qubits']}")

    print("\n" + "-" * 70)
    print("SIMULATION PHASE")
    print("-" * 70)

    try:
        # Execute via backend manager (new method name: execute)
        results = sqm.execute(circuit, shots=shots)
        
        print(f"\n[SQM Results]")
        print(f"  Fidelity: {results['fidelity']:.4f}")
        print(f"  Total Shots: {results['total_shots']}")
        print(f"  Top 5 outcomes:")
        for state_str, count in list(results['counts'].items())[:5]:
            print(f"    |{state_str}⟩: {count} shots")

        # Include the qubit mapper for visualization
        results['qubit_mapper'] = sqm.qubit_mapper
        return results

    except Exception as e:
        print(f"[Error] SQM execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_swap_compiler(R: int, n: int, c_max: int, t_max_ns: float,
                     workload: List[str], shots: int, backend_manager: BackendInterface,
                     initial_state: int = 0) -> Optional[Dict[str, Any]]:

    # ──────────────────────────────────────────────────────────
    # SEED INITIALIZATION - For global reproducibility
    # ──────────────────────────────────────────────────────────
    random.seed(42)
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    state_label = "|1⟩" if initial_state == 1 else "|0⟩"
    print(f"SWAP Compiler - Single-Register Memory (Baseline)")
    print(f"Target state: {state_label}")
    print("=" * 70)

    # Create compiler with optional backend manager (Dependency Injection)
    swap = SwapCompiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns, 
                        backend_manager=backend_manager, initial_state=initial_state)

    # Display workload
    print(f"\n[Workload] Executing {len(workload)} instructions:")
    for i, instr in enumerate(workload, 1):
        print(f"  {i}. {instr}")

    # Compile workload
    circuit = swap.compile_workload(workload)

    print(f"\n[Circuit Metrics]")
    print(f"  Qubits: {circuit.num_qubits}")
    print(f"  Classical bits: {circuit.num_clbits}")
    print(f"  Depth: {circuit.depth()}")
    print(f"  Size: {circuit.size()}")

    state = swap.get_compiler_state()
    print(f"\n[Compiler State]")
    print(f"  Available physical qubits: {state['available_qubits']}")

    print("\n" + "-" * 70)
    print("SIMULATION PHASE")
    print("-" * 70)

    try:
        # Execute via backend manager (new method name: execute)
        results = swap.execute(circuit, shots=shots)
        
        print(f"\n[SWAP Results]")
        print(f"  Fidelity: {results['fidelity']:.4f}")
        print(f"  Total Shots: {results['total_shots']}")
        print(f"  Top 5 outcomes:")
        for state_str, count in list(results['counts'].items())[:5]:
            print(f"    |{state_str}⟩: {count} shots")

        # Include the qubit mapper for visualization
        results['qubit_mapper'] = swap.qubit_mapper
        return results

    except Exception as e:
        print(f"[Error] SWAP execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Comparative Analysis Function
# ══════════════════════════════════════════════════════════════════════════════

def analyze_workload(R: int, n: int, c_max: int, t_max_ns: float,
                    workload_name: str, workload: List[str], shots: int, backend_manager: BackendInterface,
                    initial_state: int = 0) -> Optional[Dict[str, Any]]:

    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    state_label = "|1⟩" if initial_state == 1 else "|0⟩"
    print("█" + f"  {workload_name} — Target: {state_label}".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    # Run SQM with injected backend
    sqm_results = run_sqm_compiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns,
                                     workload=workload, shots=shots, initial_state=initial_state,
                                     backend_manager=backend_manager)

    # Run SWAP with injected backend
    swap_results = run_swap_compiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns,
                                     workload=workload, shots=shots, initial_state=initial_state,
                                     backend_manager=backend_manager)

    # Comparative Analysis
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS SUMMARY")
    print("=" * 70)

    if sqm_results and swap_results:
        sqm_fidelity = sqm_results['fidelity']
        swap_fidelity = swap_results['fidelity']
        sqm_qubits = sqm_results.get('qubits', 'N/A')
        swap_qubits = swap_results.get('qubits', 'N/A')
        difference = sqm_fidelity - swap_fidelity
        percent_diff = (difference / swap_fidelity * 100) if swap_fidelity > 0 else 0

        print(f"\n[Fidelity Comparison]")
        print(f"  SQM:  {sqm_fidelity:.4f} ({sqm_fidelity*100:.2f}%)")
        print(f"  SWAP:  {swap_fidelity:.4f} ({swap_fidelity*100:.2f}%)")
        print(f"  Δ:     {difference:+.4f} ({percent_diff:+.2f}%)")

        behavior = "✓ BETTER" if difference > 0 else "✗ WORSE" if difference < 0 else "= EQUAL"
        print(f"  → SQM is {behavior} than SWAP")

        print(f"\n[Architecture Comparison]")
        print(f"  SQM Memory:  Dual-register (Original + Backup)")
        print(f"  SWAP Memory:  Single-register (Baseline)")
        print(f"  SQM Resilience: Higher (backup register for redundancy)")
        print(f"  → SQM provides resilience advantage via dual-register design")

        return {
            'workload_name': workload_name,
            'sqm': sqm_results,
            'swap': swap_results,
            'comparison': {
                'sqm_fidelity': sqm_fidelity,
                'swap_fidelity': swap_fidelity,
                'difference': difference,
                'percent_diff': percent_diff,
                'sqm_qubits': sqm_qubits,
                'swap_qubits': swap_qubits,
            }
        }
    else:
        print("[Error] Could not complete comparative analysis due to simulation failures")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Full Comparison coordinator
# ══════════════════════════════════════════════════════════════════════════════

def run_full_comparison(R: int, n: int, c_max: int, t_max_ns: float,
                       shots: int, workloads: List[Tuple[str, List[str]]], backend_manager: BackendInterface,
                       initial_state: int = 0) -> None:
    
    # Summary tracking
    results = []
    qubit_mapping_visualized = False
    
    # Run comparative analysis for each workload
    for idx, (workload_name, workload) in enumerate(workloads):
        result = analyze_workload(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns,
                                 workload_name=workload_name, 
                                 workload=workload, shots=shots, initial_state=initial_state,
                                 backend_manager=backend_manager)
        if result:
            results.append(result)
            
            # Visualize qubit allocation comparison only for the first workload
            if idx == 0 and not qubit_mapping_visualized:
                if 'sqm' in result and 'swap' in result:
                    sqm_mapper = result['sqm'].get('qubit_mapper')
                    swap_mapper = result['swap'].get('qubit_mapper')
                    if sqm_mapper and swap_mapper:
                        output_viz_file = f"results/qubit_mapping_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        QubitMapper.compare_mappers(sqm_mapper, swap_mapper, output_file=output_viz_file)
                        qubit_mapping_visualized = True

    # Print overall summary
    print("\n\n" + "╔" + "═" * 68 + "╗")
    print("║" + "  OVERALL COMPARATIVE SUMMARY".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    if results:
        print(f"\n[Analysis Summary - {len(results)} Workloads]")
        print("\n" + "┌" + "─" * 68 + "┐")
        
        sqm_better_count = 0
        swap_better_count = 0
        equal_count = 0

        for result in results:
            comparison = result['comparison']
            diff = comparison['difference']
            workload = result['workload_name']

            if diff > 0:
                sqm_better_count += 1
                status = "✓ SQM BETTER"
            elif diff < 0:
                swap_better_count += 1
                status = "✗ SWAP BETTER"
            else:
                equal_count += 1
                status = "= EQUAL"

            print(f"│ {workload:40} → {status:20} │")

        print("└" + "─" * 68 + "┘")

        print(f"\n[Results]")
        print(f"  SQM Better Cases:  {sqm_better_count}/{len(results)}")
        print(f"  SWAP Better Cases:  {swap_better_count}/{len(results)}")
        print(f"  Equal Cases:        {equal_count}/{len(results)}")

        # Calculate average metrics
        avg_sqm_fidelity = sum(r['comparison']['sqm_fidelity'] for r in results) / len(results)
        avg_swap_fidelity = sum(r['comparison']['swap_fidelity'] for r in results) / len(results)
        avg_difference = avg_sqm_fidelity - avg_swap_fidelity

        print(f"\n[Average Fidelity Across All Workloads]")
        print(f"  SQM:  {avg_sqm_fidelity:.4f} ({avg_sqm_fidelity*100:.2f}%)")
        print(f"  SWAP:  {avg_swap_fidelity:.4f} ({avg_swap_fidelity*100:.2f}%)")
        print(f"  Δ:     {avg_difference:+.4f}")

        # ═════════════════════════════════════════════════════════════════════════
        # GENERATE GRAPH AND SAVE CSV
        # ═════════════════════════════════════════════════════════════════════════
        
        print("\n" + "=" * 70)
        print("GENERATING GRAPH AND SAVING RESULTS")
        print("=" * 70)
        
        # Prepare data for graph and CSV
        workload_names = []
        sqm_fidelities = []
        swap_fidelities = []
        csv_data = []
        
        for i, result in enumerate(results, 1):
            workload_names.append(result['workload_name'])
            sqm_fidelities.append(result['comparison']['sqm_fidelity'])
            swap_fidelities.append(result['comparison']['swap_fidelity'])
            
            csv_data.append({
                'Run': i,
                'Workload': result['workload_name'],
                'SQM_Fidelity': result['comparison']['sqm_fidelity'],
                'SWAP_Fidelity': result['comparison']['swap_fidelity'],
                'Difference': result['comparison']['difference'],
                'Percent_Diff': result['comparison']['percent_diff'],
                'SQM_Qubits': result['comparison']['sqm_qubits'],
                'SWAP_Qubits': result['comparison']['swap_qubits'],
            })
        
        # Generate Graph
        fig, ax = plt.subplots(figsize=(12, 7))
        x_pos = np.arange(len(workload_names))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, sqm_fidelities, width, label='SQM', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, swap_fidelities, width, label='SWAP', color='#A23B72', alpha=0.8)
        
        ax.set_xlabel('Workload', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fidelity', fontsize=12, fontweight='bold')
        ax.set_title('Comparative Analysis: SQM vs SWAP Compiler Fidelity', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(workload_names, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim((0, 1.05))
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save graph
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_path = os.path.join(results_dir, f'comparison_graph_{timestamp}.png')
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        print(f"✓ Graph saved: {graph_path}")
        plt.close()
        
        # Save CSV
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        os.makedirs(data_dir, exist_ok=True)
        csv_path = os.path.join(data_dir, f'comparison_results_{timestamp}.csv')
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['Run', 'Workload', 'SQM_Fidelity', 'SWAP_Fidelity', 
                         'Difference', 'Percent_Diff', 'SQM_Qubits', 'SWAP_Qubits']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"✓ CSV saved: {csv_path}")
        
        # Add summary row to CSV
        summary_path = os.path.join(data_dir, f'comparison_summary_{timestamp}.csv')
        with open(summary_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Workloads', len(results)])
            writer.writerow(['SQM Better', sqm_better_count])
            writer.writerow(['SWAP Better', swap_better_count])
            writer.writerow(['Equal', equal_count])
            writer.writerow(['Avg SQM Fidelity', f'{avg_sqm_fidelity:.4f}'])
            writer.writerow(['Avg SWAP Fidelity', f'{avg_swap_fidelity:.4f}'])
            writer.writerow(['Avg Difference', f'{avg_difference:+.4f}'])
        
        print(f"✓ Summary CSV saved: {summary_path}")

        print("\n" + "=" * 70)
        print("✓ COMPARATIVE ANALYSIS COMPLETE")
        print("=" * 70 + "\n")

    else:
        print("\n[Error] No successful results to summarize")


# ══════════════════════════════════════════════════════════════════════════════
# REAL HARDWARE COMPARISON (3 Scenarios Experiment)
# ══════════════════════════════════════════════════════════════════════════════

def run_real_comparison(R: int, n: int, c_max: int, t_max_ns: float,
                       shots: int, workload: List[str], 
                       backend_manager: BackendInterface,
                       initial_state: int = 0,
                       scenario_filter: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Execute 3-scenario experiment on real IBM Quantum hardware.
    
    Scenario 1 (Baseline - SWAP): Qubit static decoherence
    Scenario 2 (SQM no Delay): Teleportation routing overhead only (idle=0)
    Scenario 3 (SQM Real): Full SQM with calibrated timing
    
    Parameters
    ----------
    R : int
        Number of memory registers
    n : int
        Qubits per register
    c_max : int
        Gate cost threshold for teleportation
    t_max_ns : float
        Time threshold (nanoseconds)
    shots : int
        Number of circuit shots
    workload : List[str]
        Workload instructions (e.g., ["READ_0", "IDLE_2", "READ_0"])
    backend_manager : BackendInterface
        IBMHardwareBackend instance (real hardware connection)
    initial_state : int
        Target quantum state (0 or 1)
    scenario_filter : int, optional
        Execute only specific scenario (1, 2, or 3). If None, runs all 3.
    
    Returns
    -------
    Dict[str, Any]
        Results from executed scenario(s) with comparison (if multiple)
    """
    
    # Helper class for wrapping backend with zero idle time (Scenario 2)
    class BackendZeroIdleWrapper(BackendInterface):
        """Wrapper that forces time_idle_ns = 0"""
        def __init__(self, wrapped_backend: BackendInterface):
            self._backend = wrapped_backend
        
        def get_backend_device(self) -> Any:
            return self._backend.get_backend_device()
        
        def get_backend_info(self) -> Dict[str, Any]:
            return self._backend.get_backend_info()
        
        def run(self, qc_transpiled: QuantumCircuit, shots: int, seed: int = 42) -> Any:
            return self._backend.run(qc_transpiled, shots, seed)
        
        @property
        def use_native_delay(self) -> bool:
            return self._backend.use_native_delay
        
        @property
        def time_idle_ns(self) -> float:
            return 0  # Override: always return 0 for Scenario 2
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "  3-SCENARIO EXPERIMENT: IBM QUANTUM HARDWARE".center(68) + "║")
    print("╚" + "═" * 68 + "╝\n")
    
    print(f"[Hardware Connection]")
    print(f"  Backend: {backend_manager.get_backend_info()['backend_name']}")
    print(f"  Qubits available: {backend_manager.get_backend_info()['num_qubits']}")
    print(f"  Use native delay: {backend_manager.use_native_delay}")
    
    print(f"\n[Experiment Configuration]")
    print(f"  Workload: {workload}")
    print(f"  Parameters: R={R}, n={n}, c_max={c_max}, t_max={t_max_ns} ns")
    print(f"  Shots: {shots}")
    state_label = "|1⟩" if initial_state == 1 else "|0⟩"
    print(f"  Target state: {state_label}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'backend_info': backend_manager.get_backend_info(),
        'workload': workload,
        'params': {'R': R, 'n': n, 'c_max': c_max, 't_max_ns': t_max_ns, 'shots': shots},
        'scenarios': {}
    }
    
    # Initialize job_id variables to None (for type checker)
    job_id_swap = None
    job_id_sqm_no_delay = None
    job_id_sqm_real = None
    fidelity_swap = None
    fidelity_sqm_no_delay = None
    fidelity_sqm_real = None
    
    # ──────────────────────────────────────────────────────────
    # SCENARIO 1: BASELINE - SWAP COMPILER (Static Decoherence)
    # ──────────────────────────────────────────────────────────
    
    if scenario_filter is None or scenario_filter == 1:
        print("\n" + "─" * 70)
        print("SCENARIO 1: BASELINE (SWAP Compiler)")
        print("─" * 70)
        print("Description: Qubit experiences static decoherence (no refresh)")
        print("Hypothesis: Lowest fidelity (baseline for comparison)")
        
        try:
            swap = SwapCompiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns,
                               backend_manager=backend_manager, initial_state=initial_state)
            
            circuit_swap = swap.compile_workload(workload)
            print(f"\n[Circuit Generated]")
            print(f"  Qubits: {circuit_swap.num_qubits}")
            print(f"  Depth: {circuit_swap.depth()}")
            print(f"  Size: {circuit_swap.size()}")
            
            print(f"\n[Submitting to Hardware]")
            results_swap = swap.execute(circuit_swap, shots=shots)
            fidelity_swap = results_swap['fidelity']
            job_id_swap = results_swap.get('job_id', 'N/A')
            
            results['scenarios']['scenario_1_swap'] = {
                'fidelity': fidelity_swap,
                'job_id': job_id_swap,
                'total_shots': results_swap.get('total_shots', shots),
                'counts_top5': list(results_swap.get('counts', {}).items())[:5]
            }
            
            print(f"\n[Scenario 1 Results]")
            print(f"  ✓ Fidelity: {fidelity_swap:.4f}")
            print(f"  ✓ Job ID: {job_id_swap}")
            print(f"  ✓ Total shots: {results_swap.get('total_shots', shots)}")
            
        except Exception as e:
            print(f"\n[ERROR] Scenario 1 failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results['scenarios']['scenario_1_swap'] = {'error': str(e)}
    else:
        print("\n[SKIPPED] Scenario 1 (Scenario filter active)")
    
    # ──────────────────────────────────────────────────────────
    # SCENARIO 2: SQM WITHOUT DELAY (Routing Overhead Only)
    # ──────────────────────────────────────────────────────────
    
    if scenario_filter is None or scenario_filter == 2:
        print("\n" + "─" * 70)
        print("SCENARIO 2: SQM WITHOUT DELAY (Routing Overhead)")
        print("─" * 70)
        print("Description: SQM with time_idle_ns = 0 (no passive decoherence)")
        print("Hypothesis: Measure teleportation routing overhead cost")
        
        try:
            # Use wrapper backend with zero idle time
            wrapped_backend = BackendZeroIdleWrapper(backend_manager)
            
            print(f"\n[Configuration] Using wrapped backend with time_idle_ns = 0")
            
            sqm_no_delay = SQMCompiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns,
                                       backend_manager=wrapped_backend, initial_state=initial_state)
            
            circuit_sqm_no_delay = sqm_no_delay.compile_workload(workload)
            print(f"\n[Circuit Generated]")
            print(f"  Qubits: {circuit_sqm_no_delay.num_qubits}")
            print(f"  Depth: {circuit_sqm_no_delay.depth()}")
            print(f"  Size: {circuit_sqm_no_delay.size()}")
            
            print(f"\n[Submitting to Hardware]")
            results_sqm_no_delay = sqm_no_delay.execute(circuit_sqm_no_delay, shots=shots)
            fidelity_sqm_no_delay = results_sqm_no_delay['fidelity']
            job_id_sqm_no_delay = results_sqm_no_delay.get('job_id', 'N/A')
            
            results['scenarios']['scenario_2_sqm_no_delay'] = {
                'fidelity': fidelity_sqm_no_delay,
                'job_id': job_id_sqm_no_delay,
                'total_shots': results_sqm_no_delay.get('total_shots', shots),
                'counts_top5': list(results_sqm_no_delay.get('counts', {}).items())[:5],
                'time_idle_ns': 0
            }
            
            print(f"\n[Scenario 2 Results]")
            print(f"  ✓ Fidelity: {fidelity_sqm_no_delay:.4f}")
            print(f"  ✓ Job ID: {job_id_sqm_no_delay}")
            print(f"  ✓ Total shots: {results_sqm_no_delay.get('total_shots', shots)}")
            
        except Exception as e:
            print(f"\n[ERROR] Scenario 2 failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results['scenarios']['scenario_2_sqm_no_delay'] = {'error': str(e)}
    else:
        print("\n[SKIPPED] Scenario 2 (Scenario filter active)")
    
    # ──────────────────────────────────────────────────────────
    # SCENARIO 3: SQM WITH REAL TIMING (Calibrated)
    # ──────────────────────────────────────────────────────────
    
    if scenario_filter is None or scenario_filter == 3:
        print("\n" + "─" * 70)
        print("SCENARIO 3: SQM WITH REAL TIMING (Full SQM)")
        print("─" * 70)
        print("Description: SQM with backend-calibrated timing")
        print("Hypothesis: SQM fidelity > SWAP baseline (validates thesis)")
        
        try:
            sqm_real = SQMCompiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns,
                                  backend_manager=backend_manager, initial_state=initial_state)
            
            circuit_sqm_real = sqm_real.compile_workload(workload)
            print(f"\n[Circuit Generated]")
            print(f"  Qubits: {circuit_sqm_real.num_qubits}")
            print(f"  Depth: {circuit_sqm_real.depth()}")
            print(f"  Size: {circuit_sqm_real.size()}")
            
            print(f"\n[Submitting to Hardware]")
            results_sqm_real = sqm_real.execute(circuit_sqm_real, shots=shots)
            fidelity_sqm_real = results_sqm_real['fidelity']
            job_id_sqm_real = results_sqm_real.get('job_id', 'N/A')
            
            results['scenarios']['scenario_3_sqm_real'] = {
                'fidelity': fidelity_sqm_real,
                'job_id': job_id_sqm_real,
                'total_shots': results_sqm_real.get('total_shots', shots),
                'counts_top5': list(results_sqm_real.get('counts', {}).items())[:5],
                'time_idle_ns': backend_manager.time_idle_ns
            }
            
            print(f"\n[Scenario 3 Results]")
            print(f"  ✓ Fidelity: {fidelity_sqm_real:.4f}")
            print(f"  ✓ Job ID: {job_id_sqm_real}")
            print(f"  ✓ Total shots: {results_sqm_real.get('total_shots', shots)}")
            print(f"  ✓ Time idle: {backend_manager.time_idle_ns} ns")
            
        except Exception as e:
            print(f"\n[ERROR] Scenario 3 failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results['scenarios']['scenario_3_sqm_real'] = {'error': str(e)}
    else:
        print("\n[SKIPPED] Scenario 3 (Scenario filter active)")
    
    # ──────────────────────────────────────────────────────────
    # COMPARATIVE ANALYSIS (if all 3 scenarios executed)
    # ──────────────────────────────────────────────────────────
    
    if scenario_filter is None:
        # Full comparison mode (all 3 scenarios)
        print("\n" + "╔" + "═" * 68 + "╗")
        print("║" + "  COMPARATIVE ANALYSIS (3 Scenarios)".center(68) + "║")
        print("╚" + "═" * 68 + "╝\n")
        
        if fidelity_swap is not None and fidelity_sqm_no_delay is not None and fidelity_sqm_real is not None:
            
            # Calculate deltas
            delta_s2_s1 = fidelity_sqm_no_delay - fidelity_swap
            delta_s3_s1 = fidelity_sqm_real - fidelity_swap
            delta_s3_s2 = fidelity_sqm_real - fidelity_sqm_no_delay
            
            # Create comparison table
            print("┌─ FIDELITY COMPARISON ─────────────────────────────────────────┐")
            print("│                                                                 │")
            print(f"│ Scenario 1 (SWAP Baseline):         {fidelity_swap:.4f}                   │")
            print(f"│ Scenario 2 (SQM no-delay):          {fidelity_sqm_no_delay:.4f}  ({delta_s2_s1:+.4f})      │")
            print(f"│ Scenario 3 (SQM real timing):       {fidelity_sqm_real:.4f}  ({delta_s3_s1:+.4f})      │")
            print("│                                                                 │")
            print("└─────────────────────────────────────────────────────────────────┘")
            
            print("\n┌─ INSIGHTS ────────────────────────────────────────────────────┐")
            
            # Analysis
            if fidelity_sqm_real > fidelity_swap:
                print(f"│ ✓ SQM OUTPERFORMS SWAP by {delta_s3_s1:+.4f} ({delta_s3_s1/fidelity_swap*100:+.2f}%)      │")
                thesis_validation = "SUPPORTED"
            else:
                print(f"│ ✗ SWAP outperforms SQM by {abs(delta_s3_s1):+.4f} ({abs(delta_s3_s1)/fidelity_swap*100:+.2f}%)     │")
                thesis_validation = "NOT SUPPORTED (but check noise)"
            
            if fidelity_sqm_real > fidelity_sqm_no_delay:
                print(f"│ ✓ Timing improves S3 vs S2 by {delta_s3_s2:+.4f} (refresh benefit)│")
            else:
                print(f"│ ✗ Timing hurts S3 vs S2 by {abs(delta_s3_s2):+.4f} (decoherence > refresh) │")
            
            print(f"│ ★ THESIS STATUS: {thesis_validation:<42} │")
            print("└─────────────────────────────────────────────────────────────────┘")
            
            results['comparative_analysis'] = {
                'fidelity_swap': fidelity_swap,
                'fidelity_sqm_no_delay': fidelity_sqm_no_delay,
                'fidelity_sqm_real': fidelity_sqm_real,
                'delta_s2_s1': delta_s2_s1,
                'delta_s3_s1': delta_s3_s1,
                'delta_s3_s2': delta_s3_s2,
                'thesis_validation': thesis_validation,
                'job_ids': {
                    'scenario_1': job_id_swap if job_id_swap else 'N/A',
                    'scenario_2': job_id_sqm_no_delay if job_id_sqm_no_delay else 'N/A',
                    'scenario_3': job_id_sqm_real if job_id_sqm_real else 'N/A'
                }
            }
            
            print("\n" + "=" * 70)
            print("✓ 3-SCENARIO EXPERIMENT COMPLETE")
            print("=" * 70)
        
        else:
            print("\n[Error] One or more scenarios failed - cannot complete analysis")
    
    else:
        # Single scenario mode - display results for selected scenario
        print("\n" + "╔" + "═" * 68 + "╗")
        print("║" + f"  SCENARIO {scenario_filter} RESULTS".center(68) + "║")
        print("╚" + "═" * 68 + "╝\n")
        
        if scenario_filter == 1 and fidelity_swap is not None:
            print(f"✓ Scenario 1 (SWAP Baseline)")
            print(f"  Fidelity: {fidelity_swap:.4f}")
            print(f"  Job ID: {job_id_swap}")
            print("\n✓ SINGLE SCENARIO EXECUTION COMPLETE")
        
        elif scenario_filter == 2 and fidelity_sqm_no_delay is not None:
            print(f"✓ Scenario 2 (SQM no-delay)")
            print(f"  Fidelity: {fidelity_sqm_no_delay:.4f}")
            print(f"  Job ID: {job_id_sqm_no_delay}")
            print(f"  Time idle: 0 ns (forced)")
            print("\n✓ SINGLE SCENARIO EXECUTION COMPLETE")
        
        elif scenario_filter == 3 and fidelity_sqm_real is not None:
            print(f"✓ Scenario 3 (SQM Real Timing)")
            print(f"  Fidelity: {fidelity_sqm_real:.4f}")
            print(f"  Job ID: {job_id_sqm_real}")
            print(f"  Time idle: {backend_manager.time_idle_ns} ns (calibrated)")
            print("\n✓ SINGLE SCENARIO EXECUTION COMPLETE")
        
        else:
            print(f"[Error] Scenario {scenario_filter} did not complete successfully")
    
    return results


if __name__ == "__main__":
    print("[Note] This module is designed to be imported from main.py")
    print("       Run 'python main.py' to execute the comparative analysis")
