# ============================================================
# SQM Research Project - Comparative Analysis Module
# Compare SQM Compiler vs SWAP Compiler
# ============================================================

import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulator.sqm_simulator import SQMCompiler
from src.simulator.swap_simulator import SwapCompiler
from src.functions.qubit_mapper import QubitMapper


# ══════════════════════════════════════════════════════════════════════════════
# Compiler Execution Functions
# ══════════════════════════════════════════════════════════════════════════════

def run_sqm_compiler(R: int, n: int, c_max: int, t_max_ns: float, 
                     workload: List[str], shots: int, initial_state: int = 0) -> Optional[Dict[str, Any]]:

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

    # Create compiler with initial state parameter
    sqm = SQMCompiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns, initial_state=initial_state)

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
        results = sqm.run_simulation(circuit, shots=shots)
        
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
        print(f"[Error] SQM simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_swap_compiler(R: int, n: int, c_max: int, t_max_ns: float,
                     workload: List[str], shots: int, initial_state: int = 0) -> Optional[Dict[str, Any]]:

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

    # Create compiler with initial state parameter
    swap = SwapCompiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns, backend_name="FakeKyiv", initial_state=initial_state)

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
        results = swap.run_simulation(circuit, shots=shots)
        
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
        print(f"[Error] SWAP simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Comparative Analysis Function
# ══════════════════════════════════════════════════════════════════════════════

def analyze_workload(R: int, n: int, c_max: int, t_max_ns: float,
                    workload_name: str, workload: List[str], shots: int, initial_state: int = 0) -> Optional[Dict[str, Any]]:

    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    state_label = "|1⟩" if initial_state == 1 else "|0⟩"
    print("█" + f"  {workload_name} — Target: {state_label}".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    # Run SQM
    sqm_results = run_sqm_compiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns,
                                     workload=workload, shots=shots, initial_state=initial_state)

    # Run SWAP
    swap_results = run_swap_compiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns,
                                     workload=workload, shots=shots, initial_state=initial_state)

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
                       shots: int, workloads: List[Tuple[str, List[str]]], initial_state: int = 0) -> None:
    
    # Summary tracking
    results = []
    qubit_mapping_visualized = False
    
    # Run comparative analysis for each workload
    for idx, (workload_name, workload) in enumerate(workloads):
        result = analyze_workload(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns,
                                 workload_name=workload_name, 
                                 workload=workload, shots=shots, initial_state=initial_state)
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


if __name__ == "__main__":
    print("[Note] This module is designed to be imported from main.py")
    print("       Run 'python main.py' to execute the comparative analysis")
