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
from src.simulator.sqm_simulator_Flow import SQMFlowCompiler
from src.simulator.swap_simulator_Flow import SwapFlowCompiler
from src.functions.qubit_mapper import QubitMapper
from src.utils.hardware_results_processor import save_hardware_comparison_results


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
# Flow Compiler Execution Functions (Fidelity on Operation Register)
# ══════════════════════════════════════════════════════════════════════════════

def run_sqm_flow_compiler(R: int, n: int, c_max: int, t_max_ns: float, 
                     workload: List[str], shots: int, backend_manager: BackendInterface,
                     initial_state: int = 0) -> Optional[Dict[str, Any]]:

    # ──────────────────────────────────────────────────────────
    # SEED INITIALIZATION - For global reproducibility
    # ──────────────────────────────────────────────────────────
    random.seed(42)
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    state_label = "|1⟩" if initial_state == 1 else "|0⟩"
    print(f"SQM Flow Compiler - Dual-Register Memory with Quantum Teleportation")
    print(f"Target state: {state_label}")
    print(f"Fidelity measured on: OPERATION REGISTER (q_work)")
    print("=" * 70)

    # Create compiler with optional backend manager (Dependency Injection)
    sqm = SQMFlowCompiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns, 
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
    print("SIMULATION PHASE (FLOW - Operation Register)")
    print("-" * 70)

    try:
        results = sqm.execute(circuit, shots=shots)
        
        print(f"\n[SQM Flow Results]")
        print(f"  Fidelity: {results['fidelity']:.4f}")
        print(f"  Total Shots: {results['total_shots']}")
        print(f"  Top 5 outcomes:")
        for state_str, count in list(results['counts'].items())[:5]:
            print(f"    |{state_str}⟩: {count} shots")

        results['qubit_mapper'] = sqm.qubit_mapper
        return results

    except Exception as e:
        print(f"[Error] SQM Flow execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_swap_flow_compiler(R: int, n: int, c_max: int, t_max_ns: float,
                     workload: List[str], shots: int, backend_manager: BackendInterface,
                     initial_state: int = 0) -> Optional[Dict[str, Any]]:

    # ──────────────────────────────────────────────────────────
    # SEED INITIALIZATION - For global reproducibility
    # ──────────────────────────────────────────────────────────
    random.seed(42)
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    state_label = "|1⟩" if initial_state == 1 else "|0⟩"
    print(f"SWAP Flow Compiler - Single-Register Memory (Baseline)")
    print(f"Target state: {state_label}")
    print(f"Fidelity measured on: OPERATION REGISTER (q_work)")
    print("=" * 70)

    # Create compiler with optional backend manager (Dependency Injection)
    swap = SwapFlowCompiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns, 
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
    print("SIMULATION PHASE (FLOW - Operation Register)")
    print("-" * 70)

    try:
        results = swap.execute(circuit, shots=shots)
        
        print(f"\n[SWAP Flow Results]")
        print(f"  Fidelity: {results['fidelity']:.4f}")
        print(f"  Total Shots: {results['total_shots']}")
        print(f"  Top 5 outcomes:")
        for state_str, count in list(results['counts'].items())[:5]:
            print(f"    |{state_str}⟩: {count} shots")

        results['qubit_mapper'] = swap.qubit_mapper
        return results

    except Exception as e:
        print(f"[Error] SWAP Flow execution failed: {e}")
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



    # Run SWAP with injected backend
    swap_results = run_swap_compiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns,
                                     workload=workload, shots=shots, initial_state=initial_state,
                                     backend_manager=backend_manager)
    

    # Run SQM with injected backend
    sqm_results = run_sqm_compiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns,
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
            'workload_tasks': workload,
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


def analyze_workload_flow(R: int, n: int, c_max: int, t_max_ns: float,
                    workload_name: str, workload: List[str], shots: int, backend_manager: BackendInterface,
                    initial_state: int = 0) -> Optional[Dict[str, Any]]:
    """Flow variant of analyze_workload: measures fidelity on operation register."""

    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    state_label = "|1⟩" if initial_state == 1 else "|0⟩"
    print("█" + f"  {workload_name} — Target: {state_label} — FLOW (op_register)".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)



    # Run SWAP Flow with injected backend
    swap_results = run_swap_flow_compiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns,
                                     workload=workload, shots=shots, initial_state=initial_state,
                                     backend_manager=backend_manager)
    

    # Run SQM Flow with injected backend
    sqm_results = run_sqm_flow_compiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns,
                                     workload=workload, shots=shots, initial_state=initial_state,
                                     backend_manager=backend_manager)

    # Comparative Analysis
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS SUMMARY (FLOW - Operation Register)")
    print("=" * 70)

    if sqm_results and swap_results:
        sqm_fidelity = sqm_results['fidelity']
        swap_fidelity = swap_results['fidelity']
        sqm_qubits = sqm_results.get('qubits', 'N/A')
        swap_qubits = swap_results.get('qubits', 'N/A')
        difference = sqm_fidelity - swap_fidelity
        percent_diff = (difference / swap_fidelity * 100) if swap_fidelity > 0 else 0

        print(f"\n[Fidelity Comparison - FLOW]")
        print(f"  SQM:  {sqm_fidelity:.4f} ({sqm_fidelity*100:.2f}%)")
        print(f"  SWAP:  {swap_fidelity:.4f} ({swap_fidelity*100:.2f}%)")
        print(f"  Δ:     {difference:+.4f} ({percent_diff:+.2f}%)")

        behavior = "✓ BETTER" if difference > 0 else "✗ WORSE" if difference < 0 else "= EQUAL"
        print(f"  → SQM is {behavior} than SWAP")

        print(f"\n[Architecture Comparison]")
        print(f"  SQM Memory:  Dual-register (Original + Backup)")
        print(f"  SWAP Memory:  Single-register (Baseline)")
        print(f"  Measurement:  FLOW (operation register - q_work)")

        return {
            'workload_name': workload_name,
            'workload_tasks': workload,
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
                       initial_state: int = 0, flow: int = 0) -> None:
    
    # Summary tracking
    results = []
    qubit_mapping_visualized = False
    
    # Run comparative analysis for each workload
    # Select analyzer based on flow parameter
    flow_label = "FLOW (operation register)" if flow == 1 else "MEMORY (memory registers)"
    print(f"\n[Comparison Mode] {flow_label}")

    for idx, (workload_name, workload) in enumerate(workloads):
        if flow == 1:
            result = analyze_workload_flow(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns,
                                     workload_name=workload_name, 
                                     workload=workload, shots=shots, initial_state=initial_state,
                                     backend_manager=backend_manager)
        else:
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
                'Tasks': ' -> '.join(result.get('workload_tasks', [])),
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
        
        bars1 = ax.bar(x_pos - width/2, swap_fidelities, width, label='SWAP', color='#A23B72', alpha=0.8)
       
        bars2 = ax.bar(x_pos + width/2, sqm_fidelities, width, label='SQM', color='#06A77D', alpha=0.8)
        
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
            writer = csv.writer(csvfile)
            
            # Header with experiment parameters
            writer.writerow(['EXPERIMENT PARAMETERS'])
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['R (Logical Addresses)', R])
            writer.writerow(['n (Qubits per Register)', n])
            writer.writerow(['c_max (Max Cost Threshold)', c_max])
            writer.writerow(['t_max_ns (Max Time Threshold)', f'{t_max_ns}'])
            writer.writerow(['shots', shots])
            writer.writerow(['initial_state', initial_state])
            writer.writerow([])
            
            # Data table header
            writer.writerow(['DETAILED RESULTS'])
            fieldnames = ['Run', 'Workload', 'Tasks', 'SQM_Fidelity', 'SWAP_Fidelity', 
                         'Difference', 'Percent_Diff', 'SQM_Qubits', 'SWAP_Qubits']
            writer.writerow(fieldnames)
            
            # Write data rows
            for row in csv_data:
                writer.writerow([
                    row['Run'],
                    row['Workload'],
                    row['Tasks'],
                    row['SQM_Fidelity'],
                    row['SWAP_Fidelity'],
                    row['Difference'],
                    row['Percent_Diff'],
                    row['SQM_Qubits'],
                    row['SWAP_Qubits']
                ])
        
        print(f"✓ CSV saved: {csv_path}")
        
        # Add summary row to CSV (with experiment parameters)
        summary_path = os.path.join(data_dir, f'comparison_summary_{timestamp}.csv')
        with open(summary_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Experiment Parameters section
            writer.writerow(['EXPERIMENT PARAMETERS'])
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['R (Logical Addresses)', R])
            writer.writerow(['n (Qubits per Register)', n])
            writer.writerow(['c_max (Max Cost Threshold)', c_max])
            writer.writerow(['t_max_ns (Max Time Threshold)', f'{t_max_ns}'])
            writer.writerow(['shots', shots])
            writer.writerow(['initial_state', initial_state])
            writer.writerow(['num_workloads', len(workloads)])
            writer.writerow([])
            
            # Results section
            writer.writerow(['COMPARATIVE ANALYSIS RESULTS'])
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
# REAL HARDWARE COMPARISON (Multi-Scenario Experiment)
# ══════════════════════════════════════════════════════════════════════════════

def run_real_comparison(R: int, n: int, c_max: int, t_max_ns: float,
                       shots: int, workload: List[str],
                       backend_manager: BackendInterface,
                       initial_state: int = 0,
                       scenarios: List[int] | None = None,
                       flow: int = 0) -> Optional[Dict[str, Any]]:
    """
    Execute multi-scenario experiment on real IBM Quantum hardware.

    Scenario 1 (Baseline - SWAP): Qubit static decoherence
    Scenario 2 (SQM no Delay): Teleportation routing overhead only (idle=0)
    Scenario 3 (SQM Real): Full SQM with calibrated timing
    """

    if scenarios is None:
        scenarios = [1, 2, 3]

    scenario_str = ", ".join(map(str, scenarios))
    print("\n" + "=" * 70)
    print(f"  EXPERIMENT: SCENARIOS {scenario_str}")
    print("=" * 70)

    print(f"\n[Hardware Connection]")
    print(f"  Backend: {backend_manager.get_backend_info()['backend_name']}")
    print(f"  Qubits available: {backend_manager.get_backend_info()['num_qubits']}")
    print(f"  Use native delay: {backend_manager.use_native_delay}")

    print(f"\n[Experiment Configuration]")
    print(f"  Workload: {workload}")
    print(f"  Parameters: R={R}, n={n}, c_max={c_max}, t_max={t_max_ns} ns")
    print(f"  Shots: {shots}")
    state_label = "|1>" if initial_state == 1 else "|0>"
    print(f"  Target state: {state_label}")
    flow_label = "FLOW (operation register)" if flow == 1 else "MEMORY (memory registers)"
    print(f"  Measurement mode: {flow_label}")
    print(f"  Scenarios: {scenarios}")

    results: Dict[str, Any] = {
        'timestamp': datetime.now().isoformat(),
        'backend_info': backend_manager.get_backend_info(),
        'workload': workload,
        'params': {'R': R, 'n': n, 'c_max': c_max, 't_max_ns': t_max_ns, 'shots': shots, 'initial_state': initial_state},
        'scenarios': {}
    }

    # Track fidelity and job_id per scenario
    fidelities: Dict[int, float] = {}
    job_ids: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # SCENARIO 1: BASELINE - SWAP COMPILER (Static Decoherence)
    # ------------------------------------------------------------------

    if 1 in scenarios:
        print("\n" + "-" * 70)
        print("SCENARIO 1: BASELINE (SWAP Compiler)")
        print("-" * 70)
        print("Description: Qubit experiences static decoherence (no refresh)")
        print("Hypothesis: Lowest fidelity (baseline for comparison)")

        try:
            if flow == 1:
                swap = SwapFlowCompiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns,
                                   backend_manager=backend_manager, initial_state=initial_state)
            else:
                swap = SwapCompiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns,
                                   backend_manager=backend_manager, initial_state=initial_state)

            circuit_swap = swap.compile_workload(workload)
            print(f"\n[Circuit Generated]")
            print(f"  Qubits: {circuit_swap.num_qubits}")
            print(f"  Depth: {circuit_swap.depth()}")
            print(f"  Size: {circuit_swap.size()}")

            print(f"\n[Submitting to Hardware]")
            results_swap = swap.execute(circuit_swap, shots=shots)
            #results_swap= {  "fidelity": 0.82, "counts": 100, "total_shots": shots}
            fidelities[1] = results_swap['fidelity']
            job_ids[1] = results_swap.get('job_id', 'N/A')

            results['scenarios']['scenario_1_swap'] = {
                'fidelity': fidelities[1],
                'job_id': job_ids[1],
                'total_shots': results_swap.get('total_shots', shots),
                'counts_top5': list(results_swap.get('counts', {}).items())[:5]
            }

            print(f"\n[Scenario 1 Results]")
            print(f"  Fidelity: {fidelities[1]:.4f}")
            print(f"  Job ID: {job_ids[1]}")

        except Exception as e:
            print(f"\n[ERROR] Scenario 1 failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results['scenarios']['scenario_1_swap'] = {'error': str(e)}

    # ------------------------------------------------------------------
    # SCENARIO 2: SQM WITHOUT DELAY (Routing Overhead Only)
    # ------------------------------------------------------------------

    if 2 in scenarios:
        print("\n" + "-" * 70)
        print("SCENARIO 2: SQM WITHOUT DELAY (Routing Overhead Only)")
        print("-" * 70)
        print("Description: SQM with routing overhead only")
        print("Hypothesis: SQM fidelity > SWAP baseline (validates thesis)")

        try:
            if flow == 1:
                sqm_real = SQMFlowCompiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns,
                                      backend_manager=backend_manager, initial_state=initial_state)
            else:
                sqm_real = SQMCompiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns,
                                      backend_manager=backend_manager, initial_state=initial_state)

            filtered = [w for w in workload if not w.startswith("IDLE_5")]
            circuit_sqm_real = sqm_real.compile_workload(filtered       )
            print(f"\n[Circuit Generated]")
            print(f"  Qubits: {circuit_sqm_real.num_qubits}")
            print(f"  Depth: {circuit_sqm_real.depth()}")
            print(f"  Size: {circuit_sqm_real.size()}")

            print(f"\n[Submitting to Hardware]")
            results_sqm_real = sqm_real.execute(circuit_sqm_real, shots=shots)
            #results_sqm_real = {  "fidelity": 0.95,    "counts": 100,  "total_shots": shots, }
            fidelities[2] = results_sqm_real['fidelity']
            job_ids[2] = results_sqm_real.get('job_id', 'N/A')

            results['scenarios']['scenario_2_sqm_no_delay'] = {
                'fidelity': fidelities[2],
                'job_id': job_ids[2],
                'total_shots': results_sqm_real.get('total_shots', shots),
                'counts_top5': list(results_sqm_real.get('counts', {}).items())[:5],
                'time_idle_ns': backend_manager.time_idle_ns
            }

            print(f"\n[Scenario 2 Results]")
            print(f"  Fidelity: {fidelities[2]:.4f}")
            print(f"  Job ID: {job_ids[2]}")
            print(f"  Time idle: {backend_manager.time_idle_ns} ns")

        except Exception as e:
            print(f"\n[ERROR] Scenario 2 failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results['scenarios']['scenario_2_sqm_no_delay'] = {'error': str(e)}

    # ------------------------------------------------------------------
    # SCENARIO 3: SQM WITH REAL TIMING (Calibrated)
    # ------------------------------------------------------------------

    if 3 in scenarios:
        print("\n" + "-" * 70)
        print("SCENARIO 3: SQM WITH REAL TIMING (Full SQM)")
        print("-" * 70)
        print("Description: SQM with backend-calibrated timing")
        print("Hypothesis: SQM fidelity > SWAP baseline (validates thesis)")

        try:
            if flow == 1:
                sqm_real = SQMFlowCompiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns,
                                      backend_manager=backend_manager, initial_state=initial_state)
            else:
                sqm_real = SQMCompiler(R=R, n=n, c_max=c_max, t_max_ns=t_max_ns,
                                      backend_manager=backend_manager, initial_state=initial_state)

            circuit_sqm_real = sqm_real.compile_workload(workload)
            print(f"\n[Circuit Generated]")
            print(f"  Qubits: {circuit_sqm_real.num_qubits}")
            print(f"  Depth: {circuit_sqm_real.depth()}")
            print(f"  Size: {circuit_sqm_real.size()}")

            print(f"\n[Submitting to Hardware]")
            results_sqm_real = sqm_real.execute(circuit_sqm_real, shots=shots)
            #results_sqm_real = { "fidelity": 0.95, "counts": 100, "total_shots": shots}
            fidelities[3] = results_sqm_real['fidelity']
            job_ids[3] = results_sqm_real.get('job_id', 'N/A')

            results['scenarios']['scenario_3_sqm_real'] = {
                'fidelity': fidelities[3],
                'job_id': job_ids[3],
                'total_shots': results_sqm_real.get('total_shots', shots),
                'counts_top5': list(results_sqm_real.get('counts', {}).items())[:5],
                'time_idle_ns': backend_manager.time_idle_ns
            }

            print(f"\n[Scenario 3 Results]")
            print(f"  Fidelity: {fidelities[3]:.4f}")
            print(f"  Job ID: {job_ids[3]}")
            print(f"  Time idle: {backend_manager.time_idle_ns} ns")

        except Exception as e:
            print(f"\n[ERROR] Scenario 3 failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results['scenarios']['scenario_3_sqm_real'] = {'error': str(e)}

    # ------------------------------------------------------------------
    # COMPARATIVE ANALYSIS (Adaptive - 1, 2, or 3 scenarios)
    # ------------------------------------------------------------------

    if len(fidelities) > 0:
        print("\n" + "=" * 70)
        print(f"  COMPARATIVE ANALYSIS ({len(fidelities)} Scenario(s))")
        print("=" * 70)

        # Build comparative info based on available scenarios
        scenario_names = []
        scenario_fidelities = []
        comparisons = {}

        if 1 in fidelities:
            scenario_names.append("Scenario 1 (SWAP Baseline)")
            scenario_fidelities.append(fidelities[1])
            comparisons[1] = fidelities[1]

        if 2 in fidelities:
            scenario_names.append("Scenario 2 (SQM no-delay)")
            scenario_fidelities.append(fidelities[2])
            comparisons[2] = fidelities[2]

        if 3 in fidelities:
            scenario_names.append("Scenario 3 (SQM real timing)")
            scenario_fidelities.append(fidelities[3])
            comparisons[3] = fidelities[3]

        # Print all available results
        print(f"\n  Results:")
        for name, fid in zip(scenario_names, scenario_fidelities):
            print(f"  {name:40} {fid:.4f}")

        # Comparative analysis for different scenario combinations
        thesis_validation = "PENDING"
        if {1, 2, 3}.issubset(fidelities.keys()):
            delta_s2_s1 = fidelities[2] - fidelities[1]
            delta_s3_s1 = fidelities[3] - fidelities[1]
            delta_s3_s2 = fidelities[3] - fidelities[2]

            print(f"\n  Deltas:")
            print(f"    S2 - S1: {delta_s2_s1:+.4f}")
            print(f"    S3 - S1: {delta_s3_s1:+.4f}")
            print(f"    S3 - S2: {delta_s3_s2:+.4f}")

            if fidelities[3] > fidelities[1]:
                print(f"\n  -> SQM OUTPERFORMS SWAP by {delta_s3_s1:+.4f} ({delta_s3_s1/fidelities[1]*100:+.2f}%)")
                thesis_validation = "SUPPORTED"
            else:
                print(f"\n  -> SWAP outperforms SQM by {abs(delta_s3_s1):.4f} ({abs(delta_s3_s1)/fidelities[1]*100:.2f}%)")
                thesis_validation = "NOT SUPPORTED"

            if fidelities[3] > fidelities[2]:
                print(f"  -> Timing improves S3 vs S2 by {delta_s3_s2:+.4f} (refresh benefit)")
            else:
                print(f"  -> Timing hurts S3 vs S2 by {abs(delta_s3_s2):.4f} (decoherence > refresh)")

            results['comparative_analysis'] = {
                'fidelity_swap': fidelities[1],
                'fidelity_sqm_no_delay': fidelities[2],
                'fidelity_sqm_real': fidelities[3],
                'delta_s2_s1': delta_s2_s1,
                'delta_s3_s1': delta_s3_s1,
                'delta_s3_s2': delta_s3_s2,
                'thesis_validation': thesis_validation,
                'job_ids': {
                    'scenario_1': job_ids.get(1, 'N/A'),
                    'scenario_2': job_ids.get(2, 'N/A'),
                    'scenario_3': job_ids.get(3, 'N/A')
                }
            }

        elif {1, 2}.issubset(fidelities.keys()):
            delta_s2_s1 = fidelities[2] - fidelities[1]
            print(f"\n  Delta (S2 - S1): {delta_s2_s1:+.4f}")
            if fidelities[2] > fidelities[1]:
                print(f"  -> SQM outperforms SWAP baseline by {delta_s2_s1:+.4f}")
                thesis_validation = "PARTIAL (2/3 scenarios)"
            else:
                print(f"  -> SWAP outperforms SQM by {abs(delta_s2_s1):.4f}")
                thesis_validation = "PARTIAL (2/3 scenarios)"

            results['comparative_analysis'] = {
                'fidelity_swap': fidelities[1],
                'fidelity_sqm_no_delay': fidelities[2],
                'delta_s2_s1': delta_s2_s1,
                'thesis_validation': thesis_validation,
                'job_ids': {
                    'scenario_1': job_ids.get(1, 'N/A'),
                    'scenario_2': job_ids.get(2, 'N/A'),
                }
            }

        elif 1 in fidelities and 3 in fidelities:
            delta_s3_s1 = fidelities[3] - fidelities[1]
            print(f"\n  Delta (S3 - S1): {delta_s3_s1:+.4f}")
            if fidelities[3] > fidelities[1]:
                print(f"  -> SQM outperforms SWAP by {delta_s3_s1:+.4f}")
                thesis_validation = "PARTIAL (2/3 scenarios)"
            else:
                print(f"  -> SWAP outperforms SQM by {abs(delta_s3_s1):.4f}")
                thesis_validation = "PARTIAL (2/3 scenarios)"

            results['comparative_analysis'] = {
                'fidelity_swap': fidelities[1],
                'fidelity_sqm_real': fidelities[3],
                'delta_s3_s1': delta_s3_s1,
                'thesis_validation': thesis_validation,
                'job_ids': {
                    'scenario_1': job_ids.get(1, 'N/A'),
                    'scenario_3': job_ids.get(3, 'N/A'),
                }
            }

        elif len(fidelities) == 1:
            first_scenario = list(fidelities.keys())[0]
            print(f"\n  Single scenario result - awaiting other scenarios for full comparison")
            thesis_validation = "INCOMPLETE (1/3 scenarios)"
            results['comparative_analysis'] = {
                f'fidelity_scenario_{first_scenario}': fidelities[first_scenario],
                'thesis_validation': thesis_validation,
                'job_ids': {f'scenario_{first_scenario}': job_ids.get(first_scenario, 'N/A')}
            }

        print(f"\n  ANALYSIS STATUS: {thesis_validation}")
        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)

    else:
        print("\n[Error] No scenarios completed successfully")

    # ------------------------------------------------------------------
    # AUTO-SAVE: Generate CSV and graph for ANY completed scenarios
    # ------------------------------------------------------------------
    if 'comparative_analysis' in results and len(fidelities) > 0:
        try:
            save_hardware_comparison_results(results)
            print("\n✓ Results saved successfully")
        except Exception as e:
            print(f"\n[WARNING] Failed to auto-save results: {e}")

    return results


if __name__ == "__main__":
    print("[Note] This module is designed to be imported from main.py")
    print("       Run 'python main.py' to execute the comparative analysis")
