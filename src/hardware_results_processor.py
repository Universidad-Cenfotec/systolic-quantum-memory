# ============================================================
# Hardware Results Processor
# Save hardware experiment results to CSV and generate graphs
# ============================================================

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional


def save_hardware_comparison_results(experiment_results: Dict[str, Any]) -> None:
    """
    Save hardware experiment results to CSV files and generate comparison graph.
    
    Creates files in data/ (CSV) and results/ (PNG) directories, matching
    the format used by simulation experiments.
    
    Parameters
    ----------
    experiment_results : Dict[str, Any]
        Results dictionary from run_real_comparison()
        Expected keys: 'comparative_analysis', 'scenarios', 'params', 'workload'
    """
    
    # Check if we have comparative analysis (all 3 scenarios completed)
    if 'comparative_analysis' not in experiment_results:
        print("[Warning] No comparative analysis found - skipping CSV/graph generation")
        return
    
    analysis = experiment_results['comparative_analysis']
    
    # Extract fidelity values
    fidelity_swap = analysis['fidelity_swap']
    fidelity_sqm_no_delay = analysis['fidelity_sqm_no_delay']
    fidelity_sqm_real = analysis['fidelity_sqm_real']
    
    # Create data directory if it doesn't exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ──────────────────────────────────────────────────────────
    # Generate CSV with scenario results
    # ──────────────────────────────────────────────────────────
    
    csv_path = os.path.join(data_dir, f'hardware_comparison_{timestamp}.csv')
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Scenario', 'Configuration', 'Fidelity', 'Job_ID', 'Notes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Scenario 1: SWAP
        writer.writerow({
            'Scenario': 1,
            'Configuration': 'SWAP Compiler (Baseline)',
            'Fidelity': f'{fidelity_swap:.4f}',
            'Job_ID': analysis['job_ids']['scenario_1'],
            'Notes': 'Static decoherence, no refresh'
        })
        
        # Scenario 2: SQM no delay
        writer.writerow({
            'Scenario': 2,
            'Configuration': 'SQM (No Delay - Routing Only)',
            'Fidelity': f'{fidelity_sqm_no_delay:.4f}',
            'Job_ID': analysis['job_ids']['scenario_2'],
            'Notes': 'time_idle_ns=0, measures routing overhead'
        })
        
        # Scenario 3: SQM real
        writer.writerow({
            'Scenario': 3,
            'Configuration': 'SQM (Real Timing - Calibrated)',
            'Fidelity': f'{fidelity_sqm_real:.4f}',
            'Job_ID': analysis['job_ids']['scenario_3'],
            'Notes': f"Backend-calibrated timing ({experiment_results.get('backend_info', {}).get('idle_time_ns', 'N/A')} ns)"
        })
    
    print(f"✓ Hardware comparison CSV saved: {csv_path}")
    
    # ──────────────────────────────────────────────────────────
    # Generate summary CSV
    # ──────────────────────────────────────────────────────────
    
    summary_path = os.path.join(data_dir, f'hardware_summary_{timestamp}.csv')
    
    with open(summary_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Experiment_Type', 'Hardware (3-Scenario)'])
        writer.writerow(['Backend', experiment_results.get('backend_info', {}).get('name', 'Unknown')])
        writer.writerow(['Workload', ','.join(experiment_results.get('workload', []))])
        writer.writerow([''])
        writer.writerow(['Scenario_1_Fidelity', f'{fidelity_swap:.4f}'])
        writer.writerow(['Scenario_2_Fidelity', f'{fidelity_sqm_no_delay:.4f}'])
        writer.writerow(['Scenario_3_Fidelity', f'{fidelity_sqm_real:.4f}'])
        writer.writerow([''])
        writer.writerow(['Delta_S2_vs_S1', f'{analysis["delta_s2_s1"]:+.4f}'])
        writer.writerow(['Delta_S3_vs_S1', f'{analysis["delta_s3_s1"]:+.4f}'])
        writer.writerow(['Delta_S3_vs_S2', f'{analysis["delta_s3_s2"]:+.4f}'])
        writer.writerow([''])
        writer.writerow(['Thesis_Status', analysis['thesis_validation']])
    
    print(f"✓ Hardware summary CSV saved: {summary_path}")
    
    # ──────────────────────────────────────────────────────────
    # Generate comparison graph
    # ──────────────────────────────────────────────────────────
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = ['Scenario 1\n(SWAP Baseline)', 'Scenario 2\n(SQM No Delay)', 'Scenario 3\n(SQM Real)']
    fidelities = [fidelity_swap, fidelity_sqm_no_delay, fidelity_sqm_real]
    colors = ['#A23B72', '#2E86AB', '#06A77D']
    
    bars = ax.bar(scenarios, fidelities, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, fid) in enumerate(zip(bars, fidelities)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{fid:.4f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Fidelity', fontsize=12, fontweight='bold')
    ax.set_title('Hardware Experiment: 3-Scenario Fidelity Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim((0, 1.05))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add thesis validation annotation
    thesis_color = '#06A77D' if analysis['thesis_validation'] == 'SUPPORTED' else '#E63946'
    ax.text(0.98, 0.05, f"Thesis: {analysis['thesis_validation']}", 
            transform=ax.transAxes, fontsize=11, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor=thesis_color, alpha=0.2))
    
    plt.tight_layout()
    
    graph_path = os.path.join(results_dir, f'hardware_comparison_graph_{timestamp}.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"✓ Hardware comparison graph saved: {graph_path}")
    plt.close()
    
    # ──────────────────────────────────────────────────────────
    # Print summary
    # ──────────────────────────────────────────────────────────
    
    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    print(f"✓ CSV Results:  {csv_path}")
    print(f"✓ CSV Summary:  {summary_path}")
    print(f"✓ Graph:        {graph_path}")
    print("=" * 70 + "\n")