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
    
    Adapts to 1, 2, or 3 scenarios. Creates files in data/ (CSV) and results/ (PNG) directories.
    
    Parameters
    ----------
    experiment_results : Dict[str, Any]
        Results dictionary from run_real_comparison()
        Expected keys: 'comparative_analysis', 'scenarios', 'params', 'workload'
    """
    
    # Check if we have comparative analysis
    if 'comparative_analysis' not in experiment_results:
        print("[Warning] No comparative analysis found - skipping CSV/graph generation")
        return
    
    analysis = experiment_results['comparative_analysis']
    
    # Create data directory if it doesn't exist
    # __file__ is in src/utils, so we need to go up 3 levels to reach the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ──────────────────────────────────────────────────────────
    # Detect available scenarios (1, 2, or 3)
    # ──────────────────────────────────────────────────────────
    
    available_scenarios = []
    scenario_data = {}
    
    # Check for each possible scenario
    if 'fidelity_swap' in analysis:
        available_scenarios.append(1)
        scenario_data[1] = {
            'name': 'Scenario 1 (SWAP Baseline)',
            'config': 'SWAP Compiler (Baseline)',
            'fidelity': analysis['fidelity_swap'],
            'job_id': analysis['job_ids'].get('scenario_1', 'N/A'),
            'notes': 'Static decoherence, no refresh'
        }
    
    if 'fidelity_sqm_no_delay' in analysis:
        available_scenarios.append(2)
        scenario_data[2] = {
            'name': 'Scenario 2 (SQM No Delay)',
            'config': 'SQM (No Delay - Routing Only)',
            'fidelity': analysis['fidelity_sqm_no_delay'],
            'job_id': analysis['job_ids'].get('scenario_2', 'N/A'),
            'notes': 'time_idle_ns=0, measures routing overhead'
        }
    
    if 'fidelity_sqm_real' in analysis:
        available_scenarios.append(3)
        scenario_data[3] = {
            'name': 'Scenario 3 (SQM Real)',
            'config': 'SQM (Real Timing - Calibrated)',
            'fidelity': analysis['fidelity_sqm_real'],
            'job_id': analysis['job_ids'].get('scenario_3', 'N/A'),
            'notes': f"Backend-calibrated timing ({experiment_results.get('backend_info', {}).get('idle_time_ns', 'N/A')} ns)"
        }
    
    num_scenarios = len(available_scenarios)
    print(f"\n[Saving Results] Detected {num_scenarios} scenario(s): {available_scenarios}")
    
    # ──────────────────────────────────────────────────────────
    # Generate CSV with scenario results
    # ──────────────────────────────────────────────────────────
    
    params = experiment_results.get('params', {})
    r_val = params.get('R', 'N/A')
    n_val = params.get('n', 'N/A')
    c_max_val = params.get('c_max', 'N/A')
    t_max_val = params.get('t_max_ns', 'N/A')
    shots_val = params.get('shots', 'N/A')
    init_state = params.get('initial_state', 'N/A')
    
    csv_path = os.path.join(data_dir, f'hardware_comparison_{timestamp}.csv')
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Scenario', 'Configuration', 'Fidelity', 'Job_ID', 'R', 'N', 'C_Max', 'T_Max_ns', 'Shots', 'Initial_State', 'Notes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write all available scenarios
        for scenario_num in available_scenarios:
            data = scenario_data[scenario_num]
            writer.writerow({
                'Scenario': scenario_num,
                'Configuration': data['config'],
                'Fidelity': f'{data["fidelity"]:.4f}',
                'Job_ID': data['job_id'],
                'R': r_val,
                'N': n_val,
                'C_Max': c_max_val,
                'T_Max_ns': t_max_val,
                'Shots': shots_val,
                'Initial_State': init_state,
                'Notes': data['notes']
            })
    
    print(f"✓ Hardware comparison CSV saved: {csv_path}")
    
    # ──────────────────────────────────────────────────────────
    # Generate summary CSV (adaptive to scenario count)
    # ──────────────────────────────────────────────────────────
    
    summary_path = os.path.join(data_dir, f'hardware_summary_{timestamp}.csv')
    
    with open(summary_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Experiment_Type', f'Hardware ({num_scenarios}-Scenario)'])
        writer.writerow(['Backend', experiment_results.get('backend_info', {}).get('name', 'Unknown')])
        writer.writerow(['Workload', ','.join(experiment_results.get('workload', []))])
        writer.writerow(['R', r_val])
        writer.writerow(['N', n_val])
        writer.writerow(['C_Max', c_max_val])
        writer.writerow(['T_Max_ns', t_max_val])
        writer.writerow(['Shots', shots_val])
        writer.writerow(['Initial_State', init_state])
        writer.writerow([''])
        
        # Write fidelities for available scenarios
        for scenario_num in available_scenarios:
            data = scenario_data[scenario_num]
            writer.writerow([f'Scenario_{scenario_num}_Fidelity', f'{data["fidelity"]:.4f}'])
        
        writer.writerow([''])
        
        # Write deltas if available
        if 'delta_s2_s1' in analysis:
            writer.writerow(['Delta_S2_vs_S1', f'{analysis["delta_s2_s1"]:+.4f}'])
        if 'delta_s3_s1' in analysis:
            writer.writerow(['Delta_S3_vs_S1', f'{analysis["delta_s3_s1"]:+.4f}'])
        if 'delta_s3_s2' in analysis:
            writer.writerow(['Delta_S3_vs_S2', f'{analysis["delta_s3_s2"]:+.4f}'])
        
        writer.writerow([''])
        writer.writerow(['Analysis_Status', analysis['thesis_validation']])
    
    print(f"✓ Hardware summary CSV saved: {summary_path}")
    
    # ──────────────────────────────────────────────────────────
    # Generate comparison graph (adaptive: 1, 2, or 3 bars)
    # ──────────────────────────────────────────────────────────
    
    # Prepare data for graph
    scenario_labels = []
    fidelity_values = []
    colors = ['#A23B72', '#2E86AB', '#06A77D']
    
    for i, scenario_num in enumerate(available_scenarios):
        scenario_labels.append(scenario_data[scenario_num]['name'])
        fidelity_values.append(scenario_data[scenario_num]['fidelity'])
    
    # Create figure with adaptive width
    fig_width = max(8, 4 + num_scenarios * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    # Select colors based on scenarios
    bar_colors = [colors[num - 1] for num in available_scenarios]
    
    bars = ax.bar(scenario_labels, fidelity_values, color=bar_colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, fid in zip(bars, fidelity_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{fid:.4f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Fidelity', fontsize=12, fontweight='bold')
    title = f'Hardware Experiment: {num_scenarios}-Scenario Fidelity Comparison'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim((0, 1.05))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add thesis validation annotation
    thesis_color = '#06A77D' if 'SUPPORTED' in analysis['thesis_validation'] else '#E63946'
    ax.text(0.98, 0.05, f"Status: {analysis['thesis_validation']}", 
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


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Workload Results Processor
# ══════════════════════════════════════════════════════════════════════════════

def save_hardware_multi_workload_results(all_workload_results: list, 
                                        backend_info: Dict[str, Any],
                                        params: Dict[str, Any]) -> None:
    """
    Save multi-workload hardware experiment results to CSV and generate comparison graph.
    
    Aggregates results from multiple workloads with multiple scenarios each.
    Creates unified CSV with all runs and comparison graph.
    
    Parameters
    ----------
    all_workload_results : list
        List of result dictionaries from run_real_comparison(), one per workload.
        Each includes 'workload_name', 'workload_data', and 'comparative_analysis'.
    backend_info : Dict[str, Any]
        Backend information from IBMHardwareBackend
    params : Dict[str, Any]
        Experiment parameters (R, n, c_max, t_max_ns, shots, initial_state)
    """
    
    if not all_workload_results:
        print("[Warning] No workload results - skipping CSV/graph generation")
        return
    
    # Create data directory if it doesn't exist
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract parameters
    r_val = params.get('R', 'N/A')
    n_val = params.get('n', 'N/A')
    c_max_val = params.get('c_max', 'N/A')
    t_max_val = params.get('t_max_ns', 'N/A')
    shots_val = params.get('shots', 'N/A')
    init_state = params.get('initial_state', 'N/A')
    
    # ──────────────────────────────────────────────────────────
    # Aggregate all results for CSV
    # ──────────────────────────────────────────────────────────
    
    all_csv_rows = []
    all_scenarios_data = []  # For graph generation
    
    for workload_result in all_workload_results:
        workload_name = workload_result.get('workload_name', 'Unknown')
        workload_data = workload_result.get('workload_data', [])
        
        if 'comparative_analysis' not in workload_result:
            continue
        
        analysis = workload_result['comparative_analysis']
        
        # Detect available scenarios for this workload
        available_scenarios = []
        scenario_data = {}
        
        if 'fidelity_swap' in analysis:
            available_scenarios.append(1)
            scenario_data[1] = {
                'config': 'SWAP Compiler (Baseline)',
                'fidelity': analysis['fidelity_swap'],
                'job_id': analysis['job_ids'].get('scenario_1', 'N/A'),
                'notes': 'Static decoherence, no refresh'
            }
        
        if 'fidelity_sqm_no_delay' in analysis:
            available_scenarios.append(2)
            scenario_data[2] = {
                'config': 'SQM (No Delay - Routing Only)',
                'fidelity': analysis['fidelity_sqm_no_delay'],
                'job_id': analysis['job_ids'].get('scenario_2', 'N/A'),
                'notes': 'time_idle_ns=0, measures routing overhead'
            }
        
        if 'fidelity_sqm_real' in analysis:
            available_scenarios.append(3)
            scenario_data[3] = {
                'config': 'SQM (Real Timing - Calibrated)',
                'fidelity': analysis['fidelity_sqm_real'],
                'job_id': analysis['job_ids'].get('scenario_3', 'N/A'),
                'notes': 'Backend-calibrated timing'
            }
        
        # Add rows for each scenario
        for scenario_num in available_scenarios:
            data = scenario_data[scenario_num]
            csv_row = {
                'Scenario': scenario_num,
                'Configuration': data['config'],
                'Fidelity': f'{data["fidelity"]:.4f}',
                'Job_ID': data['job_id'],
                'R': r_val,
                'N': n_val,
                'C_Max': c_max_val,
                'T_Max_ns': t_max_val,
                'Shots': shots_val,
                'Initial_State': init_state,
                'Notes': data['notes'],
                'Workload': ','.join(map(str, workload_data))
            }
            all_csv_rows.append(csv_row)
            
            # Add to graph data
            all_scenarios_data.append({
                'workload_name': workload_name,
                'scenario': scenario_num,
                'fidelity': data['fidelity'],
                'config': data['config']
            })
    
    # ──────────────────────────────────────────────────────────
    # Generate unified CSV
    # ──────────────────────────────────────────────────────────
    
    csv_path = os.path.join(data_dir, f'hardware_comparison_multi_{timestamp}.csv')
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Scenario', 'Configuration', 'Fidelity', 'Job_ID', 'R', 'N', 
                     'C_Max', 'T_Max_ns', 'Shots', 'Initial_State', 'Notes', 'Workload']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_csv_rows)
    
    print(f"\n✓ Multi-workload CSV saved: {csv_path}")
    
    # ──────────────────────────────────────────────────────────
    # Generate summary CSV
    # ──────────────────────────────────────────────────────────
    
    summary_path = os.path.join(data_dir, f'hardware_summary_multi_{timestamp}.csv')
    
    with open(summary_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Experiment_Type', f'Hardware (Multi-Workload)'])
        writer.writerow(['Backend', backend_info.get('backend_name', 'Unknown')])
        writer.writerow(['Total_Workloads', len(all_workload_results)])
        writer.writerow(['R', r_val])
        writer.writerow(['N', n_val])
        writer.writerow(['C_Max', c_max_val])
        writer.writerow(['T_Max_ns', t_max_val])
        writer.writerow(['Shots', shots_val])
        writer.writerow(['Initial_State', init_state])
        writer.writerow([''])
        
        # Workload list
        for idx, result in enumerate(all_workload_results, 1):
            workload_name = result.get('workload_name', f'Workload {idx}')
            writer.writerow([f'Workload_{idx}', workload_name])
        
        writer.writerow([''])
        writer.writerow(['Total_Rows', len(all_csv_rows)])
    
    print(f"✓ Multi-workload summary CSV saved: {summary_path}")
    
    # ──────────────────────────────────────────────────────────
    # Generate comparison graph (all workloads x all scenarios)
    # ──────────────────────────────────────────────────────────
    
    if all_scenarios_data:
        # Group data by workload first for easier visualization
        workload_dict = {}
        for item in all_scenarios_data:
            workload = item['workload_name']
            if workload not in workload_dict:
                workload_dict[workload] = {}
            scenario = item['scenario']
            workload_dict[workload][scenario] = item['fidelity']
        
        # Prepare data for grouped bar chart
        sorted_workloads = sorted(workload_dict.keys(), 
                                 key=lambda x: int(x.split()[1]))
        all_scenarios_present = sorted(set(item['scenario'] for item in all_scenarios_data))
        
        # Scenario labels and colors
        scenario_colors_map = {
            1: '#A23B72',  # SWAP Baseline (magenta)
            2: '#2E86AB',  # SQM No Delay (blue)
            3: '#06A77D'   # SQM Real (teal)
        }
        
        scenario_labels_map = {
            1: 'Scenario 1 (SWAP)',
            2: 'Scenario 2 (SQM No-Delay)',
            3: 'Scenario 3 (SQM Real)'
        }
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x_pos = np.arange(len(sorted_workloads))
        num_scenarios = len(all_scenarios_present)
        width = 0.25  # Width of each bar
        offset = (num_scenarios - 1) * width / 2
        
        # Plot bars for each scenario
        bars_list = []
        for idx, scenario_num in enumerate(sorted(all_scenarios_present)):
            scenario_fidelities = []
            for workload in sorted_workloads:
                fidelity = workload_dict[workload].get(scenario_num, 0)
                scenario_fidelities.append(fidelity)
            
            # Position bars
            bar_position = x_pos + (idx * width) - offset
            color = scenario_colors_map.get(scenario_num, '#999999')
            label = scenario_labels_map.get(scenario_num, f'Scenario {scenario_num}')
            
            bars = ax.bar(bar_position, scenario_fidelities, width, label=label, 
                         color=color, alpha=0.8, edgecolor='black', linewidth=1)
            bars_list.append(bars)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Set labels and title
        ax.set_xlabel('Workload', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fidelity', fontsize=12, fontweight='bold')
        ax.set_title('Hardware Experiment: Multi-Workload Fidelity Comparison', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        workload_labels = [f'WL{i+1}' for i in range(len(sorted_workloads))]
        ax.set_xticklabels(workload_labels, fontsize=11)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim((0, 1.05))
        
        plt.tight_layout()
        
        graph_path = os.path.join(results_dir, f'hardware_comparison_multi_graph_{timestamp}.png')
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        print(f"✓ Multi-workload comparison graph saved: {graph_path}")
        plt.close()
    
    
    # ──────────────────────────────────────────────────────────
    # Print summary
    # ──────────────────────────────────────────────────────────
    
    print("\n" + "=" * 70)
    print("MULTI-WORKLOAD RESULTS SAVED")
    print("=" * 70)
    print(f"✓ CSV Results:  {csv_path}")
    print(f"✓ CSV Summary:  {summary_path}")
    if all_scenarios_data:
        print(f"✓ Graph:        {graph_path}")
    print(f"✓ Total Runs:   {len(all_csv_rows)}")
    print(f"✓ Workloads:    {len(all_workload_results)}")
    print("=" * 70 + "\n")