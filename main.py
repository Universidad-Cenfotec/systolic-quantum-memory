#!/usr/bin/env python3
# ============================================================================
# SQM Research Project - Unified Main Entry Point
# Systolic Quantum Teleportation Memory
# ============================================================================
"""
Unified CLI entry point supporting both simulator and hardware execution.
Configuration is loaded from default.yaml and can be overridden by CLI arguments.

USAGE EXAMPLES:

  Run with default configuration:
    python main.py

  Custom configuration file:
    python main.py --config custom.yaml

  Override specific parameters:
    python main.py --R 2 --n 1 --shots 8000

  Delay sweep workload:
    python main.py --workload delay

  Hardware mode:
    python main.py --backend hardware --scenario 1,3
"""

import sys
import os
import yaml
from typing import List, Tuple, Dict, Any, Optional

# Configure matplotlib BEFORE any other imports
import matplotlib
matplotlib.use('Agg')

# Ensure project root in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.comparison import run_full_comparison, run_real_comparison
from src.backends.aer_simulator_backend import AerSimulatorBackend
from src.backends.ibm_hardware_backend import IBMHardwareBackend
from src.utils.hardware_results_processor import save_hardware_multi_workload_results


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_file: str = "default.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


# ============================================================================
# Workload Generation
# ============================================================================

def generate_workloads_from_config(config: Dict[str, Any], workload_type: str) -> List[Tuple[str, List[str]]]:
    """Generate workloads based on configuration."""
    
    workload_cfg = config['workloads']
    
    if workload_type == "standard":
        workloads = []
        count = workload_cfg['standard']['count']
        for i in range(1, count + 1):
            read_count = 2 * i
            workload = (
                ["READ_0"] * read_count +
                ["WRITE_0"] +
                ["READ_0"] * read_count +
                ["READ_0"]
            )
            workloads.append((f"Workload {i} ({len(workload)} instr)", workload))
        return workloads
    
    elif workload_type == "delay":
        delays = workload_cfg['delay']['delays_ns']
        return [
            (f"Delay {d}ns", ["WRITE_0", f"IDLE_{d}", "READ_0"])
            for d in delays
        ]
    
    elif workload_type == "custom":
        workloads = []
        for wl in workload_cfg['custom']['workloads']:
            workloads.append((wl['name'], wl['instructions']))
        return workloads
    
    else:
        raise ValueError(f"Unknown workload type: {workload_type}")


def parse_scenarios(scenario_str: Optional[str]) -> Optional[List[int]]:
    """Parse scenarios argument (e.g., '1,3' -> [1, 3])."""
    if not scenario_str:
        return None
    try:
        return sorted([int(s.strip()) for s in scenario_str.split(',')])
    except ValueError:
        raise ValueError(f"Invalid format: {scenario_str}. Use '1,3' for scenarios 1 and 3")


# ============================================================================
# Simulator Mode
# ============================================================================

def run_simulator_mode(config: Dict[str, Any]):
    """Execute simulator-based comparative analysis."""
    
    print("\n" + "=" * 80)
    print("SQM RESEARCH - SIMULATOR MODE")
    print("=" * 80)
    
    cfg = config['simulator']
    
    # Backend configuration
    print("\n[Phase 1: Backend Configuration]")
    if not cfg['thermal']['use_default']:
        print("  Mode: CUSTOM thermal parameters")
        backend = AerSimulatorBackend(
            t1_ns=cfg['thermal']['T1_ns'],
            t2_ns=cfg['thermal']['T2_ns'],
            idle_time_ns=cfg['thermal']['idle_time_ns']
        )
        print(f"  T1: {cfg['thermal']['T1_ns']} ns ({cfg['thermal']['T1_ns']/1000:.1f} microseconds)")
        print(f"  T2: {cfg['thermal']['T2_ns']} ns ({cfg['thermal']['T2_ns']/1000:.1f} microseconds)")
        print(f"  Idle: {cfg['thermal']['idle_time_ns']} ns ({cfg['thermal']['idle_time_ns']/1000:.1f} microseconds)")
    else:
        print("  Mode: DEFAULT thermal parameters")
        backend = AerSimulatorBackend()
        print("  T1: 149149 ns (149.1 microseconds)")
        print("  T2: 38194 ns (38.2 microseconds)")
        print("  Idle: 7000 ns (7.0 microseconds)")
    
    # Compiler configuration
    print("\n[Phase 2: Compiler Configuration]")
    print(f"  Memory Registers (R): {cfg['compiler']['R']}")
    print(f"  Qubits per Register (n): {cfg['compiler']['n']}")
    print(f"  Gate Cost Threshold: {cfg['compiler']['c_max']}")
    print(f"  Time Threshold: {cfg['compiler']['t_max_ns']} ns")
    print(f"  Shots: {cfg['execution']['shots']}")
    state_labels = {0: "|0>", 1: "|1>", 2: "|+> (H)", 3: "|-> (XH)"}
    print(f"  Target State: {state_labels.get(cfg['execution']['initial_state'], 'unknown')}")
    
    # Workload selection
    print("\n[Phase 3: Workload Selection]")
    workload_type = cfg['execution']['workload_type']
    print(f"  Type: {workload_type.upper()}")
    
    workloads = generate_workloads_from_config(config, workload_type)
    print(f"  Total: {len(workloads)} workloads")
    for name, wl in workloads[:2]:
        print(f"    - {name}")
    if len(workloads) > 2:
        print(f"    - ... and {len(workloads) - 2} more")
    
    # Run experiment
    print("\n[Phase 4: Comparative Analysis]")
    print("=" * 80)
    
    run_full_comparison(
        R=cfg['compiler']['R'],
        n=cfg['compiler']['n'],
        c_max=cfg['compiler']['c_max'],
        t_max_ns=cfg['compiler']['t_max_ns'],
        shots=cfg['execution']['shots'],
        workloads=workloads,
        initial_state=cfg['execution']['initial_state'],
        backend_manager=backend,
        flow=config['advanced']['flow_mode']
    )
    
    print("\n[OK] Simulator mode completed successfully")


# ============================================================================
# Hardware Mode
# ============================================================================

def run_hardware_mode(config: Dict[str, Any]):
    """Execute hardware-based comparative analysis on IBM Quantum."""
    
    print("\n" + "=" * 80)
    print("SQM RESEARCH - HARDWARE MODE".center(80))
    print("Systolic Quantum Teleportation Memory".center(80))
    print("=" * 80 + "\n")
    
    cfg = config['hardware']
    
    # Phase 1: Backend connection
    print("PHASE 1: IBM QUANTUM BACKEND CONNECTION")
    print("=" * 80)
    print("\n[Initializing QiskitRuntimeService]")
    print("  > Loading credentials from local cache")
    print("  > Connecting to IBM Quantum...")
    
    try:
        backend = IBMHardwareBackend(
            backend_name=cfg['backend_name'],
            channel=cfg['channel']
        )
        backend_info = backend.get_backend_info()
        
        print(f"\n[OK] Connected successfully!")
        print(f"  Backend: {backend_info['backend_name']}")
        print(f"  Qubits: {backend_info['num_qubits']}")
        print(f"  Basis Gates: {backend_info['basis_gates']}")
        print(f"  Idle Timing: {backend_info['idle_time_ns']} ns")
        
    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}")
        print(f"\n[Troubleshooting]")
        print(f"  1. Verify IBM credentials: python -c 'from qiskit_ibm_runtime import QiskitRuntimeService'")
        print(f"  2. Check internet connection")
        print(f"  3. Verify account quota (minimum 10 min/month)")
        return
    
    # Phase 2: Configuration
    print("\n\nPHASE 2: CONFIGURATION")
    print("=" * 80)
    
    print(f"\n[Compiler Parameters]")
    print(f"  R={cfg['compiler']['R']}, n={cfg['compiler']['n']}, c_max={cfg['compiler']['c_max']}, t_max={cfg['compiler']['t_max_ns']} ns")
    print(f"  Shots: {cfg['execution']['shots']}")
    state_labels = {0: "|0>", 1: "|1>", 2: "|+> (H)", 3: "|-> (XH)"}
    print(f"  Target State: {state_labels.get(cfg['execution']['initial_state'], 'unknown')}")
    
    # Workloads
    workload_type = cfg['execution']['workload_type']
    workloads = generate_workloads_from_config(config, workload_type)
    
    print(f"\n[Workloads]")
    print(f"  Total: {len(workloads)}")
    for i, (name, wl) in enumerate(workloads[:2], 1):
        print(f"  {i}. {name}: {len(wl)} instructions")
    if len(workloads) > 2:
        print(f"  ... {len(workloads) - 2} more")
    
    # Scenarios
    scenarios = parse_scenarios(cfg['execution']['scenarios'])
    print(f"\n[Scenarios]")
    if scenarios:
        print(f"  Selected: {scenarios}")
    else:
        print(f"  Mode: ALL (1, 2, 3)")
    
    print(f"\n[OK] Configuration validated")
    print(f"  Qubits required: ~{cfg['compiler']['n'] * (cfg['compiler']['R'] + 2)}")
    print(f"  Qubits available: {backend_info['num_qubits']}")
    
    # Phase 3: Execution
    print(f"\n\nPHASE 3: MULTI-WORKLOAD EXECUTION")
    print("=" * 80)
    
    all_results = []
    
    try:
        for idx, (workload_name, workload_data) in enumerate(workloads, 1):
            print(f"\n{'-' * 80}")
            print(f"WORKLOAD {idx}/{len(workloads)}: {workload_name}")
            print(f"{'-' * 80}")
            
            results = run_real_comparison(
                R=cfg['compiler']['R'],
                n=cfg['compiler']['n'],
                c_max=cfg['compiler']['c_max'],
                t_max_ns=cfg['compiler']['t_max_ns'],
                shots=cfg['execution']['shots'],
                workload=workload_data,
                backend_manager=backend,
                initial_state=cfg['execution']['initial_state'],
                scenarios=scenarios,
                flow=config['advanced']['flow_mode']
            )
            
            if results:
                results['workload_name'] = workload_name
                results['workload_data'] = workload_data
                all_results.append(results)
                print(f"\n[OK] Workload {idx} completed")
            else:
                print(f"\n[ERROR] Workload {idx} failed")
        
        # Phase 4: Results
        print(f"\n\nPHASE 4: SAVE RESULTS")
        print("=" * 80)
        
        if all_results:
            save_hardware_multi_workload_results(
                all_workload_results=all_results,
                backend_info=backend_info,
                params={
                    'R': cfg['compiler']['R'],
                    'n': cfg['compiler']['n'],
                    'c_max': cfg['compiler']['c_max'],
                    't_max_ns': cfg['compiler']['t_max_ns'],
                    'shots': cfg['execution']['shots'],
                    'initial_state': cfg['execution']['initial_state']
                }
            )
            
            print("\n[OK] HARDWARE EXPERIMENT COMPLETE")
            print("  All results saved to results/")
        else:
            print("\n[ERROR] No results to save")
        
    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Load config and dispatch to execution mode."""
    
    # Load configuration
    config_file = 'default.yaml'
    print("[Loading configuration]")
    config = load_config(config_file)
    print(f"  > Loaded from {config_file}")
    
    backend = config['global']['backend']
    print(f"  > Backend: {backend}")
    
    # Auto-adjust shots for hardware if using default
    if backend == 'hardware' and config['hardware']['execution']['shots'] == 4000:
        config['hardware']['execution']['shots'] = 1024
        print("  > Auto-adjusted hardware shots to 1024 (from 4000)")
    
    print()
    
    # Dispatch
    if backend == 'simulator':
        run_simulator_mode(config)
    else:
        run_hardware_mode(config)


if __name__ == "__main__":
    main()
