"""
Tmax Calculator Module for Systolic Quantum Memory (SQM).

This module implements worst-case scenario analysis for quantum coherence times,
protecting against relaxation fluctuations using percentile-based statistical methods.

Reference: Berritta et al., 2026 - Spatial distribution of coherence times on quantum chips.
"""

import numpy as np
import math
from qiskit_ibm_runtime.fake_provider import FakeKyiv

try:
    from src.time_calculation.ibm_backend_helper import get_ibm_backend
except ModuleNotFoundError:
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from src.time_calculation.ibm_backend_helper import get_ibm_backend


class TMaxCalculator:
    """
    Calculates maximum allowed idle time (Tmax) for quantum states in SQM.
    
    Uses worst-case scenario approach with percentile analysis to ensure
    fidelity preservation across the entire quantum processor.
    """
    
    def __init__(self, target_fidelity: float = 0.75, safety_percentile: int = 10, backend=None):
        """
        Initialize the Tmax calculator with coherence time statistics.
        
        Args:
            target_fidelity: Target fidelity for quantum state preservation (default: 0.75).
            safety_percentile: Percentile threshold for worst-case analysis (default: 10).
            backend: Optional external backend (e.g. real IBM). If None, uses FakeKyiv.
        """
        self.target_fidelity = target_fidelity
        self.safety_percentile = safety_percentile
        
        # Initialize backend and extract coherence properties
        if backend is not None:
            self.backend = backend
            self.is_ibm = True
            print(f"[TMaxCalculator] Using IBM hardware backend: {self.backend.name}")
        else:
            self.backend = FakeKyiv()
            self.is_ibm = False
        
        properties = self.backend.properties()
        
        # Extract T1 and T2 times from all qubits
        t1_list = []
        t2_list = []
        
        # Iterate through all qubits using properties methods
        num_qubits = len(properties.qubits)
        for qubit_index in range(num_qubits):
            # Extract T1 (in seconds from Qiskit, convert to nanoseconds)
            t1_value = properties.t1(qubit_index)
            if t1_value is not None and t1_value > 0:
                t1_list.append(t1_value * 1e9)  # Convert to nanoseconds
            
            # Extract T2 (in seconds from Qiskit, convert to nanoseconds)
            t2_value = properties.t2(qubit_index)
            if t2_value is not None and t2_value > 0:
                t2_list.append(t2_value * 1e9)  # Convert to nanoseconds
        
        # Apply percentile-based worst-case analysis
        self.t1_worst = float(np.percentile(t1_list, self.safety_percentile))
        self.t2_worst = float(np.percentile(t2_list, self.safety_percentile))
        
        # Determine the critical (limiting) coherence time
        self.t_critical = min(self.t1_worst, self.t2_worst)
    
    def calculate_tmax(self) -> float:
        """
        Calculate maximum allowed idle time using exponential decay formula.
        
        Formula: T_MAX = -T_critical * ln(target_fidelity)
        
        Returns:
            float: Maximum idle time in nanoseconds.
        """
        tmax = float(-self.t_critical * math.log(self.target_fidelity))
        return tmax
    
    def print_thermodynamic_report(self) -> None:
        """
        Print a comprehensive thermodynamic report of coherence time analysis.
        
        Displays worst-case T1 and T2 times, critical time, and calculated Tmax
        in both nanoseconds and microseconds for practical reference.
        """
        tmax = self.calculate_tmax()
        
        # Convert nanoseconds to microseconds for readability
        t1_worst_us = self.t1_worst / 1000
        t2_worst_us = self.t2_worst / 1000
        tmax_us = tmax / 1000
        
        print("\n" + "="*70)
        print("SYSTOLIC QUANTUM MEMORY (SQM)")
        print("Thermodynamic Coherence Time Report")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Target Fidelity:    {self.target_fidelity:.2f}")
        print(f"  Safety Percentile:  {self.safety_percentile}th percentile (worst-case)")
        print(f"\nWorst-Case Coherence Times (Percentile {self.safety_percentile}):")
        print(f"  T1 (Amplitude Damping):  {t1_worst_us:.3f} μs ({self.t1_worst:.0f} ns)")
        print(f"  T2 (Phase Damping):      {t2_worst_us:.3f} μs ({self.t2_worst:.0f} ns)")
        print(f"\nCritical Limiting Time:")
        print(f"  T_critical:  {self.t_critical:.0f} ns ({self.t_critical/1000:.3f} μs)")
        print(f"\nMaximum Idle Time (Tmax):")
        print(f"  T_MAX:  {tmax:.0f} ns ({tmax_us:.3f} μs)")
        print("="*70 + "\n")


def main():
    """
    Main execution function for Tmax Calculator.
    Demonstrates worst-case scenario analysis for SQM.
    """
    # =========================================================================
    # BACKEND MODE: "default" = FakeKyiv simulator | "IBM" = real IBM hardware
    # =========================================================================
    backend_mode = "default"  # Change to "IBM" to run on real IBM hardware

    print("\n🔬 Initializing SQM Tmax Calculator...")
    
    # Create calculator with appropriate backend
    if backend_mode == "IBM":
        ibm_backend = get_ibm_backend("ibm_kingston")
        calculator = TMaxCalculator(target_fidelity=0.75, safety_percentile=10, backend=ibm_backend)
    else:
        calculator = TMaxCalculator(target_fidelity=0.75, safety_percentile=10)
    
    # Calculate and display results
    tmax = calculator.calculate_tmax()
    
    # Print comprehensive report
    calculator.print_thermodynamic_report()
    
    print(f"✅ Calculation Complete!")
    print(f"   Maximum allowed idle time (Tmax): {tmax:.0f} ns")


if __name__ == "__main__":
    main()
