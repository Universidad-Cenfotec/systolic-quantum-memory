import os
import csv
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from qiskit_ibm_runtime.fake_provider import FakeKyiv


# =============================================================================
# Exponential decay model: F(t) = A * exp(-t/tau) + B
# =============================================================================

def exponential_decay(t: float, A: float, tau: float, B: float) -> float:
    """
    Exponential decay model.
    
    F(t) = A * exp(-t/tau) + B
    
    where tau = 1/lambda is the decay time constant.
    """
    return A * np.exp(-t / tau) + B


# =============================================================================
# Main class: TmaxValidator
# =============================================================================

class TmaxValidator:
    """
    Measure fidelity decay using ONLY Identity (ID) gates.
    
    Simple method to characterize T1/T2 decoherence:
    - Prepare |1⟩ state
    - Apply N identity gates (each triggers thermal relaxation)
    - Measure population in |1⟩ state
    - See if decay is exponential (T1/T2 model) or linear
    """

    def __init__(self, N: int = 1, T1_ns: float = 100_000, T2_ns: float = 50_000) -> None:
        """
        Initialize validator.
        
        Args:
            N: Number of qubits per register
            T1_ns: T1 relaxation time in nanoseconds
            T2_ns: T2 dephasing time in nanoseconds
        """
        self.N = N
        self.d = 2 ** self.N
        
        # T1/T2 parameters
        self.T1_ns = T1_ns
        self.T2_ns = T2_ns
        self.ID_DURATION_NS = 35.5  # IBM ID gate duration
        
        # Use FakeKyiv backend (realistic IBM noise model)
        self.backend = FakeKyiv()
        
        # Create thermal relaxation noise model
        self.noise_model = self._create_thermal_noise()
        
        # Create simulator with noise model
        self.simulator = AerSimulator(noise_model=self.noise_model, method='automatic')
        
        # For storing measurement data
        self.delay_list: list[float] = []
        self.fidelity_list: list[float] = []

    def _create_thermal_noise(self) -> NoiseModel:
        """
        Create thermal relaxation noise model.
        Adds T1/T2 decoherence errors to ID gates only.
        """
        noise_model = NoiseModel()
        
        # Convert nanoseconds to seconds
        T1_sec = self.T1_ns * 1e-9
        T2_sec = self.T2_ns * 1e-9
        gate_duration_ns = 35.5  # ID gate duration (for noise injection)
        gate_duration_sec = gate_duration_ns * 1e-9
        
        # Create thermal relaxation error
        relax_error = thermal_relaxation_error(T1_sec, T2_sec, gate_duration_sec)
        
        # Add error to ID gates only (no active gate errors)
        noise_model.add_all_qubit_quantum_error(relax_error, "id")
        
        return noise_model

    def measure_fidelity_with_ids(
        self,
        num_ids: int,
        shots: int = 4000
    ) -> float:
        """
        Measure fidelity (survival probability in |1⟩) after N identity gates.
        
        Args:
            num_ids: Number of identity gates to apply
            shots: Measurement shots
        
        Returns:
            Fidelity = P(|111...1⟩) = count of |111...1⟩ / shots
        """
        qc = QuantumCircuit(self.N)
        
        # Initialize to |1⟩^N
        for i in range(self.N):
            qc.x(i)
        
        qc.barrier()
        
        # Apply ID gates (pure idle with thermal relaxation)
        for i in range(self.N):
            for _ in range(num_ids):
                qc.id(i)  # 35.5 ns idle + thermal relaxation noise
        
        qc.barrier()
        qc.measure_all()
        
        # Transpile for the backend (preserve structure with optimization_level=0)
        qc_t = transpile(qc, backend=self.backend, optimization_level=0)
        job = self.simulator.run(qc_t, shots=shots)
        counts = job.result().get_counts()
        
        target_state = "1" * self.N
        fidelity = counts.get(target_state, 0) / shots
        
        return fidelity

    def run_id_sweep(
        self,
        num_ids_list: list[int],
        shots: int = 4000,
        plot_path: str | None = "results/id_decay.png",
        csv_path: str | None = None
    ) -> None:
        """
        Sweep over different numbers of ID gates and measure fidelity decay.
        
        Args:
            num_ids_list: List of number of ID gates to apply
            shots: Measurement shots per point
            plot_path: Path to save plot
            csv_path: Path to save measurement data as CSV (default: data/id_sweep_<timestamp>.csv)
        """
        print("=" * 70)
        print("  SQM -- ID Gate Sweep: Measuring T1/T2 Decay")
        print("=" * 70)
        print(f"\n  N_qubits    = {self.N}")
        print(f"  T1          = {self.T1_ns / 1000:.1f} µs")
        print(f"  T2          = {self.T2_ns / 1000:.1f} µs")
        print(f"  ID duration = {self.ID_DURATION_NS} ns")
        print(f"  shots       = {shots}\n")
        
        self.delay_list = []
        self.fidelity_list = []
        
        for num_ids in num_ids_list:
            delay_ns = num_ids * self.ID_DURATION_NS
            fidelity = self.measure_fidelity_with_ids(num_ids, shots=shots)
            
            self.delay_list.append(delay_ns)
            self.fidelity_list.append(fidelity)
            
            print(f"  num_ids={num_ids:4d}  delay={delay_ns:8.1f} ns  F={fidelity:.6f}")
        
        print()
        
        # Save to CSV
        if csv_path is None:
            # Generate default CSV path with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"data/id_sweep_N{self.N}_T1_{self.T1_ns:.0f}_T2_{self.T2_ns:.0f}_{timestamp}.csv"
        
        self._save_to_csv(csv_path, num_ids_list)
        
        # Plot
        if plot_path:
            self._plot_decay(plot_path)

    def _save_to_csv(self, csv_path: str, num_ids_list: list[int]) -> None:
        """
        Save measurement data to CSV file.
        
        Args:
            csv_path: Path to save CSV
            num_ids_list: List of number of ID gates applied
        """
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['num_ids', 'delay_ns', 'delay_us', 'fidelity', 'N_qubits', 'T1_ns', 'T2_ns', 'ID_duration_ns']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write data rows
            for num_ids, delay_ns, fidelity in zip(num_ids_list, self.delay_list, self.fidelity_list):
                writer.writerow({
                    'num_ids': num_ids,
                    'delay_ns': f"{delay_ns:.1f}",
                    'delay_us': f"{delay_ns / 1000:.3f}",
                    'fidelity': f"{fidelity:.6f}",
                    'N_qubits': self.N,
                    'T1_ns': f"{self.T1_ns:.0f}",
                    'T2_ns': f"{self.T2_ns:.0f}",
                    'ID_duration_ns': self.ID_DURATION_NS
                })
        
        print(f"  [CSV] Saved to: {csv_path}")

    def _plot_decay(self, plot_path: str) -> None:
        """Plot the measured fidelity decay."""
        os.makedirs(os.path.dirname(plot_path) if os.path.dirname(plot_path) else ".", exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        delay_arr = np.array(self.delay_list)
        fidelity_arr = np.array(self.fidelity_list)
        
        # Convert delays to microseconds for better readability
        delay_us = delay_arr / 1000.0
        
        # Plot measured data
        ax.scatter(delay_us, fidelity_arr, s=120, color='steelblue', zorder=5, 
                   marker='o', edgecolor='darkblue', linewidth=1.5, label='Measured F(t)')
        ax.plot(delay_us, fidelity_arr, 'o-', color='steelblue', alpha=0.4, linewidth=2)
        
        # Add exponential and linear fits for comparison
        if len(delay_arr) > 1 and delay_arr.max() > 0:
            # Prepare dense delay array for plotting
            delay_dense = np.linspace(0, delay_arr.max(), 300)
            delay_dense_us = delay_dense / 1000.0
            
            # Exponential fit: F(t) = A * exp(-t/tau)
            try:
                popt_exp = np.polyfit(delay_arr, np.log(np.maximum(fidelity_arr - 0.001, 1e-6)), 1)
                tau_exp = -1 / popt_exp[0] if popt_exp[0] != 0 else float('inf')
                A_exp = np.exp(popt_exp[1])
                f_exp = A_exp * np.exp(-delay_dense / tau_exp)
                ax.plot(delay_dense_us, f_exp, '--', color='crimson', linewidth=2.5, 
                        label=f'Exp fit: τ={tau_exp:.1f} ns (A={A_exp:.3f})')
            except Exception as e:
                print(f"    [WARNING] Exponential fit failed: {e}")
            
            # Linear fit: F(t) = a*t + b
            popt_lin = np.polyfit(delay_arr, fidelity_arr, 1)
            f_lin = np.polyval(popt_lin, delay_dense)
            ax.plot(delay_dense_us, f_lin, ':', color='green', linewidth=2.5,
                    label=f'Linear fit: slope={popt_lin[0]:.2e} /ns')
        
        # Theory curve based on T1/T2
        tau_theory = 1.0 / (1.0 / self.T2_ns + 1.0 / (2 * self.T1_ns))
        if delay_arr.max() > 0:
            delay_dense = np.linspace(0, delay_arr.max(), 300)
            delay_dense_us = delay_dense / 1000.0
            f_theory = np.exp(-delay_dense / tau_theory)
            ax.plot(delay_dense_us, f_theory, '-.', color='orange', linewidth=2.5, alpha=0.8,
                    label=f'Theory (τ={tau_theory:.1f} ns)')
        
        # Formatting
        ax.set_xlabel('Delay time (µs)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Fidelity F(t) = P(|1⟩)', fontsize=13, fontweight='bold')
        ax.set_title(f'ID Gate Decay Analysis | N={self.N}, T1={self.T1_ns/1000:.0f}µs, T2={self.T2_ns/1000:.0f}µs, Gate={self.ID_DURATION_NS}ns',
                     fontsize=13, fontweight='bold')
        
        # Info box with parameters
        info_text = (
            f'T1 = {self.T1_ns/1000:.1f} µs\n'
            f'T2 = {self.T2_ns/1000:.1f} µs\n'
            f'τ_theory = {tau_theory:.1f} ns\n'
            f'N_points = {len(delay_arr)}'
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(-0.05, 1.1)
        
        # Add secondary x-axis for gate count
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim()[0] * 1000 / self.ID_DURATION_NS, 
                      ax.get_xlim()[1] * 1000 / self.ID_DURATION_NS)
        ax2.set_xlabel('Number of ID gates', fontsize=12, fontweight='bold')
        
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  [PLOT] Saved to: {plot_path}")

    def analyze_decay_type(self) -> dict:
        """
        Compare exponential vs linear decay to determine which model fits better.
        
        Returns:
            Dictionary with analysis results
        """
        if len(self.delay_list) < 3:
            print("  [ERROR] Need at least 3 data points")
            return {}
        
        delay_arr = np.array(self.delay_list)
        fidelity_arr = np.array(self.fidelity_list)
        
        # Linear fit
        popt_lin = np.polyfit(delay_arr, fidelity_arr, 1)
        f_lin_pred = np.polyval(popt_lin, delay_arr)
        ss_res_lin = np.sum((fidelity_arr - f_lin_pred) ** 2)
        ss_tot = np.sum((fidelity_arr - np.mean(fidelity_arr)) ** 2)
        r2_lin = 1 - (ss_res_lin / ss_tot) if ss_tot > 0 else 0
        
        # Exponential fit (simple log-linear)
        # Fit log(F) = -t/tau + log(A)
        fidelity_safe = np.maximum(fidelity_arr - 0.001, 1e-6)
        try:
            popt_exp = np.polyfit(delay_arr, np.log(fidelity_safe), 1)
            tau_exp = -1 / popt_exp[0] if popt_exp[0] != 0 else float('inf')
            f_exp_pred = np.exp(np.polyval(popt_exp, delay_arr))
            ss_res_exp = np.sum((fidelity_arr - f_exp_pred) ** 2)
            r2_exp = 1 - (ss_res_exp / ss_tot) if ss_tot > 0 else 0
        except:
            tau_exp = 0
            r2_exp = 0
        
        print("\n" + "=" * 70)
        print("  Decay Analysis: Exponential vs Linear")
        print("=" * 70)
        print(f"\n  Linear Model: F(t) = a*t + b")
        print(f"    Slope (a)      = {popt_lin[0]:.2e}")
        print(f"    Intercept (b) = {popt_lin[1]:.6f}")
        print(f"    R² = {r2_lin:.4f}")
        
        print(f"\n  Exponential Model: F(t) = A * exp(-t/τ)")
        print(f"    τ (decay time) = {tau_exp:.2f} ns")
        print(f"    R² = {r2_exp:.4f}")
        
        print(f"\n  Theory Prediction (T1/T2):")
        # tau_eff = 1 / (1/T2 + 1/(2*T1))
        tau_theory = 1.0 / (1.0 / self.T2_ns + 1.0 / (2 * self.T1_ns))
        print(f"    τ_theory = {tau_theory:.2f} ns")
        print(f"    (from 1/τ = 1/T2 + 1/(2*T1))")
        
        if r2_exp > r2_lin:
            print(f"\n  [RESULT] EXPONENTIAL decay dominates (ΔR²={r2_exp - r2_lin:.4f})")
        else:
            print(f"\n  [RESULT] LINEAR decay dominates (ΔR²={r2_lin - r2_exp:.4f})")
        
        print("=" * 70)
        
        return {
            'linear_r2': float(r2_lin),
            'exp_r2': float(r2_exp),
            'exp_tau': float(tau_exp),
            'theory_tau': float(tau_theory)
        }



# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    
    # =========================================================================
    # SQM Configuration: 10th Percentile (Worst-Case) Parameters
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("  SQM Tmax Validator: Worst-Case Coherence Analysis")
    print("=" * 70)
    print("\n  Configuration:")
    print("    Target Fidelity:    0.75")
    print("    Safety Percentile:  10th percentile (worst-case)")
    print("\n  Worst-Case Parameters:")
    print("    T1 (10th percentile):  149.149 μs (149149 ns)")
    print("    T2 (10th percentile):  38.194 μs (38194 ns)")
    print("    Critical limit: T_critical = 38194 ns")
    print("\n" + "=" * 70 + "\n")
    
    # TEST 1: Single Qubit, ID Gate Sweep with Worst-Case Parameters
    print("\n" + "=" * 70)
    print("  TEST 1: Single Qubit, ID Gate Sweep (Worst-Case T1/T2)")
    print("=" * 70)
    
    N = 1
    T1_ns = 192.566  # 10th percentile
    T2_ns = 35.887   # 10th percentile (critical limiting time)
    target_fidelity = 0.75
    
    validator = TmaxValidator(N=N, T1_ns=T1_ns, T2_ns=T2_ns)
    
    # Calculate tau_theory to set sweep range appropriately
    tau_theory = 1.0 / (1.0 / T2_ns + 1.0 / (2 * T1_ns))
    print(f"\n  Theory Prediction:")
    print(f"    τ_theory = {tau_theory:.2f} ns")
    print(f"    T_MAX (target F={target_fidelity}) ≈ {-tau_theory * np.log(target_fidelity):.2f} ns")
    
    # Sweep from 0 to 5*tau_theory for comprehensive coverage
    max_gates = int(5 * tau_theory / 35.5)
    num_ids_log = np.logspace(0, np.log10(max_gates + 1), 25, dtype=int)
    num_ids_list = sorted(list(set(num_ids_log)) + [0])
    
    print(f"\n  Sweep Configuration:")
    print(f"    Gate duration: {validator.ID_DURATION_NS} ns")
    print(f"    Max gates:     {max_gates}")
    print(f"    Max delay:     {max_gates * validator.ID_DURATION_NS:.1f} ns ({max_gates * validator.ID_DURATION_NS / 1000:.2f} μs)")
    print(f"    Data points:   {len(num_ids_list)}\n")
    
    validator.run_id_sweep(
        num_ids_list,
        shots=4000,
        plot_path="results/id_sweep_worst_case.png"
    )
    
    # Analyze decay type
    results = validator.analyze_decay_type()
    
    print("\n" + "=" * 70)
    print("  TEST 2: Comparison with Different Scenarios")
    print("=" * 70)
