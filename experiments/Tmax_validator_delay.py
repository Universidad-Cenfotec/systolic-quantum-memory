import math
import os
import csv
from datetime import datetime
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeKyiv

try:
    from experiments.utils.ibm_backend_helper import get_ibm_backend, run_on_ibm
    from src.utils.pauli_twirling import PauliTwirler
    from src.mitigation import ReadoutMitigator, ZNEFolder, ZNEExtrapolator
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from experiments.utils.ibm_backend_helper import get_ibm_backend, run_on_ibm
    from src.utils.pauli_twirling import PauliTwirler
    from src.mitigation import ReadoutMitigator, ZNEFolder, ZNEExtrapolator


# =============================================================================
# Exponential decay model: F(t) = A * exp(-t/tau) + B
# =============================================================================

def exp_decay_model(t: float, A: float, tau: float, B: float) -> float:
    """
    Exponential decay model for idle decoherence.

    F(t) = A * exp(-t / tau) + B

    Parameters:
        t   : idle time (nanoseconds)
        A   : SPAM contrast
        tau : characteristic decay time (ns)
        B   : asymptotic fidelity floor (ideal: 1/d)
    """
    return A * np.exp(-t / tau) + B


# =============================================================================
# Main class
# =============================================================================

class TmaxValidatorDelay:
    """
    Characterise idle-decoherence fidelity decay using explicit DELAY
    instructions and fit the exponential model F(t) = A*exp(-t/tau) + B
    via scipy curve_fit (analogous to Magesan RB for SWAP cycles).

    The class follows the same workflow as CMaxValidator:
        1. Collect empirical data   (run_delay_characterization)
        2. Print fit results        (print_delay_results)
        3. Extrapolation test       (run_extrapolation_test)
        4. Calculate T_MAX          (calculate_final_tmax)
    """

    # -- Constructor ----------------------------------------------------------

    def __init__(self, N: int = 1, backend=None, initial_state: int = 0,
                 pauli_twirling: bool = False, twirling_seed: int | None = None,
                 twirling_variants: int = 1,
                 mitigation_config: dict | None = None) -> None:
        """
        Args:
            N: Number of qubits per register (word width).
            backend: Optional external backend (e.g. real IBM). If None, uses FakeKyiv.
            initial_state: Initial qubit state and measurement basis.
                0 = |0>  -> init |0>, measure fidelity vs '0'*N
                1 = |1>  -> apply X, measure fidelity vs '1'*N
                2 = |+>  -> apply H, apply H before measure, fidelity vs '0'*N
                3 = |->  -> apply X+H, apply H before measure, fidelity vs '1'*N
        """
        self.initial_state = initial_state
        self.N = N
        self.d = 2 ** self.N
        self.B_ideal = 1.0 / self.d      # Maximum-mixing floor (analogue of CMax)
        self.pauli_twirling = pauli_twirling
        self.twirling_seed = twirling_seed
        if twirling_variants < 1:
            raise ValueError("twirling_variants must be >= 1")
        self.twirling_variants = twirling_variants

        # Backend configuration
        if backend is not None:
            self.backend = backend
            self.is_ibm = True
            self.noise_model = None
            self.simulator = None
            print(f"[TmaxValidatorDelay] Using IBM hardware backend: {self.backend.name}")
            print(f"[TmaxValidatorDelay] NOTE: delay() is a native hardware instruction.")
            # Select best physical qubits via noise-aware ranking
            self.best_qubits = self._select_best_qubits(N)
            print(f"[TmaxValidatorDelay] Best physical qubits selected: {self.best_qubits}")
        else:
            # 1. Reference backend (calibration snapshot from real IBM Kyiv)
            self.backend = FakeKyiv()
            self.is_ibm = False

            # 2. Noise model extracted from backend (plain simulator avoids
            #    ConstrainedReschedule scheduling issues with delays/barriers)
            self.noise_model = NoiseModel.from_backend(self.backend)
            self.simulator = AerSimulator(noise_model=self.noise_model)

        # 3. Backend dt (seconds per tick) -- used to convert ns -> dt
        self.dt_sec: float = self.backend.dt if hasattr(self.backend, 'dt') and self.backend.dt else 1e-9

        # 4. Fit parameters -- assigned in print_delay_results()
        self.A_fit:   float = 0.0
        self.tau_fit:  float = 0.0
        self.B_fit:   float = 0.0

        # 5. Log initial state
        _state_labels = {0: "|0>", 1: "|1>", 2: "|+> (H)", 3: "|-> (XH)"}
        state_label = _state_labels.get(initial_state, f"unknown({initial_state})")
        print(f"[TmaxValidatorDelay] Initial state : {state_label}")

        # 6. Error mitigation configuration (opt-in, default disabled)
        self.mitigation_config = mitigation_config or {}
        self.zne_enabled = self.mitigation_config.get("zne", {}).get("enabled", False)
        self.rem_enabled = self.mitigation_config.get("rem", {}).get("enabled", False)
        self.zne_noise_factors = self.mitigation_config.get("zne", {}).get("noise_factors", [1, 3])
        self.zne_extrapolator_method = self.mitigation_config.get("zne", {}).get("extrapolator", "linear")
        self.zne_seed = self.mitigation_config.get("zne", {}).get("zne_seed", None)
        if self.zne_enabled or self.rem_enabled:
            print(f"[TmaxValidatorDelay] Mitigation: ZNE={'ON' if self.zne_enabled else 'OFF'}, "
                  f"REM={'ON' if self.rem_enabled else 'OFF'}")

    # -- Noise-aware qubit selection (BackendV2 API) ----------------------------

    def _select_best_qubits(self, n: int) -> list[int]:
        """
        Select the N best physical qubits from an IBM BackendV2 backend,
        ranked by a cost function combining readout error and 1/T1 decay.

        This mirrors the noise-aware selection in QubitMapper but works
        directly with BackendV2's target.qubit_properties (real hardware),
        instead of BackendV1's backend.properties().

        Cost per qubit = W_READOUT * readout_error + W_T1 / T1
        Lower cost = better qubit.
        """
        W_READOUT = 1.0
        W_T1 = 1e-6  # scales T1 (seconds) to comparable magnitude

        costs: list[tuple[float, int]] = []

        for q in range(self.backend.num_qubits):
            cost = 0.0
            try:
                # BackendV2: qubit_properties gives T1, T2, readout_error, etc.
                qprops = self.backend.target.qubit_properties
                if qprops is not None and q < len(qprops) and qprops[q] is not None:
                    props = qprops[q]
                    # Readout error (lower is better)
                    ro_err = getattr(props, 'readout_error', None)
                    if ro_err is not None:
                        cost += W_READOUT * ro_err
                    # T1 decay penalty (higher T1 is better -> lower cost)
                    t1_val = getattr(props, 't1', None)
                    if t1_val is not None and t1_val > 0:
                        cost += W_T1 / t1_val
                    else:
                        cost += 10.0  # heavy penalty if no T1 data
                else:
                    cost += 10.0  # heavy penalty if no properties
            except Exception:
                cost += 10.0
            costs.append((cost, q))

        # Sort by cost ascending (best qubits first)
        costs.sort(key=lambda x: x[0])

        # Log top candidates for transparency
        print(f"[TmaxValidatorDelay] Top-10 qubit candidates (cost, qubit):")
        for cost_val, q_idx in costs[:10]:
            qprops = self.backend.target.qubit_properties
            ro_err = t1_val = "N/A"
            if qprops is not None and q_idx < len(qprops) and qprops[q_idx] is not None:
                p = qprops[q_idx]
                ro_err = f"{getattr(p, 'readout_error', 'N/A')}"
                t1_raw = getattr(p, 't1', None)
                t1_val = f"{t1_raw*1e6:.1f} us" if t1_raw else "N/A"
            print(f"    qubit {q_idx:3d}  cost={cost_val:.6f}  readout_err={ro_err}  T1={t1_val}")

        selected = [q for _, q in costs[:n]]
        return selected

    # -- Empirical fidelity (noisy idle) --------------------------------------

    def build_empirical_circuit(
        self,
        delay_ns: float,
        twirling_seed: int | None = None,
    ):
        if delay_ns < 0:
            raise ValueError(f"delay_ns must be >= 0, received: {delay_ns}")

        qc = QuantumCircuit(self.N, self.N)
        twirler = PauliTwirler(
            enabled=self.pauli_twirling,
            seed=self.twirling_seed if twirling_seed is None else twirling_seed,
        )

        if self.initial_state == 1:
            for i in range(self.N): qc.x(i)
        elif self.initial_state == 2:
            for i in range(self.N): twirler.h(qc, i)
        elif self.initial_state == 3:
            for i in range(self.N):
                qc.x(i)
                twirler.h(qc, i)

        if delay_ns > 0:
            for i in range(self.N):
                qc.delay(int(delay_ns), i, unit='ns')

        if self.initial_state in (2, 3):
            for i in range(self.N):
                twirler.h(qc, i)

        qc.measure(range(self.N), range(self.N))
        #print(qc.draw(output='text'))
        if self.is_ibm:
            # Use noise-aware initial_layout to map logical qubits to best physical qubits
            initial_layout = self.best_qubits
            qc_t = transpile(qc, backend=self.backend, optimization_level=0,
                             initial_layout=initial_layout)
            print(f"    [Layout] logical->physical: {list(range(self.N))} -> {initial_layout}")
        else:
            qc_t = transpile(qc, optimization_level=0)

        target_state = ('1' * self.N) if self.initial_state in (1, 3) else ('0' * self.N)
        return qc_t, target_state

    def calculate_fidelity(self, counts, target_state, shots) -> dict:
        fidelity_count = 0
        for bitstring, count in counts.items():
            measured = bitstring.replace(' ', '')[-self.N:]
            if measured == target_state:
                fidelity_count += count
        f_raw = fidelity_count / shots

        if not getattr(self, 'zne_enabled', False) and not getattr(self, 'rem_enabled', False):
            return {'f_raw': f_raw}

        mitigation_result = {'f_raw': f_raw}
        if getattr(self, 'rem_enabled', False):
            try:
                rem = ReadoutMitigator.from_backend_properties(self.backend, list(range(self.N)))
                corrected = rem.apply(counts)
                rem_count = sum(
                    c for b, c in corrected.items()
                    if b.replace(' ', '')[-self.N:] == target_state
                )
                mitigation_result['f_rem'] = rem_count / max(sum(corrected.values()), 1)
            except Exception as e:
                print(f"    [REM] Skipped: {e}")
                mitigation_result['f_rem'] = None
        return mitigation_result

    # -- Delay characterization (curve_fit) -----------------------------------

    def run_delay_characterization(
        self,
        delay_list_ns: list[float],
        shots: int = 4000,
        plot_path: str | None = "results/delay_decay_curve.png",
    ) -> np.ndarray:
        print("=" * 65)
        print("  SQM -- Delay Characterization (Batch Mode)")
        print("=" * 65)
        print(f"\n  Backend      : {self.backend.name}")
        print(f"  N_qubits     : {self.N}")
        print(f"  dt           : {self.dt_sec * 1e9:.4f} ns")
        print(f"\n  Measuring F(t) for t = {[f'{d:.0f}' for d in delay_list_ns[:5]]} ... ns")
        print(f"  shots per point = {shots}\n")
        
        variant_count = self.twirling_variants if getattr(self, 'pauli_twirling', False) else 1
        all_circuits = []
        metadata = []
        
        for delay in delay_list_ns:
            for variant in range(variant_count):
                seed = None
                if getattr(self, 'twirling_seed', None) is not None:
                    seed = self.twirling_seed + variant * 1_000_003 + int(delay) * 1_009
                qc_t, target = self.build_empirical_circuit(delay, seed)
                
                all_circuits.append(qc_t)
                metadata.append({'delay': delay, 'variant': variant, 'factor': 1, 'target': target})
                
                if getattr(self, 'zne_enabled', False):
                    folder = ZNEFolder(seed=self.zne_seed)
                    for factor in self.zne_noise_factors:
                        if factor == 1: continue
                        folded = folder.fold_circuit(qc_t, factor)
                        all_circuits.append(folded)
                        metadata.append({'delay': delay, 'variant': variant, 'factor': factor, 'target': target})

        print(f"  Generated {len(all_circuits)} circuits for batch execution.")
        
        if self.is_ibm:
            all_counts = run_on_ibm(all_circuits, self.backend, shots=shots)
        else:
            sim = AerSimulator(noise_model=self.noise_model)
            job = sim.run(all_circuits, shots=shots)
            res = job.result()
            all_counts = [res.get_counts(i) for i in range(len(all_circuits))]

        results_map = {}
        for count_dict, meta in zip(all_counts, metadata):
            d, v, f = meta['delay'], meta['variant'], meta['factor']
            if (d, v) not in results_map:
                results_map[(d, v)] = {'raw_counts': None, 'zne_counts': {}, 'meta': meta}
            if f == 1:
                results_map[(d, v)]['raw_counts'] = count_dict
            else:
                results_map[(d, v)]['zne_counts'][f] = count_dict

        t_arr = np.array(delay_list_ns, dtype=float)
        y_data, y_std = [], []
        
        for delay in delay_list_ns:
            fidelities = []
            for variant in range(variant_count):
                rm = results_map[(delay, variant)]
                meta = rm['meta']
                res = self.calculate_fidelity(rm['raw_counts'], meta['target'], shots)
                
                if getattr(self, 'zne_enabled', False):
                    extrapolator = ZNEExtrapolator()
                    zne_fids = {1: res['f_raw']}
                    for f_zne, c_zne in rm['zne_counts'].items():
                        r_zne = self.calculate_fidelity(c_zne, meta['target'], shots)
                        zne_fids[f_zne] = r_zne['f_raw']
                        
                    factors_sorted = sorted(zne_fids.keys())
                    values = [zne_fids[fac] for fac in factors_sorted]
                    zne_result = extrapolator.extrapolate(factors_sorted, values, self.zne_extrapolator_method)
                    res['f_zne'] = zne_result.bounded
                    
                if 'f_zne' in res and res['f_zne'] is not None:
                    fidelities.append(res['f_zne'])
                elif 'f_rem' in res and res['f_rem'] is not None:
                    fidelities.append(res['f_rem'])
                else:
                    fidelities.append(res['f_raw'])
                    
            f_emp = float(np.mean(fidelities))
            f_std = float(np.std(fidelities, ddof=1)) if variant_count > 1 else 0.0
            y_data.append(f_emp)
            y_std.append(f_std)
            print(f"    t={delay:10.1f} ns  F_emp = {f_emp:.6f}  std = {f_std:.6f} ({variant_count} variants)")

        y_arr = np.array(y_data, dtype=float)
        
        tau_guess = max(t_arr.max() / 3, 1.0)
        p0     = [0.75, tau_guess, self.B_ideal]
        bounds = ([0.0, 1e-3, 0.0], [1.0, 1e12, 1.0])
        
        try:
            popt, _ = curve_fit(exp_decay_model, t_arr, y_arr, p0=p0, bounds=bounds, maxfev=10_000)
        except RuntimeError as e:
            print(f"    [Warning] Curve fit failed: {e}. Using initial parameters.")
            popt = p0

        print("\n  Fit completed.")
        print(f"    A_fit   = {popt[0]:.6f}")
        print(f"    tau_fit = {popt[1]:.2f} ns")
        print(f"    B_fit   = {popt[2]:.6f}")

        if plot_path is not None:
            self._plot_decay_curve(t_arr, y_arr, popt, plot_path)
            
        csv_path = (
            plot_path.replace(".png", ".csv").replace("results", "data")
            if plot_path
            else "data/delay_characterization.csv"
        )
        self._save_results_to_csv(t_arr, y_arr, popt, csv_path, np.array(y_std))
            
        return popt

    # -- Save results to CSV --------------------------------------------------

    def _save_results_to_csv(
        self,
        t_arr: np.ndarray,
        y_data: np.ndarray,
        popt: np.ndarray,
        csv_path: str,
        y_std: np.ndarray | None = None,
    ) -> None:
        """Save delay characterization results to CSV file."""
        os.makedirs(
            os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".",
            exist_ok=True,
        )

        A_fit, tau_fit, B_fit = popt

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["SQM Delay Characterization Results"])
            writer.writerow(["Timestamp", datetime.now().isoformat()])
            writer.writerow(["Backend", self.backend.name])
            _state_labels = {0: "|0>", 1: "|1>", 2: "|+> (H)", 3: "|-> (XH)"}
            state_label = _state_labels.get(self.initial_state, f"unknown({self.initial_state})")
            writer.writerow(["Initial State", f"{self.initial_state} ({state_label})"])
            writer.writerow(["N_qubits", self.N])
            writer.writerow(["Hilbert Dimension", f"d = 2^{self.N} = {self.d}"])
            writer.writerow(["Pauli Twirling", "enabled" if self.pauli_twirling else "disabled"])
            writer.writerow(["Twirling Variants", self.twirling_variants if self.pauli_twirling else 1])
            writer.writerow(["Twirling Seed", self.twirling_seed if self.twirling_seed is not None else "random"])
            writer.writerow(["Mitigation ZNE", "enabled" if self.zne_enabled else "disabled"])
            if getattr(self, "zne_enabled", False):
                writer.writerow(["ZNE Noise Factors", getattr(self, "zne_noise_factors", "[]")])
                writer.writerow(["ZNE Extrapolator", getattr(self, "zne_extrapolator_method", "N/A")])
            writer.writerow(["Mitigation REM", "enabled" if self.rem_enabled else "disabled"])
            writer.writerow([])

            writer.writerow(["Fit Parameters: F(t) = A * exp(-t/tau) + B"])
            writer.writerow(["A (SPAM contrast)", f"{A_fit:.6f}"])
            writer.writerow(["tau (decay time, ns)", f"{tau_fit:.2f}"])
            writer.writerow(["B (asymptote)", f"{B_fit:.6f}"])
            writer.writerow([])

            writer.writerow(["t (ns)", "t (us)", "F_emp (mean)", "F_emp (std)", "F_fit"])
            if y_std is None:
                y_std = np.zeros_like(y_data)
            for t_ns, f_emp, f_std in zip(t_arr, y_data, y_std):
                f_fit_val = exp_decay_model(t_ns, A_fit, tau_fit, B_fit)
                writer.writerow([
                    f"{t_ns:.1f}",
                    f"{t_ns / 1000:.3f}",
                    f"{f_emp:.6f}",
                    f"{f_std:.6f}",
                    f"{f_fit_val:.6f}",
                ])

        print(f"\n  [CSV] Results saved to: {csv_path}")

    # -- Fit results report ---------------------------------------------------

    def print_delay_results(self, popt: np.ndarray) -> None:
        """
        Store fit parameters, print detailed report (analogous to
        CMaxValidator.print_rb_results).
        """
        self.A_fit, self.tau_fit, self.B_fit = popt

        print("\n" + "=" * 65)
        print("  SQM -- Delay Fit Results (Exponential Decay)")
        print("=" * 65)

        print(f"\n  Model: F(t) = A * exp(-t / tau) + B")
        print(f"  {'Parameter':<12}  {'Value':>12}  Interpretation")
        print(f"  {'-'*60}")
        print(f"  {'A':<12}  {self.A_fit:>12.6f}  SPAM contrast (state prep + measurement)")
        print(f"  {'tau (ns)':<12}  {self.tau_fit:>12.2f}  Characteristic decay time")
        print(f"  {'B':<12}  {self.B_fit:>12.6f}  Asymptotic fidelity floor (ideal: 1/d={self.B_ideal:.4f})")

        print(f"\n  [DECAY RATE]")
        lambda_rate = 1.0 / self.tau_fit if self.tau_fit > 0 else float("inf")
        print(f"    lambda = 1/tau = {lambda_rate:.6e} /ns")
        print(f"    tau_fit   = {self.tau_fit:.2f} ns  ({self.tau_fit / 1000:.3f} us)")

        print("=" * 65)

    # -- Plot decay curve -----------------------------------------------------

    def _plot_decay_curve(
        self,
        t_arr: np.ndarray,
        y_data: np.ndarray,
        popt: np.ndarray,
        path: str,
    ) -> None:
        """Generate and save the delay decay curve vs empirical data."""
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".",
            exist_ok=True,
        )

        A_fit, tau_fit, B_fit = popt
        t_dense = np.linspace(0, t_arr.max(), 300)
        f_fit   = exp_decay_model(t_dense, A_fit, tau_fit, B_fit)

        # Convert to us for plotting
        t_arr_us   = t_arr / 1000.0
        t_dense_us = t_dense / 1000.0

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(
            t_arr_us, y_data, color="steelblue", zorder=5, s=80,
            edgecolor="darkblue", linewidth=1.2,
            label="F_emp(t) -- noisy simulation",
        )
        ax.plot(
            t_dense_us, f_fit, color="crimson", linewidth=2,
            label=(
                f"Exp fit: A={A_fit:.3f}, tau={tau_fit:.1f} ns, B={B_fit:.3f}"
            ),
        )
        ax.axhline(
            y=B_fit, linestyle="--", color="gray", alpha=0.6,
            label=f"Asymptote B = {B_fit:.3f}",
        )

        _state_labels = {0: "|0>", 1: "|1>", 2: "|+> (H)", 3: "|-> (XH)"}
        state_label = _state_labels.get(self.initial_state, f"state({self.initial_state})")
        target_label = ('1' * self.N) if self.initial_state in (1, 3) else ('0' * self.N)

        ax.set_xlabel("Delay time (us)", fontsize=12)
        ax.set_ylabel(f"F(t) -- P(|{target_label}>) survival probability", fontsize=12)
        ax.set_title(
            f"Delay Decay Curve (N={self.N}, d={self.d}, init={state_label})",
            fontsize=14, fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"\n  [PLOT] Decay curve saved to: {path}")

    # -- Predicted fidelity from fitted model ---------------------------------

    def theoretical_fidelity(self, delay_ns: float) -> float:
        """Return the fitted model prediction at a given delay time."""
        if delay_ns < 0:
            raise ValueError(f"delay_ns must be >= 0, received: {delay_ns}")
        return float(exp_decay_model(delay_ns, self.A_fit, self.tau_fit, self.B_fit))

    # -- Extrapolation validation ---------------------------------------------

    def run_extrapolation_test(self, t_test_ns: float = 50_000) -> None:
        """
        Compare the fitted model prediction against a fresh empirical
        measurement at *t_test_ns* (analogous to CMaxValidator's n vs 2n test).
        """
        print("=" * 65)
        print(f"  SQM -- Extrapolation Validation (t={t_test_ns:.0f} ns)")
        print("=" * 65)

        f_th  = self.theoretical_fidelity(t_test_ns)
        
        qc_t, target = self.build_empirical_circuit(t_test_ns)
        if self.is_ibm:
            from experiments.utils.ibm_backend_helper import run_on_ibm
            counts = run_on_ibm([qc_t], self.backend, shots=4000)[0]
        else:
            sim = AerSimulator(noise_model=self.noise_model)
            counts = sim.run(qc_t, shots=4000).result().get_counts()
        res = self.calculate_fidelity(counts, target, 4000)
        f_emp = res.get('f_zne') or res.get('f_rem') or res.get('f_raw')
        
        diff  = abs(f_th - f_emp)
        rel   = (diff / f_emp * 100) if f_emp > 0 else float("inf")
        print(f"\n  [t={t_test_ns:.0f} ns]  F_model={f_th:.6f}  F_emp={f_emp:.6f}  "
              f"diff={diff:.6f} ({rel:.2f} %)")

        print("=" * 65)

    # -- Final T_MAX calculation (exponential model) --------------------------

    def calculate_final_tmax(self, target_fidelity: float = 0.75) -> float:
        """
        Calculate T_MAX = maximum idle time (ns) before fidelity drops
        below *target_fidelity*, using the fitted model:

            F(t) = A * exp(-t/tau) + B  >=  F_target
            =>  t <= tau * ln(A / (F_target - B))

        Returns:
            T_MAX in nanoseconds.
        """
        f_min_physical = self.B_fit
        f_max_physical = self.A_fit + self.B_fit

        if not (f_min_physical < target_fidelity <= f_max_physical):
            raise ValueError(
                f"target_fidelity={target_fidelity:.4f} outside physical range "
                f"({f_min_physical:.4f}, {f_max_physical:.4f}]. "
                f"B={self.B_fit:.4f} is minimum asymptote; "
                f"A+B={f_max_physical:.4f} is maximum achievable."
            )

        if self.tau_fit <= 0.0:
            raise RuntimeError(
                f"tau_fit={self.tau_fit:.6f} <= 0. "
                "Decay fit is invalid; increase delay range or shots."
            )

        # T_MAX = tau * ln(A / (F_target - B))
        ratio = self.A_fit / (target_fidelity - self.B_fit)
        t_max = self.tau_fit * math.log(ratio)

        print("\n" + "=" * 65)
        print("  SQM -- Final T_MAX Calculation (Exponential Decay Model)")
        print("=" * 65)
        print(f"  Fit parameters:")
        print(f"    A_fit   = {self.A_fit:.6f}  (SPAM contrast)")
        print(f"    tau_fit = {self.tau_fit:.2f} ns  (decay time)")
        print(f"    B_fit   = {self.B_fit:.6f}  (mixing asymptote)")
        print(f"\n  Target fidelity: F_target = {target_fidelity:.2f}  "
              f"({target_fidelity * 100:.0f} %)")
        print(f"\n  Formula: T_MAX = tau * ln(A / (F_target - B))")
        print(f"         = {self.tau_fit:.2f} * ln({self.A_fit:.6f} / "
              f"({target_fidelity:.2f} - {self.B_fit:.6f}))")
        print(f"         = {self.tau_fit:.2f} * ln({ratio:.6f})")
        print(f"         = {self.tau_fit:.2f} * {math.log(ratio):.6f}")
        print(f"\n  >>> T_MAX = {t_max:.2f} ns  ({t_max / 1000:.3f} us)")
        print(f"      F(T_MAX)     = {exp_decay_model(t_max, self.A_fit, self.tau_fit, self.B_fit):.6f}  "
              f"(>= {target_fidelity:.2f})")
        print(f"      F(T_MAX+100) = {exp_decay_model(t_max + 100, self.A_fit, self.tau_fit, self.B_fit):.6f}  "
              f"(<  {target_fidelity:.2f})")
        print("=" * 65)

        return t_max


# =============================================================================
# Entry point for direct execution
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # BACKEND MODE: "default" = FakeKyiv simulator | "IBM" = real IBM hardware
    # =========================================================================
    backend_mode = "default"  # Change to "IBM" to run on real IBM hardware
    shots = 1024
    twirling = False           # Set to True to enable Pauli twirling
    twirling_variants = 10    # Number of random circuits per delay point
 

    # Mitigation toggles
    use_zne = False  
    use_rem = False  
    mitigation_config = {
        "zne": {"enabled": use_zne, "noise_factors": [1, 3], "extrapolator": "exponential"},
        "rem": {"enabled": use_rem}
    }

    # =========================================================================
    # INITIAL STATE
    #   0 = |0>  : qubit starts in |0>, fidelity measured vs |0>
    #   1 = |1>  : qubit starts in |1> (X gate), fidelity measured vs |1>
    #   2 = |+>  : qubit starts in |+> (H gate), H applied before measure,
    #              fidelity measured vs |0>
    #   3 = |->  : qubit starts in |-> (X+H gates), H applied before measure,
    #              fidelity measured vs |1>
    # =========================================================================
    initial_state = 3  # 0 = |0>, 1 = |1>, 2 = |+> (H), 3 = |-> (XH)

    # 1. DEFINE THE ARCHITECTURE (N = Word width)
    N_qubits = 1
    target_fidelity = 0.75

    # -- Phase 1: Delay characterization (curve_fit) ---------------------------
    #    Define delay times directly in nanoseconds.
    delay_list_ns = [
       0, 1_000, 2_000, 5_000, 10_000, 20_000, 40_000, 50_000,100_000, 
       150_000, 200_000, 300_000, 600_000,800_000]
    #delay_list_ns = [80_0000]  
 
    _state_labels = {0: "|0>", 1: "|1>", 2: "|+> (H)", 3: "|-> (XH)"}
    state_label = _state_labels.get(initial_state, f"unknown({initial_state})")
    print(f"[Main] Running with initial_state={initial_state} ({state_label})")
    print(f"[Main] Pauli twirling: {'enabled' if twirling else 'disabled'} ({twirling_variants} variants)")

    suffix = ""
    if use_zne: suffix += "Z"
    if use_rem: suffix += "R"
    if twirling: suffix += "T"
    if suffix: suffix = "_" + suffix
    prefix = "sm" if backend_mode != "IBM" else "rb"

    if backend_mode == "IBM":
        ibm_backend = get_ibm_backend("ibm_kingston")
        validator = TmaxValidatorDelay(
            N=N_qubits,
            backend=ibm_backend,
            initial_state=initial_state,
            pauli_twirling=twirling,
            twirling_variants=twirling_variants,
            mitigation_config=mitigation_config
        )
    else:
        validator = TmaxValidatorDelay(
            N=N_qubits,
            initial_state=initial_state,
            pauli_twirling=twirling,
            twirling_variants=twirling_variants,
            mitigation_config=mitigation_config
        )


    popt = validator.run_delay_characterization(
        delay_list_ns,
        shots=shots,
        plot_path=f"results/{prefix}_decay_curve_delay_state{initial_state}_N{N_qubits}{suffix}.png",
    )

    # -- Phase 2: Print results and validate model ----------------------------
    validator.print_delay_results(popt)

    # -- Phase 3: Calculate T_MAX with target fidelity -------------------------
    t_max = validator.calculate_final_tmax(target_fidelity=target_fidelity)
    print(f"\n[FINAL RESULT]  T_MAX = {t_max:.2f} ns  ({t_max / 1000:.3f} us)")

    # -- Phase 4: Extrapolation validation ------------------------------------
    #t_test = 50_000  # ns
    #validator.run_extrapolation_test(t_test_ns=t_test)
    #print(f"\n  Interpretation:")
    #print(f"    If diff < 5%, the exponential model extrapolates correctly to t={t_test} ns.")
