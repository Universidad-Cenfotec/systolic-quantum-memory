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
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
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
# Magesan decay model (discrete operations): F(m) = A * p^m + B
# =============================================================================

def rb_decay_model(m: float, A: float, p: float, B: float) -> float:
    """
    Magesan decay model for repeated discrete operations.

    F(m) = A * p^m + B

    Parameters:
        m : number of ID gate cycles applied
        A : SPAM contrast
        p : process fidelity decay per ID gate
        B : asymptotic fidelity floor (ideal: 1/d)
    """
    return A * (p ** m) + B


# =============================================================================
# Main class
# =============================================================================

class TmaxValidatorId:
    """
    Characterise idle-decoherence fidelity decay using repeated Identity
    (ID) gates with a CONSTANT thermal relaxation error per gate.

    The error parameters (T1, T2, idle_time_ns) are injected at
    construction time and remain fixed for all ID gates.  The class
    follows the same workflow as CMaxValidator / TmaxValidatorDelay:

        1. Collect empirical data   (run_id_characterization)
        2. Print fit results        (print_id_results)
        3. Extrapolation test       (run_extrapolation_test)
        4. Calculate T_MAX          (calculate_final_tmax)
    """

    # -- Constructor ----------------------------------------------------------

    def __init__(
        self,
        N: int = 1,
        t1_ns: float = 192_566,
        t2_ns: float = 35_887,
        idle_time_ns: float = 1000,
        backend=None,
        pauli_twirling: bool = False,
        twirling_seed: int | None = None,
        twirling_variants: int = 1,
        mitigation_config: dict | None = None,
    ) -> None:
        """
        Args:
            N            : Number of qubits per register (word width).
            t1_ns        : T1 relaxation time in nanoseconds.
            t2_ns        : T2 dephasing time in nanoseconds.
            idle_time_ns : Duration of one ID gate idle period in
                           nanoseconds (constant error per gate).
            backend      : Optional external backend (e.g. real IBM).
                           If None, uses FakeKyiv + AerSimulator.
        """
        self.N = N
        self.d = 2 ** self.N
        self.B_ideal = 1.0 / self.d

        # Noise parameters (constant, defined by the caller)
        self.t1_ns = t1_ns
        self.t2_ns = t2_ns
        self.idle_time_ns = idle_time_ns
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
            print(f"[TmaxValidatorId] Using IBM hardware backend: {self.backend.name}")
            print(f"[TmaxValidatorId] NOTE: On real hardware, noise is physical.")
            print(f"[TmaxValidatorId]       Custom thermal_relaxation_error is NOT injected.")
            # Select best physical qubits via noise-aware ranking
            self.best_qubits = self._select_best_qubits(N)
            print(f"[TmaxValidatorId] Best physical qubits selected: {self.best_qubits}")
        else:
            # 1. Reference backend
            self.backend = FakeKyiv()
            self.is_ibm = False

            # 2. Build noise model: backend base + constant thermal relax on 'id'
            self.noise_model = NoiseModel.from_backend(self.backend)
            idle_error = thermal_relaxation_error(
                self.t1_ns, self.t2_ns, self.idle_time_ns
            )
            num_qubits = self.backend.configuration().n_qubits
            for q in range(num_qubits):
                self.noise_model.add_quantum_error(idle_error, "id", [q], warnings=False)

            # 3. Simulator
            self.simulator = AerSimulator(
                noise_model=self.noise_model, method="matrix_product_state"
            )

        # 4. Fit parameters -- assigned in print_id_results()
        self.A_fit: float = 0.0
        self.p_fit: float = 0.0
        self.B_fit: float = 0.0

        # 5. Error mitigation configuration (opt-in, default disabled)
        self.mitigation_config = mitigation_config or {}
        self.zne_enabled = self.mitigation_config.get("zne", {}).get("enabled", False)
        self.rem_enabled = self.mitigation_config.get("rem", {}).get("enabled", False)
        self.zne_noise_factors = self.mitigation_config.get("zne", {}).get("noise_factors", [1, 3])
        self.zne_extrapolator_method = self.mitigation_config.get("zne", {}).get("extrapolator", "linear")
        self.zne_seed = self.mitigation_config.get("zne", {}).get("zne_seed", None)
        if self.zne_enabled or self.rem_enabled:
            print(f"[TmaxValidatorId] Mitigation: ZNE={'ON' if self.zne_enabled else 'OFF'}, "
                  f"REM={'ON' if self.rem_enabled else 'OFF'}")

    # -- Noise-aware qubit selection (BackendV2 API) ----------------------------

    def _select_best_qubits(self, n: int) -> list[int]:
        """
        Select the N best physical qubits from an IBM BackendV2 backend,
        ranked by a cost function combining readout error and 1/T1 decay.
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
        print(f"[TmaxValidatorId] Top-10 qubit candidates (cost, qubit):")
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

    # -- Empirical fidelity (noisy idle via ID gates) -------------------------

    def build_empirical_circuit(
        self,
        n_ids: int,
        twirling_seed: int | None = None,
    ):
        qc = QuantumCircuit(self.N, self.N)
        twirler = PauliTwirler(
            enabled=self.pauli_twirling,
            seed=self.twirling_seed if twirling_seed is None else twirling_seed,
        )

        for i in range(self.N): qc.x(i)

        if n_ids > 0:
            for i in range(self.N):
                for _ in range(n_ids):
                    qc.id(i)


        qc.measure(range(self.N), range(self.N))

        if self.is_ibm:
            initial_layout = self.best_qubits
            qc_t = transpile(qc, backend=self.backend, optimization_level=0,
                             initial_layout=initial_layout)
            print(f"    [Layout] logical->physical: {list(range(self.N))} -> {initial_layout}")
        else:
            qc_t = transpile(qc, optimization_level=0)

        target_state = ('1' * self.N)
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
                print(f'    [REM] Skipped: {e}')
                mitigation_result['f_rem'] = None
        return mitigation_result
    def run_id_characterization(
        self,
        m_list: list[int],
        shots: int = 4000,
        plot_path: str | None = 'results/tmax_decay_curve_id.png',
    ) -> np.ndarray:
        print('=' * 65)
        print('  Id -- Phase B: T1/T2 Characterization (Batch Mode)')
        print('=' * 65)
        print(f'  Backend      : {self.backend.name}')
        
        variant_count = self.twirling_variants if getattr(self, 'pauli_twirling', False) else 1
        all_circuits = []
        metadata = []
        
        for m in m_list:
            for variant in range(variant_count):
                seed = None
                if getattr(self, 'twirling_seed', None) is not None:
                    seed = self.twirling_seed + variant * 1_000_003 + int(m) * 1_009
                qc_t, target = self.build_empirical_circuit(m, seed)
                
                all_circuits.append(qc_t)
                metadata.append({'m': m, 'variant': variant, 'factor': 1, 'target': target})
                
                if getattr(self, 'zne_enabled', False):
                    folder = ZNEFolder(seed=self.zne_seed)
                    for factor in self.zne_noise_factors:
                        if factor == 1: continue
                        folded = folder.fold_circuit(qc_t, factor)
                        all_circuits.append(folded)
                        metadata.append({'m': m, 'variant': variant, 'factor': factor, 'target': target})

        print(f'  Generated {len(all_circuits)} circuits for batch execution.')
        
        if self.is_ibm:
            from experiments.utils.ibm_backend_helper import run_on_ibm
            all_counts = run_on_ibm(all_circuits, self.backend, shots=shots)
        else:
            sim = AerSimulator(noise_model=self.noise_model)
            job = sim.run(all_circuits, shots=shots)
            res = job.result()
            all_counts = [res.get_counts(i) for i in range(len(all_circuits))]

        results_map = {}
        for count_dict, meta in zip(all_counts, metadata):
            m, v, f = meta['m'], meta['variant'], meta['factor']
            if (m, v) not in results_map:
                results_map[(m, v)] = {'raw_counts': None, 'zne_counts': {}, 'meta': meta}
            if f == 1:
                results_map[(m, v)]['raw_counts'] = count_dict
            else:
                results_map[(m, v)]['zne_counts'][f] = count_dict

        m_arr = np.array(m_list, dtype=float)
        y_data, y_std = [], []
        
        for m in m_list:
            fidelities = []
            for variant in range(variant_count):
                rm = results_map[(m, variant)]
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
            print(f'    m={m:4d} ids  F_emp = {f_emp:.6f}  std = {f_std:.6f} ({variant_count} variants)')

        y_arr = np.array(y_data, dtype=float)
        
        B_fix = self.B_ideal
        p0 = [1.0, 0.99, B_fix]
        bounds = ([0.0, 0.0, 0.0], [1.0, 1e9, 1.0])
        
        try:
            popt, _ = curve_fit(rb_decay_model, m_arr, y_arr, p0=p0, bounds=bounds, maxfev=10_000)
        except RuntimeError as e:
            print(f'    [Warning] Curve fit failed: {e}. Using initial parameters.')
            popt = p0

        print('\n  Fit completed.')
        print(f'    A_fit = {popt[0]:.6f}')
        print(f'    p_fit = {popt[1]:.6f}')
        print(f'    B_fit = {popt[2]:.6f}')

        if plot_path is not None:
            self._plot_decay_curve(m_arr, y_arr, popt, plot_path)
            csv_path = plot_path.replace('.png', '.csv').replace('results', 'data')
            self._save_results_to_csv(m_arr, y_arr, popt, csv_path, np.array(y_std))
            
        return popt

    # -- Save results to CSV --------------------------------------------------

    def _save_results_to_csv(
        self,
        m_arr: np.ndarray,
        y_data: np.ndarray,
        popt: np.ndarray,
        csv_path: str,
        y_std: np.ndarray | None = None,
    ) -> None:
        """Save ID characterization results to CSV file."""
        os.makedirs(
            os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".",
            exist_ok=True,
        )

        A_fit, p_fit, B_fit = popt

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["SQM ID Gate Characterization Results"])
            writer.writerow(["Timestamp", datetime.now().isoformat()])
            writer.writerow(["Backend", self.backend.name])
            writer.writerow(["N_qubits", self.N])
            writer.writerow(["Hilbert Dimension", f"d = 2^{self.N} = {self.d}"])
            writer.writerow(["T1 (ns)", f"{self.t1_ns:.3f}"])
            writer.writerow(["T2 (ns)", f"{self.t2_ns:.3f}"])
            writer.writerow(["idle_time_ns", f"{self.idle_time_ns:.1f}"])
            writer.writerow(["Pauli Twirling", "enabled" if self.pauli_twirling else "disabled"])
            writer.writerow(["Twirling Variants", self.twirling_variants if self.pauli_twirling else 1])
            writer.writerow(["Twirling Seed", self.twirling_seed if self.twirling_seed is not None else "random"])
            writer.writerow(["Mitigation ZNE", "enabled" if self.zne_enabled else "disabled"])
            if getattr(self, "zne_enabled", False):
                writer.writerow(["ZNE Noise Factors", getattr(self, "zne_noise_factors", "[]")])
                writer.writerow(["ZNE Extrapolator", getattr(self, "zne_extrapolator_method", "N/A")])
            writer.writerow(["Mitigation REM", "enabled" if self.rem_enabled else "disabled"])
            writer.writerow([])

            writer.writerow(["Fit Parameters: F(m) = A * p^m + B"])
            writer.writerow(["A (SPAM contrast)", f"{A_fit:.6f}"])
            writer.writerow(["p (process decay)", f"{p_fit:.6f}"])
            writer.writerow(["B (asymptote)", f"{B_fit:.6f}"])
            writer.writerow([])

            writer.writerow(["m (ID gates)", "F_emp (mean)", "F_emp (std)", "F_fit"])
            if y_std is None:
                y_std = np.zeros_like(y_data)
            for m, f_emp, f_std in zip(m_arr, y_data, y_std):
                f_fit_val = rb_decay_model(m, A_fit, p_fit, B_fit)
                writer.writerow([
                    f"{int(m):d}",
                    f"{f_emp:.6f}",
                    f"{f_std:.6f}",
                    f"{f_fit_val:.6f}",
                ])

        print(f"\n  [CSV] Results saved to: {csv_path}")

    # -- Fit results report ---------------------------------------------------

    def print_id_results(self, popt: np.ndarray) -> None:
        """
        Store fit parameters, print detailed report (analogous to
        CMaxValidator.print_rb_results).
        """
        self.A_fit, self.p_fit, self.B_fit = popt

        r_per_gate = ((self.d - 1) * (1.0 - self.p_fit)) / self.d

        print("\n" + "=" * 65)
        print("  SQM -- ID Gate Fit Results (Magesan Decay)")
        print("=" * 65)

        print(f"\n  Model: F(m) = A * p^m + B")
        print(f"  {'Parameter':<12}  {'Value':>12}  Interpretation")
        print(f"  {'-'*60}")
        print(f"  {'A':<12}  {self.A_fit:>12.6f}  SPAM contrast (state prep + measurement)")
        print(f"  {'p':<12}  {self.p_fit:>12.6f}  Process fidelity decay per ID gate")
        print(f"  {'B':<12}  {self.B_fit:>12.6f}  Asymptotic fidelity floor (ideal: 1/d={self.B_ideal:.4f})")

        print(f"\n  [ERROR PER GATE]")
        print(f"    r_per_gate = (d-1)/d * (1 - p_fit)")
        print(f"               = ({self.d - 1})/{self.d} * (1 - {self.p_fit:.6f})")
        print(f"               = {r_per_gate:.6f}  ({r_per_gate * 100:.4f} %)")

        print(f"\n  [NOISE PARAMETERS]")
        print(f"    T1             = {self.t1_ns:.3f} ns  ({self.t1_ns / 1000:.3f} us)")
        print(f"    T2             = {self.t2_ns:.3f} ns  ({self.t2_ns / 1000:.3f} us)")
        print(f"    idle_time/gate = {self.idle_time_ns:.1f} ns")

        print("=" * 65)

    # -- Plot decay curve -----------------------------------------------------

    def _plot_decay_curve(
        self,
        m_arr: np.ndarray,
        y_data: np.ndarray,
        popt: np.ndarray,
        path: str,
    ) -> None:
        """Generate and save the ID gate decay curve vs empirical data."""
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".",
            exist_ok=True,
        )

        A_fit, p_fit, B_fit = popt
        m_dense = np.linspace(0, m_arr.max(), 300)
        f_fit   = rb_decay_model(m_dense, A_fit, p_fit, B_fit)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(
            m_arr, y_data, color="steelblue", zorder=5, s=80,
            edgecolor="darkblue", linewidth=1.2,
            label="F_emp(m) -- noisy simulation",
        )
        ax.plot(
            m_dense, f_fit, color="crimson", linewidth=2,
            label=(
                f"Magesan fit: A={A_fit:.3f}, p={p_fit:.4f}, B={B_fit:.3f}"
            ),
        )
        ax.axhline(
            y=B_fit, linestyle="--", color="gray", alpha=0.6,
            label=f"Asymptote B = {B_fit:.3f}",
        )

        ax.set_xlabel("m  (number of ID gates)", fontsize=12)
        ax.set_ylabel("F(m) -- |1> survival probability", fontsize=12)
        ax.set_title(
            f"ID Gate Decay Curve (N={self.N}, d={self.d}, "
            f"idle={self.idle_time_ns:.0f} ns/gate)",
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

    def theoretical_fidelity(self, n_ids: int) -> float:
        """Return the fitted model prediction at a given number of ID gates."""
        if n_ids < 0:
            raise ValueError(f"n_ids must be >= 0, received: {n_ids}")
        return float(rb_decay_model(n_ids, self.A_fit, self.p_fit, self.B_fit))

    # -- Extrapolation validation ---------------------------------------------

    def run_extrapolation_test(self, n_test: int = 50) -> None:
        """
        Compare the fitted model prediction against a fresh empirical
        measurement at *n_test* ID gates.
        """
        print("=" * 65)
        print(f"  SQM -- Extrapolation Validation (m={n_test} ID gates)")
        print("=" * 65)

        f_th  = self.theoretical_fidelity(n_test)
        
        qc_t, target = self.build_empirical_circuit(n_test)
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
        print(f"\n  [m={n_test}]  F_model={f_th:.6f}  F_emp={f_emp:.6f}  "
              f"diff={diff:.6f} ({rel:.2f} %)")

        print("=" * 65)

    # -- Final T_MAX calculation (Magesan model) ------------------------------

    def calculate_final_tmax(self, target_fidelity: float = 0.75) -> dict:
        """
        Calculate T_MAX in two forms:
          - C_MAX : maximum number of ID gates (discrete)
          - T_MAX : C_MAX * idle_time_ns  (continuous, in nanoseconds)

        Uses the Magesan model:
            F(m) = A * p^m + B  >=  F_target
            =>  m <= log((F_target - B) / A) / log(p)

        Returns:
            dict with c_max (int) and t_max_ns (float)
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

        if self.p_fit <= 0.0 or self.p_fit >= 1.0:
            raise RuntimeError(
                f"p_fit={self.p_fit:.6f} outside interval (0, 1). "
                "Fit is invalid; increase m_list or shots."
            )

        # C_MAX = floor( log((F_target - B) / A) / log(p) )
        ratio = (target_fidelity - self.B_fit) / self.A_fit
        c_max = math.floor(math.log(ratio) / math.log(self.p_fit))
        t_max_ns = c_max * self.idle_time_ns

        print("\n" + "=" * 65)
        print("  SQM -- Final T_MAX Calculation (ID Gate Magesan Model)")
        print("=" * 65)
        print(f"  Fit parameters:")
        print(f"    A_fit  = {self.A_fit:.6f}  (SPAM contrast)")
        print(f"    p_fit  = {self.p_fit:.6f}  (process decay per ID gate)")
        print(f"    B_fit  = {self.B_fit:.6f}  (mixing asymptote)")
        print(f"\n  Noise configuration:")
        print(f"    T1             = {self.t1_ns:.3f} ns")
        print(f"    T2             = {self.t2_ns:.3f} ns")
        print(f"    idle_time/gate = {self.idle_time_ns:.1f} ns")
        print(f"\n  Target fidelity: F_target = {target_fidelity:.2f}  "
              f"({target_fidelity * 100:.0f} %)")
        print(f"\n  Formula: C_MAX = floor[ log((F_target - B) / A) / log(p) ]")
        print(f"         = floor[ log({ratio:.6f}) / log({self.p_fit:.6f}) ]")
        print(f"         = floor[ {math.log(ratio):.6f} / {math.log(self.p_fit):.6f} ]")
        print(f"\n  >>> C_MAX = {c_max} ID gates")
        print(f"  >>> T_MAX = C_MAX * idle_time = {c_max} * {self.idle_time_ns:.1f} "
              f"= {t_max_ns:.1f} ns  ({t_max_ns / 1000:.3f} us)")
        print(f"      F(C_MAX)   = {rb_decay_model(c_max,   self.A_fit, self.p_fit, self.B_fit):.6f}  "
              f"(>= {target_fidelity:.2f})")
        print(f"      F(C_MAX+1) = {rb_decay_model(c_max+1, self.A_fit, self.p_fit, self.B_fit):.6f}  "
              f"(<  {target_fidelity:.2f})")
        print("=" * 65)

        return {"c_max": c_max, "t_max_ns": t_max_ns}


# =============================================================================
# Entry point for direct execution
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # BACKEND MODE: "default" = FakeKyiv simulator | "IBM" = real IBM hardware
    # =========================================================================
    backend_mode = "default"  # Change to "IBM" to run on real IBM hardware
    twirling = False           # Set to True to enable Pauli twirling
    twirling_variants = 10    # Number of random circuits per ID point

    # Mitigation toggles
    use_zne = False
    use_rem = False
    mitigation_config = {
        "zne": {"enabled": use_zne, "noise_factors": [1, 3], "extrapolator": "linear"},
        "rem": {"enabled": use_rem}
    }

    # =========================================================================
    # CONFIGURATION (all noise parameters defined here)
    # =========================================================================
    N_qubits       = 1
    target_fidelity = 0.75

    # Constant thermal relaxation error per ID gate (ALL in nanoseconds)
    T1_NS          = 192_566      # T1 relaxation -- 192.566 us = 192566 ns
    T2_NS          = 35_887       # T2 dephasing  -- 35.887 us  = 35887 ns
    IDLE_TIME_NS   = 1000         # Duration of one ID idle period (ns)

    initial_state = 1
    suffix = ""
    if use_zne: suffix += "Z"
    if use_rem: suffix += "R"
    if twirling: suffix += "T"
    if suffix: suffix = "_" + suffix
    prefix = "sm" if backend_mode != "IBM" else "rb"

    if backend_mode == "IBM":
        ibm_backend = get_ibm_backend("ibm_kingston")
        validator = TmaxValidatorId(
            N=N_qubits,
            t1_ns=T1_NS,
            t2_ns=T2_NS,
            idle_time_ns=IDLE_TIME_NS,
            backend=ibm_backend,
            pauli_twirling=twirling,
            twirling_variants=twirling_variants,
            mitigation_config=mitigation_config,
        )
    else:
        validator = TmaxValidatorId(
            N=N_qubits,
            t1_ns=T1_NS,
            t2_ns=T2_NS,
            idle_time_ns=IDLE_TIME_NS,
            pauli_twirling=twirling,
            twirling_variants=twirling_variants,
        )

    # -- Phase 1: ID gate characterization (curve_fit) -------------------------
    m_list = [0, 1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]

    popt = validator.run_id_characterization(
        m_list, shots=4000,
        plot_path=f"results/{prefix}_decay_curve_Id_state{initial_state}_n{N_qubits}{suffix}.png"
    )

    # -- Phase 2: Print results ------------------------------------------------
    validator.print_id_results(popt)

    # -- Phase 3: Calculate T_MAX with target fidelity -------------------------
    result = validator.calculate_final_tmax(target_fidelity=target_fidelity)
    print(f"\n[FINAL RESULT]  C_MAX = {result['c_max']} ID gates  |  "
          f"T_MAX = {result['t_max_ns']:.1f} ns  ({result['t_max_ns'] / 1000:.3f} us)")

    # -- Phase 4: Extrapolation validation ------------------------------------
    m_test = 15
    validator.run_extrapolation_test(n_test=m_test)
    print(f"\n  Interpretation:")
    print(f"    If diff < 5%, the Magesan model extrapolates correctly to m={m_test}.")
