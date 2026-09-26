import math
import os
import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend - output to file
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeKyiv

# Handle imports for both direct execution and module import
try:
    from src.functions.qubit_mapper import QubitMapper
    from src.utils.measurement_parser import MeasurementParser
    from src.utils.pauli_twirling import PauliTwirler
    from src.mitigation import ReadoutMitigator, ZNEFolder, ZNEExtrapolator
    from experiments.utils.ibm_backend_helper import get_ibm_backend, run_on_ibm
except ModuleNotFoundError:
    # Add parent directory to path for direct script execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.functions.qubit_mapper import QubitMapper
    from src.utils.measurement_parser import MeasurementParser
    from src.utils.pauli_twirling import PauliTwirler
    from src.mitigation import ReadoutMitigator, ZNEFolder, ZNEExtrapolator
    from experiments.utils.ibm_backend_helper import get_ibm_backend, run_on_ibm


# =============================================================================
# Magesan decay model (global reusable function)
# =============================================================================

def rb_decay_model(m: float, A: float, p: float, B: float) -> float:
   
    return A * (p ** m) + B


# =============================================================================
# Main class
# =============================================================================

class CMaxValidator:
   
    _TWO_QUBIT_GATE_CANDIDATES = ["ecr", "cx", "cz", "rzx"]

    # -- Constructor ----------------------------------------------------------

    def __init__(self, N: int = 1, backend=None, initial_state: int = 1,
                 pauli_twirling: bool = False, twirling_seed: int | None = None,
                 twirling_variants: int = 1,
                 mitigation_config: dict | None = None) -> None:
        """
        Args:
            N: Number of qubits per register (word width).
            backend: Optional external backend (e.g. real IBM). If None, uses FakeKyiv.
            initial_state: Initial qubit state and measurement basis.
                0 = |0⟩  -> init |0⟩, measure fidelity vs '0'*N
                1 = |1⟩  -> apply X, measure fidelity vs '1'*N  (original behaviour)
                2 = |+⟩  -> apply H, apply H before measure, fidelity vs '0'*N
                3 = |-⟩  -> apply X+H, apply H before measure, fidelity vs '1'*N
        """
        # 0. Dynamic word width parameter
        self.N = N
        self.d = 2 ** self.N
        self.B_ideal = 1.0 / self.d
        self.initial_state = initial_state
        self.pauli_twirling = pauli_twirling
        self.twirling_seed = twirling_seed
        if twirling_variants < 1:
            raise ValueError("twirling_variants must be >= 1")
        self.twirling_variants = twirling_variants

        # 1. Backend configuration
        #    If backend is provided externally (e.g. real IBM), use it directly.
        #    Otherwise, default to FakeKyiv() simulator.
        if backend is not None:
            self.backend = backend
            self.is_ibm = True
            self.noise_model = None
            print(f"[CMaxValidator] Using IBM hardware backend: {self.backend.name}")
        else:
            self.backend = FakeKyiv()
            self.is_ibm = False
            # 2. Complete noise model (depolarization + thermal relaxation)
            self.noise_model = NoiseModel.from_backend(self.backend)

        # 3. Detect native 2Q gate and extract average error.
        #    FakeKyiv uses ECR; auto-detects among [ECR, CX, CZ, RZX].
        self.native_2q_gate: str = ""   # filled by _extract_avg_cx_error
        self.cx_error: float     = self._extract_avg_cx_error()

        # 4. Theoretical SWAP error (3*N native 2Q gates in series, i.i.d.)
        #    Kept as reference to compare against empirical r from RB.
        self.p_swap_teorico: float = 1.0 - (1.0 - self.cx_error) ** (3 * N)

        # 5. RB fitting parameters - assigned in print_rb_results()
        self.A_fit:     float = 0.0
        self.p_fit:     float = 0.0
        self.B_fit:     float = 0.0
        self.r_empirico: float = 0.0

        # 6. Error mitigation configuration (opt-in, default disabled)
        self.mitigation_config = mitigation_config or {}
        self.zne_enabled = self.mitigation_config.get("zne", {}).get("enabled", False)
        self.rem_enabled = self.mitigation_config.get("rem", {}).get("enabled", False)
        self.zne_noise_factors = self.mitigation_config.get("zne", {}).get("noise_factors", [1, 3])
        self.zne_extrapolator_method = self.mitigation_config.get("zne", {}).get("extrapolator", "linear")
        self.zne_seed = self.mitigation_config.get("zne", {}).get("zne_seed", None)

        # 7. Log initial state
        _state_labels = {0: "|0⟩", 1: "|1⟩", 2: "|+⟩ (H)", 3: "|-⟩ (XH)"}
        state_label = _state_labels.get(initial_state, f"unknown({initial_state})")
        print(f"[CMaxValidator] Initial state : {state_label}")
        if self.zne_enabled or self.rem_enabled:
            print(f"[CMaxValidator] Mitigation: ZNE={'ON' if self.zne_enabled else 'OFF'}, "
                  f"REM={'ON' if self.rem_enabled else 'OFF'}")

    # -- Extraction of calibration parameters ----------------------------------

    def _extract_avg_cx_error(self) -> float:
       
        props = self.backend.properties()

        gate_errors: dict[str, list[float]] = {}
        for gate in props.gates:
            gname = gate.gate.lower()
            for param in gate.parameters:
                if param.name == "gate_error":
                    gate_errors.setdefault(gname, []).append(param.value)
                    break

        for candidate in self._TWO_QUBIT_GATE_CANDIDATES:
            if candidate in gate_errors:
                self.native_2q_gate = candidate
                return float(np.mean(gate_errors[candidate]))

        raise RuntimeError(
            f"No known 2Q gate found among "
            f"{self._TWO_QUBIT_GATE_CANDIDATES} in '{self.backend.name}'. "
            f"Available gates: {sorted(gate_errors.keys())}."
        )



    # -- Empirical fidelity (noisy WorkPhase) ----------------------------------

    def build_empirical_circuit(
        self,
        n_swaps: int,
        twirling_seed: int | None = None,
    ):
        if n_swaps < 0:
            raise ValueError(f"n_swaps must be >= 0, received: {n_swaps}")

        qc = QuantumCircuit(2 * self.N, self.N)
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

        for _ in range(n_swaps):
            for i in range(self.N):
                twirler.cx(qc, i, i + self.N)
                twirler.cx(qc, i + self.N, i)
                twirler.cx(qc, i, i + self.N)
            qc.barrier()

        if n_swaps % 2 == 0:
            measure_qubits = list(range(self.N))
            reg_label = "mem_0"
        else:
            measure_qubits = list(range(self.N, 2 * self.N))
            reg_label = "q_work"

        if self.initial_state in (2, 3):
            for i in measure_qubits:
                twirler.h(qc, i)

        qc.measure(measure_qubits, range(self.N))

        mapper = QubitMapper(self.backend)
        allocation = mapper.allocate_chain_topology(
            chain_config=[("q_work", self.N), ("mem_0", self.N)]
        )

        initial_layout = [0] * (2 * self.N)
        for i in range(self.N):
            initial_layout[i] = allocation["mem_0"][i]
            initial_layout[self.N + i] = allocation["q_work"][i]
        #print(qc.draw(output='text'))
        qc_t = transpile(qc, backend=self.backend, optimization_level=0, initial_layout=initial_layout)

        target_state = ('1' * self.N) if self.initial_state in (1, 3) else ('0' * self.N)
        return qc_t, target_state, reg_label, initial_layout

    def calculate_fidelity(self, counts, target_state, reg_label, initial_layout, shots) -> dict:
        fidelity_count = counts.get(target_state, 0)
        f_raw = fidelity_count / shots

        if not getattr(self, "zne_enabled", False) and not getattr(self, "rem_enabled", False):
            return {"f_raw": f_raw}

        mitigation_result = {"f_raw": f_raw}
        if getattr(self, "rem_enabled", False):
            try:
                if reg_label == "q_work":
                    final_physical = [initial_layout[self.N + i] for i in range(self.N)]
                else:
                    final_physical = [initial_layout[i] for i in range(self.N)]
                rem = ReadoutMitigator.from_backend_properties(self.backend, final_physical)
                corrected = rem.apply(counts)
                mitigation_result["f_rem"] = corrected.get(target_state, 0) / max(sum(corrected.values()), 1)
            except Exception as e:
                print(f"    [REM] Skipped: {e}")
                mitigation_result["f_rem"] = None
        return mitigation_result

    def run_rb_characterization(
        self,
        m_list: list[int],
        shots: int = 4000,
        plot_path: str | None = "results/rb_decay_curve_swap.png",
    ) -> np.ndarray:
        print("=" * 70)
        print("  SWAP -- Phase B: RB Characterization (Batch Mode)")
        print("=" * 70)
        print(f"  Backend      : {self.backend.name}")
        print(f"  p_swap_theory: {self.p_swap_teorico:.6f}")
        
        variant_count = self.twirling_variants if self.pauli_twirling else 1
        all_circuits = []
        metadata = []
        
        for m in m_list:
            for variant in range(variant_count):
                seed = None
                if self.twirling_seed is not None:
                    seed = self.twirling_seed + variant * 1_000_003 + m * 1_009
                qc_t, target, reg_label, initial_layout = self.build_empirical_circuit(m, seed)
                
                all_circuits.append(qc_t)
                metadata.append({"m": m, "variant": variant, "factor": 1, "target": target, "reg_label": reg_label, "ilayout": initial_layout})
                
                if getattr(self, "zne_enabled", False):
                    folder = ZNEFolder(seed=self.zne_seed)
                    for factor in self.zne_noise_factors:
                        if factor == 1: continue
                        folded = folder.fold_circuit(qc_t, factor)
                        all_circuits.append(folded)
                        metadata.append({"m": m, "variant": variant, "factor": factor, "target": target, "reg_label": reg_label, "ilayout": initial_layout})

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
            m, v, f = meta["m"], meta["variant"], meta["factor"]
            if (m, v) not in results_map:
                results_map[(m, v)] = {"raw_counts": None, "zne_counts": {}, "meta": meta}
            if f == 1:
                results_map[(m, v)]["raw_counts"] = count_dict
            else:
                results_map[(m, v)]["zne_counts"][f] = count_dict

        m_arr = np.array(m_list, dtype=float)
        y_data, y_std = [], []
        
        for m in m_list:
            fidelities = []
            for variant in range(variant_count):
                rm = results_map[(m, variant)]
                meta = rm["meta"]
                res = self.calculate_fidelity(rm["raw_counts"], meta["target"], meta["reg_label"], meta["ilayout"], shots)
                
                if getattr(self, "zne_enabled", False):
                    extrapolator = ZNEExtrapolator()
                    zne_fids = {1: res["f_raw"]}
                    for f_zne, c_zne in rm["zne_counts"].items():
                        r_zne = self.calculate_fidelity(c_zne, meta["target"], meta["reg_label"], meta["ilayout"], shots)
                        zne_fids[f_zne] = r_zne["f_raw"]
                        
                    factors_sorted = sorted(zne_fids.keys())
                    values = [zne_fids[fac] for fac in factors_sorted]
                    zne_result = extrapolator.extrapolate(factors_sorted, values, self.zne_extrapolator_method)
                    res["f_zne"] = zne_result.bounded
                    
                if "f_zne" in res and res["f_zne"] is not None:
                    fidelities.append(res["f_zne"])
                elif "f_rem" in res and res["f_rem"] is not None:
                    fidelities.append(res["f_rem"])
                else:
                    fidelities.append(res["f_raw"])
                    
            f_emp = float(np.mean(fidelities))
            f_std = float(np.std(fidelities, ddof=1)) if variant_count > 1 else 0.0
            y_data.append(f_emp)
            y_std.append(f_std)
            print(f"    m={m:3d}  F_emp = {f_emp:.6f}  std = {f_std:.6f} ({variant_count} variants)")

        y_arr = np.array(y_data, dtype=float)
        p0 = [0.75, 0.90, self.B_ideal]
        bounds = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        popt, _ = curve_fit(rb_decay_model, m_arr, y_arr, p0=p0, bounds=bounds, maxfev=10_000)
        
        print("\n  Fit completed.")
        print(f"    A_fit = {popt[0]:.6f}")
        print(f"    p_fit = {popt[1]:.6f}")
        print(f"    B_fit = {popt[2]:.6f}")

        if plot_path is not None:
            self._plot_rb_curve(m_arr, y_arr, popt, plot_path)
            csv_path = plot_path.replace('.png', '.csv').replace('results', 'data')
            self._save_rb_results_to_csv(m_arr, y_arr, popt, csv_path, np.array(y_std))
            
        return popt

    def _save_rb_results_to_csv(
        self,
        m_arr: np.ndarray,
        y_data: np.ndarray,
        popt: np.ndarray,
        csv_path: str,
        y_std: np.ndarray | None = None,
    ) -> None:
        """Save RB characterization results to CSV file."""
        import csv
        from datetime import datetime
        
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
        
        A_fit, p_fit, B_fit = popt
        r_empirico = ((self.d - 1) * (1.0 - p_fit)) / self.d
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header with metadata
            writer.writerow(["SQM RB Characterization Results (SWAP protocol)"])
            writer.writerow(["Timestamp", datetime.now().isoformat()])
            writer.writerow(["Backend", self.backend.name])
            _state_labels = {0: "|0>", 1: "|1>", 2: "|+> (H)", 3: "|-> (XH)"}
            state_label = _state_labels.get(self.initial_state, f"unknown({self.initial_state})")
            writer.writerow(["Initial State", f"{self.initial_state} ({state_label})"])
            writer.writerow(["Architecture", f"2 registers * {self.N} qubits = {2*self.N} total qubits"])
            writer.writerow(["Hilbert Dimension", f"d = 2^{self.N} = {self.d}"])
            writer.writerow(["Native Gate", self.native_2q_gate.upper()])
            writer.writerow(["Pauli Twirling", "enabled" if self.pauli_twirling else "disabled"])
            writer.writerow(["Twirling Variants", self.twirling_variants if self.pauli_twirling else 1])
            writer.writerow(["Twirling Seed", self.twirling_seed if self.twirling_seed is not None else "random"])
            writer.writerow(["Mitigation ZNE", "enabled" if self.zne_enabled else "disabled"])
            if getattr(self, "zne_enabled", False):
                writer.writerow(["ZNE Noise Factors", getattr(self, "zne_noise_factors", "[]")])
                writer.writerow(["ZNE Extrapolator", getattr(self, "zne_extrapolator_method", "N/A")])
            writer.writerow(["Mitigation REM", "enabled" if self.rem_enabled else "disabled"])
            writer.writerow([])
            
            # Write fit parameters
            writer.writerow(["Magesan Fit Parameters"])
            writer.writerow(["A (SPAM contrast)", f"{A_fit:.6f}"])
            writer.writerow(["p (process decay)", f"{p_fit:.6f}"])
            writer.writerow(["B (asymptote)", f"{B_fit:.6f}"])
            writer.writerow(["r_empirical", f"{r_empirico:.6f}"])
            writer.writerow(["p_swap_theory", f"{self.p_swap_teorico:.6f}"])
            writer.writerow([])
            
            # Write data columns
            writer.writerow(["m (SWAP cycles)", "F_emp (mean fidelity)", "F_emp (std)", "F_fit (fitted)"])
            
            if y_std is None:
                y_std = np.zeros_like(y_data)
            for m, f_emp, f_std in zip(m_arr, y_data, y_std):
                f_fit = rb_decay_model(m, A_fit, p_fit, B_fit)
                writer.writerow([f"{int(m):d}", f"{f_emp:.6f}", f"{f_std:.6f}", f"{f_fit:.6f}"])
        
        print(f"\n  [CSV] RB results saved to: {csv_path}")

    # -- RB results report -----------------------------------------------------

    def print_rb_results(self, popt: np.ndarray) -> float:
       
        self.A_fit, self.p_fit, self.B_fit = popt

        self.r_empirico = ((self.d - 1) * (1.0 - self.p_fit)) / self.d

        print("\n" + "=" * 65)
        print("  SQM -- RB Fit Results (Magesan 2012)")
        print("=" * 65)

        print(f"\n  Model: F(m) = A * p^m + B")
        print(f"  {'Parameter':<12}  {'Value':>12}  Interpretation")
        print(f"  {'-'*52}")
        print(f"  {'A':<12}  {self.A_fit:>12.6f}  SPAM contrast (state prep + measurement)")
        print(f"  {'p':<12}  {self.p_fit:>12.6f}  Process decay per SWAP")
        print(f"  {'B':<12}  {self.B_fit:>12.6f}  Maximum mixing asymptote (ideal: 1/d={self.B_ideal:.4f})")

        print(f"\n  [PURIFIED EMPIRICAL ERROR]")
        print(f"    r_empirical = (d-1)/d * (1 - p_fit)")
        print(f"                = ({self.d - 1})/{self.d} * (1 - {self.p_fit:.6f})")
        print(f"                = {self.r_empirico:.6f}  ({self.r_empirico * 100:.4f} %)")

        print(f"\n  [COMPARISON WITH THEORETICAL MODEL]")
        print(f"    p_swap_theory ({self.native_2q_gate.upper()})      = "
              f"{self.p_swap_teorico:.6f}  ({self.p_swap_teorico * 100:.4f} %)")
        print(f"    r_empirical (RB)               = "
              f"{self.r_empirico:.6f}  ({self.r_empirico * 100:.4f} %)")

        diff_abs = abs(self.r_empirico - self.p_swap_teorico)
        diff_rel = (diff_abs / self.p_swap_teorico * 100) if self.p_swap_teorico > 0 else float("inf")
        print(f"    Relative difference            = {diff_rel:.2f} %")

        print(f"\n  [VERDICT]")
        if diff_rel > 5.0:
            print(f"    [RB MODEL REQUIRED] Errors differ by {diff_rel:.2f} %.")
            print(f"    Parameters A and B capture SPAM effects that")
            print(f"    the pure i.i.d. model cannot represent.")
        else:
            print(f"    [EQUIVALENT] Difference = {diff_rel:.2f} % < 5 %.")
            print(f"    Errors are essentially equal; the i.i.d. model")
            print(f"    is sufficient for this operation range.")

        print("=" * 65)
        return self.r_empirico

    # -- RB decay curve plot ---------------------------------------------------

    def _plot_rb_curve(
        self,
        m_arr: np.ndarray,
        y_data: np.ndarray,
        popt: np.ndarray,
        path: str,
    ) -> None:
        """Generate and save the RB decay curve vs empirical data."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        A_fit, p_fit, B_fit = popt
        m_dense = np.linspace(0, m_arr.max(), 300)
        f_fit   = rb_decay_model(m_dense, A_fit, p_fit, B_fit)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(m_arr, y_data, color="steelblue", zorder=5,
                   label="F_emp(m) - noisy simulation")
        ax.plot(m_dense, f_fit, color="crimson", linewidth=2,
                label=f"Magesan fit: A={A_fit:.3f}, p={p_fit:.4f}, B={B_fit:.3f}")
        ax.axhline(y=B_fit, linestyle="--", color="gray", alpha=0.6,
                   label=f"Asymptote B = {B_fit:.3f}")
        _state_labels = {0: "|0⟩", 1: "|1⟩", 2: "|+⟩ (H)", 3: "|-⟩ (XH)"}
        state_label = _state_labels.get(self.initial_state, f"state({self.initial_state})")
        target_label = ('1' * self.N) if self.initial_state in (1, 3) else ('0' * self.N)

        ax.set_xlabel("m  (number of SWAPs)", fontsize=12)
        ax.set_ylabel(f"F(m)  - P(|{target_label}⟩) survival probability", fontsize=12)
        ax.set_title(f"SWAP Decay Curve (N={self.N}, d={self.d}, init={state_label})",
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"\n  [PLOT] RB curve saved to: {path}")

    # -- Predicted fidelity from Magesan model ---------------------------------

    def theoretical_fidelity(self, n_swaps: int) -> float:
      
        if n_swaps < 0:
            raise ValueError(f"n_swaps must be >= 0, received: {n_swaps}")
        return self.A_fit * self.p_fit ** n_swaps + self.B_fit

    # -- Extrapolation validation (n vs 2n) -------------------------------------

    def run_extrapolation_test(self, n: int = 10) -> None:
        
        gate_label = self.native_2q_gate.upper()
        print("=" * 65)
        print(f"  SQM -- Phase B.4: Extrapolation Validation (n={n})")
        print("=" * 65)
        print(f"\n  p_{gate_label.lower()} = {self.cx_error:.6f}  |  "
              f"r_empirical = {self.r_empirico/(3*self.N):.6f}")

        f_th  = self.theoretical_fidelity(n)
        f_emp = self.empirical_fidelity(n)
        diff  = abs(f_th - f_emp)
        rel   = (diff / f_emp * 100) if f_emp > 0 else float("inf")
        print(f"\n  [n={n}]  F_model={f_th:.6f}  F_emp={f_emp:.6f}  "
              f"diff={diff:.6f} ({rel:.2f} %)")

        print("=" * 65)

    # -- Final C_MAX calculation (Magesan model) -------------------------------

    def calculate_final_cmax(self, target_fidelity: float = 0.90) -> int:
       
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
                "RB fit is invalid; increase m_list or shots."
            )

        # C_MAX = floor( log((F_target - B) / A) / log(p) )
        ratio = (target_fidelity - self.B_fit) / self.A_fit
        c_max = math.floor(math.log(ratio) / math.log(self.p_fit))

        print("\n" + "=" * 65)
        print("  SQM -- Final C_MAX Calculation (Magesan RB Model)")
        print("=" * 65)
        print(f"  RB fitting parameters:")
        print(f"    A_fit  = {self.A_fit:.6f}  (SPAM contrast)")
        print(f"    p_fit  = {self.p_fit:.6f}  (process decay)")
        print(f"    B_fit  = {self.B_fit:.6f}  (mixing asymptote)")
        print(f"\n  Purified process error:")
        print(f"    r_empirical = (d-1)/d * (1 - p_fit) = {self.r_empirico:.6f}  "
              f"({self.r_empirico * 100:.4f} %)")
        print(f"    p_swap_theory = {self.p_swap_teorico:.6f}  "
              f"({self.p_swap_teorico * 100:.4f} %)")
        print(f"\n  Target fidelity: F_target = {target_fidelity:.2f}  "
              f"({target_fidelity * 100:.0f} %)")
        print(f"\n  Formula: C_MAX = floor[ log((F_target-B)/A) / log(p) ]")
        print(f"         = floor[ log({ratio:.6f}) / log({self.p_fit:.6f}) ]")
        print(f"         = floor[ {math.log(ratio):.6f} / {math.log(self.p_fit):.6f} ]")
        print(f"\n  >>> C_MAX = {c_max} SWAPs")
        print(f"      F(C_MAX)   = {rb_decay_model(c_max,   self.A_fit, self.p_fit, self.B_fit):.6f}  (>= {target_fidelity:.2f})")
        print(f"      F(C_MAX+1) = {rb_decay_model(c_max+1, self.A_fit, self.p_fit, self.B_fit):.6f}  (<  {target_fidelity:.2f})")
        print("=" * 65)

        return c_max


# =============================================================================
# Entry point for direct execution
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # BACKEND MODE: "default" = FakeKyiv simulator | "IBM" = real IBM hardware
    # =========================================================================
    backend_mode = "IBM"  # Change to "IBM" to run on real IBM hardware
    twirling = False           # Set to True to enable Pauli twirling
    twirling_variants = 10    # Number of random circuits per RB point
    shots = 1024
    # =========================================================================
    # INITIAL STATE
    #   0 = |0⟩  : qubit starts in |0⟩, fidelity measured vs |0⟩
    #   1 = |1⟩  : qubit starts in |1⟩ (X gate), fidelity measured vs |1⟩
    #   2 = |+⟩  : qubit starts in |+⟩ (H gate), H applied before measure,
    #              fidelity measured vs |0⟩
    #   3 = |-⟩  : qubit starts in |-⟩ (X+H gates), H applied before measure,
    #              fidelity measured vs |1⟩
    # =========================================================================
    initial_state = 0  # 0 = |0⟩, 1 = |1⟩, 2 = |+⟩ (H), 3 = |-⟩ (XH)

    # 1. DEFINE THE ARCHITECTURE (N = Word width)
    N_qubits = 1
    m_list = [0, 1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40,50, 60, 80, 100]
    #m_list = [0, 1, 2, 3, 4]

    # Mitigation toggles
    use_zne = False
    use_rem = False
    mitigation_config = {
        "zne": {"enabled": use_zne, "noise_factors": [1, 3,5], "extrapolator": "exponential"},
        "rem": {"enabled": use_rem}
    } 
        # Extrapolation method: linear, polynomial, exponential
    # Noise amplification factors (positive odd integers)

    _state_labels = {0: "|0⟩", 1: "|1⟩", 2: "|+⟩ (H)", 3: "|-⟩ (XH)"}
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
        validator = CMaxValidator(
            N=N_qubits, 
            backend=ibm_backend, 
            initial_state=initial_state,
            pauli_twirling=twirling, 
            twirling_variants=twirling_variants,
            mitigation_config=mitigation_config
        )
    else:
        validator = CMaxValidator(
            N=N_qubits, 
            initial_state=initial_state,
            pauli_twirling=twirling, 
            twirling_variants=twirling_variants,
            mitigation_config=mitigation_config
        )

    # -- Phase B.1: Complete RB characterization -------------------------------
   
    popt = validator.run_rb_characterization(
        m_list, shots=shots,
        plot_path=f"results/{prefix}_decay_curve_swap_state{initial_state}_n{N_qubits}{suffix}.png",
    )

    # -- Phase B.2: Print results and validate model ---------------------------
    r_emp = validator.print_rb_results(popt)

    # -- Phase B.3: Calculate C_MAX with target fidelity ----------------------
    c_max = validator.calculate_final_cmax(target_fidelity=0.75)
    print(f"\n[FINAL RESULT]  C_MAX = {c_max} SWAPs  "
          f"(r_emp = {r_emp:.4f},  p_swap_theory = {validator.p_swap_teorico:.4f})")

    # -- Phase B.4: Extrapolation validation (Magesan model vs empirical) ------
    # Change 'x' to compare the fitted model against a new measurement.
    x = 15
    #validator.run_extrapolation_test(n=x)
    print(f"\n  Interpretation:")
    print(f"    If diff < 5%, the Magesan model extrapolates correctly to n={x}.")
    print(f"    r_emp={r_emp:.4f}  vs  p_swap_theory={validator.p_swap_teorico:.4f}")