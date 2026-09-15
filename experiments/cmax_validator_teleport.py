import math
import os
import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend - output to file
import matplotlib.pyplot as plt

from typing import cast
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import QuantumRegister, ClassicalRegister, Clbit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeKyiv

# Handle imports for both direct execution and module import
try:
    from src.functions.qubit_mapper import QubitMapper
    from src.functions.teleportation import SystolicTeleportation
    from src.utils.measurement_parser import MeasurementParser
    from src.mitigation import ReadoutMitigator, ZNEFolder, ZNEExtrapolator
    from experiments.utils.ibm_backend_helper import get_ibm_backend, run_on_ibm
except ModuleNotFoundError:
    # Add parent directory to path for direct script execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.functions.qubit_mapper import QubitMapper
    from src.functions.teleportation import SystolicTeleportation
    from src.utils.measurement_parser import MeasurementParser
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

class CMaxValidatorTeleport:
    """
    CMaxValidator for **multiple teleportations only** (no SWAPs).

    Architecture: 3 registers × N qubits = 3N total qubits
        - reg_A  : source/destination (ping)
        - reg_B  : destination/source (pong)
        - ancilla: Bell channel ancilla (reused each cycle via active reset)

    Protocol per RB point:
        1. State |0⟩ starts in reg_A
        2. Teleport reg_A → reg_B  (odd  cycles)
        3. Teleport reg_B → reg_A  (even cycles)
        4. After m teleportations, measure the register holding the final state
        5. Fidelity = P(|0...0⟩)

    The Magesan model F(m) = A·p^m + B is fitted to the empirical curve
    to extract p (per-teleport process decay) and subsequently C_MAX.
    """

    _TWO_QUBIT_GATE_CANDIDATES = ["ecr", "cx", "cz", "rzx"]

    # -- Constructor ----------------------------------------------------------

    def __init__(
        self,
        N: int = 1,
        backend=None,
        initial_state: int = 1,
        pauli_twirling: bool = False,
        twirling_seed: int | None = None,
        twirling_variants: int = 1,
        mitigation_config: dict | None = None,
    ) -> None:
        """
        Args:
            N: Word width (qubits per register). Total qubits = 3*N.
            backend: Optional external backend (e.g. real IBM). If None, uses FakeKyiv.
            initial_state: Initial qubit state and measurement basis.
                0 = |0⟩  -> init |0⟩, measure fidelity vs '0'*N
                1 = |1⟩  -> apply X, measure fidelity vs '1'*N  (original behaviour)
                2 = |+⟩  -> apply H, apply H before measure, fidelity vs '0'*N
                3 = |-⟩  -> apply X+H, apply H before measure, fidelity vs '1'*N
        """
        # 0. Dynamic word width parameter
        self.N = N
        self.d = 2 ** self.N          # Hilbert-space dimension per register
        self.B_ideal = 1.0 / self.d   # Maximum-mixing asymptote
        self.initial_state = initial_state
        self.pauli_twirling = pauli_twirling
        self.twirling_seed = twirling_seed
        if twirling_variants < 1:
            raise ValueError("twirling_variants must be >= 1")
        self.twirling_variants = twirling_variants

        # 1. Backend configuration
        if backend is not None:
            self.backend = backend
            self.is_ibm = True
            self.noise_model = None
            print(f"[CMaxValidatorTeleport] Using IBM hardware backend: {self.backend.name}")
        else:
            # 1. Reference backend (calibration snapshot from real IBM Kyiv)
            self.backend = FakeKyiv()
            self.is_ibm = False
            # 2. Complete noise model (depolarization + thermal relaxation)
            self.noise_model = NoiseModel.from_backend(self.backend)

        # 3. Detect native 2Q gate and extract average error.
        self.native_2q_gate: str = ""
        self.cx_error: float     = self._extract_avg_cx_error()

        # 4. Theoretical teleport error (per teleport cycle).
        #    A single teleportation uses 2 CNOTs + 1 Hadamard + feed-forward.
        #    Dominant noise contribution: 2 native 2Q gates per qubit, i.i.d.
        self.p_teleport_teorico: float = 1.0 - (1.0 - self.cx_error) ** (2 * N)

        # 5. RB fitting parameters — assigned in print_rb_results()
        self.A_fit:     float = 0.0
        self.p_fit:     float = 0.0
        self.B_fit:     float = 0.0
        self.r_empirico: float = 0.0

        # 6. Log initial state
        _state_labels = {0: "|0⟩", 1: "|1⟩", 2: "|+⟩ (H)", 3: "|-⟩ (XH)"}
        state_label = _state_labels.get(initial_state, f"unknown({initial_state})")
        print(f"[CMaxValidatorTeleport] Initial state : {state_label}")

        # 7. Error mitigation configuration (opt-in, default disabled)
        self.mitigation_config = mitigation_config or {}
        self.zne_enabled = self.mitigation_config.get("zne", {}).get("enabled", False)
        self.rem_enabled = self.mitigation_config.get("rem", {}).get("enabled", False)
        self.zne_noise_factors = self.mitigation_config.get("zne", {}).get("noise_factors", [1, 3])
        self.zne_extrapolator_method = self.mitigation_config.get("zne", {}).get("extrapolator", "linear")
        self.zne_seed = self.mitigation_config.get("zne", {}).get("zne_seed", None)
        if self.zne_enabled or self.rem_enabled:
            print(f"[CMaxValidatorTeleport] Mitigation: ZNE={'ON' if self.zne_enabled else 'OFF'}, "
                  f"REM={'ON' if self.rem_enabled else 'OFF'}")

    # -- Extraction of calibration parameters ----------------------------------

    def _extract_avg_cx_error(self) -> float:
        """Auto-detect native 2Q gate and return its average error rate."""
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

    # -- Physical chain allocation via QubitMapper -----------------------------

    def _get_physical_chains(self) -> list[tuple[int, int, int]]:
        """
        Find N disjoint chains of exactly 3 qubits each using QubitMapper.

        Returns: list of tuples (phys_A, phys_B, phys_ancilla)
        """
        mapper = QubitMapper(self.backend)

        chain_config = [
            ("reg_A",    self.N),   # Source/destination (ping)
            ("reg_B",    self.N),   # Destination/source (pong)
            ("ancilla",  self.N),   # Bell channel ancilla
        ]

        allocation = mapper.allocate_chain_topology(chain_config)

        chains = []
        for i in range(self.N):
            chains.append((
                allocation["reg_A"][i],
                allocation["reg_B"][i],
                allocation["ancilla"][i],
            ))

        return chains

    # -- Empirical fidelity (noisy multiple teleportations) --------------------

    def build_empirical_circuit(
        self,
        m_teleports: int,
        twirling_seed: int | None = None,
    ):
        """Build and transpile the circuit for m_teleports."""
        if m_teleports < 0:
            raise ValueError(f"m_teleports must be >= 0, received: {m_teleports}")

        reg_A   = QuantumRegister(self.N, name="A")
        reg_B   = QuantumRegister(self.N, name="B")
        ancilla = QuantumRegister(self.N, name="anc")
        cr_final = ClassicalRegister(self.N, name="cr_final")
        qc = QuantumCircuit(reg_A, reg_B, ancilla, cr_final)

        if self.initial_state == 1:
            for i in range(self.N): qc.x(reg_A[i])
        elif self.initial_state == 2:
            for i in range(self.N): qc.h(reg_A[i])
        elif self.initial_state == 3:
            for i in range(self.N): qc.x(reg_A[i]); qc.h(reg_A[i])

        teleporter = SystolicTeleportation(
            name="teleport_validator",
            pauli_twirling=self.pauli_twirling,
            twirling_seed=self.twirling_seed if twirling_seed is None else twirling_seed,
        )

        cr_bell = ClassicalRegister(2 * self.N, name="cr_bell")
        qc.add_register(cr_bell)

        for k in range(m_teleports):
            src_reg, dst_reg = (reg_A, reg_B) if k % 2 == 0 else (reg_B, reg_A)
            qc = teleporter.build_circuit(qc, source_reg=src_reg, dest_reg=dst_reg, ancilla_reg=ancilla, cr_bell=cr_bell)
            qc.barrier()

        if m_teleports == 0: final_reg = reg_A
        elif m_teleports % 2 == 1: final_reg = reg_B
        else: final_reg = reg_A

        if self.initial_state in (2, 3):
            for i in range(self.N): qc.h(final_reg[i])

        for i in range(self.N):
            qc.measure(final_reg[i], cr_final[i])

        chains = self._get_physical_chains()
        initial_layout = [0] * (3 * self.N)
        for i, (phys_a, phys_b, phys_anc) in enumerate(chains):
            initial_layout[i]              = phys_a
            initial_layout[self.N + i]     = phys_b
            initial_layout[2 * self.N + i] = phys_anc
        
        qc_t = transpile(qc, backend=self.backend, optimization_level=0, initial_layout=initial_layout)

        register_layout = MeasurementParser.build_register_layout_from_order(
            register_names=["cr_final", "cr_bell"],
            register_sizes=[self.N, 2 * self.N],
            reverse_for_endianness=True,
        )
        target_state = ('1' * self.N) if self.initial_state in (1, 3) else ('0' * self.N)
        return qc_t, register_layout, target_state, final_reg, initial_layout

    def calculate_fidelity(self, counts, register_layout, target_state, final_reg, initial_layout, shots) -> dict:
        fidelity_count = 0
        for bitstring, count in counts.items():
            final_bits = MeasurementParser.extract_register_bits(bitstring, "cr_final", register_layout)
            if final_bits == target_state:
                fidelity_count += count
        f_raw = fidelity_count / shots

        if not (self.zne_enabled or self.rem_enabled):
            return {"f_raw": f_raw}

        mitigation_result = {"f_raw": f_raw}
        if self.rem_enabled:
            try:
                if final_reg.name == "B":
                    final_physical = [initial_layout[self.N + i] for i in range(self.N)]
                else:
                    final_physical = [initial_layout[i] for i in range(self.N)]
                rem = ReadoutMitigator.from_backend_properties(self.backend, final_physical)
                corrected = rem.apply(counts)
                rem_count = 0
                for b, c in corrected.items():
                    fb = MeasurementParser.extract_register_bits(b, "cr_final", register_layout)
                    if fb == target_state: rem_count += c
                mitigation_result["f_rem"] = rem_count / max(sum(corrected.values()), 1)
            except Exception as e:
                print(f"    [REM] Skipped: {e}")
                mitigation_result["f_rem"] = None
        return mitigation_result

    def run_rb_characterization(
        self,
        m_list: list[int],
        shots: int = 4000,
        plot_path: str | None = "results/rb_decay_curve_teleport.png",
    ) -> np.ndarray:
        print("=" * 70)
        print("  TELEPORT -- Phase B: RB Characterization (Batch Mode)")
        print("=" * 70)
        print(f"  Backend        : {self.backend.name}")
        print(f"  p_teleport_theory: {self.p_teleport_teorico:.6f}")
        
        variant_count = self.twirling_variants if self.pauli_twirling else 1
        all_circuits = []
        metadata = []
        
        # 1. Generate all circuits
        for m in m_list:
            for variant in range(variant_count):
                seed = None
                if self.twirling_seed is not None:
                    seed = self.twirling_seed + variant * 1_000_003 + m * 1_009
                qc_t, layout, target, final_reg, initial_layout = self.build_empirical_circuit(m, seed)
                
                all_circuits.append(qc_t)
                metadata.append({"m": m, "variant": variant, "factor": 1, "layout": layout, "target": target, "freg": final_reg, "ilayout": initial_layout})
                
                if self.zne_enabled:
                    folder = ZNEFolder(seed=self.zne_seed)
                    for factor in self.zne_noise_factors:
                        if factor == 1: continue
                        folded = folder.fold_circuit(qc_t, factor)
                        all_circuits.append(folded)
                        metadata.append({"m": m, "variant": variant, "factor": factor, "layout": layout, "target": target, "freg": final_reg, "ilayout": initial_layout})

        print(f"  Generated {len(all_circuits)} circuits for batch execution.")
        
        # 2. Execute all circuits
        if self.is_ibm:
            all_counts = run_on_ibm(all_circuits, self.backend, shots=shots)
        else:
            sim = AerSimulator(noise_model=self.noise_model)
            job = sim.run(all_circuits, shots=shots)
            res = job.result()
            all_counts = [res.get_counts(i) for i in range(len(all_circuits))]

        # 3. Process results
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
                res = self.calculate_fidelity(rm["raw_counts"], meta["layout"], meta["target"], meta["freg"], meta["ilayout"], shots)
                
                if self.zne_enabled:
                    extrapolator = ZNEExtrapolator()
                    zne_fids = {1: res["f_raw"]}
                    for f_zne, c_zne in rm["zne_counts"].items():
                        r_zne = self.calculate_fidelity(c_zne, meta["layout"], meta["target"], meta["freg"], meta["ilayout"], shots)
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
            writer.writerow(["RB Characterization Results (Multiple Teleportation Protocol)"])
            writer.writerow(["Timestamp", datetime.now().isoformat()])
            writer.writerow(["Backend", self.backend.name])
            _state_labels = {0: "|0>", 1: "|1>", 2: "|+> (H)", 3: "|-> (XH)"}
            state_label = _state_labels.get(self.initial_state, f"unknown({self.initial_state})")
            writer.writerow(["Initial State", f"{self.initial_state} ({state_label})"])
            writer.writerow(["Architecture", f"3 registers × {self.N} qubits = {3*self.N} total qubits"])
            writer.writerow(["Registers", "reg_A (ping), reg_B (pong), ancilla (Bell channel)"])
            writer.writerow(["Hilbert Dimension", f"d = 2^{self.N} = {self.d}"])
            writer.writerow(["Native Gate", self.native_2q_gate.upper()])
            writer.writerow(["Protocol", "SystolicTeleportation (ping-pong, active reset)"])
            writer.writerow(["Pauli Twirling", "enabled" if self.pauli_twirling else "disabled"])
            writer.writerow(["Twirling Variants", self.twirling_variants if self.pauli_twirling else 1])
            writer.writerow(["Twirling Seed", self.twirling_seed if self.twirling_seed is not None else "random"])
            writer.writerow(["Mitigation ZNE", "enabled" if self.zne_enabled else "disabled"])
            writer.writerow(["Mitigation REM", "enabled" if self.rem_enabled else "disabled"])
            writer.writerow([])
            
            # Write fit parameters
            writer.writerow(["Magesan Fit Parameters"])
            writer.writerow(["A (SPAM contrast)", f"{A_fit:.6f}"])
            writer.writerow(["p (process decay per teleport)", f"{p_fit:.6f}"])
            writer.writerow(["B (asymptote)", f"{B_fit:.6f}"])
            writer.writerow(["r_empirical", f"{r_empirico:.6f}"])
            writer.writerow(["p_teleport_theory", f"{self.p_teleport_teorico:.6f}"])
            writer.writerow([])
            
            # Write data columns
            writer.writerow(["m (teleport cycles)", "F_emp (mean fidelity)", "F_emp (std)", "F_fit (fitted)"])
            
            if y_std is None:
                y_std = np.zeros_like(y_data)
            for m, f_emp, f_std in zip(m_arr, y_data, y_std):
                f_fit = rb_decay_model(m, A_fit, p_fit, B_fit)
                writer.writerow([f"{int(m):d}", f"{f_emp:.6f}", f"{f_std:.6f}", f"{f_fit:.6f}"])
        
        print(f"\n  [CSV] RB results saved to: {csv_path}")

    # -- RB results report -----------------------------------------------------

    def print_rb_results(self, popt: np.ndarray) -> float:
        """Print detailed RB results and return the purified empirical error."""
        self.A_fit, self.p_fit, self.B_fit = popt

        self.r_empirico = ((self.d - 1) * (1.0 - self.p_fit)) / self.d

        print("\n" + "=" * 70)
        print("  TELEPORT -- RB Fit Results (Magesan 2012, Teleport-Only)")
        print("=" * 70)

        print(f"\n  Model: F(m) = A · p^m + B")
        print(f"  {'Parameter':<15}  {'Value':>12}  Interpretation")
        print(f"  {'-'*65}")
        print(f"  {'A':<15}  {self.A_fit:>12.6f}  SPAM + teleport protocol overhead")
        print(f"  {'p':<15}  {self.p_fit:>12.6f}  Process decay per teleportation")
        print(f"  {'B':<15}  {self.B_fit:>12.6f}  Max mixing asymptote (ideal: 1/d={self.B_ideal:.4f})")

        print(f"\n  [ARCHITECTURE]  3 registers × {self.N} qubits = {3*self.N} total qubits")
        print(f"                  (reg_A, reg_B, ancilla)")
        print(f"                  Hilbert space dimension: d = 2^{self.N} = {self.d}")

        print(f"\n  [PURIFIED EMPIRICAL ERROR]")
        print(f"    r_empirical = (d-1)/d × (1 - p_fit)")
        print(f"                = ({self.d - 1})/{self.d} × (1 - {self.p_fit:.6f})")
        print(f"                = {self.r_empirico:.6f}  ({self.r_empirico * 100:.4f} %)")

        print(f"\n  [COMPARISON WITH THEORETICAL MODEL]")
        print(f"    p_teleport_theory ({self.native_2q_gate.upper()})    = "
              f"{self.p_teleport_teorico:.6f}  ({self.p_teleport_teorico * 100:.4f} %)")
        print(f"    r_empirical (RB)               = "
              f"{self.r_empirico:.6f}  ({self.r_empirico * 100:.4f} %)")

        diff_abs = abs(self.r_empirico - self.p_teleport_teorico)
        diff_rel = (diff_abs / self.p_teleport_teorico * 100) if self.p_teleport_teorico > 0 else float("inf")
        print(f"    Relative difference            = {diff_rel:.2f} %")

        print(f"\n  [VERDICT]")
        if diff_rel > 5.0:
            print(f"    [RB MODEL REQUIRED] Errors differ by {diff_rel:.2f} %.")
            print(f"    Parameters A and B capture SPAM + teleportation protocol effects")
            print(f"    (active reset noise, mid-circuit measurement, feed-forward).")
        else:
            print(f"    [EQUIVALENT] Difference = {diff_rel:.2f} % < 5 %.")
            print(f"    Errors are essentially equal within tolerance.")

        print(f"\n  [TELEPORTATION PROTOCOL NOTES]")
        print(f"    Each cycle = Bell pair + BSM + feed-forward + active reset.")
        print(f"    Error sources per cycle:")
        print(f"      - 2 CNOT gates (dominant): ~{self.cx_error:.6f} per gate")
        print(f"      - 1 Hadamard gate: negligible single-qubit error")
        print(f"      - Mid-circuit measurement + classically conditioned X/Z")
        print(f"      - Active reset on source and ancilla (decoherence during reset)")
        print(f"    The parameter p={self.p_fit:.6f} captures ALL these effects combined.")

        print("=" * 70)
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

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(m_arr, y_data, color="darkorange", s=100, zorder=5,
                   label="F_emp(m) – noisy teleportation simulation")
        ax.plot(m_dense, f_fit, color="darkviolet", linewidth=2.5,
                label=f"Magesan fit: A={A_fit:.3f}, p={p_fit:.4f}, B={B_fit:.3f}")
        ax.axhline(y=B_fit, linestyle="--", color="gray", alpha=0.6,
                   label=f"Asymptote B = {B_fit:.3f}")
        _state_labels = {0: "|0⟩", 1: "|1⟩", 2: "|+⟩ (H)", 3: "|-⟩ (XH)"}
        state_label = _state_labels.get(self.initial_state, f"state({self.initial_state})")
        target_label = ('1' * self.N) if self.initial_state in (1, 3) else ('0' * self.N)

        ax.set_xlabel("m  (number of teleportation cycles)", fontsize=13, fontweight='bold')
        ax.set_ylabel(f"F(m)  – P(|{target_label}⟩) survival probability", fontsize=13, fontweight='bold')
        ax.set_title(f"Teleport-Only Decay Curve (N={self.N}, d={self.d}, init={state_label})",
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"\n  [PLOT] RB curve saved to: {path}")

    # -- Predicted fidelity from Magesan model ---------------------------------

    def theoretical_fidelity(self, n_teleports: int) -> float:
        """F(m) = A · p^m + B — predict fidelity after n teleportations."""
        if n_teleports < 0:
            raise ValueError(f"n_teleports must be >= 0, received: {n_teleports}")
        return self.A_fit * self.p_fit ** n_teleports + self.B_fit

    # -- Extrapolation validation (model vs fresh measurement) -----------------

    def run_extrapolation_test(self, n: int = 10) -> None:
        """Compare model prediction vs fresh empirical measurement at m=n."""
        gate_label = self.native_2q_gate.upper()
        print("=" * 70)
        print(f"  TELEPORT -- Phase B.4: Extrapolation Validation (n={n})")
        print("=" * 70)
        print(f"\n  Architecture: 3 registers × {self.N} qubits = {3*self.N} total qubits")
        print(f"  p_{gate_label.lower()} = {self.cx_error:.6f}  |  "
              f"r_empirical = {self.r_empirico/(2*self.N):.6f}")

        f_th  = self.theoretical_fidelity(n)
        f_emp = self.empirical_fidelity(n)
        diff  = abs(f_th - f_emp)
        rel   = (diff / f_emp * 100) if f_emp > 0 else float("inf")
        print(f"\n  [n={n}]  F_model={f_th:.6f}  F_emp={f_emp:.6f}  "
              f"diff={diff:.6f} ({rel:.2f} %)")

        if rel < 5.0:
            print(f"  [OK] Model extrapolates well (diff < 5 %)")
        else:
            print(f"  [WARN] Model deviation {rel:.2f} % — may need higher shots")

        print("=" * 70)

    # -- Final C_MAX calculation (Magesan model) -------------------------------

    def calculate_final_cmax(self, target_fidelity: float = 0.90) -> int:
        """
        Compute the maximum number of teleportation cycles C_MAX such that
        F(C_MAX) >= target_fidelity.

        Formula:  C_MAX = floor[ log((F_target - B) / A) / log(p) ]
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
                "RB fit is invalid; increase m_list or shots."
            )

        # C_MAX = floor( log((F_target - B) / A) / log(p) )
        ratio = (target_fidelity - self.B_fit) / self.A_fit
        c_max = math.floor(math.log(ratio) / math.log(self.p_fit))

        print("\n" + "=" * 70)
        print("  TELEPORT -- Final C_MAX Calculation (Magesan RB Model)")
        print("=" * 70)
        print(f"  RB fitting parameters:")
        print(f"    A_fit  = {self.A_fit:.6f}  (SPAM + teleport overhead)")
        print(f"    p_fit  = {self.p_fit:.6f}  (process decay per teleport)")
        print(f"    B_fit  = {self.B_fit:.6f}  (mixing asymptote)")
        print(f"\n  Purified process error:")
        print(f"    r_empirical = (d-1)/d × (1 - p_fit) = {self.r_empirico:.6f}  "
              f"({self.r_empirico * 100:.4f} %)")
        print(f"    p_teleport_theory = {self.p_teleport_teorico:.6f}  "
              f"({self.p_teleport_teorico * 100:.4f} %)")
        print(f"\n  Target fidelity: F_target = {target_fidelity:.2f}  "
              f"({target_fidelity * 100:.0f} %)")
        print(f"\n  Formula: C_MAX = floor[ log((F_target-B)/A) / log(p) ]")
        print(f"         = floor[ log({ratio:.6f}) / log({self.p_fit:.6f}) ]")
        print(f"         = floor[ {math.log(ratio):.6f} / {math.log(self.p_fit):.6f} ]")
        print(f"\n  >>> C_MAX = {c_max} teleportations")
        print(f"      F(C_MAX)   = {rb_decay_model(c_max,   self.A_fit, self.p_fit, self.B_fit):.6f}  (>= {target_fidelity:.2f})")
        print(f"      F(C_MAX+1) = {rb_decay_model(c_max+1, self.A_fit, self.p_fit, self.B_fit):.6f}  (<  {target_fidelity:.2f})")
        print("=" * 70)

        return c_max


# =============================================================================
# Entry point for direct execution
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # BACKEND MODE: "default" = FakeKyiv simulator | "IBM" = real IBM hardware
    # =========================================================================
    backend_mode = "default"  # Change to "IBM" to run on real IBM hardware
    twirling = False           # Set to True to enable Pauli twirling
    twirling_variants = 5    # Number of random circuits per teleport point
    shots = 1024             # Number of shots per circuit 
    # =========================================================================
    # INITIAL STATE
    #   0 = |0⟩  : qubit starts in |0⟩, fidelity measured vs |0⟩
    #   1 = |1⟩  : qubit starts in |1⟩ (X gate), fidelity measured vs |1⟩
    #   2 = |+⟩  : qubit starts in |+⟩ (H gate), H applied before measure,
    #              fidelity measured vs |0⟩
    #   3 = |-⟩  : qubit starts in |-⟩ (X+H gates), H applied before measure,
    #              fidelity measured vs |1⟩
    # =========================================================================
    initial_state = 2  # 0 = |0⟩, 1 = |1⟩, 2 = |+⟩ (H), 3 = |-⟩ (XH)
    m_list = [0, 1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
    #m_list = [0, 1, 2]
    # 1. DEFINE THE ARCHITECTURE (N = Word width)
    N_qubits = 1 

    # Mitigation toggles 
    use_zne = False
    use_rem = False  
    mitigation_config = {
        "zne": {"enabled": use_zne, "noise_factors": [1, 3], "extrapolator": "exponential"},
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
        validator = CMaxValidatorTeleport(
            N=N_qubits,
            backend=ibm_backend,
            initial_state=initial_state,
            pauli_twirling=twirling,
            twirling_variants=twirling_variants,
            mitigation_config=mitigation_config
        )
    else:
        validator = CMaxValidatorTeleport(
            N=N_qubits,
            initial_state=initial_state,
            pauli_twirling=twirling,
            twirling_variants=twirling_variants,
            mitigation_config=mitigation_config
        )

    # -- Phase B.1: Complete RB characterization (teleport-only) ---------------
 
    popt = validator.run_rb_characterization(
        m_list, shots=shots,
        plot_path=f"results/{prefix}_decay_curve_teleport_state{initial_state}_n{N_qubits}{suffix}.png",
    )

    # -- Phase B.2: Print results and validate model ---------------------------
    r_emp = validator.print_rb_results(popt)

    # -- Phase B.3: Calculate C_MAX with target fidelity -----------------------
    c_max = validator.calculate_final_cmax(target_fidelity=0.75)
    print(f"\n[FINAL RESULT]  C_MAX = {c_max} teleportations  "
          f"(r_emp = {r_emp:.4f},  p_teleport_theory = {validator.p_teleport_teorico:.4f})")

    # -- Phase B.4: Extrapolation validation (Magesan model vs empirical) ------
    x = 15
    #validator.run_extrapolation_test(n=x)
    print(f"\n  Interpretation:")
    print(f"    If diff < 5%, the Magesan model extrapolates correctly to n={x}.")
    print(f"    r_emp={r_emp:.4f}  vs  p_teleport_theory={validator.p_teleport_teorico:.4f}")
