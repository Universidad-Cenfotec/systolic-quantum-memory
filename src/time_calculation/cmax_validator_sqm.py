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
    from src.time_calculation.ibm_backend_helper import get_ibm_backend, run_on_ibm
except ModuleNotFoundError:
    # Add parent directory to path for direct script execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from src.functions.qubit_mapper import QubitMapper
    from src.functions.teleportation import SystolicTeleportation
    from src.utils.measurement_parser import MeasurementParser
    from src.time_calculation.ibm_backend_helper import get_ibm_backend, run_on_ibm


# =============================================================================
# Magesan decay model (global reusable function)
# =============================================================================

def rb_decay_model(m: float, A: float, p: float, B: float) -> float:
   
    return A * (p ** m) + B


# =============================================================================
# Main class
# =============================================================================

class CMaxValidator:
   
    
    # Candidate 2-qubit gates, in order of priority.
    # IBM Kyiv (Eagle r3) uses ECR natively; other platforms use CX.
    _TWO_QUBIT_GATE_CANDIDATES = ["ecr", "cx", "cz", "rzx"]

    # -- Constructor ----------------------------------------------------------

    def __init__(self, N: int = 1, backend=None) -> None:
      
        # 0. Dynamic word width parameter (now uses 4*N total qubits)
        self.N = N
        self.d = 2 ** self.N  # Hilbert space dimension per register
        self.B_ideal = 1.0 / self.d

        # 1. Backend configuration
        if backend is not None:
            self.backend = backend
            self.is_ibm = True
            self.noise_model = None
            print(f"[CMaxValidator] Using IBM hardware backend: {self.backend.name}")
        else:
            # 1. Reference backend (calibration snapshot from real IBM Brisbane)
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

        # 6. Initialize SystolicTeleportation module
        self.teleportation = SystolicTeleportation(name="cmax_validator_teleportation")

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

    # -- Physical chain allocation via QubitMapper (NEW UNIFIED APPROACH) ------

    def _get_physical_chains(self) -> list[tuple[int, int, int, int]]:
        """
        Find N disjoint chains of exactly 4 qubits each using QubitMapper.
        
        Uses SQM per-bit topology where:
        - q_work[i]: operation qubit for bit i
        - mem_orig_0[i]: storage qubit for bit i
        - tele_ancilla_0[i]: link alice qubit for bit i
        - mem_backup_0[i]: link bob qubit for bit i
        
        Returns: list of tuples (storage, operation, link_alice, link_bob)
        """
        # Create an instance of QubitMapper from the backend
        mapper = QubitMapper(self.backend)
        
        # Build chain config for SQM with 1 register (R=1)
        # This will use allocate_sqm_per_bit_topology internally
        chain_config = [
            ("q_work", self.N),           # Operation register
            ("mem_orig_0", self.N),       # Storage register (in mem_orig)
            ("mem_backup_0", self.N),     # Link Bob (in mem_backup)
            ("tele_ancilla_0", self.N),   # Link Alice (in tele_ancilla)
        ]
        
        # Allocate using SQM per-bit topology
        allocation = mapper.allocate_chain_topology(chain_config)
        
        # Convert allocation back to tuple format: (storage, operation, link_alice, link_bob)
        chains = []
        for i in range(self.N):
            o_qubit = allocation["q_work"][i]
            s_qubit = allocation["mem_orig_0"][i]
            la_qubit = allocation["tele_ancilla_0"][i]
            lb_qubit = allocation["mem_backup_0"][i]
            chains.append((s_qubit, o_qubit, la_qubit, lb_qubit))
        
        return chains

    # -- Empirical fidelity with SystolicTeleportation protocol ----------------

    def empirical_fidelity(self, m_swaps: int, shots: int = 4000) -> float:
       
        if m_swaps < 0:
            raise ValueError(f"m_swaps must be >= 0, received: {m_swaps}")

        # -- Build 4-register quantum circuit (same as cmax_validator.py) -------
        reg_s = QuantumRegister(self.N, name="S")      # Storage
        reg_o = QuantumRegister(self.N, name="O")      # Operation
        reg_la = QuantumRegister(self.N, name="LA")    # Link Alice
        reg_lb = QuantumRegister(self.N, name="LB")    # Link Bob
        
        cr_lb = ClassicalRegister(self.N, name="cr_lb")         # LB measurements
        
        qc = QuantumCircuit(reg_s, reg_o, reg_la, reg_lb, cr_lb)

        # --------- PHASE 1: PREPARATION (all |0>) - implicit by circuit initialization
        
        # --------- PHASE 2: m SWAP CYCLES (Storage <-> Operation)
        
        for _ in range(m_swaps):
            for i in range(self.N):
                q_s = reg_s[i]
                q_o = reg_o[i]
                
                # SWAP implementation: 3 CNOTs
                qc.cx(q_s, q_o)
                qc.cx(q_o, q_s)
                qc.cx(q_s, q_o)
            
            qc.barrier()  # Prevent inter-SWAP optimization

        # --------- PHASE 3: TELEPORTATION PROTOCOL (S -> LA -> LB)
        # Use SystolicTeleportation.build_circuit() which includes active reset
        # This replaces manual EPR, BSM, feed-forward with unified implementation
        
        cr_bell = ClassicalRegister(2 * self.N, name="cr_bell")
        qc.add_register(cr_bell)
        
        qc = self.teleportation.build_circuit(
            qc, 
            source_reg=reg_s,
            dest_reg=reg_lb,
            ancilla_reg=reg_la,
            cr_bell=cr_bell,
        )
        
        qc.barrier()
        
        # --------- PHASE 4: FINAL MEASUREMENT (LinkBob only)
        
        # Measure LinkBob register (final state for fidelity calculation)
        for i in range(self.N):
            qc.measure(reg_lb[i], cr_lb[i])

        # --------- HARDWARE-AWARE QUBIT MAPPING
        chains = self._get_physical_chains()
        
        initial_layout = [0] * (4 * self.N)
        for i, (phys_s, phys_o, phys_la, phys_lb) in enumerate(chains):
            initial_layout[i] = phys_s                           # Storage
            initial_layout[self.N + i] = phys_o                  # Operation
            initial_layout[2 * self.N + i] = phys_la             # Link Alice
            initial_layout[3 * self.N + i] = phys_lb             # Link Bob
            
        # --------- TRANSPILE AND SIMULATE
        
        qc_t = transpile(qc, backend=self.backend, optimization_level=0, initial_layout=initial_layout)

        if self.is_ibm:
            # Run on real IBM hardware via SamplerV2
            counts = run_on_ibm(qc_t, self.backend, shots=shots)
        else:
            # Run on local AerSimulator with noise model
            sim  = AerSimulator(noise_model=self.noise_model)
            job  = sim.run(qc_t, shots=shots)
            counts = job.result().get_counts()

        # --------- DIRECT VERIFICATION OF TELEPORTED STATUS
        
        # Build layout for register extraction (Little-Endian aware)
        # Registers added in order: cr_lb (first), then cr_bell (second)
        # Qiskit Little-Endian: last added appears first in bitstring
        register_layout = MeasurementParser.build_register_layout_from_order(
            register_names=["cr_lb", "cr_bell"],
            register_sizes=[self.N, 2 * self.N],
            reverse_for_endianness=True
        )
        
        fidelity_count = 0
        target_state = '0' * self.N
        
        for bitstring, count in counts.items():
            # Extract Link Bob bits using the register layout (endianness-safe)
            lb_bits = MeasurementParser.extract_register_bits(
                bitstring, 
                "cr_lb", 
                register_layout
            )
            
            if lb_bits == target_state:
                fidelity_count += count
                
        return fidelity_count / shots

    # -- RB Characterization (Magesan) -----------------------------------------

    def run_rb_characterization(
        self,
        m_list: list[int],
        shots: int = 2048,
        plot_path: str | None = "results/rb_decay_curve_.png",
    ) -> np.ndarray:

        print("=" * 75)
        print("  SQM -- Phase B: RB Characterization with SystolicTeleportation")
        print("=" * 75)
        print(f"\n  Backend      : {self.backend.name}")
        print(f"  Architecture : 4 registers * {self.N} qubits = {4*self.N} total qubits")
        print(f"                 (Storage, Operation, LinkAlice, LinkBob)")
        print(f"  Hilbert dim  : d = 2^{self.N} = {self.d}")
        print(f"  Native gate  : {self.native_2q_gate.upper()}")
        print(f"  p_swap_theory: {self.p_swap_teorico:.6f}  "
              f"({self.p_swap_teorico * 100:.4f} %)")
        print(f"\n  Measuring F_emp(m) for m = {m_list} with SystolicTeleportation...")
        print(f"  shots per point = {shots}\n")

        # -- Empirical data collection -----------------------------------------
        m_arr  = np.array(m_list, dtype=float)
        y_data: list[float] = []

        for m in m_list:
            f_emp = self.empirical_fidelity(m, shots=shots)
            y_data.append(f_emp)
            print(f"    m={m:3d}  F_emp = {f_emp:.6f}")

        y_arr = np.array(y_data, dtype=float)

        # -- curve_fit adjustment ----------------------------------------------
        p0     = [0.75, 0.90, self.B_ideal]
        bounds = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

        popt, _ = curve_fit(
            rb_decay_model,
            m_arr,
            y_arr,
            p0=p0,
            bounds=bounds,
            maxfev=10_000,
        )

        print(f"\n  Fit completed.")
        print(f"    A_fit = {popt[0]:.6f}")
        print(f"    p_fit = {popt[1]:.6f}")
        print(f"    B_fit = {popt[2]:.6f}")

        # -- Plot (optional) ---------------------------------------------------
        if plot_path is not None:
            self._plot_rb_curve(m_arr, y_arr, popt, plot_path)

        # -- Save results to CSV -----------------------------------------------
        csv_path = plot_path.replace('.png', '.csv').replace('results', 'data') if plot_path else "data/rb_characterization.csv"
        self._save_rb_results_to_csv(m_arr, y_arr, popt, csv_path)

        return popt

    # -- Save RB results to CSV -----------------------------------------------

    def _save_rb_results_to_csv(
        self,
        m_arr: np.ndarray,
        y_data: np.ndarray,
        popt: np.ndarray,
        csv_path: str,
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
            writer.writerow(["SQM RB Characterization Results (SystolicTeleportation)"])
            writer.writerow(["Timestamp", datetime.now().isoformat()])
            writer.writerow(["Backend", self.backend.name])
            writer.writerow(["Architecture", f"4 registers * {self.N} qubits = {4*self.N} total qubits"])
            writer.writerow(["Hilbert Dimension", f"d = 2^{self.N} = {self.d}"])
            writer.writerow(["Native Gate", self.native_2q_gate.upper()])
            writer.writerow(["Teleportation Module", "SystolicTeleportation (with active reset)"])
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
            writer.writerow(["m (SWAP cycles)", "F_emp (fidelity)", "F_fit (fitted)"])
            
            for m, f_emp in zip(m_arr, y_data):
                f_fit = rb_decay_model(m, A_fit, p_fit, B_fit)
                writer.writerow([f"{int(m):d}", f"{f_emp:.6f}", f"{f_fit:.6f}"])
        
        print(f"\n  [CSV] RB results saved to: {csv_path}")

    # -- RB results report -----------------------------------------------------

    def print_rb_results(self, popt: np.ndarray) -> float:
       
        self.A_fit, self.p_fit, self.B_fit = popt

        self.r_empirico = ((self.d - 1) * (1.0 - self.p_fit)) / self.d

        print("\n" + "=" * 75)
        print("  SQM -- RB Fit Results (Magesan 2012 + SystolicTeleportation)")
        print("=" * 75)

        print(f"\n  Model: F(m) = A * p^m + B")
        print(f"  {'Parameter':<15}  {'Value':>12}  Interpretation")
        print(f"  {'-'*65}")
        print(f"  {'A':<15}  {self.A_fit:>12.6f}  SPAM + teleportation 'toll'")
        print(f"  {'p':<15}  {self.p_fit:>12.6f}  Process decay per SWAP cycle")
        print(f"  {'B':<15}  {self.B_fit:>12.6f}  Max mixing asymptote (ideal: 1/d={self.B_ideal:.4f})")

        print(f"\n  [ARCHITECTURE]  4 registers * {self.N} qubits = {4*self.N} total qubits")
        print(f"                  (Storage, Operation, LinkAlice, LinkBob)")
        print(f"                  Hilbert space dimension: d = 2^{self.N} = {self.d}")

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
            print(f"    Parameters A, p, and B capture SPAM + teleportation effects.")
        else:
            print(f"    [EQUIVALENT] Difference = {diff_rel:.2f} % < 5 %.")
            print(f"    Errors are essentially equal within tolerance.")

        print("\n  [TELEPORTATION PROTOCOL IMPACT]")
        print(f"    The parameter A={self.A_fit:.6f} now represents")
        print(f"    the fidelity after surviving the teleportation 'tollbooth'.")
        print(f"    Active reset (qc.reset) in SystolicTeleportation eliminates")
        print(f"    entanglement residue, enabling more accurate C_MAX calculation.")
        print(f"    C_MAX calculation using this A guarantees a post-rescue")
        print(f"    coherence bound that accounts for protocol overhead.")

        print("=" * 75)
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
        ax.scatter(m_arr, y_data, color="steelblue", s=100, zorder=5,
                   label="F_emp(m) - SystolicTeleportation with active reset")
        ax.plot(m_dense, f_fit, color="crimson", linewidth=2.5,
                label=f"Magesan fit: A={A_fit:.3f}, p={p_fit:.4f}, B={B_fit:.3f}")
        ax.axhline(y=B_fit, linestyle="--", color="gray", alpha=0.6,
                   label=f"Asymptote B = {B_fit:.3f}")
        ax.set_xlabel("m  (SWAP cycles)", fontsize=13, fontweight='bold')
        ax.set_ylabel("F(m)  - survival probability", fontsize=13, fontweight='bold')
        ax.set_title(f"SQM RB Decay Curve with SystolicTeleportation (N={self.N}, d={self.d})", 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')
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
        print("=" * 75)
        print(f"  SQM -- Phase B.4: RB Extrapolation Validation (n={n})")
        print("=" * 75)
        print(f"\n  Architecture: 4 registers * {self.N} qubits = {4*self.N} total qubits")
        print(f"  p_{gate_label.lower()} = {self.cx_error:.6f}  |  "
              f"r_empirical = {self.r_empirico/(3*self.N):.6f}")

        f_th  = self.theoretical_fidelity(n)
        f_emp = self.empirical_fidelity(n)
        diff  = abs(f_th - f_emp)
        rel   = (diff / f_emp * 100) if f_emp > 0 else float("inf")
        print(f"\n  [n={n}]  F_model={f_th:.6f}  F_emp={f_emp:.6f}  "
              f"diff={diff:.6f} ({rel:.2f} %)")
        
        if rel < 5.0:
            print(f"  [OK] Model extrapolates well (diff < 5 %)")
        else:
            print(f"  [WARN] Model deviation {rel:.2f} % -- may need higher shots")

        print("=" * 75)

    # -- Final C_MAX calculation (Magesan model with teleportation) -----------

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
    backend_mode = "default"  # Change to "IBM" to run on real IBM hardware

    # -- DEFINE THE ARCHITECTURE (N = Word width per register) -----------------
    N_qubits = 2

    if backend_mode == "IBM":
        ibm_backend = get_ibm_backend("ibm_kingston")
        validator = CMaxValidator(N=N_qubits, backend=ibm_backend)
    else:
        validator = CMaxValidator(N=N_qubits)
    

    # -- Phase B.1: Complete RB characterization with teleportation ------------
    m_list = [0, 1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
    popt = validator.run_rb_characterization(m_list, shots=4000, plot_path = "results/rb_decay_curve_sqm n="+ str(N_qubits) +".png")

    # -- Phase B.2: Print results and validate model ---------------------------
    r_emp = validator.print_rb_results(popt)

    # -- Phase B.3: Calculate C_MAX with target fidelity ----------------------
  
    c_max = validator.calculate_final_cmax(target_fidelity=0.75)
    print(f"\n[FINAL RESULT]  C_MAX = {c_max} SWAPs  "
          f"(r_emp = {r_emp:.4f},  p_swap_theory = {validator.p_swap_teorico:.4f})")

    # -- Phase B.4: Extrapolation validation (optional) -----------------------
    # -- Phase B.4: Extrapolation validation (Magesan model vs empirical) ------
    # Change 'x' to compare the fitted model against a new measurement.
    x = 15
    validator.run_extrapolation_test(n=x)
    print(f"\n  Interpretation:")
    print(f"    If diff < 5%, the Magesan model extrapolates correctly to n={x}.")
    print(f"    r_emp={r_emp:.4f}  vs  p_swap_theory={validator.p_swap_teorico:.4f}")
