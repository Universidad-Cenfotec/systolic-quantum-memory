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
    from src.time_calculation.ibm_backend_helper import get_ibm_backend, run_on_ibm
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from src.time_calculation.ibm_backend_helper import get_ibm_backend, run_on_ibm


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

    def __init__(self, N: int = 1, backend=None, initial_state: int = 0) -> None:
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

        # Backend configuration
        if backend is not None:
            self.backend = backend
            self.is_ibm = True
            self.noise_model = None
            self.simulator = None
            print(f"[TmaxValidatorDelay] Using IBM hardware backend: {self.backend.name}")
            print(f"[TmaxValidatorDelay] NOTE: delay() is a native hardware instruction.")
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

    # -- Empirical fidelity (noisy idle) --------------------------------------

    def empirical_fidelity(self, delay_ns: float, shots: int = 4000) -> float:
        """
        Measure fidelity after idling for *delay_ns* nanoseconds.

        State preparation & measurement basis follow ``self.initial_state``:

            0 = |0>  : no init gates,  no pre-measurement gate,  target '0'*N
            1 = |1>  : X gates,        no pre-measurement gate,  target '1'*N
            2 = |+>  : H gates,        H before measurement,     target '0'*N
            3 = |->  : X+H gates,      H before measurement,     target '1'*N
        """
        if delay_ns < 0:
            raise ValueError(f"delay_ns must be >= 0, received: {delay_ns}")

        qc = QuantumCircuit(self.N, self.N)

        # ── State preparation ─────────────────────────────────────────────────
        if self.initial_state == 0:
            pass  # |0> is the default reset state; no gates needed
        elif self.initial_state == 1:
            for i in range(self.N):
                qc.x(i)       # |0> -> |1>
        elif self.initial_state == 2:
            for i in range(self.N):
                qc.h(i)       # |0> -> |+>
        elif self.initial_state == 3:
            for i in range(self.N):
                qc.x(i)       # |0> -> |1>
                qc.h(i)       # |1> -> |->
        else:
            raise ValueError(f"initial_state must be 0-3, received: {self.initial_state}")

        # ── Idle delay ────────────────────────────────────────────────────────
        if delay_ns > 0:
            for i in range(self.N):
                qc.delay(int(delay_ns), i, unit='ns')

        # ── Basis rotation before measurement (superposition states) ──────────
        # For |+> and |->: apply H to rotate back to computational basis
        if self.initial_state in (2, 3):
            for i in range(self.N):
                qc.h(i)

        qc.measure(range(self.N), range(self.N))
        #print(qc.draw(output="text"))

        # ── Transpile & run ───────────────────────────────────────────────────
        qc_t = transpile(qc, optimization_level=0)

        if self.is_ibm:
            qc_t = transpile(qc, backend=self.backend, optimization_level=0)
            counts = run_on_ibm(qc_t, self.backend, shots=shots)
        else:
            job = self.simulator.run(qc_t, shots=shots)
            counts = job.result().get_counts()

        # ── Target state selection ────────────────────────────────────────────
        # states 0, 2 -> '0'*N  |  states 1, 3 -> '1'*N
        target_state = ('1' * self.N) if self.initial_state in (1, 3) else ('0' * self.N)

        fidelity_count = 0
        for bitstring, count in counts.items():
            measured = bitstring.replace(" ", "")[-self.N:]
            if measured == target_state:
                fidelity_count += count

        return fidelity_count / shots

    # -- Delay characterization (curve_fit) -----------------------------------

    def run_delay_characterization(
        self,
        delay_list_ns: list[float],
        shots: int = 4000,
        plot_path: str | None = "results/delay_decay_curve.png",
    ) -> np.ndarray:
        """
        Sweep over delay durations, collect fidelity, and fit the
        exponential decay model F(t) = A*exp(-t/tau) + B via curve_fit.

        Returns:
            popt  --  array [A_fit, tau_fit, B_fit]
        """
        print("=" * 65)
        print("  SQM -- Delay Characterization (Exponential Decay Model)")
        print("=" * 65)
        print(f"\n  Backend      : {self.backend.name}")
        print(f"  N_qubits     : {self.N}")
        print(f"  dt           : {self.dt_sec * 1e9:.4f} ns")
        print(f"\n  Measuring F(t) for t = {[f'{d:.0f}' for d in delay_list_ns[:5]]} ... ns")
        print(f"  shots per point = {shots}\n")

        # -- Empirical data collection -----------------------------------------
        t_arr = np.array(delay_list_ns, dtype=float)
        y_data: list[float] = []

        for t_ns in delay_list_ns:
            f_emp = self.empirical_fidelity(t_ns, shots=shots)
            y_data.append(f_emp)
            print(f"    t={t_ns:10.1f} ns  F_emp = {f_emp:.6f}")

        y_arr = np.array(y_data, dtype=float)

        # -- curve_fit: F(t) = A * exp(-t/tau) + B ----------------------------
        tau_guess = max(t_arr.max() / 3, 1.0)
        p0     = [0.75, tau_guess, self.B_ideal]
        bounds = ([0.0, 1e-3, 0.0], [1.0, 1e12, 1.0])

        popt, _ = curve_fit(
            exp_decay_model,
            t_arr,
            y_arr,
            p0=p0,
            bounds=bounds,
            maxfev=10_000,
        )

        print(f"\n  Fit completed.")
        print(f"    A_fit   = {popt[0]:.6f}")
        print(f"    tau_fit = {popt[1]:.2f} ns")
        print(f"    B_fit   = {popt[2]:.6f}")

        # -- Plot (optional) ---------------------------------------------------
        if plot_path is not None:
            self._plot_decay_curve(t_arr, y_arr, popt, plot_path)

        # -- Save to CSV -------------------------------------------------------
        csv_path = (
            plot_path.replace(".png", ".csv").replace("results", "data")
            if plot_path
            else "data/delay_characterization.csv"
        )
        self._save_results_to_csv(t_arr, y_arr, popt, csv_path)

        return popt

    # -- Save results to CSV --------------------------------------------------

    def _save_results_to_csv(
        self,
        t_arr: np.ndarray,
        y_data: np.ndarray,
        popt: np.ndarray,
        csv_path: str,
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
            writer.writerow(["N_qubits", self.N])
            writer.writerow(["Hilbert Dimension", f"d = 2^{self.N} = {self.d}"])
            writer.writerow([])

            writer.writerow(["Fit Parameters: F(t) = A * exp(-t/tau) + B"])
            writer.writerow(["A (SPAM contrast)", f"{A_fit:.6f}"])
            writer.writerow(["tau (decay time, ns)", f"{tau_fit:.2f}"])
            writer.writerow(["B (asymptote)", f"{B_fit:.6f}"])
            writer.writerow([])

            writer.writerow(["t (ns)", "t (us)", "F_emp", "F_fit"])
            for t_ns, f_emp in zip(t_arr, y_data):
                f_fit_val = exp_decay_model(t_ns, A_fit, tau_fit, B_fit)
                writer.writerow([
                    f"{t_ns:.1f}",
                    f"{t_ns / 1000:.3f}",
                    f"{f_emp:.6f}",
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
        f_emp = self.empirical_fidelity(t_test_ns)
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
    backend_mode = "IBM"  # Change to "IBM" to run on real IBM hardware

    # =========================================================================
    # INITIAL STATE
    #   0 = |0>  : qubit starts in |0>, fidelity measured vs |0>
    #   1 = |1>  : qubit starts in |1> (X gate), fidelity measured vs |1>
    #   2 = |+>  : qubit starts in |+> (H gate), H applied before measure,
    #              fidelity measured vs |0>
    #   3 = |->  : qubit starts in |-> (X+H gates), H applied before measure,
    #              fidelity measured vs |1>
    # =========================================================================
    initial_state = 0  # 0 = |0>, 1 = |1>, 2 = |+> (H), 3 = |-> (XH)

    # 1. DEFINE THE ARCHITECTURE (N = Word width)
    N_qubits = 1
    target_fidelity = 0.75

    _state_labels = {0: "|0>", 1: "|1>", 2: "|+> (H)", 3: "|-> (XH)"}
    state_label = _state_labels.get(initial_state, f"unknown({initial_state})")
    print(f"[Main] Running with initial_state={initial_state} ({state_label})")

    if backend_mode == "IBM":
        ibm_backend = get_ibm_backend("ibm_kingston")
        validator = TmaxValidatorDelay(N=N_qubits, backend=ibm_backend, initial_state=initial_state)
    else:
        validator = TmaxValidatorDelay(N=N_qubits, initial_state=initial_state)

    # -- Phase 1: Delay characterization (curve_fit) ---------------------------
    #    Define delay times directly in nanoseconds.
    delay_list_ns = [
        0, 500, 1_000, 2_000, 4_000, 6_000, 8_000,
        10_000, 15_000, 20_000, 30_000, 40_000, 50_000,
        60_000, 80_000, 100_000, 120_000, 150_000, 200_000, 400_000, 600_000]

    popt = validator.run_delay_characterization(
        delay_list_ns,
        shots=4000,
        plot_path=f"results/rb_decay_curve_delay_state{initial_state}_N{N_qubits}.png",
    )

    # -- Phase 2: Print results and validate model ----------------------------
    validator.print_delay_results(popt)

    # -- Phase 3: Calculate T_MAX with target fidelity -------------------------
    t_max = validator.calculate_final_tmax(target_fidelity=target_fidelity)
    print(f"\n[FINAL RESULT]  T_MAX = {t_max:.2f} ns  ({t_max / 1000:.3f} us)")

    # -- Phase 4: Extrapolation validation ------------------------------------
    t_test = 50_000  # ns
    #validator.run_extrapolation_test(t_test_ns=t_test)
    print(f"\n  Interpretation:")
    print(f"    If diff < 5%, the exponential model extrapolates correctly to t={t_test} ns.")
