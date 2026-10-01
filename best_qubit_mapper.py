# ============================================================
# SQM Research Project - Best Qubit Mapper (Centralized)
# Authors: Danny Valerio-Ramirez & Santiago Nunez-Corrales
# Role: Noise-aware pre-computation of optimal qubit layouts.
#       Run ONCE before experiments; results consumed as CSV.
# ============================================================

"""
best_qubit_mapper.py
====================
Analyzes backend calibration data and determines the optimal physical-qubit
topology for N logical qubits.  The result is written to a CSV lookup table
that every experiment script reads at startup instead of recalculating the
mapping on every circuit build.

Supported experiment types
--------------------------
* **swap**     — 2N qubits  (q_work + mem_0), linear chain pairs
* **sqm**      — 4N qubits  (q_work + mem_orig + tele_ancilla + mem_backup)
* **teleport** — 4N allocated (uses 3N: reg_A + reg_B + ancilla; q_work reserved)
* **delay**    — N qubits   (individual best-cost qubits, no connectivity req.)

Usage
-----
    python -m src.functions.best_qubit_mapper          # uses defaults
    python src/functions/best_qubit_mapper.py           # direct execution
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

# Ensure project root is in sys.path for direct execution
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════
# CSV schema constants (column names)
# ═══════════════════════════════════════════════════════════════════════
CSV_COLUMNS = [
    "experiment_id",
    "backend_name",
    "n_qubits",
    "experiment_type",
    "physical_qubits_list",
    "coupling_edges",
    "avg_gate_error",
    "readout_error_avg",
    "t1_avg_us",
    "t2_avg_us",
    "timestamp",
]


# ═══════════════════════════════════════════════════════════════════════
# BestQubitMapper
# ═══════════════════════════════════════════════════════════════════════

class BestQubitMapper:
    """
    Inspects QPU calibration metrics and finds the lowest-noise subgraph
    for a given experiment type and qubit count N.

    Metrics used
    ------------
    * T1, T2                  — coherence times
    * E_ro                    — readout assignment error
    * E_cx / E_ecr            — 2-qubit gate error on each coupling edge

    Cost function
    -------------
    For chain-based experiments (swap, sqm, teleport):
        cost(chain) = sum edge_error(u,v)
                    + W_RO * sum readout_error(q)
                    + W_T1 * sum 1/T1(q)

    For single-qubit experiments (delay):
        cost(q) = W_RO * readout_error(q) + W_T1 / T1(q)
    """

    # Cost-function weights (same as original QubitMapper)
    _W_READOUT = 1.0
    _W_T1 = 1e-6       # scales T1 (seconds) to ~us contribution

    # Gate priority for 2-qubit native gate detection
    _TWO_QUBIT_GATE_CANDIDATES = ["cz", "cx", "ecr"]

    def __init__(self, backend: Any) -> None:
        """
        Parameters
        ----------
        backend
            A Qiskit BackendV1 or BackendV2 (FakeKyiv, real IBM, etc.)
        """
        self.backend = backend
        self.backend_name: str = getattr(backend, "name", str(backend))

        # Build undirected connectivity graph
        coupling_map = backend.configuration().coupling_map
        self.n_qubits_hw: int = backend.configuration().n_qubits
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.n_qubits_hw))
        for u, v in coupling_map:
            self.graph.add_edge(u, v)

        # Cache calibration data
        self._is_v2: bool = False
        self._props, self._gate_name = self._get_calibration_data()

        print(f"[BestQubitMapper] Backend: {self.backend_name} "
              f"({self.n_qubits_hw} qubits, native 2Q: {self._gate_name})")

    @property
    def n_qubits(self) -> int:
        """Backward-compatible alias for n_qubits_hw."""
        return self.n_qubits_hw

    # ------------------------------------------------------------------
    # Calibration data extraction
    # ------------------------------------------------------------------

    def _get_calibration_data(self) -> Tuple[Any, Optional[str]]:
        """Detect BackendV1/V2 API and cache calibration + native 2Q gate."""
        props = None
        gate_name = None

        # 1. Try BackendV2 Target API first
        if hasattr(self.backend, "target") and self.backend.target is not None:
            self._is_v2 = True
            props = self.backend.target
            for candidate in self._TWO_QUBIT_GATE_CANDIDATES:
                if candidate in props:
                    gate_name = candidate
                    break
            return props, gate_name

        # 2. Fallback to BackendV1 Properties API
        self._is_v2 = False
        try:
            if hasattr(self.backend, "properties"):
                props = self.backend.properties()
        except Exception:
            pass
        if props is not None:
            gate_names_in_backend = {g.gate for g in props.gates}
            for candidate in self._TWO_QUBIT_GATE_CANDIDATES:
                if candidate in gate_names_in_backend:
                    gate_name = candidate
                    break
        return props, gate_name

    # ------------------------------------------------------------------
    # Per-qubit metrics
    # ------------------------------------------------------------------

    def _qubit_readout_error(self, q: int) -> float:
        """Return readout error for qubit q; 1.0 if unavailable."""
        if self._props is None:
            return 0.0
        try:
            if self._is_v2:
                return self._props["measure"].get((q,)).error or 0.0
            else:
                return self._props.readout_error(q)
        except Exception:
            return 1.0

    def _qubit_t1(self, q: int) -> Optional[float]:
        """Return T1 in seconds for qubit q; None if unavailable."""
        if self._props is None:
            return None
        try:
            if self._is_v2:
                qp = self._props.qubit_properties
                if qp and q < len(qp) and qp[q] is not None:
                    return getattr(qp[q], "t1", None)
            else:
                return self._props.t1(q)
        except Exception:
            return None

    def _qubit_t2(self, q: int) -> Optional[float]:
        """Return T2 in seconds for qubit q; None if unavailable."""
        if self._props is None:
            return None
        try:
            if self._is_v2:
                qp = self._props.qubit_properties
                if qp and q < len(qp) and qp[q] is not None:
                    return getattr(qp[q], "t2", None)
            else:
                return self._props.t2(q)
        except Exception:
            return None

    def _edge_gate_error(self, u: int, v: int) -> float:
        """Return 2Q-gate error on edge (u,v); 1.0 if unavailable."""
        if self._props is None or self._gate_name is None:
            return 0.0
        try:
            if self._is_v2:
                err = self._props[self._gate_name].get((u, v))
                if err is not None and err.error is not None:
                    return err.error
                err = self._props[self._gate_name].get((v, u))
                if err is not None and err.error is not None:
                    return err.error
            else:
                try:
                    return self._props.gate_error(self._gate_name, [u, v])
                except Exception:
                    return self._props.gate_error(self._gate_name, [v, u])
        except Exception:
            pass
        return 1.0

    # ------------------------------------------------------------------
    # Chain cost evaluation (reusable by all chain-based experiments)
    # ------------------------------------------------------------------

    def _evaluate_chain_cost(self, chain: List[int], include_readout_all: bool = False) -> float:
        """
        Compute aggregate noise cost for a linear chain of qubits.

        Parameters
        ----------
        chain : list[int]
            Ordered list of physical qubit indices.
        include_readout_all : bool
            If True, include readout error of ALL qubits (not just ancilla).
        """
        cost = 0.0
        # 2Q gate error on consecutive edges
        for i in range(len(chain) - 1):
            cost += self._edge_gate_error(chain[i], chain[i + 1])
        # Readout error
        if include_readout_all:
            for q in chain:
                cost += self._W_READOUT * self._qubit_readout_error(q)
        # T1 decay penalty
        for q in chain:
            t1 = self._qubit_t1(q)
            if t1 is not None and t1 > 0:
                cost += self._W_T1 / t1
            else:
                cost += 10.0  # heavy penalty if no T1
        return cost

    def _single_qubit_cost(self, q: int) -> float:
        """Cost for delay experiments: readout + 1/T1."""
        cost = self._W_READOUT * self._qubit_readout_error(q)
        t1 = self._qubit_t1(q)
        if t1 is not None and t1 > 0:
            cost += self._W_T1 / t1
        else:
            cost += 10.0
        return cost

    # ------------------------------------------------------------------
    # Chain metrics for CSV export
    # ------------------------------------------------------------------

    def _chain_metrics(self, qubits: List[int]) -> Dict[str, Any]:
        """Compute aggregate metrics for a set of qubits."""
        ro_errors = [self._qubit_readout_error(q) for q in qubits]
        t1_vals = [self._qubit_t1(q) for q in qubits]
        t2_vals = [self._qubit_t2(q) for q in qubits]

        # Gate errors on consecutive edges
        gate_errors: List[float] = []
        for i in range(len(qubits) - 1):
            ge = self._edge_gate_error(qubits[i], qubits[i + 1])
            if ge < 1.0:  # only count valid edges
                gate_errors.append(ge)

        # Coupling edges within the set
        edges = []
        for i in range(len(qubits)):
            for j in range(i + 1, len(qubits)):
                if self.graph.has_edge(qubits[i], qubits[j]):
                    edges.append((qubits[i], qubits[j]))

        t1_valid = [t * 1e6 for t in t1_vals if t is not None]
        t2_valid = [t * 1e6 for t in t2_vals if t is not None]

        return {
            "avg_gate_error": float(sum(gate_errors) / max(len(gate_errors), 1)),
            "readout_error_avg": float(sum(ro_errors) / max(len(ro_errors), 1)),
            "t1_avg_us": float(sum(t1_valid) / max(len(t1_valid), 1)),
            "t2_avg_us": float(sum(t2_valid) / max(len(t2_valid), 1)),
            "coupling_edges": edges,
        }

    # ==================================================================
    # PUBLIC: Find optimal mapping per experiment type
    # ==================================================================

    def find_best_swap(self, n: int) -> Dict[str, List[int]]:
        """
        Find N disjoint, adjacent 2-qubit pairs (q_work[i] <-> mem_0[i]).

        Returns
        -------
        dict with keys "q_work" and "mem_0", each a list[int] of length n.
        """
        result: Dict[str, List[int]] = {"q_work": [], "mem_0": []}
        used: Set[int] = set()

        for bit_idx in range(n):
            best_pair: Optional[List[int]] = None
            best_cost = float("inf")
            for u in range(self.n_qubits_hw):
                if u in used:
                    continue
                for v in self.graph.neighbors(u):
                    if v in used:
                        continue
                    chain = [u, v]
                    cost = self._evaluate_chain_cost(chain, include_readout_all=True)
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = chain
            if best_pair is None:
                raise RuntimeError(f"[BestQubitMapper] Cannot find SWAP pair for bit {bit_idx}")
            result["q_work"].append(best_pair[0])
            result["mem_0"].append(best_pair[1])
            used.update(best_pair)

        return result

    def find_best_sqm(self, n: int) -> Dict[str, List[int]]:
        """
        Find N disjoint 4-qubit linear chains:
        q_work <-> mem_orig_0 <-> tele_ancilla_0 <-> mem_backup_0

        Returns
        -------
        dict with keys "q_work", "mem_orig_0", "tele_ancilla_0", "mem_backup_0"
        """
        result: Dict[str, List[int]] = {
            "q_work": [], "mem_orig_0": [],
            "tele_ancilla_0": [], "mem_backup_0": [],
        }
        used: Set[int] = set()

        for bit_idx in range(n):
            best_chain: Optional[List[int]] = None
            best_cost = float("inf")

            for a in range(self.n_qubits_hw):
                if a in used:
                    continue
                for b in self.graph.neighbors(a):
                    if b in used or b == a:
                        continue
                    for c in self.graph.neighbors(b):
                        if c in used or c in (a, b):
                            continue
                        for d in self.graph.neighbors(c):
                            if d in used or d in (a, b, c):
                                continue
                            chain = [a, b, c, d]
                            cost = self._evaluate_chain_cost(chain, include_readout_all=True)
                            if cost < best_cost:
                                best_cost = cost
                                best_chain = chain

            if best_chain is None:
                raise RuntimeError(f"[BestQubitMapper] Cannot find SQM 4-chain for bit {bit_idx}")

            result["q_work"].append(best_chain[0])
            result["mem_orig_0"].append(best_chain[1])
            result["tele_ancilla_0"].append(best_chain[2])
            result["mem_backup_0"].append(best_chain[3])
            used.update(best_chain)

        return result

    def find_best_teleport(self, n: int) -> Dict[str, List[int]]:
        """
        Teleport uses 3N qubits but allocates 4N for topology compatibility.
        Returns same structure as SQM (q_work allocated but unused in circuit).
        """
        return self.find_best_sqm(n)

    def find_best_delay(self, n: int) -> Dict[str, List[int]]:
        """
        Find the N individual qubits with lowest single-qubit cost.
        No connectivity constraint.

        Returns
        -------
        dict with key "delay_qubits".
        """
        costs: List[Tuple[float, int]] = []
        for q in range(self.n_qubits_hw):
            costs.append((self._single_qubit_cost(q), q))
        costs.sort(key=lambda x: x[0])

        selected = [q for _, q in costs[:n]]

        # Log top candidates
        print(f"[BestQubitMapper] Delay: Top-{min(10, n)} qubits (cost, qubit):")
        for cost_val, q_idx in costs[:min(10, n)]:
            t1 = self._qubit_t1(q_idx)
            ro = self._qubit_readout_error(q_idx)
            t1_str = f"{t1 * 1e6:.1f} us" if t1 else "N/A"
            print(f"    qubit {q_idx:3d}  cost={cost_val:.6f}  ro_err={ro:.6f}  T1={t1_str}")

        return {"delay_qubits": selected}

    # ==================================================================
    # PUBLIC: Run all experiment types and export CSV
    # ==================================================================

    def generate_all_mappings(self, n: int, output_csv: str) -> str:
        """
        Compute optimal qubit mappings for all experiment types (swap, sqm,
        teleport, delay) and write them to a single CSV file.

        Parameters
        ----------
        n : int
            Word width (N qubits per register).
        output_csv : str
            Path to output CSV file.

        Returns
        -------
        str : Absolute path of the written CSV.
        """
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        rows: List[Dict[str, Any]] = []

        # --- SWAP ---
        print(f"\n{'='*60}")
        print(f"  [BestQubitMapper] Computing SWAP mapping (N={n}, 2N={2*n} qubits)")
        print(f"{'='*60}")
        swap_alloc = self.find_best_swap(n)
        swap_qubits = swap_alloc["q_work"] + swap_alloc["mem_0"]
        swap_metrics = self._chain_metrics(swap_qubits)
        self._print_allocation("SWAP", swap_alloc)
        rows.append({
            "experiment_id": f"swap_N{n}",
            "backend_name": self.backend_name,
            "n_qubits": n,
            "experiment_type": "swap",
            "physical_qubits_list": str(swap_alloc),
            "coupling_edges": str(swap_metrics["coupling_edges"]),
            "avg_gate_error": f"{swap_metrics['avg_gate_error']:.8f}",
            "readout_error_avg": f"{swap_metrics['readout_error_avg']:.8f}",
            "t1_avg_us": f"{swap_metrics['t1_avg_us']:.2f}",
            "t2_avg_us": f"{swap_metrics['t2_avg_us']:.2f}",
            "timestamp": timestamp,
        })

        # --- SQM ---
        print(f"\n{'='*60}")
        print(f"  [BestQubitMapper] Computing SQM mapping (N={n}, 4N={4*n} qubits)")
        print(f"{'='*60}")
        sqm_alloc = self.find_best_sqm(n)
        sqm_qubits = (sqm_alloc["q_work"] + sqm_alloc["mem_orig_0"]
                       + sqm_alloc["tele_ancilla_0"] + sqm_alloc["mem_backup_0"])
        sqm_metrics = self._chain_metrics(sqm_qubits)
        self._print_allocation("SQM", sqm_alloc)
        rows.append({
            "experiment_id": f"sqm_N{n}",
            "backend_name": self.backend_name,
            "n_qubits": n,
            "experiment_type": "sqm",
            "physical_qubits_list": str(sqm_alloc),
            "coupling_edges": str(sqm_metrics["coupling_edges"]),
            "avg_gate_error": f"{sqm_metrics['avg_gate_error']:.8f}",
            "readout_error_avg": f"{sqm_metrics['readout_error_avg']:.8f}",
            "t1_avg_us": f"{sqm_metrics['t1_avg_us']:.2f}",
            "t2_avg_us": f"{sqm_metrics['t2_avg_us']:.2f}",
            "timestamp": timestamp,
        })

        # --- TELEPORT ---
        print(f"\n{'='*60}")
        print(f"  [BestQubitMapper] Computing TELEPORT mapping (N={n}, 3N={3*n}+1 reserve)")
        print(f"{'='*60}")
        teleport_alloc = self.find_best_teleport(n)
        teleport_qubits = (teleport_alloc["q_work"] + teleport_alloc["mem_orig_0"]
                            + teleport_alloc["tele_ancilla_0"] + teleport_alloc["mem_backup_0"])
        teleport_metrics = self._chain_metrics(teleport_qubits)
        self._print_allocation("TELEPORT", teleport_alloc)
        rows.append({
            "experiment_id": f"teleport_N{n}",
            "backend_name": self.backend_name,
            "n_qubits": n,
            "experiment_type": "teleport",
            "physical_qubits_list": str(teleport_alloc),
            "coupling_edges": str(teleport_metrics["coupling_edges"]),
            "avg_gate_error": f"{teleport_metrics['avg_gate_error']:.8f}",
            "readout_error_avg": f"{teleport_metrics['readout_error_avg']:.8f}",
            "t1_avg_us": f"{teleport_metrics['t1_avg_us']:.2f}",
            "t2_avg_us": f"{teleport_metrics['t2_avg_us']:.2f}",
            "timestamp": timestamp,
        })

        # --- DELAY ---
        print(f"\n{'='*60}")
        print(f"  [BestQubitMapper] Computing DELAY mapping (N={n} individual qubits)")
        print(f"{'='*60}")
        delay_alloc = self.find_best_delay(n)
        delay_qubits = delay_alloc["delay_qubits"]
        delay_metrics = self._chain_metrics(delay_qubits)
        rows.append({
            "experiment_id": f"delay_N{n}",
            "backend_name": self.backend_name,
            "n_qubits": n,
            "experiment_type": "delay",
            "physical_qubits_list": str(delay_alloc),
            "coupling_edges": str(delay_metrics["coupling_edges"]),
            "avg_gate_error": f"{delay_metrics['avg_gate_error']:.8f}",
            "readout_error_avg": f"{delay_metrics['readout_error_avg']:.8f}",
            "t1_avg_us": f"{delay_metrics['t1_avg_us']:.2f}",
            "t2_avg_us": f"{delay_metrics['t2_avg_us']:.2f}",
            "timestamp": timestamp,
        })

        # --- Write CSV ---
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        abs_path = os.path.abspath(output_csv)
        print(f"\n{'='*60}")
        print(f"  [BestQubitMapper] CSV written: {abs_path}")
        print(f"  [BestQubitMapper] {len(rows)} experiment mappings exported")
        print(f"{'='*60}")
        return abs_path

    def _print_allocation(self, label: str, alloc: Dict[str, List[int]]) -> None:
        """Pretty-print a register allocation."""
        print(f"  [{label}] Allocation:")
        for reg_name, qubits in sorted(alloc.items()):
            print(f"    {reg_name:25s} -> {qubits}")

    # ==================================================================
    # STATIC: Load mapping from CSV (used by all experiment scripts)
    # ==================================================================

    @staticmethod
    def load_mapping(
        csv_path: str,
        experiment_type: str,
        n_qubits: int,
        backend_name: Optional[str] = None,
    ) -> Dict[str, List[int]]:
        """
        Read the CSV lookup table and return the physical qubit allocation
        for the requested experiment type and qubit count.

        Parameters
        ----------
        csv_path : str
            Path to the qubit_mapping_N{n}.csv file.
        experiment_type : str
            One of: "swap", "sqm", "teleport", "delay".
        n_qubits : int
            Word width N.
        backend_name : str, optional
            If specified, filter by backend name.

        Returns
        -------
        dict[str, list[int]]
            Register allocation dictionary (same format as find_best_*).

        Raises
        ------
        FileNotFoundError
            If CSV file doesn't exist.
        ValueError
            If no matching row found.
        """
        import ast

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"[BestQubitMapper] Mapping CSV not found: {csv_path}\n"
                f"Run best_qubit_mapper.py first to generate it."
            )

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row["experiment_type"] == experiment_type
                        and int(row["n_qubits"]) == n_qubits):
                    if backend_name and row["backend_name"] != backend_name:
                        continue
                    alloc = ast.literal_eval(row["physical_qubits_list"])
                    print(f"[BestQubitMapper] Loaded {experiment_type} mapping "
                          f"for N={n_qubits} from {csv_path}")
                    return alloc

        raise ValueError(
            f"[BestQubitMapper] No mapping found for experiment_type='{experiment_type}', "
            f"n_qubits={n_qubits} in {csv_path}"
        )

    @staticmethod
    def find_mapping_csv(n_qubits: int, is_ibm: bool, search_dir: str = "IBM_Configuration") -> str:
        """
        Search for an existing mapping CSV for the given N and backend type.

        Looks for files matching pattern {prefix}_qubit_mapping_N{n}.csv in search_dir.
        Returns the path to the most recent one (by filename/timestamp).

        Parameters
        ----------
        n_qubits : int
            Word width N.
        is_ibm : bool
            True if running on hardware, False if simulator.
        search_dir : str
            Directory to search in.
        
        Returns
        -------
        str : path to the CSV file.
        """
        prefix = "rb" if is_ibm else "sm"
        target_name = f"{prefix}_qubit_mapping_N{n_qubits}.csv"
        candidate = os.path.join(search_dir, target_name)
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(
            f"[BestQubitMapper] No mapping CSV found for N={n_qubits} in {search_dir}/\n"
            f"Expected: {candidate}\n"
            f"Run: python best_qubit_mapper.py"
        )


# ===================================================================
# Entry point (__main__)
# ===================================================================

if __name__ == "__main__":
    # ==================================================================
    # Panel de Control
    # ==================================================================
    N_QUBITS = 1                          # Word width N
    BACKEND_NAME = "simulator"            # "simulator" | "ibm_kingston" | etc.
    
    prefix = "sm" if BACKEND_NAME == "simulator" else "rb"
    OUTPUT_CSV = f"IBM_Configuration/{prefix}_qubit_mapping_N{N_QUBITS}.csv"
    # ==================================================================

    if BACKEND_NAME == "simulator":
        from qiskit_ibm_runtime.fake_provider import FakeKyiv
        backend = FakeKyiv()
        print(f"[Main] Using simulator backend: FakeKyiv ({backend.configuration().n_qubits} qubits)")
    else:
        from experiments.utils.ibm_backend_helper import get_ibm_backend
        backend = get_ibm_backend(BACKEND_NAME)
        print(f"[Main] Using real backend: {BACKEND_NAME}")

    mapper = BestQubitMapper(backend)
    csv_path = mapper.generate_all_mappings(n=N_QUBITS, output_csv=OUTPUT_CSV)

    print(f"\n[Main] Done. Mapping CSV: {csv_path}")
    print(f"[Main] All experiment scripts will read this file for initial_layout.")
