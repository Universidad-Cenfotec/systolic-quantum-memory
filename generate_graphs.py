#!/usr/bin/env python3
"""
Visualización de Rendimiento de Mitigación Pasiva de Errores Cuánticos
en Arquitecturas de Memoria Sistólica (SQM vs SWAP).

Genera figuras de calidad IEEE con SciencePlots y exporta a TikZ para LaTeX.

──────────────────────────────────────────────────────────────────────
PANEL DE CONTROL — 4 Variables:

  State:        "0", "1", "2", "3", "1,3", "ALL"
  Mitigation:   "base", "ZNE", "Twirling", "base,ZNE", "base,ZNE-Twirling", "ALL"
                   separar por ',' = OR,   separar por '-' = AND (combinado)
  Expr Type:    "experimentos" → usa decay curves (delay, teleport, swap)
                "simulador"    → usa sm_comparison (base, sqm)
  Architecture: (depende de Expr Type)
                experimentos → "delay", "teleport", "swap", "delay,teleport", "ALL"
                simulador    → "base", "sqm", "ALL"

Reglas de visualización:
  - Experimentos → gráfico de líneas continuas (un gráfico global)
  - Simulador    → gráficos con puntos, emparejados por state
  - Siempre comparando las técnicas de mitigación seleccionadas
──────────────────────────────────────────────────────────────────────

Autor: Danny Valerio-Ramírez (CENFOTEC) · Santiago Núñez-Corrales (UIUC)
"""

from __future__ import annotations

import os
import re
import glob
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 – registers styles on import

# ──────────────────────────────────────────────────────────────────────
#  tikzplotlib compatibility patches for matplotlib >= 3.6
# ──────────────────────────────────────────────────────────────────────
import matplotlib.backends.backend_pgf
if not hasattr(matplotlib.backends.backend_pgf, "common_texification"):
    def _common_texification(text, *args, **kwargs):
        import re as _re
        return _re.sub(r'(?<!\\)([{}_&#%])', r'\\\1', str(text))
    matplotlib.backends.backend_pgf.common_texification = _common_texification

import matplotlib.axes
if not hasattr(matplotlib.axes.Axes, "get_default_bbox_extra_artists"):
    matplotlib.axes.Axes.get_default_bbox_extra_artists = (
        lambda self: self.get_children()
    )

import webcolors
if not hasattr(webcolors, "CSS3_HEX_TO_NAMES"):
    webcolors.CSS3_HEX_TO_NAMES = webcolors._definitions._CSS3_HEX_TO_NAMES

import matplotlib.legend
if not hasattr(matplotlib.legend.Legend, "legendHandles"):
    matplotlib.legend.Legend.legendHandles = property(lambda self: self.legend_handles)
if not hasattr(matplotlib.legend.Legend, "_ncol"):
    matplotlib.legend.Legend._ncol = property(lambda self: self._ncols)

import matplotlib.lines
if not hasattr(matplotlib.lines.Line2D, "_us_dashSeq"):
    matplotlib.lines.Line2D._us_dashSeq = property(lambda self: self._dash_pattern[1])
if not hasattr(matplotlib.lines.Line2D, "_us_dashOffset"):
    matplotlib.lines.Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])

import tikzplotlib


# =====================================================================
#                       CONSTANTS & MAPPINGS
# =====================================================================

# Mitigation tag → human label
MITIGATION_LABELS = {
    "":    "No Mitigation",
    "Z":   "ZNE",
    "R":   "REM",
    "T":   "Pauli Twirling",
    "ZR":  "ZNE + REM",
    "ZT":  "ZNE + Pauli Twirling",
    "RT":  "REM + Pauli Twirling",
    "ZRT": "ZNE + REM + Twirling",
    "MAR": "Adaptive Margin",
}

# User-facing mitigation name → internal file tag(s)
MITIGATION_NAME_TO_TAGS = {
    "base":     [""],
    "zne":      ["Z"],
    "rem":      ["R"],
    "twirling": ["T"],
    "twirl":    ["T"],
    "zr":       ["ZR"],
    "zt":       ["ZT"],
    "rt":       ["RT"],
    "zrt":      ["ZRT"],
    "mar":      ["MAR"],
}

# Colors for each mitigation tag (accessibility + IEEE)
MITIGATION_COLORS = {
    "":    "#2c3e50",   # dark charcoal
    "Z":   "#2980b9",   # strong blue
    "R":   "#27ae60",   # green
    "T":   "#e67e22",   # orange
    "ZR":  "#8e44ad",   # purple
    "ZT":  "#c0392b",   # red
    "RT":  "#16a085",   # teal
    "ZRT": "#d4ac0d",   # gold
    "MAR": "#7f8c8d",   # gray
}

MITIGATION_MARKERS = {
    "":    "o",
    "Z":   "s",
    "R":   "^",
    "T":   "D",
    "ZR":  "v",
    "ZT":  "P",
    "RT":  "<",
    "ZRT": "*",
    "MAR": "X",
}

STATE_NAMES = {
    0: r"$|0\rangle$",
    1: r"$|1\rangle$",
    2: r"$|+\rangle$",
    3: r"$|-\rangle$",
}

STATE_COLORS = {
    0: "#2980b9", # Azul
    1: "#c0392b", # Rojo
    2: "#f1c40f", # Amarillo
    3: "#27ae60", # Verde
}

MITIGATION_LINESTYLES = {
    "": "-",
    "Z": "--",
    "R": "-.",
    "T": ":",
    "ZR": (0, (3, 1, 1, 1)),
    "ZT": (0, (5, 1)),
    "RT": (0, (3, 1, 1, 1, 1, 1)),
    "ZRT": (0, (1, 1)),
    "MAR": (0, (5, 5))
}

PROTOCOL_NAMES = {
    "swap":     "Base",
    "teleport": "Teleportation (SQM)",
    "delay":    "Delay (Idle)",
    "not":      "NOT",
}

ALL_MITIGATION_TAGS = ["", "Z", "R", "T", "ZR", "ZT", "RT", "ZRT", "MAR"]


# =====================================================================
#                       DATA STRUCTURES
# =====================================================================

@dataclass
class RBDecayCurve:
    """Parsed randomized-benchmarking decay-curve CSV."""
    protocol: str           # "swap", "teleport", "delay"
    state: int              # 0, 1, 2, 3
    n_qubits: int           # 1 or 2
    backend: str            # "fake_kyiv", "ibm_marrakesh", ...
    mitigation_tag: str     # "" (base), "Z" (ZNE), "ZT" (ZNE+Twirl), etc.
    pauli_twirling: bool
    zne_enabled: bool
    rem_enabled: bool
    zne_noise_factors: list[int]
    zne_extrapolator: str
    # Magesan fit parameters
    A: float
    p: float
    B: float
    r_empirical: float
    # Data points
    m_values: np.ndarray
    fidelity_mean: np.ndarray
    fidelity_std: np.ndarray
    fidelity_fit: np.ndarray
    source_file: str


@dataclass
class SimComparison:
    """Parsed simulator SQM-vs-SWAP comparison CSV."""
    state: int
    mitigation_tag: str     # "" or "Z", "ZT", etc.
    zne_enabled: bool
    entries: list[dict] = field(default_factory=list)
    source_file: str = ""


# =====================================================================
#                       CONFIG PARSER
# =====================================================================

@dataclass
class GraphConfig:
    """Parsed user configuration from the 4 control variables."""
    states: list[int]
    mitigation_tags: list[str]          # combined (AND) tags resolved
    mitigation_tags_sqm: list[str]
    mitigation_tags_swap: list[str]
    mitigation_sql_groups: list[list[str]]  # for '-' separated AND groups
    expr_type: str                      # "experimentos" | "simulador"
    architectures: list[str]            # experiment: delay/teleport/swap; sim: base/sqm
    # Rendering params
    n_qubits: int = 1
    shots: int = 1024
    show_fit: bool = True
    export_tikz: bool = True
    combine_states_in_one_graph: str = "one"
    dpi: int = 600
    output_dir: str = ""
    data_dirs: list[str] = field(default_factory=list)


def parse_state(raw: str) -> list[int]:
    """Parse State variable → list of ints."""
    raw = raw.strip().upper()
    if raw == "ALL":
        return [0, 1, 2, 3]
    return [int(s.strip()) for s in raw.split(",") if s.strip().isdigit()]


def parse_mitigation(raw: str) -> tuple[list[str], list[list[str]]]:
    """Parse Mitigation variable → (flat list of tags, grouped AND-lists).

    Examples:
        "base"             → ([""], [[""]])
        "base,ZNE"         → (["", "Z"], [[""], ["Z"]])
        "base,ZNE-Twirling"→ (["", "ZT"], [[""], ["Z", "T"]])  # ZNE+Twirl = "ZT"
        "ALL"              → all tags
    """
    raw = raw.strip().upper()
    if raw == "ALL":
        return list(ALL_MITIGATION_TAGS), [[t] for t in ALL_MITIGATION_TAGS]

    groups: list[list[str]] = []
    flat: list[str] = []

    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        # Split by '-' for AND combination
        parts = [p.strip().lower() for p in token.split("-")]
        if len(parts) == 1:
            # Single mitigation
            resolved = _resolve_mitigation_name(parts[0])
            for tag in resolved:
                if tag not in flat:
                    flat.append(tag)
            groups.append(resolved)
        else:
            # Combined: e.g. "ZNE-Twirling" → combine tags
            combined_tag = _combine_mitigation_parts(parts)
            if combined_tag not in flat:
                flat.append(combined_tag)
            individual_tags = []
            for p in parts:
                individual_tags.extend(_resolve_mitigation_name(p))
            groups.append(individual_tags)

    return flat, groups


def _resolve_mitigation_name(name: str) -> list[str]:
    """Map user-facing name to internal tag(s)."""
    name = name.strip().lower()
    if name in MITIGATION_NAME_TO_TAGS:
        return MITIGATION_NAME_TO_TAGS[name]
    # Try direct tag match (e.g. "Z", "ZT")
    upper = name.upper()
    if upper in MITIGATION_LABELS:
        return [upper]
    return [name.upper()]


def _combine_mitigation_parts(parts: list[str]) -> str:
    """Combine multiple mitigation parts into a single tag.

    E.g. ["zne", "twirling"] → "ZT"
         ["zne", "rem"]      → "ZR"
    """
    tag_chars = set()
    for p in parts:
        resolved = _resolve_mitigation_name(p)
        for r in resolved:
            tag_chars.update(r)
    # Sort canonically: Z, R, T
    ordered = ""
    for ch in ["Z", "R", "T"]:
        if ch in tag_chars:
            ordered += ch
    return ordered if ordered else "".join(sorted(tag_chars))


def parse_architecture(raw: str, expr_type: str) -> list[str]:
    """Parse Architecture variable based on Expr Type."""
    raw = raw.strip().lower()
    if raw == "all":
        if expr_type == "experimentos":
            return ["delay", "teleport", "swap"]
        else:
            return ["base", "sqm"]

    return [a.strip() for a in raw.split(",") if a.strip()]


def build_config(
    state_raw: str,
    mitigation_raw: str,
    expr_type_raw: str,
    architecture_raw: str,
    n_qubits: int = 1,
    shots: int = 1024,
    show_fit: bool = True,
    export_tikz: bool = True,
    combine_states_in_one_graph: str = "one",
    dpi: int = 600,
    output_dir: str = "",
    data_dirs: list[str] | None = None,
    mitigation_sqm_raw: str = "",
    mitigation_swap_raw: str = "",
) -> GraphConfig:
    """Build a validated GraphConfig from raw user strings."""
    expr_type = "experimentos" if "exp" in expr_type_raw.lower() else "simulador"

    states = parse_state(state_raw)
    tags, groups = parse_mitigation(mitigation_raw)
    
    tags_sqm, _ = parse_mitigation(mitigation_sqm_raw) if mitigation_sqm_raw else (tags, groups)
    tags_swap, _ = parse_mitigation(mitigation_swap_raw) if mitigation_swap_raw else (tags, groups)
    
    union_tags = list(set(tags + tags_sqm + tags_swap))
    
    archs = parse_architecture(architecture_raw, expr_type)

    return GraphConfig(
        states=states,
        mitigation_tags=union_tags,
        mitigation_tags_sqm=tags_sqm,
        mitigation_tags_swap=tags_swap,
        mitigation_sql_groups=groups,
        expr_type=expr_type,
        architectures=archs,
        n_qubits=n_qubits,
        shots=shots,
        show_fit=show_fit,
        export_tikz=export_tikz,
        combine_states_in_one_graph=combine_states_in_one_graph.strip().lower(),
        dpi=dpi,
        output_dir=output_dir,
        data_dirs=data_dirs or [],
    )


# =====================================================================
#                       CSV PARSERS
# =====================================================================

def _parse_metadata_value(line: str) -> str:
    """Extract value from 'Key,Value' lines."""
    parts = line.split(",", 1)
    return parts[1].strip() if len(parts) > 1 else ""


def _extract_mitigation_tag(filename: str) -> str:
    """Extract mitigation suffix tag from filename.

    Convention:
        base file:  sm_decay_curve_swap_state1_n1       -> ""
        ZNE:        sm_decay_curve_swap_state1_n1_Z     -> "Z"
        ZNE+REM:    sm_decay_curve_swap_state1_n1_ZR    -> "ZR"
        ZNE+Twirl:  sm_decay_curve_swap_state1_n1_ZT    -> "ZT"
        ZNE+R+T:    sm_decay_curve_swap_state1_n1_ZRT   -> "ZRT"
        REM only:   sm_decay_curve_swap_state1_n1_R     -> "R"
        Twirl only: sm_decay_curve_swap_state1_n1_T     -> "T"
        REM+Twirl:  sm_decay_curve_swap_state1_n1_RT    -> "RT"
    """
    # Match suffix after the last _nN or _N1 pattern
    match = re.search(r"_n\d+_([A-Z]+)$", filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"_N\d+_([A-Z]+)$", filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # For files like rb_decay_curve_swap_state1_n1_Z
    match = re.search(r"state\d+_n\d+_([A-Z]+)$", filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # For MAR suffix
    if filename.endswith("MAR"):
        return "MAR"
    # For comparison files: sm_comparison_results_state1_Z_TIMESTAMP
    match = re.search(r"state\d+_([A-Z]+)_\d{8}", filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""


def parse_rb_decay_csv(filepath: str) -> Optional[RBDecayCurve]:
    """Parse an RB / SM decay-curve CSV (both sim and hardware formats)."""
    try:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            lines = [l.rstrip("\r\n") for l in f.readlines()]
    except UnicodeDecodeError:
        # Fallback to cp1252 for Windows generated files with symbols like '×' (0xd7)
        with open(filepath, "r", encoding="cp1252") as f:
            lines = [l.rstrip("\r\n") for l in f.readlines()]
    except Exception:
        return None

    if not lines:
        return None

    # Extract metadata from header
    metadata = {}
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith("m ("):
            data_start = i
            break
        if "," in line:
            key = line.split(",", 1)[0].strip()
            val = _parse_metadata_value(line)
            metadata[key] = val

    if data_start is None:
        return None

    # Parse protocol from first line
    first_line_lower = lines[0].lower()
    if "teleport" in first_line_lower:
        protocol = "teleport"
    elif "swap" in first_line_lower:
        protocol = "swap"
    elif "delay" in first_line_lower:
        protocol = "delay"
    elif "not" in first_line_lower:
        protocol = "not"
    else:
        protocol = "unknown"

    # Parse state
    state_str = metadata.get("Initial State", "0")
    state_match = re.search(r"(\d)", state_str)
    state = int(state_match.group(1)) if state_match else 0

    # Parse n_qubits from architecture
    arch = metadata.get("Architecture", "1 qubits")
    n_match = re.search(r"(\d+)\s*qubits", arch)
    n_qubits_total = int(n_match.group(1)) if n_match else 1
    # For swap: 2 registers * n; for teleport: 3 registers * n
    if protocol == "teleport":
        n_per_reg = max(1, n_qubits_total // 3)
    elif protocol == "swap":
        n_per_reg = max(1, n_qubits_total // 2)
    else:
        n_per_reg = 1

    backend = metadata.get("Backend", "unknown")
    pauli_twirling = metadata.get("Pauli Twirling", "disabled").lower() == "enabled"
    zne_enabled = metadata.get("Mitigation ZNE", "disabled").lower() == "enabled"
    rem_enabled = metadata.get("Mitigation REM", "disabled").lower() == "enabled"

    zne_factors_str = metadata.get("ZNE Noise Factors", "[]")
    try:
        zne_factors = eval(zne_factors_str) if zne_factors_str else []
    except Exception:
        zne_factors = []
    zne_extrapolator = metadata.get("ZNE Extrapolator", "")

    # Parse Magesan fit
    A = float(metadata.get("A (SPAM contrast)", "0"))
    p_decay = float(metadata.get("p (process decay)", metadata.get("p (process decay per teleport)", "0")))
    B_asym = float(metadata.get("B (asymptote)", "0"))
    r_emp = float(metadata.get("r_empirical", "0"))

    # Parse data rows
    m_vals, f_mean, f_std, f_fit = [], [], [], []
    for line in lines[data_start + 1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) >= 3:
            try:
                m_vals.append(float(parts[0]))
                f_mean.append(float(parts[1]))
                f_std.append(float(parts[2]) if len(parts) > 2 else 0.0)
                f_fit.append(float(parts[3]) if len(parts) > 3 else float(parts[1]))
            except ValueError:
                continue

    if not m_vals:
        return None

    # Determine mitigation tag from filename
    fname = Path(filepath).stem
    tag = _extract_mitigation_tag(fname)

    return RBDecayCurve(
        protocol=protocol,
        state=state,
        n_qubits=n_per_reg,
        backend=backend,
        mitigation_tag=tag,
        pauli_twirling=pauli_twirling,
        zne_enabled=zne_enabled,
        rem_enabled=rem_enabled,
        zne_noise_factors=zne_factors,
        zne_extrapolator=zne_extrapolator,
        A=A, p=p_decay, B=B_asym, r_empirical=r_emp,
        m_values=np.array(m_vals),
        fidelity_mean=np.array(f_mean),
        fidelity_std=np.array(f_std),
        fidelity_fit=np.array(f_fit),
        source_file=filepath,
    )


def _split_csv_line(line: str) -> list[str]:
    """Split CSV line respecting quoted fields."""
    result = []
    current = ""
    in_quotes = False
    for ch in line:
        if ch == '"':
            in_quotes = not in_quotes
        elif ch == "," and not in_quotes:
            result.append(current.strip())
            current = ""
        else:
            current += ch
    result.append(current.strip())
    return result


def parse_sim_comparison_csv(filepath: str) -> Optional[SimComparison]:
    """Parse simulator SQM-vs-SWAP comparison CSV."""
    try:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            lines = [l.rstrip("\r\n") for l in f.readlines()]
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="cp1252") as f:
            lines = [l.rstrip("\r\n") for l in f.readlines()]
    except Exception:
        return None

    if not lines:
        return None

    # Extract metadata
    metadata = {}
    for line in lines:
        if line.startswith("Run,") or line.startswith("Scenario,"):
            break
        if "," in line:
            key = line.split(",", 1)[0].strip()
            val = _parse_metadata_value(line)
            metadata[key] = val

    # Find header line
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Run,") or line.startswith("Scenario,"):
            header_idx = i
            break

    if header_idx is None:
        return None

    headers = lines[header_idx].split(",")

    # Determine state
    fname = Path(filepath).stem
    state_match = re.search(r"state(\d+)", fname)
    state = int(state_match.group(1)) if state_match else -1

    # If state not in filename, try metadata
    if state < 0:
        state_val = metadata.get("initial_state", "")
        if state_val.isdigit():
            state = int(state_val)

    # Determine mitigation tag from filename
    mit_tag = _extract_mitigation_tag(fname)

    # ZNE from metadata
    zne_enabled = metadata.get("ZNE Enabled", "False") == "True"
    if "Z" in mit_tag:
        zne_enabled = True

    entries = []
    for line in lines[header_idx + 1:]:
        line = line.strip()
        if not line:
            continue
        parts = _split_csv_line(line)
        if len(parts) < len(headers):
            continue
        entry = {}
        for h, v in zip(headers, parts):
            entry[h] = v
        entries.append(entry)

    return SimComparison(
        state=state,
        mitigation_tag=mit_tag,
        zne_enabled=zne_enabled,
        entries=entries,
        source_file=filepath,
    )


# =====================================================================
#                       DATA DISCOVERY
# =====================================================================

def discover_decay_curves(
    data_dirs: list[str],
    protocols: list[str],
    states: list[int],
    mitigation_tags: list[str],
    n_qubits: int = 1,
) -> list[RBDecayCurve]:
    """Find all RB decay curves matching the config filters.

    Deduplicates by (protocol, state, mitigation_tag), keeping only the
    most recently modified file for each combination.
    """
    # Collect all candidates keyed by (protocol, state, tag)
    candidates: dict[tuple[str, int, str], list[tuple[str, RBDecayCurve]]] = {}
    seen_files: set[str] = set()

    for base_dir in data_dirs:
        if not os.path.isdir(base_dir):
            continue
        # Search recursively
        for filepath in glob.glob(os.path.join(base_dir, "**", "*decay_curve*.csv"), recursive=True):
            norm_path = os.path.normpath(filepath)
            if norm_path in seen_files:
                continue
            seen_files.add(norm_path)

            curve = parse_rb_decay_csv(filepath)
            if curve is None:
                continue

            # Filter by protocol
            if curve.protocol not in protocols:
                continue
            # Filter by state
            if curve.state not in states:
                continue
            # Filter by n_qubits
            if curve.n_qubits != n_qubits:
                continue
            # Filter by mitigation tag
            if curve.mitigation_tag not in mitigation_tags:
                continue

            key = (curve.protocol, curve.state, curve.mitigation_tag)
            candidates.setdefault(key, []).append((norm_path, curve))

    # Deduplicate: keep only the most recently modified file per key
    curves: list[RBDecayCurve] = []
    for key, entries in candidates.items():
        if len(entries) == 1:
            curves.append(entries[0][1])
        else:
            # Sort by file modification time (newest first)
            entries.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)
            curves.append(entries[0][1])

    return curves


def discover_sim_comparisons(
    data_dirs: list[str],
    states: list[int],
    mitigation_tags: list[str],
) -> list[SimComparison]:
    """Find all simulator comparison CSVs matching the config filters.

    Only matches sm_comparison_results_* files (new format with Run,
    SQM_Fidelity, SWAP_Fidelity columns). Skips old comparison_results_*
    files from Data_All/ that use Scenario-based format.

    Deduplicates by (state, mitigation_tag), keeping the most recent file.
    """
    candidates: dict[tuple[int, str], list[tuple[str, SimComparison]]] = {}
    seen_files: set[str] = set()

    for base_dir in data_dirs:
        if not os.path.isdir(base_dir):
            continue
        # Only match sm_comparison_results_* (not old comparison_results_*)
        for filepath in glob.glob(os.path.join(base_dir, "**", "sm_comparison_results*.csv"), recursive=True):
            norm_path = os.path.normpath(filepath)
            if norm_path in seen_files:
                continue
            seen_files.add(norm_path)

            # Skip summary files
            if "summary" in Path(filepath).stem:
                continue

            comp = parse_sim_comparison_csv(filepath)
            if comp is None:
                continue

            # Filter by state
            if comp.state not in states:
                continue
            # Filter by mitigation tag
            if comp.mitigation_tag not in mitigation_tags:
                continue

            key = (comp.state, comp.mitigation_tag)
            candidates.setdefault(key, []).append((norm_path, comp))

    # Deduplicate: keep only the most recently modified file per key
    comparisons: list[SimComparison] = []
    for key, entries in candidates.items():
        if len(entries) == 1:
            comparisons.append(entries[0][1])
        else:
            entries.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)
            comparisons.append(entries[0][1])

    return comparisons


# =====================================================================
#                       STATISTICS
# =====================================================================

def compute_binomial_std(fidelity: float, shots: int = 1024) -> float:
    """Compute binomial standard deviation for shot-based fidelity.

    For a Bernoulli process with probability p measured over N shots:
    sigma = sqrt(p(1-p)/N)
    """
    p = np.clip(fidelity, 0.0, 1.0)
    return np.sqrt(p * (1.0 - p) / shots)


def ensure_error_bars(curve: RBDecayCurve, shots: int = 1024) -> np.ndarray:
    """Return error bars: use CSV std if nonzero, else compute binomial."""
    if np.any(curve.fidelity_std > 0):
        return curve.fidelity_std
    return np.array([compute_binomial_std(f, shots) for f in curve.fidelity_mean])


# =====================================================================
#                       PLOT HELPERS
# =====================================================================

def _apply_style():
    """Apply SciencePlots IEEE style."""
    try:
        plt.style.use(["science", "ieee"])
        plt.rcParams['text.usetex'] = False  # Disable LaTeX requirement for local rendering
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
    except Exception:
        warnings.warn("SciencePlots 'science'+'ieee' style not found. Using default.")


def _save_figure(fig, output_dir: str, name: str, export_tikz: bool = True,
                 dpi: int = 300):
    """Save figure as PDF, PNG, and optionally TikZ."""
    os.makedirs(output_dir, exist_ok=True)

    pdf_path = os.path.join(output_dir, f"{name}.pdf")
    png_path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"  -> Saved: {pdf_path}")
    print(f"  -> Saved: {png_path}")

    if export_tikz:
        tikz_path = os.path.join(output_dir, f"{name}.tikz")
        try:
            tikzplotlib.save(
                tikz_path,
                figure=fig,
                axis_width=r"\columnwidth",
                axis_height="6cm",
                extra_axis_parameters=[
                    "legend style={font=\\footnotesize}",
                    "label style={font=\\small}",
                    "tick label style={font=\\footnotesize}",
                ],
            )
            print(f"  -> Saved: {tikz_path}")
        except Exception as e:
            warnings.warn(f"tikzplotlib export failed: {e}")

    plt.close(fig)


def _extract_idle_units(workload_str: str) -> Optional[int]:
    """Extract IDLE units from workload string like 'WRITE_0,IDLE_100,READ_0' or 'Delay 10ns'."""
    match = re.search(r"IDLE_(\d+)", workload_str)
    if match:
        return int(match.group(1))
    match = re.search(r"Delay (\d+)ns", workload_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


# =====================================================================
#                       EXPERIMENT PLOTS (líneas continuas)
# =====================================================================

def plot_experiment_decay(
    curves: list[RBDecayCurve],
    config: GraphConfig,
):
    """Plot RB decay curves comparing mitigation techniques.

    Experiments → continuous lines (no markers on data points).
    One global figure per protocol, comparing all selected mitigations.
    """
    if not curves:
        return

    # Group curves by protocol
    by_protocol: dict[str, list[RBDecayCurve]] = {}
    for c in curves:
        by_protocol.setdefault(c.protocol, []).append(c)

    for protocol, proto_curves in by_protocol.items():
        proto_label = PROTOCOL_NAMES.get(protocol, protocol)

        # Group by state within this protocol
        by_state: dict[int, list[RBDecayCurve]] = {}
        for c in proto_curves:
            by_state.setdefault(c.state, []).append(c)

        # Si hay múltiples estados, decidir cómo graficarlos basado en combine_states_in_one_graph
        if len(by_state) > 1:
            if config.combine_states_in_one_graph == "multi":
                _plot_experiment_multistate(by_state, protocol, proto_label, config)
            elif config.combine_states_in_one_graph == "combine":
                _plot_experiment_combined_states(by_state, protocol, proto_label, config)
            else: # "one" o cualquier otra cosa
                for state in sorted(by_state.keys()):
                    state_curves = by_state[state]
                    state_label = STATE_NAMES.get(state, f"State {state}")
        
                    _plot_experiment_single(
                        state_curves, protocol, state,
                        title=f"{proto_label} — {state_label}",
                        filename=f"exp_{protocol}_state{state}_n{config.n_qubits}",
                        config=config,
                    )
        else:
            # Si solo hay un estado, siempre es un solo gráfico (one)
            for state in sorted(by_state.keys()):
                state_curves = by_state[state]
                state_label = STATE_NAMES.get(state, f"State {state}")
    
                _plot_experiment_single(
                    state_curves, protocol, state,
                    title=f"{proto_label} — {state_label}",
                    filename=f"exp_{protocol}_state{state}_n{config.n_qubits}",
                    config=config,
                )

def _plot_experiment_combined_states(
    by_state: dict[int, list[RBDecayCurve]],
    protocol: str,
    proto_label: str,
    config: GraphConfig,
):
    """Combined experiment plot: all states and mitigations on a single axis."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    state_linestyles = {
        0: "-",
        1: "--",
        2: "-.",
        3: ":"
    }

    for state in sorted(by_state.keys()):
        state_label = STATE_NAMES.get(state, f"State {state}")
        curves = by_state[state]
        ls = state_linestyles.get(state, "-")

        for curve in sorted(curves, key=lambda c: c.mitigation_tag):
            tag = curve.mitigation_tag
            label_mit = MITIGATION_LABELS.get(tag, tag or "Base")
            color = STATE_COLORS.get(state, "#333333")
            err = ensure_error_bars(curve, config.shots)
            
            label = f"{state_label} - {label_mit}"
            ls_mit = MITIGATION_LINESTYLES.get(tag, "-")

            ax.plot(
                curve.m_values, curve.fidelity_mean,
                color=color, label=label,
                linewidth=1.2, linestyle=ls_mit,
            )

            if config.show_fit and curve.A != 0:
                m_fine = np.linspace(curve.m_values.min(), curve.m_values.max(), 200)
                f_fit = curve.A * (curve.p ** m_fine) + curve.B
                ax.plot(m_fine, f_fit, color=color, linewidth=0.7,
                        linestyle=ls_mit, alpha=0.6)

    xlabel_map = {
        "swap": "SWAP Cycles ($m$)",
        "teleport": "Teleportation Cycles ($m$)",
        "delay": "Delay Cycles ($m$)",
    }
    ax.set_xlabel(xlabel_map.get(protocol, "Cycles ($m$)"))
    ax.set_ylabel("Fidelity")
    ax.set_title(f"{proto_label} — All States", fontsize=7)
    ax.set_ylim(0.0, 1.05)
    
    # Colocar la leyenda dentro del gráfico (loc="best")
    ax.legend(loc="best", fontsize=5, framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.3)

    fig.tight_layout()
    state_suffix = "_".join(str(s) for s in sorted(by_state.keys()))
    filename = f"exp_{protocol}_combined_{state_suffix}_n{config.n_qubits}"
    _save_figure(fig, config.output_dir, filename, config.export_tikz, config.dpi)


def _plot_experiment_single(
    curves: list[RBDecayCurve],
    protocol: str,
    state: int,
    title: str,
    filename: str,
    config: GraphConfig,
):
    """Single-panel experiment plot: continuous lines, comparing mitigations."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for curve in sorted(curves, key=lambda c: c.mitigation_tag):
        tag = curve.mitigation_tag
        label = MITIGATION_LABELS.get(tag, tag or "Base")
        color = STATE_COLORS.get(state, "#333333")
        ls = MITIGATION_LINESTYLES.get(tag, "-")
        err = ensure_error_bars(curve, config.shots)

        # Experiment: continuous line (no markers, no shadow)
        ax.plot(
            curve.m_values, curve.fidelity_mean,
            color=color, label=label,
            linewidth=1.2, linestyle=ls,
        )

        # Magesan fit overlay
        if config.show_fit and curve.A != 0:
            m_fine = np.linspace(curve.m_values.min(), curve.m_values.max(), 200)
            f_fit = curve.A * (curve.p ** m_fine) + curve.B
            ax.plot(m_fine, f_fit, color=color, linewidth=0.7,
                    linestyle=ls, alpha=0.6)

    xlabel_map = {
        "swap": "Base Cycles ($m$)",
        "teleport": "Teleportation Cycles ($m$)",
        "delay": "Delay Cycles ($m$)",
    }
    ax.set_xlabel(xlabel_map.get(protocol, "Cycles ($m$)"))
    ax.set_ylabel("Fidelity")
    ax.set_title(title, fontsize=7)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best", fontsize=5, framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.3)

    fig.tight_layout()
    _save_figure(fig, config.output_dir, filename, config.export_tikz, config.dpi)


def _plot_experiment_multistate(
    by_state: dict[int, list[RBDecayCurve]],
    protocol: str,
    proto_label: str,
    config: GraphConfig,
):
    """Multi-state experiment plot: one subplot per state, continuous lines."""
    _apply_style()
    n_states = len(by_state)
    fig, axes = plt.subplots(1, n_states, figsize=(3.5 * n_states, 2.8), squeeze=False)
    axes = axes.flatten()

    sorted_states = sorted(by_state.keys())

    for idx, state in enumerate(sorted_states):
        ax = axes[idx]
        state_label = STATE_NAMES.get(state, f"State {state}")
        curves = by_state[state]

        for curve in sorted(curves, key=lambda c: c.mitigation_tag):
            tag = curve.mitigation_tag
            label = MITIGATION_LABELS.get(tag, tag or "Base")
            color = STATE_COLORS.get(state, "#333333")
            ls = MITIGATION_LINESTYLES.get(tag, "-")
            err = ensure_error_bars(curve, config.shots)

            ax.plot(
                curve.m_values, curve.fidelity_mean,
                color=color, label=label,
                linewidth=1.2, linestyle=ls,
            )

            if config.show_fit and curve.A != 0:
                m_fine = np.linspace(curve.m_values.min(), curve.m_values.max(), 200)
                f_fit = curve.A * (curve.p ** m_fine) + curve.B
                ax.plot(m_fine, f_fit, color=color, linewidth=0.7,
                        linestyle=ls, alpha=0.6)

        ax.set_xlabel("Cycles ($m$)")
        if idx == 0:
            ax.set_ylabel("Fidelity")
        ax.set_title(state_label, fontsize=7)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="best", fontsize=5, framealpha=0.9)
        ax.grid(True, alpha=0.3, linewidth=0.3)

    fig.suptitle(f"{proto_label} — Mitigation Comparison", fontsize=8, y=1.02)
    fig.tight_layout()

    state_suffix = "_".join(str(s) for s in sorted_states)
    filename = f"exp_{protocol}_states_{state_suffix}_n{config.n_qubits}"
    _save_figure(fig, config.output_dir, filename, config.export_tikz, config.dpi)


# =====================================================================
#                 SIMULATION PLOTS (líneas con puntos, por state)
# =====================================================================

def plot_simulation_comparison(
    comparisons: list[SimComparison],
    config: GraphConfig,
):
    """Plot simulator SQM-vs-SWAP comparison data.

    Simulation → lines with markers, paired by state.
    Only generates idle sweep plots from sm_comparison_results CSVs.
    Always comparing mitigations: base vs ZNE vs Twirling etc.
    """
    # Group comparisons by state
    by_state: dict[int, list[SimComparison]] = {}
    for comp in comparisons:
        by_state.setdefault(comp.state, []).append(comp)

    # Determine which architectures to show
    show_base = "base" in config.architectures or "swap" in config.architectures
    show_sqm = "sqm" in config.architectures or "teleport" in config.architectures

    # Generate paired plots per state
    for state in sorted(by_state.keys()):
        state_label = STATE_NAMES.get(state, f"State {state}")
        comps = by_state[state]
        _plot_sim_idle_sweep(comps, state, state_label, config, show_base, show_sqm)


def _plot_sim_idle_sweep(
    comparisons: list[SimComparison],
    state: int,
    state_label: str,
    config: GraphConfig,
    show_base: bool = True,
    show_sqm: bool = True,
):
    """Plot SQM vs SWAP idle sweep for simulation data (markers + lines)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))



    for comp in sorted(comparisons, key=lambda c: c.mitigation_tag):
        if not comp.entries:
            continue

        tag = comp.mitigation_tag
        tag_label = MITIGATION_LABELS.get(tag, tag or "Base")
        color_sqm = "#2980b9" # Azul
        color_swap = "#7f8c8d" # Gris
        
        # Para distinguir si hay más de una mitigación (ya que el color ahora es fijo por arquitectura), 
        # seguimos usando markers diferentes basados en el tag
        marker_sqm = MITIGATION_MARKERS.get(tag, "s")
        marker_swap = MITIGATION_MARKERS.get(tag, "o")

        idle_units = []
        sqm_fids = []
        swap_fids = []

        for e in comp.entries:
            wl = e.get("Workload", e.get("Tasks", ""))
            idle = _extract_idle_units(wl)
            if idle is None:
                # Try the Workload column for "Delay Xns"
                wl2 = e.get("Workload", "")
                match = re.search(r"(\d+)", wl2)
                if match:
                    idle = int(match.group(1))
                else:
                    continue
            idle_units.append(idle)
            sqm_fids.append(float(e.get("SQM_Fidelity", 0)))
            swap_fids.append(float(e.get("SWAP_Fidelity", 0)))

        if not idle_units:
            continue

        # Sort by idle units
        order = np.argsort(idle_units)
        idle_units = [idle_units[i] for i in order]
        sqm_fids = [sqm_fids[i] for i in order]
        swap_fids = [swap_fids[i] for i in order]

        # SWAP baseline
        if show_base and swap_fids and (comp.mitigation_tag in config.mitigation_tags_swap):
            swap_label = f"Base ({tag_label})" if tag else "Base (No Mitigation)"
            ax.plot(
                idle_units, swap_fids,
                f"-{marker_swap}", color=color_swap, label=swap_label,
                markersize=3.5, linewidth=1.0,
            )

        # SQM with this mitigation
        if show_sqm and sqm_fids and (comp.mitigation_tag in config.mitigation_tags_sqm):
            sqm_label = f"SQM ({tag_label})" if tag else "SQM (No Mitigation)"
            ax.plot(
                idle_units, sqm_fids,
                f"-{marker_sqm}", color=color_sqm, label=sqm_label,
                markersize=3.5, linewidth=1.0,
            )

    ax.set_xlabel("Idle Duration (ns)")
    ax.set_ylabel("Fidelity")
    ax.set_title(f"SQM vs Base — {state_label}", fontsize=7)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best", fontsize=5, framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.3)

    fig.tight_layout()
    filename = f"sim_idle_sweep_state{state}"
    _save_figure(fig, config.output_dir, filename, config.export_tikz, config.dpi)


# =====================================================================
#                       MAIN ORCHESTRATOR
# =====================================================================

def generate_graphs(config: GraphConfig):
    """Main entry point: discover data and generate all graphs per config."""
    print("=" * 70)
    print("  Quantum Error Mitigation Visualization — SQM / SWAP")
    print("=" * 70)
    print(f"  State:         {config.states}")
    print(f"  Mitigation:    {[MITIGATION_LABELS.get(t, t) for t in config.mitigation_tags]}")
    print(f"  Expr Type:     {config.expr_type}")
    print(f"  Architecture:  {config.architectures}")
    print(f"  n (qubits):    {config.n_qubits}")
    print(f"  Shots:         {config.shots}")
    print(f"  TikZ export:   {config.export_tikz}")
    print(f"  Output:        {config.output_dir}")
    print(f"  Data dirs:     {config.data_dirs}")
    print("=" * 70)

    if config.expr_type == "experimentos":
        _run_experiment_mode(config)
    else:
        _run_simulator_mode(config)

    print(f"\n{'=' * 70}")
    print(f"  DONE: Visualization completed.")
    print(f"  Output dir: {config.output_dir}")
    if config.export_tikz:
        print(f"  .tikz files ready for \\input{{}} in LaTeX")
    print(f"{'=' * 70}")


def _run_experiment_mode(config: GraphConfig):
    """Experiment mode: discover decay curves and plot continuous lines."""
    protocols = config.architectures  # delay, teleport, swap
    print(f"\n  >> Experiment mode: protocols = {protocols}")

    curves = discover_decay_curves(
        data_dirs=config.data_dirs,
        protocols=protocols,
        states=config.states,
        mitigation_tags=config.mitigation_tags,
        n_qubits=config.n_qubits,
    )

    if not curves:
        print("  WARNING: No decay curves found matching the configuration.")
        print("  Searched in:")
        for d in config.data_dirs:
            print(f"    - {d}")
        return

    print(f"\n  Found {len(curves)} decay curves:")
    for c in curves:
        tag_desc = MITIGATION_LABELS.get(c.mitigation_tag, c.mitigation_tag or "Base")
        print(f"    - {c.protocol:<10s} state{c.state} {tag_desc:<30s} p={c.p:.6f}  [{Path(c.source_file).name}]")

    plot_experiment_decay(curves, config)


def _run_simulator_mode(config: GraphConfig):
    """Simulator mode: discover sm_comparison CSVs and plot idle sweeps with markers."""
    print(f"\n  >> Simulator mode: architectures = {config.architectures}")

    # Discover comparison CSVs (sm_comparison_results_*)
    comparisons = discover_sim_comparisons(
        data_dirs=config.data_dirs,
        states=config.states,
        mitigation_tags=config.mitigation_tags,
    )

    if not comparisons:
        print("  WARNING: No sm_comparison_results CSVs found matching the configuration.")
        print("  Searched in:")
        for d in config.data_dirs:
            print(f"    - {d}")
        return

    print(f"\n  Found {len(comparisons)} comparison CSVs:")
    for c in comparisons:
        tag_desc = MITIGATION_LABELS.get(c.mitigation_tag, c.mitigation_tag or "Base")
        print(f"    - state{c.state} {tag_desc:<30s} [{Path(c.source_file).name}]")

    plot_simulation_comparison(comparisons, config)


# =====================================================================
#                       MAIN CONTROL PANEL
# =====================================================================

if __name__ == "__main__":

    # =================================================================
    #  PANEL DE CONTROL — 4 VARIABLES PRINCIPALES
    # =================================================================

    # --- State ---
    # Opciones: "0", "1", "2", "3", "1,3", "ALL"
    SELECTED_STATE = "1"
    # --- Combine States ---
    # "multi"   : Multiples gráficos en un solo panel (side-by-side)
    # "one"     : Un gráfico individual separado por cada state
    # "combine" : Un solo gráfico con todos los states mezclados
    COMBINE_STATES_IN_ONE_GRAPH = "one"
 
    # --- Mitigation (Experimentos) ---
    # Opciones: "base", "ZNE", "Twirling", "base,ZNE", "base,ZNE-Twirling", "ALL"
    #   - Separar por ',' = OR (base O ZNE)
    #   - Separar por '-' = AND combinado (ZNE+Twirling = ZT) 
    SELECTED_MITIGATION_EXP = "base"
    
    # --- Mitigation (Simulador) ---
    # Configuraciones independientes para SQM y SWAP (solo aplican para el simulador). 
    SELECTED_MITIGATION_SQM = "base,ZNE"
    SELECTED_MITIGATION_SWAP = "base"

    # --- Expr Type ---
    # "experimentos" → usa decay curves (delay, teleport, swap)
    # "simulador"    → usa sm_comparison (base, sqm)
    SELECTED_EXPR_TYPE = "simulador"

    # --- Architecture ---
    # Si Expr Type = "experimentos": "delay", "teleport", "swap", "delay,teleport", "ALL"
    # Si Expr Type = "simulador":    "base", "sqm", "ALL"
    SELECTED_ARCHITECTURE = "ALL"

    #   =================================================================
    #  PARAMETROS ADICIONALES
    # =================================================================
    N_QUBITS = 1
    SHOTS = 1024
    SHOW_FIT = False
    EXPORT_TIKZ = True
    DPI = 600

    # =================================================================
    #  DIRECTORIOS
    # =================================================================
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "graphs_latex")

    DATA_DIRS = [
        os.path.join(PROJECT_ROOT, "data"),
        os.path.join(PROJECT_ROOT, "Final_results", "Version2SM", "Simulacion"),
        os.path.join(PROJECT_ROOT, "Final_results", "Version2SM", "Real"),
        os.path.join(PROJECT_ROOT, "Final_results", "results_Article"),
        os.path.join(PROJECT_ROOT, "Final_results", "Data_All"),
        os.path.join(PROJECT_ROOT, "results_Article", "Version1"),
    ]

    # =================================================================
    #  BUILD CONFIG & RUN
    # =================================================================
    config = build_config(
        state_raw=SELECTED_STATE,
        mitigation_raw=SELECTED_MITIGATION_EXP,
        expr_type_raw=SELECTED_EXPR_TYPE,
        architecture_raw=SELECTED_ARCHITECTURE,
        n_qubits=N_QUBITS,
        shots=SHOTS,
        show_fit=SHOW_FIT,
        export_tikz=EXPORT_TIKZ,
        combine_states_in_one_graph=COMBINE_STATES_IN_ONE_GRAPH,
        dpi=DPI,
        output_dir=OUTPUT_DIR,
        data_dirs=DATA_DIRS,
        mitigation_sqm_raw=SELECTED_MITIGATION_SQM,
        mitigation_swap_raw=SELECTED_MITIGATION_SWAP,
    )

    generate_graphs(config)
