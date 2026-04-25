# Systolic Quantum Memory (SQM)

**Authors:** Danny Valerio-Ramírez (CENFOTEC) · Santiago Núñez-Corrales (UIUC)

Quantum memory architecture based on systolic teleportation to mitigate decoherence in the NISQ era. Includes comparative analysis with SWAP-based baseline, randomized benchmarking (RB) characterization, and real IBM Quantum hardware experiments.

---

## Project Structure

```
SQM/
├── src/                                    # Source code
│   ├── __init__.py
│   ├── comparison.py                       # Comparative analysis (SQM vs SWAP, simulation & hardware)
│   ├── backends/                           # Backend abstraction layer (DI pattern)
│   │   ├── __init__.py
│   │   ├── backend_interface.py            # Abstract interface (BackendInterface)
│   │   ├── aer_simulator_backend.py        # Local noisy simulation (AerSimulatorBackend)
│   │   └── ibm_hardware_backend.py         # Real IBM Quantum hardware (IBMHardwareBackend)
│   ├── modular_circuits/                   # Core circuit components
│   │   ├── __init__.py
│   │   ├── qpc.py                          # QPC odometer (hybrid desgaste tracker)
│   │   ├── memory_register.py              # StorageRegister (passive memory)
│   │   └── operation_register.py           # OperationRegister (active CPU)
│   ├── functions/                          # Quantum algorithms & operations
│   │   ├── __init__.py
│   │   ├── qubit_mapper.py                 # Hardware-aware qubit allocation (chain topology)
│   │   ├── teleportation.py                # SystolicTeleportation (3-parallel bus)
│   │   └── work_phase.py                   # SystolicWorkPhase (NISQ SWAP)
│   ├── simulator/                          # Quantum compilers & simulators
│   │   ├── __init__.py
│   │   ├── sqm_simulator.py               # SQMCompiler (fidelity on memory registers)
│   │   ├── sqm_simulator_Flow.py           # SQMFlowCompiler (fidelity on operation register)
│   │   ├── swap_simulator.py               # SwapCompiler (fidelity on memory registers)
│   │   └── swap_simulator_Flow.py          # SwapFlowCompiler (fidelity on operation register)
│   ├── time_calculation/                   # RB characterization & threshold validators
│   │   ├── __init__.py
│   │   ├── ibm_backend_helper.py           # Shared IBM backend + SamplerV2 utilities
│   │   ├── tmax_calculator.py              # Passive desgaste threshold (analytical)
│   │   ├── Tmax_validator_delay.py         # Passive desgaste via native delay() instructions
│   │   ├── tmax_validator_Id.py            # Passive desgaste via identity gates + thermal noise
│   │   ├── cmax_validator_sqm.py           # Active desgaste (SQM + SystolicTeleportation)
│   │   ├── cmax_validator_swap.py          # Active desgaste (SWAP pairs)
│   │   ├── cmax_validator_teleport.py      # Active desgaste (teleportation only, ping-pong)
│   │   └── cmax_validator_not.py           # Active desgaste (NOT gate pairs)
│   └── utils/                              # Shared utility modules
│       ├── __init__.py
│       ├── measurement_parser.py           # MeasurementParser (endianness-aware bit extraction)
│       └── hardware_results_processor.py   # CSV/PNG export for hardware experiments
├── tests/                                  # Test suite
│   ├── __init__.py
│   ├── qpc_test.py                         # BipartiteQPC validation
│   ├── teleportation_test.py               # End-to-end teleportation
│   ├── work_phase_test.py                  # Work phase simulation
│   ├── simulator_integration_test.py       # SQM/SWAP compiler integration
│   ├── cmax_validator_integration_test.py  # CMax validator integration
│   ├── swap_validator_integration_test.py  # SWAP validator integration
│   ├── measurement_parser_test.py          # MeasurementParser unit tests
│   └── Test_IBMHardwareBackend.py          # IBM hardware backend connectivity test
├── context/                                # Documentation & research context
│   ├── context.md                          # Full research context document
│   ├── Api.txt                             # IBM Quantum API token reference
│   └── setup_token.py                      # IBM credential setup helper
├── data/                                   # Experiment output data (CSV)
├── results/                                # Simulation graphs (PNG)
├── results_Article/                        # Published article results (CSV + PNG)
├── main.py                                 # Entry point: local simulation (AerSimulator)
├── main_hardware_experiment.py             # Entry point: IBM Quantum hardware experiment
├── requirements.txt                        # Python dependencies
├── pyrightconfig.json                      # Type checker configuration
├── LICENSE                                 # Open-source license
└── README.md                               # This file
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `qiskit` | ≥ 1.3.0 | Core SDK: circuits, transpiler, primitives |
| `qiskit-aer` | ≥ 0.14.0 | High-performance noisy simulation (T1/T2) |
| `qiskit-ibm-runtime` | 0.46.1 | IBM backend access & SamplerV2 primitives |
| `numpy` | 1.26.4 | N-dimensional arrays & linear algebra |
| `scipy` | 1.13.1 | Advanced math: curve fitting, optimization |
| `networkx` | 3.3 | Graph algorithms & circuit topology analysis |
| `matplotlib` | 3.9.2 | Plots, histograms, fidelity decay curves |
| `pylatexenc` | 2.10 | LaTeX-style circuit rendering in Qiskit |

---

## Quick Start

### 1. Create Virtual Environment (Python 3.11)
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure IBM Quantum Credentials (for hardware experiments)
```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel='ibm_quantum_platform',
    token='YOUR_IBM_QUANTUM_TOKEN'
)
```

### 4. Run Local Comparative Analysis
```powershell
python main.py
```

### 5. Run Hardware Experiment (IBM Quantum)
```powershell
python main_hardware_experiment.py
```

---

## Entry Points

### `main.py` — Local Simulation

Executes SQM ↔ SWAP comparative analysis using the AerSimulator backend with a FakeKyiv noise model.

**Configuration parameters (edit `main.py`):**
```python
# Compiler parameters
R = 1              # Number of logical memory registers
n = 1              # Qubits per register (quantum word width)
c_max = 10         # Gate cost threshold
t_max_ns = 1350    # Time threshold (nanoseconds)
shots = 4000       # Simulation shots per workload
flow = 1           # 0 = measure memory registers, 1 = measure operation register

# Backend thermal parameters (custom mode)
backend_mode = "custom"   # "default" or "custom"
t1_ns = 149149            # T1 relaxation time (ns)
t2_ns = 38194             # T2 dephasing time (ns)
idle_time_ns = 1000       # Idle period duration (ns)

# Quantum state configuration
initial_state = 3  # 0 = |0⟩, 1 = |1⟩, 2 = |+⟩ (H), 3 = |−⟩ (XH)

# Workloads with parametric IDLE durations
workload1 = ["WRITE_0", "IDLE_10", "READ_0"]
workload2 = ["WRITE_0", "IDLE_20", "READ_0"]
# ... up to workload14
```

**What it does:**
- Creates an `AerSimulatorBackend` with configurable thermal parameters
- Runs SQM and SWAP compilers for each workload
- Generates fidelity comparison graphs (`results/`) and CSV data (`data/`)
- Supports both memory-register and operation-register fidelity measurement

---

### `main_hardware_experiment.py` — IBM Quantum Hardware

Executes multi-scenario, multi-workload experiments on real IBM Quantum hardware.

**Scenarios:**
| Scenario | Configuration | Purpose |
|----------|---------------|---------|
| **1** | SWAP Compiler (baseline) | Static decoherence, no refresh |
| **2** | SQM without delay | Routing overhead only (time_idle_ns=0) |
| **3** | SQM with real timing | Backend-calibrated decoherence |

**Configuration:**
```python
# Hardware execution
shots = 1000       # Keep low to preserve IBM quota
flow = 1           # 0 = memory registers, 1 = operation register
initial_state = 3  # 0 = |0⟩, 1 = |1⟩, 2 = |+⟩, 3 = |−⟩

# Scenario selection
scenario = [1, 3]  # Run specific scenarios (or [1, 2, 3] for all)
```

**Output:**
- Multi-workload CSV results → `data/hardware_comparison_multi_*.csv`
- Multi-workload comparison graph → `results/hardware_comparison_multi_graph_*.png`

---

## Module Reference

### Backend Abstraction Layer (`src/backends/`)

| Module | Class | Purpose |
|--------|-------|---------|
| `backend_interface.py` | `BackendInterface` | Abstract interface: `run()`, `time_idle_ns`, `use_native_delay` |
| `aer_simulator_backend.py` | `AerSimulatorBackend` | Local FakeKyiv + thermal relaxation noise model |
| `ibm_hardware_backend.py` | `IBMHardwareBackend` | Real IBM Quantum via SamplerV2 + dynamic calibration |

The backend layer uses **dependency injection**: backends are created in `main.py` / `main_hardware_experiment.py` and passed to compilers and comparison functions. This decouples circuit compilation from execution.

**Key features:**
- `AerSimulatorBackend`: Configurable T1/T2/idle noise, `id()` gate for idle periods
- `IBMHardwareBackend`: Native `delay()` instructions, immutable `time_idle_ns` after calibration, `MockResult`/`MockJob` wrappers for SamplerV2 → V1 compatibility

---

### Compilers & Simulators (`src/simulator/`)

| Module | Class | Fidelity Target | Description |
|--------|-------|-----------------|-------------|
| `sqm_simulator.py` | `SQMCompiler` | Memory registers | Dual-register + teleportation + QPC odometer |
| `sqm_simulator_Flow.py` | `SQMFlowCompiler` | Operation register | Same compilation, fidelity on `q_work` |
| `swap_simulator.py` | `SwapCompiler` | Memory registers | Single-register baseline + SWAP gates |
| `swap_simulator_Flow.py` | `SwapFlowCompiler` | Operation register | Same compilation, fidelity on `q_work` |

**Memory vs Flow modes:**
- **Memory mode** (`flow=0`): Measures fidelity on the memory registers where data is stored at rest
- **Flow mode** (`flow=1`): Measures fidelity on the operation register (`q_work`) — evaluates the quantum state "in transit"

**Superposition state support:**
| `initial_state` | State | Preparation | Measurement target |
|-----------------|-------|-------------|--------------------|
| 0 | \|0⟩ | None | `0…0` |
| 1 | \|1⟩ | X gate | `1…1` |
| 2 | \|+⟩ | H gate | `0…0` (after H decode) |
| 3 | \|−⟩ | X + H gates | `1…1` (after H decode) |

---

### Workload Instruction Set

| Instruction | Format | Description |
|-------------|--------|-------------|
| `WRITE_<addr>` | `WRITE_0` | SWAP data from `q_work` → memory register at address |
| `READ_<addr>` | `READ_0` | SWAP data from memory register → `q_work` |
| `IDLE_<units>` | `IDLE_100` | Apply `units × time_idle_ns` nanoseconds of decoherence |
| `WORKING_<pairs>` | `WORKING_3` | Apply `2×pairs` X gates on `q_work` (computation phase) |

Addresses are binary-encoded (e.g., `WRITE_0` = address 0, `READ_10` = address 2).

---

### Time Characterization & Validators (`src/time_calculation/`)

Seven independent validators characterize fidelity decay using the **Magesan model** `F(m) = A·p^m + B` or **exponential decay** `F(t) = A·exp(−t/τ) + B`:

| Validator | Protocol | Model | Gate Type | Output |
|-----------|----------|-------|-----------|--------|
| `cmax_validator_sqm.py` | SQM (SWAP + teleport) | Magesan | 2Q (ECR/CX) | C_MAX (cycles) |
| `cmax_validator_swap.py` | SWAP pairs | Magesan | 2Q (ECR/CX) | C_MAX (SWAP pairs) |
| `cmax_validator_teleport.py` | Teleportation only (ping-pong) | Magesan | 2Q (ECR/CX) | C_MAX (teleportations) |
| `cmax_validator_not.py` | NOT gate pairs | Magesan | 1Q (X/SX/RZ) | C_MAX (NOT pairs) |
| `Tmax_validator_delay.py` | Native `delay()` instruction | Exponential | — | T_MAX (nanoseconds) |
| `tmax_validator_Id.py` | Identity gates + thermal noise | Magesan | id | T_MAX (ns via C_MAX × idle_time) |
| `tmax_calculator.py` | Analytical (T1/T2 only) | Analytical | — | T_MAX (nanoseconds) |

Each validator supports:
- **FakeKyiv simulation** (default) or **real IBM hardware** (`backend_mode = "IBM"`)
- Full RB characterization with `curve_fit`
- Decay curve plots (PNG) and data export (CSV) to `results/` and `data/`
- Extrapolation validation (model vs fresh measurement)
- `ibm_backend_helper.py` provides shared `get_ibm_backend()` and `run_on_ibm()` for hardware execution

---

### Utility Modules (`src/utils/`)

| Module | Class/Function | Purpose |
|--------|----------------|---------|
| `measurement_parser.py` | `MeasurementParser` | Endianness-aware bitstring extraction for multi-register outcomes |
| `hardware_results_processor.py` | `save_hardware_comparison_results()` | Single-workload CSV + comparison graph |
| `hardware_results_processor.py` | `save_hardware_multi_workload_results()` | Multi-workload aggregated CSV + grouped bar chart |

**MeasurementParser features:**
- `split_registers()` — Split outcome by spaces
- `extract_register_bits()` — Extract bits using explicit layout dictionary
- `build_register_layout_from_order()` — Build layout from circuit registration order (handles Qiskit little-endian)
- `validate_layout()` — Verify layout consistency

---

### Quantum Circuit Components (`src/modular_circuits/`)

| Module | Class | Purpose |
|--------|-------|---------|
| `memory_register.py` | `StorageRegister` | Passive memory qubits |
| `operation_register.py` | `OperationRegister` | Active operation workspace |
| `qpc.py` | `QPC` | Quantum Processor Component — hybrid odometer for gate cost (`c_max`) and time-based (`t_max`) desgaste tracking |

---

### Comparison Module (`src/comparison.py`)

Orchestrates all comparative analyses with four execution modes:

| Function | Mode | Description |
|----------|------|-------------|
| `run_full_comparison()` | Local simulation | Multi-workload SQM ↔ SWAP comparison (graph + CSV) |
| `run_real_comparison()` | IBM hardware | Multi-scenario experiment (1/2/3 scenarios, selectable) |
| `analyze_workload()` | Memory fidelity | Single-workload comparison |
| `analyze_workload_flow()` | Flow fidelity | Single-workload comparison (operation register) |

---

## Architecture Overview

### SQM Compiler (Dual-Register Memory)

```
Workload: [WRITE_0, IDLE_100, READ_0, ...] ──→ Parser
                                                  ↓
                                    QubitMapper (FakeKyiv / IBM Kingston)
                                                  ↓
                        ┌─────────────────────────────────────┐
                        │ Chain Topology Allocation:          │
                        │ q_work → mem_orig_0 → mem_backup_0 │
                        │ → tele_ancilla_0 → mem_orig_1 → ...│
                        └─────────────────────────────────────┘
                                                  ↓
                        QPC Odometer (c_max + t_max thresholds)
                            → Triggers SystolicTeleportation
                              when thresholds exceeded
                                                  ↓
                        Backend Manager (AerSimulator or IBM Hardware)
                        + T1/T2 Thermal Relaxation (id or delay)
                                                  ↓
                              Fidelity Measurement
                              (Memory or Operation register)
```

### SWAP Compiler (Baseline Comparison)

```
Workload: [WRITE_0, IDLE_100, READ_0, ...] ──→ Parser
                                                  ↓
                                    QubitMapper (FakeKyiv / IBM Kingston)
                                                  ↓
                        ┌─────────────────────────────────────┐
                        │ Chain Topology Allocation:          │
                        │ q_work → mem_0 → mem_1 → ...       │
                        └─────────────────────────────────────┘
                                                  ↓
                        SystolicWorkPhase (NISQ SWAP only)
                        No teleportation, no refresh mechanism
                                                  ↓
                        Backend Manager (AerSimulator or IBM Hardware)
                        + T1/T2 Thermal Relaxation (id or delay)
                                                  ↓
                              Fidelity Measurement
                              (Memory or Operation register)
```

### Key Differences

| Feature | SQM | SWAP |
|---------|-----|------|
| Registers per address | 2 (original + backup) | 1 (single copy) |
| Ancilla qubits | Yes (teleportation channel) | No |
| Refresh mechanism | Quantum teleportation | None |
| Desgaste tracking | QPC odometer (c_max + t_max) | None |
| Total qubits | n × (3R + 1) | n × (R + 1) |
| Noise mitigation | Active (tele-refresh resets decoherence) | Passive (accumulates) |

---

## Running Tests

### Unit Tests
```powershell
# Teleportation test
python tests/teleportation_test.py

# Work phase test
python tests/work_phase_test.py

# BipartiteQPC test
python tests/qpc_test.py

# Measurement parser test
python tests/measurement_parser_test.py

# Simulator integration test
python tests/simulator_integration_test.py

# CMax validator integration test
python tests/cmax_validator_integration_test.py

# SWAP validator integration test
python tests/swap_validator_integration_test.py

# IBM hardware backend connectivity test
python tests/Test_IBMHardwareBackend.py
```

### Running RB Characterization Validators
```powershell
# SWAP pair decay characterization
python src/time_calculation/cmax_validator_swap.py

# SQM (SWAP + teleportation) characterization
python src/time_calculation/cmax_validator_sqm.py

# Teleportation-only characterization
python src/time_calculation/cmax_validator_teleport.py

# NOT gate pair characterization
python src/time_calculation/cmax_validator_not.py

# Delay-based idle decoherence characterization
python src/time_calculation/Tmax_validator_delay.py

# Identity gate idle decoherence characterization
python src/time_calculation/tmax_validator_Id.py
```

Each validator can be switched between `"default"` (FakeKyiv) and `"IBM"` (real hardware) by editing the `backend_mode` variable at the bottom of each file.

---

## Output Directory Structure

| Directory | Contents |
|-----------|----------|
| `data/` | CSV files: comparison results, hardware experiments, RB characterization |
| `results/` | PNG files: fidelity comparison graphs, RB decay curves, qubit mapping visualizations |
| `results_Article/` | Curated results for publication (SM decay curves, comparison graphs) |
| `Final_results/` | Consolidated archive of all experiment data and results |

### Output File Naming Convention
- `comparison_results_YYYYMMDD_HHMMSS.csv` — Local simulation comparison data
- `comparison_graph_YYYYMMDD_HHMMSS.png` — Local simulation comparison graph
- `hardware_comparison_multi_YYYYMMDD_HHMMSS.csv` — Hardware multi-workload data
- `hardware_comparison_multi_graph_YYYYMMDD_HHMMSS.png` — Hardware multi-workload graph
- `rb_decay_curve_<protocol>_n=<N>.png` — RB decay curve for specific protocol
- `SM_decay_curve_<protocol>_N<n>.csv` — Published decay curve data

---

## Development Phases

| Phase | Focus | Status |
|-------|-------|--------|
| **0 — Environment** | venv, dependencies, project structure | ✅ Complete |
| **A — Building Blocks** | Registers, teleportation, work phase circuits | ✅ Complete |
| **B — Hardware Mapping** | QubitMapper, chain topology allocation | ✅ Complete |
| **C — Noise Model** | T1/T2 from FakeKyiv, thermal relaxation, backend abstraction | ✅ Complete |
| **D — Comparative Analysis** | SQM ↔ SWAP compiler validation & metrics | ✅ Complete |
| **E — RB Characterization** | Magesan model fitting, C_MAX/T_MAX validation (7 protocols) | ✅ Complete |
| **F — Real Hardware** | IBM Quantum execution (Kingston), multi-scenario experiments | ✅ Complete |
| **G — Flow Measurement** | Operation register fidelity, superposition state support | ✅ Complete |
| **H — Optimization** | ML-based adaptive thresholds, topological improvements | 🔲 Future |

---

## Recent Changes (April 2026)

### ✅ Backend Abstraction Layer
- Added `BackendInterface` abstract class with dependency injection pattern
- `AerSimulatorBackend` wraps FakeKyiv with configurable thermal relaxation
- `IBMHardwareBackend` wraps real IBM hardware with SamplerV2, `MockResult`/`MockJob` for V1 compatibility, immutable `time_idle_ns` after calibration

### ✅ IBM Quantum Hardware Execution
- `main_hardware_experiment.py` — Multi-scenario, multi-workload hardware experiments
- 3 experimental scenarios (SWAP baseline, SQM no-delay, SQM real timing)
- Selectable scenario execution (`scenario = [1, 3]`)
- Hardware results processor with adaptive graph generation (1, 2, or 3 bars)

### ✅ Flow Simulators (Operation Register Fidelity)
- `SQMFlowCompiler` and `SwapFlowCompiler` — measure fidelity on `q_work` instead of memory registers
- `flow` parameter in `main.py` and `main_hardware_experiment.py` toggles between memory and flow modes

### ✅ Superposition State Support
- `initial_state = 2` → |+⟩ (Hadamard gate), fidelity target = |0⟩
- `initial_state = 3` → |−⟩ (X + Hadamard gates), fidelity target = |1⟩
- Measurement basis rotation (H before measurement) for proper fidelity validation

### ✅ RB Characterization Suite (7 Validators)
- SWAP, SQM, Teleportation, NOT, Delay, Identity gate characterization
- Magesan model `F(m) = A·p^m + B` and exponential decay `F(t) = A·exp(−t/τ) + B`
- Shared `ibm_backend_helper.py` for dynamic backend selection (FakeKyiv ↔ IBM hardware)
- Automatic decay curve plots and CSV export

### ✅ Measurement Parser Utility
- Unified `MeasurementParser` class for endianness-aware bitstring extraction
- Handles Qiskit's little-endian register ordering with explicit layout dictionaries
- Used by all compilers and validators

### ✅ Idle Gate Logic Fix
- Fixed `_get_active_qubits_for_idle` to apply decoherence only on qubits currently holding active data
- Prevents unnecessary fidelity degradation on inactive qubits after teleportation

### ✅ Code Quality & Maintenance
- Removed outdated, redundant, and contradictory comments (~35 lines cleaned)
- Fixed noise model documentation across all simulators
- Removed debug print statements and commented-out code
- All 8+ source files reviewed and optimized
