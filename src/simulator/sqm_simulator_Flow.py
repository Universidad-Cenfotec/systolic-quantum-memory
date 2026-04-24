# ============================================================
# SQM Research Project - Flow Simulator & Compiler
# Systolic Quantum Memory - Fidelity measured on Operation Register
# Authors: Danny Valerio-Ramírez & Santiago Núñez-Corrales
# Role: Quantum Compiler Architect (Senior)
# ============================================================

from typing import Any, Dict, List, Tuple, Optional
import sys
import os
import numpy as np
import random

# Ensure project root is in path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Instruction

from src.modular_circuits.qpc import QPC, MemLocation
from src.modular_circuits.memory_register import StorageRegister
from src.modular_circuits.operation_register import OperationRegister
from src.functions.work_phase import SystolicWorkPhase
from src.functions.teleportation import SystolicTeleportation
from src.functions.qubit_mapper import QubitMapper
from src.utils.measurement_parser import MeasurementParser
from src.backends.backend_interface import BackendInterface
from src.backends.aer_simulator_backend import AerSimulatorBackend


class SQMFlowCompiler:
    """
    SQM Compiler variant that measures fidelity on the operation_register (q_work).
    
    Identical compilation logic to SQMCompiler (teleportation, IDLE, READ, WRITE with QPC),
    but the execute() method measures only the operation register instead of memory registers.
    This allows evaluating the quantum state "in transit" through the systolic pipeline.
    """

    # Constant: SWAP operation duration in nanoseconds (NISQ-level)
    SWAP_TIME_NS = 1350

    def __init__(
        self,
        R: int,
        n: int,
        c_max: int,
        t_max_ns: float,
        backend_manager: BackendInterface,
        initial_state: int = 0,
    ):
        
        self.R = R
        self.n = n
        self.c_max = c_max
        self.t_max_ns = t_max_ns
        self.initial_state = initial_state
        
        # Dependency Injection: Backend manager must be provided from main.py
        self.backend_manager = backend_manager
        
        # Retrieve base backend device from backend manager for qubit allocation
        self.backend = self.backend_manager.get_backend_device()

        # Initialize QubitMapper for intelligent qubit allocation
        self.qubit_mapper = QubitMapper(self.backend)
        
        # Store thermal parameters for use in compile_workload
        self.time_idle_ns = self.backend_manager.time_idle_ns
        
        self.logical_to_physical_map: Dict[Any, int] = {}
        """
        Maps qubit_obj (Qubit) -> physical_qubit_id.
        Stores the allocated physical qubit index for each logical qubit object.
        """
        
        self.qubit_register_map: Dict[int, str] = {}
        """
        Maps physical_qubit_id -> register_name.
        Used for validation and tracking.
        """

        # ----------------------------------------------------------
        # 2. Initialize Storage Registers (2*R total: Original + Backup)
        # ----------------------------------------------------------

        self.memory_registers_original: List[StorageRegister] = [
            StorageRegister(n_qubits=n, reg_id=f"mem_orig_{i}") for i in range(R)
        ]

        self.memory_registers_backup: List[StorageRegister] = [
            StorageRegister(n_qubits=n, reg_id=f"mem_backup_{i}") for i in range(R)
        ]

        # ----------------------------------------------------------
        # 2b. Teleportation Ancilla Registers (1 per Original-Backup pair)
        # Pre-allocated Bell channels for quantum teleportation refresh protocol
        # ----------------------------------------------------------

        self.tele_ancilla_registers: List[QuantumRegister] = [
            QuantumRegister(n, name=f"tele_ancilla_{i}") for i in range(R)
        ]
        self.tele_cr_bell_registers: List[ClassicalRegister] = [
            ClassicalRegister(2 * n, name=f"cr_bell_{i}") for i in range(R)
        ]

        # ----------------------------------------------------------
        # 3. Initialize Operation Register (Work Phase)
        # ----------------------------------------------------------

        self.operation_register = OperationRegister(n_qubits=n, reg_id="work")

        # ----------------------------------------------------------
        # 4. Initialize QPC (Odometer - Hybrid desgaste tracker)
        # ----------------------------------------------------------

        self.qpc = QPC(logical_size=R, c_max=c_max, t_max=t_max_ns)
        # Note: QPC fully manages wear-down tracking (costs and idle times).
        # No local tracking needed - consult QPC via get_cost() and get_idle_time().

        # ----------------------------------------------------------
        # 5. Initialize Functional Modules
        # ----------------------------------------------------------

        self.work_phase = SystolicWorkPhase(name="sqm_work_phase")
        self.teleportation = SystolicTeleportation(name="sqm_teleportation")

        # Cache for built registers (to avoid rebuilding)
        self._built_registers: Dict[str, QuantumRegister] = {}

        _state_labels = {0: "|0>", 1: "|1>", 2: "|+> (H)", 3: "|-> (XH)"}
        state_label = _state_labels.get(initial_state, f"unknown({initial_state})")
        print(f"[SQM Flow Compiler] Initialized: R={R}, n={n}, c_max={c_max}, t_max={t_max_ns} ns")
        print(f"[SQM Flow Compiler] Fidelity target state: {state_label}")
        print(f"[SQM Flow Compiler] Fidelity measured on: OPERATION REGISTER (q_work)")
        print(f"[Backend] {self.backend.__class__.__name__} with {self.qubit_mapper.n_qubits} qubits")

    # --------------------------------------------------------------
    # QUBIT ALLOCATION & PHYSICAL MAPPING
    # --------------------------------------------------------------
    # MEASUREMENT OUTCOME PARSING (Unified via MeasurementParser)
    # --------------------------------------------------------------

    @staticmethod
    def _parse_measurement_outcome(outcome: str) -> str:
        """
        Parse measurement outcome string from Qiskit, handling multiple registers.
        
        Uses unified MeasurementParser utility for endianness-aware extraction.
        Returns the first register after splitting on spaces.
        
        In SQM: outcome contains multiple spaces (e.g., "data_bits bell_bits")
        In SWAP: outcome contains single register with no spaces
        
        Returns: First register (measurement bits) after splitting
        """
        registers = MeasurementParser.split_registers(outcome)
        return registers[0] if registers else ""

    # --------------------------------------------------------------

    def _get_initial_layout(self, qc: QuantumCircuit) -> List[int]:

        initial_layout = []
        for qubit in qc.qubits:
            if qubit in self.logical_to_physical_map:
                initial_layout.append(self.logical_to_physical_map[qubit])
            else:
                raise RuntimeError(f"Qubit {qubit} not physically allocated! Cannot simulate.")
        return initial_layout

    # --------------------------------------------------------------
    # COMPILER MAIN METHOD: COMPILE WORKLOAD
    # (Identical to SQMCompiler - same teleportation, IDLE, READ, WRITE logic)
    # --------------------------------------------------------------

    def compile_workload(self, workload: List[str]) -> QuantumCircuit:

        # ----------------------------------------------------------
        # SEED INITIALIZATION - For reproducibility
        # ----------------------------------------------------------
        random.seed(42)
        np.random.seed(42)

        # ----------------------------------------------------------
        # Phase 0: Build register instances and allocate physical qubits
        # ----------------------------------------------------------

        qc = QuantumCircuit()

        # Build all register instances (logical) first
        for i in range(self.R):
            qr_orig = self.memory_registers_original[i].build()
            qr_backup = self.memory_registers_backup[i].build()
            qr_tele_ancilla = self.tele_ancilla_registers[i]
            cr_bell_i = self.tele_cr_bell_registers[i]
            
            qc.add_register(qr_orig)
            qc.add_register(qr_backup)
            qc.add_register(qr_tele_ancilla)
            qc.add_register(cr_bell_i)
            
            self._built_registers[f"mem_orig_{i}"] = qr_orig
            self._built_registers[f"mem_backup_{i}"] = qr_backup
            self._built_registers[f"tele_ancilla_{i}"] = qr_tele_ancilla

        qr_work = self.operation_register.build()
        qc.add_register(qr_work)
        self._built_registers["q_work"] = qr_work

        # Chain Topology: q_work - (Mem_Orig_i - Mem_Backup_i - TeleAncilla_i) x R
        # Linear structure ensures direct connectivity without routing SWAPs
        
        print("\n[Compilation] Allocating chain topology...")
        chain_config = [("q_work", self.n)]
        for i in range(self.R):
            chain_config.append((f"mem_orig_{i}", self.n))
            chain_config.append((f"mem_backup_{i}", self.n))
            chain_config.append((f"tele_ancilla_{i}", self.n))
        
        total_qubits_needed = sum(size for _, size in chain_config)
        print(f"[Compilation] Total qubits needed: {total_qubits_needed}")
        print(f"[Compilation] Chain config: {chain_config}")
        
        # Allocate the linear chain
        allocation_map = self.qubit_mapper.allocate_chain_topology(chain_config)
        
        # Map logical qubits to physical qubits for all registers
        for reg_id, physical_qubits in allocation_map.items():
            if reg_id == "q_work":
                qr = self._built_registers["q_work"]
            else:
                qr = self._built_registers[reg_id]
            
            for local_idx, physical_qubit in enumerate(physical_qubits):
                qubit_obj = qr[local_idx]
                self.logical_to_physical_map[qubit_obj] = physical_qubit
                self.qubit_register_map[physical_qubit] = reg_id

        print(f"\n[Compilation] Physical qubit mapping:")
        for reg_id, phys_qubits in allocation_map.items():
            print(f"  {reg_id:20s} -> {sorted(phys_qubits)}")
        
        print(f"\n[Compilation] Total qubits allocated: {len(self.logical_to_physical_map)}")

        # ----------------------------------------------------------
        # Phase 0b: Prepare initial quantum state
        # ----------------------------------------------------------
        
        if self.initial_state == 1:
            print("\n[Compilation] Preparing initial state |1...1> (applying X to q_work)...")
            for qubit in qr_work:
                qc.x(qubit)
            qc.barrier()
        elif self.initial_state == 2:
            print("\n[Compilation] Preparing initial state |+> (applying H to q_work)...")
            for qubit in qr_work:
                qc.h(qubit)
            qc.barrier()
        elif self.initial_state == 3:
            print("\n[Compilation] Preparing initial state |-> (applying X+H to q_work)...")
            for qubit in qr_work:
                qc.x(qubit)
                qc.h(qubit)
            qc.barrier()
        else:
            print("\n[Compilation] Using default initial state |0...0> (no gates applied)")

        # ----------------------------------------------------------
        # Phase 1: Process workload instructions
        # ----------------------------------------------------------

        for instruction in workload:

            if instruction.startswith("IDLE_"):
                # IDLE instruction: Apply thermal relaxation with intermediate tele-refresh checks
                # If time_ns exceeds t_max_ns, we split into blocks and check for refresh between them
                num_units = int(instruction.split("_")[1])
                time_ns = num_units * self.time_idle_ns
                
                print(f"    -> Wear-down sequence: {num_units} idle units ({time_ns:.0f} ns total)")
                
                # Calculate number of blocks based on t_max_ns threshold
                num_blocks = max(1, int(np.ceil(time_ns / self.t_max_ns)))
                
                if num_blocks > 1:
                    print(f"    -> Splitting {time_ns:.0f} ns into {num_blocks} block(s) (t_max_ns={self.t_max_ns})")
                
                # Apply delays in blocks, checking for tele-refresh between blocks
                remaining_time = time_ns
                for block_idx in range(num_blocks):
                    block_time = min(self.t_max_ns, remaining_time)
                    
                    if num_blocks > 1:
                        print(f"      Block {block_idx + 1}/{num_blocks}: Applying {block_time:.0f} ns delay")
                    
                    # MUST recalculate active_qubits EACH block iteration because
                    # a tele-refresh in the previous block flips QPC location
                    # (ORIGINAL↔BACKUP), so the register holding the data changes.
                    active_qubits = self._get_active_qubits_for_idle()
                    
                    if self.backend_manager.use_native_delay:
                        if block_time > 0:
                            for qubit in active_qubits:
                                qc.delay(int(block_time), qubit, unit='ns')
                        else:
                            print(f"      [Hardware] Skipping zero-delay instruction (block_time=0)")
                    else:
                        num_id_gates = max(1, int(block_time / self.time_idle_ns))
                        for _ in range(num_id_gates):
                            for qubit in active_qubits:
                                qc.id(qubit)
                    
                    qc.barrier()
                    
                    for logical_addr in range(self.R):
                        self._check_and_apply_tele_refresh(qc, logical_addr, gate_cost=0, time_dt=block_time)
                    
                    remaining_time -= block_time
                
            elif instruction.startswith("READ_"):
                address_binary = instruction.split("_")[1]
                logical_addr = int(address_binary, 2)

                if logical_addr >= self.R:
                    raise ValueError(f"Logical address {logical_addr} out of range [0, {self.R - 1}]")

                print(f"    -> READ from Mem[{logical_addr}]")

                if self.qpc.get_location(logical_addr) == MemLocation.ORIGINAL:
                    source_reg = self._built_registers[f"mem_orig_{logical_addr}"]
                else:
                    source_reg = self._built_registers[f"mem_backup_{logical_addr}"]

                qc = self.work_phase.apply_swap(qc, source_reg, qr_work)

                qc.barrier()
                self._check_and_apply_tele_refresh(qc, logical_addr, gate_cost=1, time_dt=self.SWAP_TIME_NS)

            elif instruction.startswith("WRITE_"):
                address_binary = instruction.split("_")[1]
                logical_addr = int(address_binary, 2)

                if logical_addr >= self.R:
                    raise ValueError(f"Logical address {logical_addr} out of range [0, {self.R - 1}]")

                print(f"    -> WRITE to Mem[{logical_addr}]")

                if self.qpc.get_location(logical_addr) == MemLocation.ORIGINAL:
                    dest_reg = self._built_registers[f"mem_orig_{logical_addr}"]
                else:
                    dest_reg = self._built_registers[f"mem_backup_{logical_addr}"]

                qc = self.work_phase.apply_swap(qc, qr_work, dest_reg)

                qc.barrier()
                self._check_and_apply_tele_refresh(qc, logical_addr, gate_cost=1, time_dt=self.SWAP_TIME_NS)

            elif instruction.startswith("WORKING_"):
                # WORKING instruction: WORKING_# where # is number of X-gate pairs
                # Each pair is 2 X gates, so WORKING_1 = 2X, WORKING_3 = 6X
                num_pairs = int(instruction.split("_")[1])
                num_x_gates = 2 * num_pairs
                
                print(f"    -> WORKING phase: Applying {num_x_gates} X gates to q_work ({num_pairs} pairs)")
                
                # Apply X gates to all qubits in work register
                for _ in range(num_x_gates):
                    for qubit in qr_work:
                        qc.x(qubit)
                
                qc.barrier()  # Prevent inter-SWAP optimization
                # WORKING phase does not trigger refresh (pure operation phase, no time cost)

            else:
                print(f"  [WARNING] Unknown instruction: {instruction}")

        print(f"[Compilation] Workload processing complete")
        return qc

    def _get_active_qubits_for_idle(self) -> List:

        active_qubits = []
        
        # 1. Add all qubits from operation register (work register)
        qr_work = self._built_registers["q_work"]
        for qubit in qr_work:
            active_qubits.append(qubit)
        
        # 2. For each logical address, add qubits from the register that holds the data
        for logical_addr in range(self.R):
            if self.qpc.get_location(logical_addr) == MemLocation.ORIGINAL:
                qr_data = self._built_registers[f"mem_orig_{logical_addr}"]
            else:
                qr_data = self._built_registers[f"mem_backup_{logical_addr}"]
            
            for qubit in qr_data:
                active_qubits.append(qubit)
        
        return active_qubits

    def _check_and_apply_tele_refresh(self, qc: QuantumCircuit, logical_addr: int, gate_cost: int = 1, time_dt: float = 0) -> None:

        requires_refresh = self.qpc.update_odometer(
            logical_addr,
            gate_cost=gate_cost,
            time_dt=time_dt
        )

        if requires_refresh:
            print(f"    [Odometer] Threshold exceeded for Mem[{logical_addr}] -> Tele-refreshing")

            current_location = self.qpc.get_location(logical_addr)
            if current_location == MemLocation.ORIGINAL:
                source_reg = self._built_registers[f"mem_orig_{logical_addr}"]
                dest_reg = self._built_registers[f"mem_backup_{logical_addr}"]
            else:
                source_reg = self._built_registers[f"mem_backup_{logical_addr}"]
                dest_reg = self._built_registers[f"mem_orig_{logical_addr}"]

            tele_ancilla = self._built_registers[f"tele_ancilla_{logical_addr}"]
            cr_bell = self.tele_cr_bell_registers[logical_addr]

            qc = self.teleportation.build_circuit(
                qc, source_reg, dest_reg,
                ancilla_reg=tele_ancilla,
                cr_bell=cr_bell,
            )

            new_location = self.qpc.tick(logical_addr)

            print(f"    [Tele-Refresh] Mem[{logical_addr}] now stored in {new_location.value}")

    # --------------------------------------------------------------
    # EXECUTION VIA BACKEND MANAGER
    # FLOW VARIANT: Measures fidelity on OPERATION REGISTER (q_work)
    # --------------------------------------------------------------

    def execute(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, Any]:
        """
        Execute the quantum circuit via the injected backend manager.
        
        FLOW VARIANT: Measures fidelity on the operation register (q_work)
        instead of memory registers. This measures only n qubits.
        
        Parameters
        ----------
        circuit : QuantumCircuit
            Compiled quantum circuit ready for execution
        shots : int, optional
            Number of circuit executions (default: 1024)
        
        Returns
        -------
        Dict[str, Any]
            Execution results including fidelity, counts, etc.
        """

        # ----------------------------------------------------------
        # SEED SETUP FOR REPRODUCIBILITY
        # ----------------------------------------------------------
        random.seed(42)
        np.random.seed(42)
        
        print(f"\n[Execution] Preparing circuit for {shots} shots")
        print(f"[Execution] Seeds configured for reproducibility")
        print(f"[Execution] FLOW MODE: Measuring fidelity on operation register (q_work)")

        try:
            qc_measured = circuit.copy()
            
            # ──────────────────────────────────────────────────────
            # FLOW: Measure ONLY the operation register (q_work)
            # Only n qubits instead of n*R
            # ──────────────────────────────────────────────────────
            cr_final = ClassicalRegister(self.n, name="final_meas")
            qc_measured.add_register(cr_final)

            # For superposition states (2, 3): apply H before measurement
            # to rotate back to computational basis
            qr_work = self._built_registers["q_work"]
            if self.initial_state in (2, 3):
                print("[Execution] Applying H before measurement (superposition decode)")
                for qubit in qr_work:
                    qc_measured.h(qubit)

            # Measure the operation register (q_work)
            for i in range(self.n):
                qc_measured.measure(qr_work[i], cr_final[i])

            # ------------------------------------------------------------------
            # TRANSPILATION WITH EXPLICIT SEEDS
            # ------------------------------------------------------------------
            print("[Transpile] Translating to hardware topology with seed=42...")
            
            initial_layout = self._get_initial_layout(qc_measured)
            print(qc_measured.draw(output='text'))
            print("[Noise Model] Extracting noise characteristics...")
           
            qc_transpiled = transpile(
                qc_measured,
                backend=self.backend,
                optimization_level=0,
                initial_layout=initial_layout,
                seed_transpiler=42
            )

            print(f"[Execution] Sending circuit to Backend Manager...")
            result = self.backend_manager.run(qc_transpiled, shots=shots, seed=42)
            
            counts = result.get_counts()
            total_counts = sum(counts.values())
            
            # ──────────────────────────────────────────────────────
            # FLOW: Compare against target state for n qubits only
            # States 0, 2 -> target '0'*n
            # States 1, 3 -> target '1'*n (state 3: XH init + H measure = |1>)
            # ──────────────────────────────────────────────────────
            fidelity_count = 0
            target_state_bits = self.n  # Only n qubits (operation register)
            target_state = ('1' * target_state_bits) if self.initial_state in (1, 3) else ('0' * target_state_bits)
            
            for outcome, count in counts.items():
                final_meas_bits = self._parse_measurement_outcome(outcome)
            
                if final_meas_bits == target_state:
                    fidelity_count += count
            
            fidelity = fidelity_count / total_counts if total_counts > 0 else 0.0

            print(f"\n[Circuit] Generated transpiled circuit:")
            print(f"  Qubits: {qc_transpiled.num_qubits}")
            print(f"  Clbits: {qc_transpiled.num_clbits}")
            print(f"  Depth: {qc_transpiled.depth()}")
            print(f"  Size: {qc_transpiled.size()}")
            
            _state_labels = {0: "|0...0>", 1: "|1...1>", 2: "|+> (H->|0>)", 3: "|-> (XH->|1>)"}
            state_label = _state_labels.get(self.initial_state, "|0...0>")
            print(f"  Fidelity FLOW ({state_label} success for {self.n} qubits in operation register): {fidelity:.4f}")
            
            # Show top outcomes
            if counts:
                sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                print(f"  Top 3 outcomes: {dict(sorted_counts[:3])}")

            return {
                "fidelity": fidelity,
                "counts": counts,
                "total_shots": shots,
            }

        except Exception as e:
            print(f"[ERROR] Execution failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "fidelity": 0.0,
                "counts": {},
                "total_shots": shots,
                "error": str(e),
            }

    def get_compiler_state(self) -> Dict:
        """
        Return compiler internal state for debugging/inspection.

        Returns
        -------
        Dict
            State dictionary containing maps, counters, etc.
        """
        return {
            "R": self.R,
            "n": self.n,
            "c_max": self.c_max,
            "t_max_ns": self.t_max_ns,
            "qpc_locations": {
                i: self.qpc.get_location(i).value for i in range(self.R)
            },
            "qpc_costs": {
                i: self.qpc.get_cost(i) for i in range(self.R)
            },
            "qpc_idle_times": {
                i: self.qpc.get_idle_time(i) for i in range(self.R)
            },
            "logical_to_physical_map": self.logical_to_physical_map,
            "available_qubits": len(self.qubit_mapper.available_qubits),
        }
