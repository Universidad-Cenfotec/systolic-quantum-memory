# ============================================================
# SQTM Research Project — Main Simulator & Compiler
# Systolic Quantum Teleportation Memory
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
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeBrisbane
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error

from src.modular_circuits.qpc import QPC
from src.modular_circuits.memory_register import StorageRegister
from src.modular_circuits.operation_register import OperationRegister
from src.functions.work_phase import SystolicWorkPhase
from src.functions.teleportation import SystolicTeleportation
from src.functions.qubit_mapper import QubitMapper


class SQTMCompiler:

    # Constant: SWAP operation duration in nanoseconds (NISQ-level)
    SWAP_TIME_NS = 1350

    def __init__(
        self,
        R: int,
        n: int,
        c_max: int,
        t_max_ns: float,
        backend_name: str = "FakeKyiv",
        initial_state: int = 0,
    ):
        
        self.R = R
        self.n = n
        self.c_max = c_max
        self.t_max_ns = t_max_ns
        self.initial_state = initial_state

        # ──────────────────────────────────────────────────────────
        # 1. Initialize backend and qubit resources
        # ──────────────────────────────────────────────────────────
        self.backend = FakeKyiv()
        self.noise_model = NoiseModel.from_backend(self.backend)

        # ──────────────────────────────────────────────────────────
        # 1b. Create thermal relaxation error linked to 'id' gate
        # ──────────────────────────────────────────────────────────
        # T1 and T2 times for FakeKyiv (typical NISQ parameters)
        t1_ns = 150_000  # 150 μs
        t2_ns = 100_000  # 100 μs
        self.time_idle_ns = 700  # One IDLE unit = 7000 ns (equivalent to one gate cycle)
        
        # Create thermal relaxation error for the idle period
        idle_error = thermal_relaxation_error(t1_ns, t2_ns, self.time_idle_ns)
        
        # Inject thermal relaxation to 'id' gate (applied during IDLE periods)
        # NOTE: Apply to a limited set of qubits to avoid serialization issues with Aer backend
        num_physical_qubits = self.backend.configuration().n_qubits
        max_qubits_for_error = min(10, num_physical_qubits)  # Limit to first 10 qubits
        for q in range(max_qubits_for_error):
            self.noise_model.add_quantum_error(idle_error, 'id', [q], warnings=False)
        
        print(f"[SQTM Compiler] Thermal relaxation configured: T1={t1_ns/1000:.1f}μs, T2={t2_ns/1000:.1f}μs")
        print(f"[SQTM Compiler] Idle period per unit: {self.time_idle_ns} ns")
        print(f"[SQTM Compiler] Applied thermal decay to 'id' gate on first {max_qubits_for_error} qubits")

        # Initialize QubitMapper for intelligent qubit allocation
        self.qubit_mapper = QubitMapper(self.backend)
        
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

        # ──────────────────────────────────────────────────────────
        # 2. Initialize Storage Registers (2*R total: Original + Backup)
        # ──────────────────────────────────────────────────────────

        self.memory_registers_original: List[StorageRegister] = [
            StorageRegister(n_qubits=n, reg_id=f"mem_orig_{i}") for i in range(R)
        ]

        self.memory_registers_backup: List[StorageRegister] = [
            StorageRegister(n_qubits=n, reg_id=f"mem_backup_{i}") for i in range(R)
        ]

        # ──────────────────────────────────────────────────────────
        # 2b. Initialize Ancilla Registers (1 per Original-Backup pair)
        # Used for quantum teleportation protocol
        # ──────────────────────────────────────────────────────────

        self.ancilla_registers: List[OperationRegister] = [
            OperationRegister(n_qubits=n, reg_id=f"ancilla_{i}") for i in range(R)
        ]

        # ──────────────────────────────────────────────────────────
        # 3. Initialize Operation Register (Work Phase)
        # ──────────────────────────────────────────────────────────

        self.operation_register = OperationRegister(n_qubits=n, reg_id="work")

        # ──────────────────────────────────────────────────────────
        # 4. Initialize QPC (Odometer - Hybrid desgaste tracker)
        # ──────────────────────────────────────────────────────────

        self.qpc = QPC(logical_size=R, c_max=c_max, t_max=t_max_ns)

        # Location map: 'O' = Original, 'B' = Backup
        self.location_map: Dict[int, str] = {i: "O" for i in range(R)}

        # Current counters for active tracking (per register)
        self.current_c: Dict[int, int] = {i: 0 for i in range(R)}
        self.current_t: Dict[int, float] = {i: 0.0 for i in range(R)}

        # ──────────────────────────────────────────────────────────
        # 5. Initialize Functional Modules
        # ──────────────────────────────────────────────────────────

        self.work_phase = SystolicWorkPhase(name="sqtm_work_phase")
        self.teleportation = SystolicTeleportation(name="sqtm_teleportation")

        # Cache for built registers (to avoid rebuilding)
        self._built_registers: Dict[str, QuantumRegister] = {}

        state_label = "|0⟩" if initial_state == 0 else "|1⟩"
        print(f"[SQTM Compiler] Initialized: R={R}, n={n}, c_max={c_max}, t_max={t_max_ns} ns")
        print(f"[SQTM Compiler] Fidelity target state: {state_label}")
        print(f"[Backend] {self.backend.__class__.__name__} with {self.qubit_mapper.n_qubits} qubits")

    # ──────────────────────────────────────────────────────────────
    # QUBIT ALLOCATION & PHYSICAL MAPPING
    # ──────────────────────────────────────────────────────────────

    def _allocate_physical_qubits(
        self,
        register_type: str,
        logical_addr: int,
        quantum_register: QuantumRegister,
    ) -> None:

        required_qubits = quantum_register.size
        register_id = quantum_register.name

        # Use QubitMapper to find connected subgraph
        allocated = self.qubit_mapper.allocate_register(
            register_type=register_type,
            register_id=register_id,
            size=required_qubits
        )

        # Map the actual qubit objects to physical indices
        for local_idx, physical_qubit in enumerate(allocated):
            qubit_obj = quantum_register[local_idx]
            self.logical_to_physical_map[qubit_obj] = physical_qubit
            self.qubit_register_map[physical_qubit] = register_id

    def _get_initial_layout(self, qc: QuantumCircuit) -> List[int]:

        initial_layout = []
        for qubit in qc.qubits:
            if qubit in self.logical_to_physical_map:
                initial_layout.append(self.logical_to_physical_map[qubit])
            else:
                raise RuntimeError(f"Qubit {qubit} not physically allocated! Cannot simulate.")
        return initial_layout

    # ──────────────────────────────────────────────────────────────
    # COMPILER MAIN METHOD: COMPILE WORKLOAD
    # ──────────────────────────────────────────────────────────────

    def compile_workload(self, workload: List[str]) -> QuantumCircuit:
        """
        Core compilation engine: Process workload instructions and generate circuit.

        Workload format:
        - "IDLE_X": Idle X nanoseconds
        - "READ_ij": Read from logical register i (binary j = lower bits, i = upper bits)
        - "WRITE_ij": Write to logical register i (binary j format)

        Parameters
        ----------
        workload : List[str]
            List of instruction strings.

        Returns
        -------
        QuantumCircuit
            Compiled quantum circuit ready for simulation/execution.
        """

        # ──────────────────────────────────────────────────────────
        # SEED INITIALIZATION - For reproducibility
        # ──────────────────────────────────────────────────────────
        random.seed(42)
        np.random.seed(42)

        # ──────────────────────────────────────────────────────────
        # Phase 0: Build register instances and allocate physical qubits
        # ──────────────────────────────────────────────────────────

        qc = QuantumCircuit()

        # Build all register instances (logical) first
        for i in range(self.R):
            qr_orig = self.memory_registers_original[i].build()
            qr_backup = self.memory_registers_backup[i].build()
            qr_ancilla = self.ancilla_registers[i].build()
            
            qc.add_register(qr_orig)
            qc.add_register(qr_ancilla)
            qc.add_register(qr_backup)
            
            self._built_registers[f"mem_orig_{i}"] = qr_orig
            self._built_registers[f"ancilla_{i}"] = qr_ancilla
            self._built_registers[f"mem_backup_{i}"] = qr_backup

        qr_opreg = self.operation_register.build()
        qc.add_register(qr_opreg)
        self._built_registers["opreg"] = qr_opreg

        # ──────────────────────────────────────────────────────────
        # CHAIN TOPOLOGY ALLOCATION (WITH ANCILLAS)
        # ──────────────────────────────────────────────────────────
        # Structure: OpReg — Mem_Orig_0 — Ancilla_0 — Mem_Backup_0 — Mem_Orig_1 — Ancilla_1 — Mem_Backup_1 — ...
        # This ensures direct connectivity without routing SWAPs with optimization_level=0
        # ──────────────────────────────────────────────────────────
        
        print("\n[Compilation] Allocating chain topology with ancillas...")
        print(f"[Compilation] Structure: OpReg—Mem_Orig—Ancilla—Mem_Backup — ...")
        
        # Build chain configuration: OpReg + (Original + Ancilla + Backup) x R
        chain_config = [("opreg", self.n)]
        for i in range(self.R):
            chain_config.append((f"mem_orig_{i}", self.n))
            chain_config.append((f"ancilla_{i}", self.n))
            chain_config.append((f"mem_backup_{i}", self.n))
        
        total_qubits_needed = sum(size for _, size in chain_config)
        print(f"[Compilation] Total qubits needed: {total_qubits_needed}")
        print(f"[Compilation] Chain config: {chain_config}")
        
        # Allocate the linear chain
        allocation_map = self.qubit_mapper.allocate_chain_topology(chain_config)
        
        # Map logical qubits to physical qubits for all registers
        for reg_id, physical_qubits in allocation_map.items():
            if reg_id == "opreg":
                qr = self._built_registers["opreg"]
            else:
                # mem_orig_*, ancilla_*, mem_backup_*
                qr = self._built_registers[reg_id]
            
            for local_idx, physical_qubit in enumerate(physical_qubits):
                qubit_obj = qr[local_idx]
                self.logical_to_physical_map[qubit_obj] = physical_qubit
                self.qubit_register_map[physical_qubit] = reg_id

        print(f"\n[Compilation] Physical qubit mapping:")
        for reg_id, phys_qubits in allocation_map.items():
            print(f"  {reg_id:20s} → {sorted(phys_qubits)}")
        
        print(f"\n[Compilation] Total qubits allocated: {len(self.logical_to_physical_map)}")

        # ──────────────────────────────────────────────────────────
        # Phase 0b: Prepare initial quantum state
        # ──────────────────────────────────────────────────────────
        
        if self.initial_state == 1:
            print("\n[Compilation] Preparing initial state |1...1⟩ (applying X to all qubits: memory + operation register)...")
            # Apply X gates to ALL qubits (memory and operation register) to prepare |1...1⟩ state
            # This ensures SWAPs don't affect the final state validation
            
            # Apply X to operation register
            for qubit in qr_opreg:
                qc.x(qubit)
            
            # Apply X to all memory qubits
            for i in range(self.R):
                        qr_orig = self._built_registers[f"mem_orig_{i}"]
                        for qubit in qr_orig:
                            qc.x(qubit)
            
            qc.barrier()
            print("[Compilation] Initial state |1...1⟩ prepared for all qubits")
        else:
            print("\n[Compilation] Using default initial state |0...0⟩ (no X gates applied)")

        print(f"\n[Compilation] Starting workload processing: {len(workload)} instructions")

        # ──────────────────────────────────────────────────────────
        # Phase 1: Process workload instructions
        # ──────────────────────────────────────────────────────────

        for instruction in workload:
            print(f"  [Instruction] {instruction}")

            if instruction.startswith("IDLE_"):
                # IDLE instruction: Apply thermal relaxation via native 'id' gate
                # Each IDLE unit maps to one identity gate with attached thermal noise
                num_units = int(instruction.split("_")[1])
                time_ns = num_units * self.time_idle_ns
                
                print(f"    -> Wear-down sequence: {num_units} idle units ({time_ns:.0f} ns total)")
                
                # Get only the qubits that are currently active (holding data or in operation register)
                active_qubits = self._get_active_qubits_for_idle()
                
                # Apply thermal relaxation only to active qubits
                # This represents realistic wear-down: only used qubits experience decoherence
                for _ in range(num_units):
                    for qubit in active_qubits:
                        qc.id(qubit)
                
                # Increment time for all active registers
                for i in range(self.R):
                    self.current_t[i] += time_ns
                for logical_addr in range(self.R):    
                    self._check_and_apply_tele_refresh(qc, logical_addr, gate_cost=0,
                    time_dt=self.time_idle_ns)
                
            elif instruction.startswith("READ_"):
                # READ instruction: READ_ij where ij is binary address
                address_binary = instruction.split("_")[1]
                logical_addr = int(address_binary, 2)  # Binary to decimal

                if logical_addr >= self.R:
                    raise ValueError(f"Logical address {logical_addr} out of range [0, {self.R - 1}]")

                print(f"    -> READ from Mem[{logical_addr}]")

                # Increment counters
                self.current_c[logical_addr] += 1
                self.current_t[logical_addr] += self.SWAP_TIME_NS

                # Determine source register (Original or Backup)
                if self.location_map[logical_addr] == "O":
                    source_reg = self._built_registers[f"mem_orig_{logical_addr}"]
                else:
                    source_reg = self._built_registers[f"mem_backup_{logical_addr}"]

                # Apply SWAP between source and OpReg
                qc = self.work_phase.apply_swap(qc, source_reg, qr_opreg)

                # Check if odometer threshold exceeded
                self._check_and_apply_tele_refresh(qc, logical_addr, gate_cost=1, time_dt=self.SWAP_TIME_NS)

            elif instruction.startswith("WRITE_"):
                # WRITE instruction: WRITE_ij
                address_binary = instruction.split("_")[1]
                logical_addr = int(address_binary, 2)

                if logical_addr >= self.R:
                    raise ValueError(f"Logical address {logical_addr} out of range [0, {self.R - 1}]")

                print(f"    -> WRITE to Mem[{logical_addr}]")

                # Increment counters
                self.current_c[logical_addr] += 1
                self.current_t[logical_addr] += self.SWAP_TIME_NS

                # Determine destination register
                if self.location_map[logical_addr] == "O":
                    dest_reg = self._built_registers[f"mem_orig_{logical_addr}"]
                else:
                    dest_reg = self._built_registers[f"mem_backup_{logical_addr}"]

                # Apply SWAP between OpReg and destination
                qc = self.work_phase.apply_swap(qc, qr_opreg, dest_reg)


                # Check if odometer threshold exceeded
                self._check_and_apply_tele_refresh(qc, logical_addr, gate_cost=1, time_dt=self.SWAP_TIME_NS)

            else:
                print(f"  [WARNING] Unknown instruction: {instruction}")
            qc.barrier()  # Prevent inter-SWAP optimization
            #print(qc.draw(output="text"))   

        print(f"[Compilation] Workload processing complete")
        return qc

    def _get_active_qubits_for_idle(self) -> List:

        active_qubits = []
        
        # 1. Add all qubits from operation register (work register)
        qr_work = self._built_registers["opreg"]
        for qubit in qr_work:
            active_qubits.append(qubit)
        
        # 2. For each logical address, add qubits from the register that holds the data
        for logical_addr in range(self.R):
            if self.location_map[logical_addr] == "O":
                # Data is in Original memory
                qr_data = self._built_registers[f"mem_orig_{logical_addr}"]
            else:
                # Data is in Backup memory
                qr_data = self._built_registers[f"mem_backup_{logical_addr}"]
            
            # Add all qubits from the active data register
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

            # Determine source and destination registers
            if self.location_map[logical_addr] == "O":
                source_reg = self._built_registers[f"mem_orig_{logical_addr}"]
                dest_reg = self._built_registers[f"mem_backup_{logical_addr}"]
                future_location = "B"
            else:
                source_reg = self._built_registers[f"mem_backup_{logical_addr}"]
                dest_reg = self._built_registers[f"mem_orig_{logical_addr}"]
                future_location = "O"

            # Apply quantum teleportation
            qc = self.teleportation.build_circuit(qc, source_reg, dest_reg)

            # FIX: ¡Interceptar registros ancilla creados dinámicamente y asignarles hardware!
            for qreg in qc.qregs:
                if qreg.name not in self._built_registers:
                    self._allocate_physical_qubits("ancilla", logical_addr, qreg)
                    self._built_registers[qreg.name] = qreg

            # Update location map
            self.location_map[logical_addr] = future_location

            # Reset counters via QPC
            self.qpc.tick(logical_addr)
            self.current_c[logical_addr] = 0
            self.current_t[logical_addr] = 0.0

            print(f"    [Tele-Refresh] Mem[{logical_addr}] now stored in {future_location}")

    # ──────────────────────────────────────────────────────────────
    # SIMULATION WITH NOISE
    # ──────────────────────────────────────────────────────────────

    def run_simulation(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, Any]:
 

        # ──────────────────────────────────────────────────────────
        # SEED SETUP FOR REPRODUCIBILITY
        # ──────────────────────────────────────────────────────────
        random.seed(42)
        np.random.seed(42)
        
        print(f"\n[Simulation] Preparing circuit for {shots} shots")
        print(f"[Simulation] Seeds configured for reproducibility")

        try:
            qc_measured = circuit.copy()
            
   
            # Measure ALL n*R qubits (all memory registers) for correct fidelity
            cr_final = ClassicalRegister(self.n * self.R, name="final_meas")
            qc_measured.add_register(cr_final)

   
            # The data location changes with each tele-refresh operation

            classical_bit_index = 0
            for logical_addr in range(self.R):
                # Determine which register holds the data for this logical address
                if self.location_map[logical_addr] == "O":
                    # Data is in Original memory
                    target_reg = self._built_registers[f"mem_orig_{logical_addr}"]
                else:
                    # Data is in Backup memory
                    target_reg = self._built_registers[f"mem_backup_{logical_addr}"]
                
                # Measure only the register containing the data
                for i in range(self.n):
                    qc_measured.measure(target_reg[i], cr_final[classical_bit_index])
                    classical_bit_index += 1

            # ------------------------------------------------------------------
            # TRANSPILATION WITH EXPLICIT SEEDS
            # ------------------------------------------------------------------
            print("[Transpile] Translating to hardware topology with seed=42...")
            
            # Extract initial layout before transpilation
            initial_layout = self._get_initial_layout(qc_measured)
            
            print("[Noise Model] Extracting noise characteristics...")
            print(qc_measured.draw(output="text"))
           
            
            # Initialize simulator with MPS method and fixed seed
            simulator = AerSimulator(
                noise_model=self.noise_model, 
                method='matrix_product_state',
                seed_simulator=42
            )
            
            # Transpile with seed for reproducibility
            qc_transpiled = transpile(
                qc_measured,
                backend=self.backend,
                optimization_level=0,
                initial_layout=initial_layout,
                seed_transpiler=42
            )
            
            print(f"[Simulator] Running {shots} shots with fixed seed...")
            job = simulator.run(qc_transpiled, shots=shots, seed=42)
            result = job.result()
            
            counts = result.get_counts()
            print(  f"[Simulation] Raw counts: {counts}")
            total_counts = sum(counts.values())
            
            # Extract results: Compare measurement bits against target state
            # We measure n*R bits total (n qubits per logical address, only from the register holding the data)
            fidelity_count = 0
            target_state_bits = self.n * self.R
            target_state = ('1' * target_state_bits) if self.initial_state == 1 else ('0' * target_state_bits)
            
            for outcome, count in counts.items():
                # The outcome string contains all measured bits (no spaces since single register)
                separated_registers = outcome.split()
                final_meas_bits = separated_registers[0]
            
                if final_meas_bits == target_state:
                    fidelity_count += count
            
            fidelity = fidelity_count / total_counts if total_counts > 0 else 0.0

            print(f"\n[Circuit] Generated transpiled circuit:")
            print(f"  Qubits: {qc_transpiled.num_qubits}")
            print(f"  Clbits: {qc_transpiled.num_clbits}")
            print(f"  Depth: {qc_transpiled.depth()}")
            print(f"  Size: {qc_transpiled.size()}")
            
            state_label = "|1...1>" if self.initial_state == 1 else "|0...0>"
            print(f"  Fidelity ({state_label} success for {self.n * self.R} data qubits in their current location): {fidelity:.4f}")
            
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
            print(f"[ERROR] Simulation failed: {e}")
            import traceback
            traceback.print_exc()
            # Return graceful degradation result
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
            "location_map": self.location_map,
            "current_c": self.current_c,
            "current_t": self.current_t,
            "logical_to_physical_map": self.logical_to_physical_map,
            "available_qubits": len(self.qubit_mapper.available_qubits),
        }

