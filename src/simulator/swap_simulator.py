# ============================================================
# SQM Research Project - Main Simulator & Compiler
# Systolic Quantum Teleportation Memory
# Authors: Danny Valerio-Ram?rez & Santiago N??ez-Corrales
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

from src.modular_circuits.memory_register import StorageRegister
from src.modular_circuits.operation_register import OperationRegister
from src.functions.work_phase import SystolicWorkPhase
from src.functions.qubit_mapper import QubitMapper
from src.utils.measurement_parser import MeasurementParser
from src.backend.backend_interface import BackendInterface
from src.backend.aer_simulator_backend import AerSimulatorBackend


class SwapCompiler:

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
        # 2. Initialize Storage Registers (Single copy per register)
        # ----------------------------------------------------------

        self.memory_registers: List[StorageRegister] = [
            StorageRegister(n_qubits=n, reg_id=f"mem_{i}") for i in range(R)
        ]

        # ----------------------------------------------------------
        # 3. Initialize Operation Register (Work Phase)
        # ----------------------------------------------------------

        self.operation_register = OperationRegister(n_qubits=n, reg_id="work")

        # ----------------------------------------------------------
        # 4. Initialize Functional Modules
        # ----------------------------------------------------------

        self.work_phase = SystolicWorkPhase(name="sqm_work_phase")

        # Cache for built registers (to avoid rebuilding)
        self._built_registers: Dict[str, QuantumRegister] = {}

        state_label = "|0>" if initial_state == 0 else "|1>"
        print(f"[Swap Compiler] Initialized: R={R}, n={n}, c_max={c_max}, t_max={t_max_ns} ns")
        print(f"[Swap Compiler] Fidelity target state: {state_label}")
        print(f"[Backend] {self.backend.__class__.__name__} with {self.qubit_mapper.n_qubits} qubits")

    # --------------------------------------------------------------
    # MEASUREMENT OUTCOME PARSING (Unified Parser)
    # --------------------------------------------------------------

    @staticmethod
    def _parse_measurement_outcome(outcome: str) -> str:
        """
        Parse measurement outcome string from Qiskit, handling multiple registers.
        
        In SQM: outcome contains multiple spaces (e.g., "data_bits bell_bits")
        In SWAP: outcome contains single register with no spaces
        
        Returns: First register (measurement bits) after cleaning
        """
        cleaned = outcome.strip()
        # If multiple registers, extract first one; otherwise return as-is
        registers = cleaned.split()
        return registers[0] if registers else cleaned

    # --------------------------------------------------------------
    # QUBIT ALLOCATION & PHYSICAL MAPPING
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
    # --------------------------------------------------------------

    def compile_workload(self, workload: List[str]) -> QuantumCircuit:

        # ----------------------------------------------------------
        # SEED INITIALIZATION - For reproducibility
        # ----------------------------------------------------------
        random.seed(42)
        np.random.seed(42)

        # ----------------------------------------------------------
        # Phase 0: Build register instances and allocate physical qubits
        # CRITICAL: Use chain topology to ensure fair comparison with SQM
        # ----------------------------------------------------------

        qc = QuantumCircuit()

        # Build all register instances (logical) first
        for i in range(self.R):
            qr_mem = self.memory_registers[i].build()
            qc.add_register(qr_mem)
            self._built_registers[f"mem_{i}"] = qr_mem

        qr_work = self.operation_register.build()
        qc.add_register(qr_work)
        self._built_registers["q_work"] = qr_work

        # ----------------------------------------------------------
        # PER-BIT TOPOLOGY ALLOCATION (QUBIT-CENTRIC)
        # ----------------------------------------------------------
        # Each qubit i of q_work connects to all mem_r_i registers
        # Structure: Each bit i has q_work_i connected to mem_0_i, mem_1_i, ..., mem_(R-1)_i
        # ----------------------------------------------------------
        
        print("\n[Compilation] Allocating per-bit topology (q_work[i] connects to all mem_r[i])...")
        
        # Build chain configuration: q_work first, then all memory registers
        chain_config = [("q_work", self.n)]  # Operation register as first element
        for i in range(self.R):
            chain_config.append((f"mem_{i}", self.n))
        
        total_qubits_needed = sum(size for _, size in chain_config)
        print(f"[Compilation] Total qubits needed: {total_qubits_needed}")
        
        # Allocate using per-bit topology
        allocation_map = self.qubit_mapper.allocate_chain_topology(chain_config)
        
        # Map logical qubits to physical qubits for all registers
        for reg_id, physical_qubits in allocation_map.items():
            if reg_id == "q_work":
                qr = self._built_registers["q_work"]
            else:
                # mem_0, mem_1, ...
                qr = self._built_registers[reg_id]
            
            for local_idx, physical_qubit in enumerate(physical_qubits):
                qubit_obj = qr[local_idx]
                self.logical_to_physical_map[qubit_obj] = physical_qubit
                self.qubit_register_map[physical_qubit] = reg_id

        print(f"[Compilation] Physical qubit mapping:")
        for reg_id, phys_qubits in allocation_map.items():
            print(f"  {reg_id:12s} -> {sorted(phys_qubits)}")

        print(f"\n[Compilation] Total qubits allocated: {len(self.logical_to_physical_map)}")

        # ----------------------------------------------------------
        # Phase 0b: Prepare initial quantum state
        # ----------------------------------------------------------
        
        if self.initial_state == 1:
            print("\n[Compilation] Preparing initial state |1...1> (applying X to all qubits: memory + operation register)...")
            # Apply X gates to ALL qubits (memory and operation register) to prepare |1...1> state
            # This ensures SWAPs don't affect the final state validation
            
            # Apply X to operation register
            for qubit in qr_work:
                qc.x(qubit)
            
            # Apply X to all memory qubits
            for i in range(self.R):
                qr_mem = self._built_registers[f"mem_{i}"]
                # Apply X to memory
                for qubit in qr_mem:
                    qc.x(qubit)
            
            qc.barrier()
            print("[Compilation] Initial state |1...1> prepared for all qubits")
        else:
            print("\n[Compilation] Using default initial state |0...0> (no X gates applied)")

        #print(f"\n[Compilation] Starting workload processing: {len(workload)} instructions")

        # ----------------------------------------------------------
        # Phase 1: Process workload instructions
        # ----------------------------------------------------------

        for instruction in workload:
            #print(f"  [Instruction] {instruction}")

            if instruction.startswith("IDLE_"):
                # IDLE instruction: Apply thermal relaxation via native 'id' gate
                # Each IDLE unit maps to one identity gate with attached thermal noise
                num_units = int(instruction.split("_")[1])
                time_ns = num_units * self.time_idle_ns
                
                print(f"    -> Wear-down sequence: {num_units} idle units ({time_ns:.0f} ns total)")
                
                # Apply native identity gate to operation register
                # The thermal_relaxation_error attached to 'id' will be applied
                for _ in range(num_units):
                    qc.id(qr_work)
                
                # Optional: Also apply to inactive memory registers for complete wear-down modeling
                for i in range(self.R):
                    qr_mem = self._built_registers[f"mem_{i}"]
                    for _ in range(num_units):
                        for qubit in qr_mem:
                            qc.id(qubit)
                
            elif instruction.startswith("READ_"):
                # READ instruction: READ_ij where ij is binary address
                address_binary = instruction.split("_")[1]
                logical_addr = int(address_binary, 2)  # Binary to decimal

                if logical_addr >= self.R:
                    raise ValueError(f"Logical address {logical_addr} out of range [0, {self.R - 1}]")

                print(f"    -> READ from Mem[{logical_addr}]")

                # Get source register
                source_reg = self._built_registers[f"mem_{logical_addr}"]

                # Apply SWAP between source and work register
                qc = self.work_phase.apply_swap(qc, source_reg, qr_work)



            elif instruction.startswith("WRITE_"):
                # WRITE instruction: WRITE_ij
                address_binary = instruction.split("_")[1]
                logical_addr = int(address_binary, 2)

                if logical_addr >= self.R:
                    raise ValueError(f"Logical address {logical_addr} out of range [0, {self.R - 1}]")

                print(f"    -> WRITE to Mem[{logical_addr}]")

                # Get destination register
                dest_reg = self._built_registers[f"mem_{logical_addr}"]

                # Apply SWAP between work register and destination
                qc = self.work_phase.apply_swap(qc, qr_work, dest_reg)



            else:
                print(f"  [WARNING] Unknown instruction: {instruction}")
            qc.barrier()  # Prevent inter-SWAP optimization
            #print(qc.draw(output="text"))   

        print(f"[Compilation] Workload processing complete")
        return qc

    # --------------------------------------------------------------
    # EXECUTION VIA BACKEND MANAGER
    # --------------------------------------------------------------

    def execute(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, Any]:
        """
        Execute the quantum circuit via the injected backend manager.
        
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

        try:
            qc_measured = circuit.copy()
            
            # ALWAYS add a specific register for final fidelity
            # CRITICAL: Measure ALL n*R qubits (all memory registers) for correct fidelity
            cr_final = ClassicalRegister(self.n * self.R, name="final_meas")
            qc_measured.add_register(cr_final)

            # Measure ALL memory registers into cr_final
            classical_bit_index = 0
            for logical_addr in range(self.R):
                # Each memory register stores n qubits
                memory_reg = self._built_registers[f"mem_{logical_addr}"]
                for i in range(self.n):
                    qc_measured.measure(memory_reg[i], cr_final[classical_bit_index])
                    classical_bit_index += 1

            print("[Transpile] Translating to hardware topology with seed=42...")
            
            # Extract initial layout before transpilation
            initial_layout = self._get_initial_layout(qc_measured)

            # DEBUG: Uncomment below to visualize circuit before transpilation
            print(qc_measured.draw(output="text"))
            print("[Noise Model] Extracting noise characteristics...")
            
            # Transpile with seed for reproducibility
            qc_transpiled = transpile(
                qc_measured,
                backend=self.backend,
                optimization_level=0,
                initial_layout=initial_layout,
                seed_transpiler=42
            )
            #print(qc_transpiled.draw(output="text"))
            
            print(f"[Execution] Sending circuit to Backend Manager...")
            # Execute via backend manager (no more manual simulator instantiation)
            result = self.backend_manager.run(qc_transpiled, shots=shots, seed=42)
            
            counts = result.get_counts()
            total_counts = sum(counts.values())
            
            # Extract results: Compare ALL n*R measurement bits against target state
            fidelity_count = 0
            target_state_bits = self.n * self.R
            target_state = ('1' * target_state_bits) if self.initial_state == 1 else ('0' * target_state_bits)
            
            for outcome, count in counts.items():
                # Parse measurement outcome (unified parser for multiple registers)
                measured_bits = self._parse_measurement_outcome(outcome)
                if measured_bits == target_state:
                    fidelity_count += count
            
            fidelity = fidelity_count / total_counts if total_counts > 0 else 0.0

            print(f"\n[Circuit] Generated transpiled circuit:")
            print(f"  Qubits: {qc_transpiled.num_qubits}")
            print(f"  Clbits: {qc_transpiled.num_clbits}")
            print(f"  Depth: {qc_transpiled.depth()}")
            print(f"  Size: {qc_transpiled.size()}")
            
            state_label = "|1...1>" if self.initial_state == 1 else "|0...0>"
            print(f"  Fidelity ({state_label} success for all {self.n * self.R} memory qubits): {fidelity:.4f}")
            
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
            "logical_to_physical_map": self.logical_to_physical_map,
            "available_qubits": len(self.qubit_mapper.available_qubits),
        }


