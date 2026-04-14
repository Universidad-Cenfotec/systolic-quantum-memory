# ============================================================
# SQM Research Project — Systolic Teleportation Module
# Systolic Quantum Teleportation Memory
# Authors: Danny Valerio-Ramírez & Santiago Núñez-Corrales
# ============================================================

from typing import cast, Dict, Optional, Tuple
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister, Clbit


class SystolicTeleportation:

    def __init__(self, name: str = "systolic_teleportation") -> None:

        self.name = name
        # Cache de registros ancilla y clásicos por par lógico (source_name, dest_name)
        self._ancilla_cache: Dict[Tuple[str, str], QuantumRegister] = {}
        self._crbell_cache: Dict[Tuple[str, str], ClassicalRegister] = {}

    def build_circuit(
        self,
        qc: QuantumCircuit,
        source_reg: QuantumRegister,
        dest_reg: QuantumRegister,
        ancilla_reg: Optional[QuantumRegister] = None,
        cr_bell: Optional[ClassicalRegister] = None,
    ) -> QuantumCircuit:
        """
        Aplica el protocolo de tele-refresco (teleportación cuántica) en el circuito.

        Args:
            qc:          Circuito sobre el que se construye el protocolo.
            source_reg:  Registro fuente (qubit a teleportar).
            dest_reg:    Registro destino.
            ancilla_reg: Registro ancilla pre-asignado con hardware físico.
                         Si se provee, se usa directamente (no se crea uno nuevo).
                         Si es None, se crea dinámicamente (comportamiento original).
            cr_bell:     Registro clásico pre-creado para medición Bell.
                         Si es None, se crea dinámicamente.
        """
        
        # ──────────────────────────────────────────────────────────────
        # 0. VALIDATION: Ensure registers have compatible sizes
        # ──────────────────────────────────────────────────────────────
        
        N = source_reg.size
        if dest_reg.size != N:
            raise ValueError(
                f"Register size mismatch: source_reg has {N} qubits, "
                f"but dest_reg has {dest_reg.size} qubits. Must be equal."
            )
        
        # ──────────────────────────────────────────────────────────────
        # 1. GET OR CREATE BUS INTERNAL RESOURCES (Reuse on multiple calls)
        # ──────────────────────────────────────────────────────────────
        
        cache_key = (source_reg.name, dest_reg.name)
        
        if ancilla_reg is not None:
            # MODO EXTERNO: usar ancilla pre-asignada (con hardware físico)
            # Registrar en caché para reutilización futura en el mismo par
            if cache_key not in self._ancilla_cache:
                if ancilla_reg not in qc.qregs:
                    qc.add_register(ancilla_reg)
                if cr_bell is None:
                    cr_bell_name = f"cr_bell_{source_reg.name}_to_{dest_reg.name}"
                    cr_bell = ClassicalRegister(2 * source_reg.size, name=cr_bell_name)
                    qc.add_register(cr_bell)
                elif cr_bell not in qc.cregs:
                    qc.add_register(cr_bell)
                self._ancilla_cache[cache_key] = ancilla_reg
                self._crbell_cache[cache_key] = cr_bell
                print(f"[Teleportation] Using pre-assigned ancilla {ancilla_reg.name} for {source_reg.name} -> {dest_reg.name}")
            else:
                ancilla_reg = self._ancilla_cache[cache_key]
                cr_bell = self._crbell_cache[cache_key]
                for q in ancilla_reg:
                    qc.reset(q)
                print(f"[Teleportation] Reusing pre-assigned ancilla for {source_reg.name} -> {dest_reg.name}")
        elif cache_key not in self._ancilla_cache:
            # MODO DINÁMICO: crear ancilla internamente (comportamiento original)
            ancilla_name = f"ancilla_{source_reg.name}_to_{dest_reg.name}"
            cr_bell_name = f"cr_bell_{source_reg.name}_to_{dest_reg.name}"
            
            ancilla_reg = QuantumRegister(source_reg.size, name=ancilla_name)
            qc.add_register(ancilla_reg)
            self._ancilla_cache[cache_key] = ancilla_reg
            
            cr_bell = ClassicalRegister(2 * source_reg.size, name=cr_bell_name)
            qc.add_register(cr_bell)
            self._crbell_cache[cache_key] = cr_bell
            
            print(f"[Teleportation] Created ancilla pair for {source_reg.name} -> {dest_reg.name}")
        else:
            ancilla_reg = self._ancilla_cache[cache_key]
            cr_bell = self._crbell_cache[cache_key]
            for q in ancilla_reg:
                qc.reset(q)
            print(f"[Teleportation] Reusing ancilla pair for {source_reg.name} -> {dest_reg.name}")
        
        # ──────────────────────────────────────────────────────────────
        # 2. PARALLEL TELEPORTATION LOOP (over all N qubits)
        # ──────────────────────────────────────────────────────────────
        
        for i in range(N):
            source_q = source_reg[i]
            dest_q = dest_reg[i]
            link_q = ancilla_reg[i]
            
            # ────────────────────────────────────────────────────────
            # 2a. BELL CHANNEL GENERATION (between link_q and dest_q)
            # ────────────────────────────────────────────────────────
            qc.h(link_q)
            qc.cx(link_q, dest_q)
            
            # ────────────────────────────────────────────────────────
            # 2b. SOURCE INTERACTION (BSM preparation)
            # ────────────────────────────────────────────────────────
            
            qc.cx(source_q, link_q)
            qc.h(source_q)
            
            # ────────────────────────────────────────────────────────
            # 2c. BELL STATE MEASUREMENT (BSM)
            # Classical bits: cr_bell[2*i] = source_q, cr_bell[2*i+1] = link_q
            # ────────────────────────────────────────────────────────
            
            qc.measure(source_q, cr_bell[2 * i])
            qc.measure(link_q, cr_bell[2 * i + 1])
            
            # ────────────────────────────────────────────────────────
            # 2d. FEED-FORWARD (Data Movement Correction)
            # ────────────────────────────────────────────────────────
            
            # If cr_bell[2*i+1] == 1 (bit position 2*i+1), apply X
            with qc.if_test((cast(Clbit, cr_bell[2 * i + 1]), 1)):
                qc.x(dest_q)
            
            # If cr_bell[2*i] == 1 (bit position 2*i), apply Z
            with qc.if_test((cast(Clbit, cr_bell[2 * i]), 1)):
                qc.z(dest_q)
            
            # ────────────────────────────────────────────────────────
            # 2e. ACTIVE RESET (Clean source and ancilla, NOT destination)
            # ────────────────────────────────────────────────────────
            
            qc.reset(link_q)
            qc.reset(source_q)
        
        return qc

    def __repr__(self) -> str:
        """String representation of the Systolic Teleportation module."""
        return f"{self.__class__.__name__}(name='{self.name}')"
