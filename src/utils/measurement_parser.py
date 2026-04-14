# ============================================================
# Measurement Parser Utility - Unified Outcome Handling
# Systolic Quantum Memory
# Authors: Danny Valerio-Ramírez & Santiago Núñez-Corrales
# ============================================================
"""
Unified measurement outcome parser for validadores and simuladores.

CRITICAL: Qiskit uses LITTLE-ENDIAN register ordering.
If you add creg_A then creg_B to a circuit, the bitstring output is:
    "valor_creg_B valor_creg_A"

This means the LAST registered classical register appears FIRST (leftmost) in the bitstring.

Example:
    circuit.add_register(creg_A)  # 2 bits
    circuit.add_register(creg_B)  # 2 bits
    
    Bitstring: "11 10"  (creg_B=11, creg_A=10)
    
Don't assume blind indexing! Always use metadata layout.
"""

from typing import Dict, Tuple, List, Optional


class MeasurementParser:
    """
    Unified measurement outcome parser with explicit endianness awareness.
    """

    @staticmethod
    def split_registers(outcome: str) -> List[str]:
        """
        Split outcome string by spaces to extract individual registers.
        
        Handles cases where measurement outcome may have multiple classical registers
        separated by spaces (e.g., "1010 0011" for two 4-bit registers).
        
        Args:
            outcome: Raw measurement outcome string from Qiskit
        
        Returns:
            List of register values (strings), one per classical register
        
        Example:
            >>> MeasurementParser.split_registers("1010 0011")
            ['1010', '0011']
            >>> MeasurementParser.split_registers("1111")
            ['1111']
        """
        cleaned = outcome.strip()
        registers = cleaned.split()
        return registers if registers else [cleaned]

    @staticmethod
    def extract_register_bits(
        bitstring: str,
        register_name: str,
        register_layout: Dict[str, Tuple[int, int]],
    ) -> str:
        """
        Extract bits for a specific register using explicit metadata layout.
        
        This method respects Qiskit's Little-Endian ordering by using a pre-computed
        layout dictionary that specifies the exact bit positions for each register.
        
        Args:
            bitstring: Full measurement outcome string (possibly with spaces)
            register_name: Name of the register to extract (e.g., "cr_bell", "cr_lb")
            register_layout: Dict mapping {register_name -> (start_idx, end_idx)}
                           The indices refer to positions in the cleaned bitstring
                           (after removing all spaces).
        
        Returns:
            Bit string for the requested register
        
        Raises:
            ValueError: If register_name not in layout or invalid indices
            KeyError: If register_name not found in layout
        
        Example:
            >>> layout = {
            ...     "cr_bell": (0, 4),
            ...     "cr_lb": (4, 6)
            ... }
            >>> bitstring = "00110011 11"  # After cleaning: "0011001111"
            >>> MeasurementParser.extract_register_bits(
            ...     bitstring, "cr_bell", layout
            ... )
            '0011'
            >>> MeasurementParser.extract_register_bits(
            ...     bitstring, "cr_lb", layout
            ... )
            '0011'
        """
        if register_name not in register_layout:
            raise KeyError(
                f"Register '{register_name}' not in layout. Available: "
                f"{list(register_layout.keys())}"
            )

        # Clean bitstring (remove spaces)
        clean = bitstring.replace(" ", "")

        # Validate layout indices
        start, end = register_layout[register_name]
        if not (0 <= start <= end <= len(clean)):
            raise ValueError(
                f"Invalid layout indices ({start}, {end}) for bitstring of length {len(clean)}\n"
                f"Bitstring: {clean}"
            )

        return clean[start:end]

    @staticmethod
    def extract_first_n_bits(bitstring: str, n: int) -> str:
        """
        Extract the first N bits from a measurement outcome (simple case).
        
        Useful for single-register scenarios or when the register of interest
        appears first in the measurement outcome.
        
        Args:
            bitstring: Measurement outcome (may contain spaces)
            n: Number of bits to extract from the beginning
        
        Returns:
            First N bits after cleaning
        
        Example:
            >>> MeasurementParser.extract_first_n_bits("0011 1010", 4)
            '0011'
        """
        clean = bitstring.replace(" ", "")
        if n > len(clean):
            raise ValueError(
                f"Requested {n} bits but bitstring only has {len(clean)} bits"
            )
        return clean[:n]

    @staticmethod
    def extract_last_n_bits(bitstring: str, n: int) -> str:
        """
        Extract the last N bits from a measurement outcome.
        
        Useful when the register of interest was added last to the circuit
        (and thus appears first in the Little-Endian bitstring).
        
        Args:
            bitstring: Measurement outcome (may contain spaces)
            n: Number of bits to extract from the end
        
        Returns:
            Last N bits after cleaning
        
        Example:
            >>> MeasurementParser.extract_last_n_bits("1010 0011", 4)
            '0011'
        """
        clean = bitstring.replace(" ", "")
        if n > len(clean):
            raise ValueError(
                f"Requested {n} bits but bitstring only has {len(clean)} bits"
            )
        return clean[-n:] if n > 0 else ""

    @staticmethod
    def build_register_layout_from_order(
        register_names: List[str],
        register_sizes: List[int],
        reverse_for_endianness: bool = True
    ) -> Dict[str, Tuple[int, int]]:
        """
        Build a register layout dictionary from registration order.
        
        This helper accounts for Qiskit's Little-Endian ordering:
        if registers are added in order [creg_A, creg_B], they appear
        in the bitstring as "creg_B creg_A" (reversed).
        
        Args:
            register_names: List of register names in order of circuit.add_register()
            register_sizes: Corresponding sizes for each register
            reverse_for_endianness: If True, reverse for Little-Endian (default: True)
        
        Returns:
            Dict mapping {reg_name -> (start_bit, end_bit)} for bitstring indexing
        
        Example:
            >>> layout = MeasurementParser.build_register_layout_from_order(
            ...     ["cr_bell", "cr_lb"],
            ...     [4, 2],
            ...     reverse_for_endianness=True
            ... )
            >>> layout
            {'cr_lb': (0, 2), 'cr_bell': (2, 6)}
        """
        if len(register_names) != len(register_sizes):
            raise ValueError(
                f"Mismatch: {len(register_names)} names vs {len(register_sizes)} sizes"
            )

        # If reverse_for_endianness, flip the order (Qiskit Little-Endian)
        if reverse_for_endianness:
            register_names = list(reversed(register_names))
            register_sizes = list(reversed(register_sizes))

        layout = {}
        bit_offset = 0
        for name, size in zip(register_names, register_sizes):
            layout[name] = (bit_offset, bit_offset + size)
            bit_offset += size

        return layout

    @staticmethod
    def validate_layout(
        bitstring: str,
        layout: Dict[str, Tuple[int, int]]
    ) -> bool:
        """
        Validate that a layout is consistent with a given bitstring.
        
        Checks that:
        1. All layout indices are within bitstring bounds
        2. Layout regions don't overlap
        3. No gaps in coverage (optional check)
        
        Args:
            bitstring: Measurement outcome string (will remove spaces)
            layout: Register layout dictionary
        
        Returns:
            True if layout is valid, raises ValueError otherwise
        """
        clean = bitstring.replace(" ", "")
        
        # Check bounds
        max_end = 0
        for reg_name, (start, end) in layout.items():
            if not (0 <= start <= end <= len(clean)):
                raise ValueError(
                    f"Register '{reg_name}': indices ({start}, {end}) out of bounds "
                    f"for bitstring length {len(clean)}"
                )
            max_end = max(max_end, end)
        
        # Check for overlaps
        regions = sorted(layout.values())
        for i in range(len(regions) - 1):
            if regions[i][1] > regions[i + 1][0]:
                raise ValueError(
                    f"Overlapping register regions: {regions[i]} and {regions[i+1]}"
                )
        
        return True
