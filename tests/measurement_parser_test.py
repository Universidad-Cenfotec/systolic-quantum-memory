# ============================================================
# Test Suite: Measurement Parser with Qiskit Endianness
# ============================================================
"""
Comprehensive test suite for MeasurementParser.

CRITICAL: These tests validate correct handling of Qiskit's Little-Endian ordering.
Tests include tricky cases like "1010 0011 11" to ensure layout extraction is precise.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.measurement_parser import MeasurementParser


class TestMeasurementParserBasic(unittest.TestCase):
    """Test basic parsing utilities."""

    def test_split_registers_single(self):
        """Extract single register (no spaces)."""
        result = MeasurementParser.split_registers("11101010")
        self.assertEqual(result, ["11101010"])

    def test_split_registers_multiple(self):
        """Extract multiple registers separated by spaces."""
        result = MeasurementParser.split_registers("1010 0011")
        self.assertEqual(result, ["1010", "0011"])

    def test_split_registers_tricky_multiple(self):
        """Extract tricky case with 3 registers."""
        result = MeasurementParser.split_registers("1010 0011 11")
        self.assertEqual(result, ["1010", "0011", "11"])

    def test_split_registers_with_whitespace(self):
        """Handle extra whitespace."""
        result = MeasurementParser.split_registers("  1010   0011  ")
        self.assertEqual(result, ["1010", "0011"])


class TestExtractFirstNBits(unittest.TestCase):
    """Test extraction of first N bits."""

    def test_extract_first_n_bits_single_register(self):
        """Extract first N bits from single register."""
        result = MeasurementParser.extract_first_n_bits("11101010", 4)
        self.assertEqual(result, "1110")

    def test_extract_first_n_bits_multi_register(self):
        """Extract first N bits, ignoring spaces."""
        result = MeasurementParser.extract_first_n_bits("1010 0011", 4)
        self.assertEqual(result, "1010")

    def test_extract_first_n_bits_tricky(self):
        """Extract from complex bitstring."""
        # "1010 0011 11" -> "101000111" (after removing spaces)
        # First 6 bits: "101000"
        result = MeasurementParser.extract_first_n_bits("1010 0011 11", 6)
        self.assertEqual(result, "101000")

    def test_extract_first_n_bits_entire_string(self):
        """Extract all bits."""
        result = MeasurementParser.extract_first_n_bits("1111 0000", 8)
        self.assertEqual(result, "11110000")

    def test_extract_first_n_bits_overflow(self):
        """Requesting more bits than available raises error."""
        with self.assertRaises(ValueError):
            MeasurementParser.extract_first_n_bits("1010", 5)


class TestExtractLastNBits(unittest.TestCase):
    """Test extraction of last N bits."""

    def test_extract_last_n_bits_single_register(self):
        """Extract last N bits from single register."""
        result = MeasurementParser.extract_last_n_bits("11101010", 4)
        self.assertEqual(result, "1010")

    def test_extract_last_n_bits_multi_register(self):
        """Extract last N bits, ignoring spaces."""
        result = MeasurementParser.extract_last_n_bits("1010 0011", 4)
        self.assertEqual(result, "0011")

    def test_extract_last_n_bits_tricky(self):
        """Extract from complex bitstring."""
        result = MeasurementParser.extract_last_n_bits("1010 0011 11", 4)
        self.assertEqual(result, "1111")

    def test_extract_last_n_bits_overflow(self):
        """Requesting more bits than available raises error."""
        with self.assertRaises(ValueError):
            MeasurementParser.extract_last_n_bits("1010", 5)


class TestRegisterLayoutBuilding(unittest.TestCase):
    """Test building register layouts from order."""

    def test_build_layout_two_registers(self):
        """Build layout for 2 registers with Little-Endian reversal."""
        layout = MeasurementParser.build_register_layout_from_order(
            ["creg_A", "creg_B"],
            [4, 2],
            reverse_for_endianness=True
        )
        # After reversal: creg_B should appear first (leftmost)
        self.assertEqual(layout["creg_B"], (0, 2))
        self.assertEqual(layout["creg_A"], (2, 6))

    def test_build_layout_no_endianness_reversal(self):
        """Build layout without endianness reversal."""
        layout = MeasurementParser.build_register_layout_from_order(
            ["creg_A", "creg_B"],
            [4, 2],
            reverse_for_endianness=False
        )
        # No reversal: keep original order
        self.assertEqual(layout["creg_A"], (0, 4))
        self.assertEqual(layout["creg_B"], (4, 6))

    def test_build_layout_three_registers(self):
        """Build layout for 3 registers."""
        layout = MeasurementParser.build_register_layout_from_order(
            ["cr_bell", "cr_lb", "cr_data"],
            [4, 2, 3],
            reverse_for_endianness=True
        )
        # After reversal: cr_data, cr_lb, cr_bell
        self.assertEqual(layout["cr_data"], (0, 3))
        self.assertEqual(layout["cr_lb"], (3, 5))
        self.assertEqual(layout["cr_bell"], (5, 9))

    def test_build_layout_size_mismatch(self):
        """Mismatch between names and sizes raises error."""
        with self.assertRaises(ValueError):
            MeasurementParser.build_register_layout_from_order(
                ["creg_A", "creg_B"],
                [4, 2, 3]  # Extra size
            )


class TestExtractRegisterBits(unittest.TestCase):
    """Test extraction of specific registers using layout."""

    def test_extract_register_bits_simple(self):
        """Extract bits for a single register."""
        layout = {
            "cr_bell": (0, 4),
            "cr_lb": (4, 6)
        }
        bitstring = "00110011"
        
        result = MeasurementParser.extract_register_bits(
            bitstring, "cr_bell", layout
        )
        self.assertEqual(result, "0011")

        result = MeasurementParser.extract_register_bits(
            bitstring, "cr_lb", layout
        )
        self.assertEqual(result, "00")

    def test_extract_register_bits_with_spaces(self):
        """Extract bits, handling spaces in bitstring."""
        layout = {
            "cr_bell": (0, 4),
            "cr_lb": (4, 6)
        }
        bitstring = "0011 0011"  # Spaces should be ignored
        
        result = MeasurementParser.extract_register_bits(
            bitstring, "cr_bell", layout
        )
        self.assertEqual(result, "0011")

    def test_extract_register_bits_tricky_bitstring(self):
        """Extract from a tricky bitstring with pattern '1010 0011 11'."""
        # Bitstring: "1010 0011 11" -> cleaned: "101000111"
        # Layout simulates: cr_A=4bits, cr_B=3bits, cr_C=2bits
        layout = {
            "cr_A": (0, 4),
            "cr_B": (4, 7),
            "cr_C": (7, 9)
        }
        bitstring = "1010 0011 11"
        
        self.assertEqual(
            MeasurementParser.extract_register_bits(bitstring, "cr_A", layout),
            "1010"
        )
        self.assertEqual(
            MeasurementParser.extract_register_bits(bitstring, "cr_B", layout),
            "001"
        )
        self.assertEqual(
            MeasurementParser.extract_register_bits(bitstring, "cr_C", layout),
            "11"
        )

    def test_extract_register_bits_unknown_register(self):
        """Requesting unknown register raises KeyError."""
        layout = {"cr_bell": (0, 4)}
        bitstring = "11110000"
        
        with self.assertRaises(KeyError):
            MeasurementParser.extract_register_bits(
                bitstring, "unknown_reg", layout
            )

    def test_extract_register_bits_invalid_indices(self):
        """Invalid layout indices raise ValueError."""
        layout = {"cr_bell": (0, 10)}  # Exceeds bitstring length
        bitstring = "11110000"
        
        with self.assertRaises(ValueError):
            MeasurementParser.extract_register_bits(
                bitstring, "cr_bell", layout
            )


class TestValidateLayout(unittest.TestCase):
    """Test layout validation."""

    def test_validate_layout_valid(self):
        """Valid layout passes validation."""
        layout = {
            "cr_bell": (0, 4),
            "cr_lb": (4, 6)
        }
        bitstring = "11110000"
        
        result = MeasurementParser.validate_layout(bitstring, layout)
        self.assertTrue(result)

    def test_validate_layout_out_of_bounds(self):
        """Out of bounds indices raise ValueError."""
        layout = {"cr_bell": (0, 10)}  # Exceeds length
        bitstring = "11110000"
        
        with self.assertRaises(ValueError) as context:
            MeasurementParser.validate_layout(bitstring, layout)
        self.assertIn("out of bounds", str(context.exception))

    def test_validate_layout_overlapping(self):
        """Overlapping regions raise ValueError."""
        layout = {
            "cr_A": (0, 5),
            "cr_B": (3, 8)  # Overlaps with cr_A
        }
        bitstring = "1111111111"
        
        with self.assertRaises(ValueError) as context:
            MeasurementParser.validate_layout(bitstring, layout)
        self.assertIn("Overlapping", str(context.exception))

    def test_validate_layout_with_spaces(self):
        """Layout validation works with spaces in bitstring."""
        layout = {
            "cr_bell": (0, 4),
            "cr_lb": (4, 6)
        }
        bitstring = "1111 0000"  # Spaces are handled
        
        result = MeasurementParser.validate_layout(bitstring, layout)
        self.assertTrue(result)


class TestEndiannessBehavior(unittest.TestCase):
    """
    Test Qiskit Little-Endian behavior explicitly.
    
    Real scenario from SQM Teleportation:
    - creg_bell: 2*N bits (added first)
    - creg_lb: N bits (added second)
    
    In Qiskit Little-Endian: creg_lb appears first (leftmost) in bitstring.
    """

    def test_qiskit_endianness_scenario_sqm(self):
        """Simulate real SQM measurement scenario."""
        # Simulate: creg_bell (4 bits), then creg_lb (2 bits)
        # After Little-Endian, bitstring shows: "cr_lb cr_bell"
        
        register_names = ["cr_bell", "cr_lb"]
        register_sizes = [4, 2]
        
        layout = MeasurementParser.build_register_layout_from_order(
            register_names, register_sizes, reverse_for_endianness=True
        )
        
        # Layout should reflect: cr_lb first, then cr_bell
        self.assertEqual(layout["cr_lb"], (0, 2))
        self.assertEqual(layout["cr_bell"], (2, 6))
        
        # Simulate measurement: "10" (cr_lb) + "0011" (cr_bell) = "100011"
        bitstring = "10 0011"
        
        lb_bits = MeasurementParser.extract_register_bits(
            bitstring, "cr_lb", layout
        )
        bell_bits = MeasurementParser.extract_register_bits(
            bitstring, "cr_bell", layout
        )
        
        self.assertEqual(lb_bits, "10")
        self.assertEqual(bell_bits, "0011")

    def test_qiskit_endianness_scenario_state_patterns(self):
        """Test with different state patterns (not just 0's)."""
        # Verify that extraction works for arbitrary patterns
        layout = {
            "cr_dest": (0, 4),
            "cr_src": (4, 8)
        }
        
        # Bitstring with pattern "1010" for each
        bitstring = "1010 1010"
        
        dest = MeasurementParser.extract_register_bits(
            bitstring, "cr_dest", layout
        )
        src = MeasurementParser.extract_register_bits(
            bitstring, "cr_src", layout
        )
        
        self.assertEqual(dest, "1010")
        self.assertEqual(src, "1010")
        
        # Now mixed pattern
        bitstring = "0011 1100"
        
        dest = MeasurementParser.extract_register_bits(
            bitstring, "cr_dest", layout
        )
        src = MeasurementParser.extract_register_bits(
            bitstring, "cr_src", layout
        )
        
        self.assertEqual(dest, "0011")
        self.assertEqual(src, "1100")


if __name__ == "__main__":
    unittest.main()
