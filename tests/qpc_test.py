
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from src.modular_circuits.sqc import SQC, CacheLocation


class TestSQC(unittest.TestCase):
    """Test del SQC Odómetro Híbrido con desgaste activo (gates) y pasivo (tiempo)."""

    def test_initial_location_is_original(self):
        """La ubicación inicial de todos los qubits debe ser ORIGINAL."""
        sqc = SQC(logical_size=4, c_max=10, t_max=5.0)
        for addr in range(4):
            self.assertEqual(sqc.get_cache_location(addr), CacheLocation.ORIGINAL)

    def test_complete_bipartite_sqc_scenario(self):
        """Test del ciclo corto: desgaste activo + pasivo con evaluación unificada."""
        
        # 1. Crear SQC: 4 direcciones lógicas, C_MAX=10, T_MAX=5.0
        sqc = SQC(logical_size=4, c_max=10, t_max=5.0)
        logical_addr = 0
        
        # 2. Obtener ubicación inicial (debe ser ORIGINAL)
        location_initial = sqc.get_cache_location(logical_addr)
        print(f"✓ Ubicación inicial de dirección lógica {logical_addr}: {location_initial}")
        self.assertEqual(location_initial, CacheLocation.ORIGINAL)
        
        # 3. Registrar operación con costo 6 y tiempo 2.0 (no alcanza C_MAX ni T_MAX)
        needs_tick_1 = sqc.update_odometer(logical_addr, gate_cost=6, time_dt=2.0)
        print(f"✓ Operación 1 (costo=6, tiempo=2.0):")
        print(f"    needs_tick={needs_tick_1}")
        print(f"    costo_acumulado={sqc.get_cost(logical_addr)}")
        print(f"    tiempo_acumulado={sqc.get_idle_time(logical_addr)}")
        self.assertFalse(needs_tick_1)  # No dispara (6 < 10 y 2.0 < 5.0)
        self.assertEqual(sqc.get_cost(logical_addr), 6)
        self.assertEqual(sqc.get_idle_time(logical_addr), 2.0)
        
        # 4. Registrar operación con costo 4 y tiempo 3.5 más (total gate=10 >= C_MAX)
        needs_tick_2 = sqc.update_odometer(logical_addr, gate_cost=4, time_dt=3.5)
        print(f"\n✓ Operación 2 (costo=4, tiempo=3.5):")
        print(f"    needs_tick={needs_tick_2}")
        print(f"    costo_acumulado={sqc.get_cost(logical_addr)}")
        print(f"    tiempo_acumulado={sqc.get_idle_time(logical_addr)}")
        self.assertTrue(needs_tick_2)  # DISPARA (6 + 4 = 10 >= C_MAX)
        self.assertEqual(sqc.get_cost(logical_addr), 10)
        self.assertEqual(sqc.get_idle_time(logical_addr), 5.5)
        
        # 5. Llamar tick() para alternar ubicación y resetear AMBOS contadores
        new_location = sqc.tick(logical_addr)
        print(f"\n✓ Tick ejecutado:")
        print(f"    nueva ubicación={new_location}")
        print(f"    costo reseteado a {sqc.get_cost(logical_addr)}")
        print(f"    tiempo reseteado a {sqc.get_idle_time(logical_addr)}")
        self.assertEqual(new_location, CacheLocation.BACKUP)  # Debe estar en BACKUP
        self.assertEqual(sqc.get_cost(logical_addr), 0)     # Costo reseteado
        self.assertEqual(sqc.get_idle_time(logical_addr), 0.0)  # Tiempo reseteado
        
        # 6. Verificar nueva ubicación y presupuesto restante (ambos)
        location_after_tick = sqc.get_cache_location(logical_addr)
        budget_remaining = sqc.get_coherence_budget_remaining(logical_addr)
        time_remaining = sqc.get_time_coherence_remaining(logical_addr)
        print(f"\n✓ Nueva ubicación y presupuestos:")
        print(f"    ubicación={location_after_tick}")
        print(f"    presupuesto gates={budget_remaining}/{sqc.c_max}")
        print(f"    presupuesto tiempo={time_remaining:.1f}/{sqc.t_max}")
        self.assertEqual(location_after_tick, CacheLocation.BACKUP)
        self.assertEqual(budget_remaining, 10)   # Presupuesto completo disponible
        self.assertEqual(time_remaining, 5.0)    # Presupuesto de tiempo completo disponible

        # 7. Segundo tick: debe volver a ORIGINAL
        sqc.tick(logical_addr)
        self.assertEqual(sqc.get_cache_location(logical_addr), CacheLocation.ORIGINAL)
        
        print("\n✅ Test completado exitosamente")

    def test_invalid_address_raises(self):
        """Acceder a una dirección fuera de rango debe lanzar ValueError."""
        sqc = SQC(logical_size=2, c_max=10, t_max=5.0)
        with self.assertRaises(ValueError):
            sqc.get_cache_location(99)
        with self.assertRaises(ValueError):
            sqc.update_odometer(-1, gate_cost=1)
        with self.assertRaises(ValueError):
            sqc.tick(5)


if __name__ == "__main__":
    unittest.main()
