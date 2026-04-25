
from enum import Enum
from typing import Dict


class CacheLocation(Enum):
    """
    Representa el registro físico donde se almacena actualmente un qubit lógico.
    
    ORIGINAL → mem_orig_i  (registro de memoria principal)
    BACKUP   → mem_backup_i (registro de memoria de respaldo)
    """
    ORIGINAL = "O"
    BACKUP = "B"


class SQC:
    """
    Odómetro Híbrido para Quantum Sequential Cache (SQC).

    Rastrea tanto desgaste activo (compuertas) como pasivo (tiempo) y
    gestiona la ubicación física de los qubits lógicos en cache (ORIGINAL vs BACKUP).
    """

    def __init__(self, logical_size: int, c_max: int, t_max: float) -> None:
        """
        Inicializa el controlador SQC.
        
        Args:
            logical_size: Número de direcciones lógicas
            c_max: Umbral máximo de costo térmico (desgaste activo - gates)
            t_max: Umbral máximo de tiempo inactivo (desgaste pasivo - tiempo)
        """
        self.logical_size = logical_size
        self.c_max = c_max
        self.t_max = t_max
        
        # Ubicación en cache por dirección lógica: ORIGINAL o BACKUP
        self._cache_location: Dict[int, CacheLocation] = {
            i: CacheLocation.ORIGINAL for i in range(logical_size)
        }
        
        # Costo térmico acumulado por dirección lógica (desgaste activo)
        self._costs: Dict[int, int] = {i: 0 for i in range(logical_size)}
        
        # Tiempo inactivo acumulado por dirección lógica (desgaste pasivo)
        self._idle_times: Dict[int, float] = {i: 0.0 for i in range(logical_size)}

    def _validate_address(self, logical_address: int) -> None:
        """Valida que la dirección lógica esté dentro del rango permitido."""
        if logical_address < 0 or logical_address >= self.logical_size:
            raise ValueError(
                f"Dirección lógica {logical_address} fuera de rango "
                f"[0, {self.logical_size - 1}]"
            )

    # ==================== Gestión de Ubicación en Cache ====================

    def get_cache_location(self, logical_address: int) -> CacheLocation:
        """
        Retorna el registro físico activo donde se encuentra el qubit lógico en cache.

        Returns:
            CacheLocation.ORIGINAL si el qubit está en mem_orig_i,
            CacheLocation.BACKUP   si el qubit está en mem_backup_i.
        """
        self._validate_address(logical_address)
        return self._cache_location[logical_address]

    # ==================== Odómetro Híbrido ====================

    def update_odometer(self, logical_address: int, gate_cost: int = 0, time_dt: float = 0.0) -> bool:
        """
        Actualiza el odómetro híbrido (desgaste activo + pasivo).
        
        Args:
            logical_address: Dirección lógica de la memoria
            gate_cost: Costo térmico de las compuertas (desgaste activo)
            time_dt: Tiempo transcurrido en inactividad (desgaste pasivo)
        
        Returns:
            True si CUALQUIERA de los umbrales se alcanzó o superó, False en caso contrario.
        """
        self._validate_address(logical_address)
        
        # Acumular desgaste activo (gates)
        self._costs[logical_address] += gate_cost
        
        # Acumular desgaste pasivo (tiempo inactivo)
        self._idle_times[logical_address] += time_dt
        
        # Evaluar ambos umbrales: retorna True si CUALQUIERA se alcanza o supera
        return (
            self._costs[logical_address] >= self.c_max
            or self._idle_times[logical_address] >= self.t_max
        )

    def tick(self, logical_address: int) -> CacheLocation:
        """
        Ejecuta un tele-refresco: alterna la ubicación activa (ORIGINAL ↔ BACKUP)
        y resetea el odómetro híbrido.
        
        Args:
            logical_address: Dirección lógica de la memoria
        
        Returns:
            La nueva ubicación activa (CacheLocation.ORIGINAL o CacheLocation.BACKUP).
        """
        self._validate_address(logical_address)
        
        # Alternar ubicación en cache: ORIGINAL → BACKUP → ORIGINAL
        if self._cache_location[logical_address] == CacheLocation.ORIGINAL:
            self._cache_location[logical_address] = CacheLocation.BACKUP
        else:
            self._cache_location[logical_address] = CacheLocation.ORIGINAL
        
        # Resetear AMBOS contadores del odómetro
        self._costs[logical_address] = 0
        self._idle_times[logical_address] = 0.0
        
        return self._cache_location[logical_address]

    # ==================== Métodos de Inspección ====================

    def get_cost(self, logical_address: int) -> int:
        """Retorna el costo térmico acumulado (desgaste activo) de una dirección lógica."""
        self._validate_address(logical_address)
        return self._costs[logical_address]

    def get_idle_time(self, logical_address: int) -> float:
        """Retorna el tiempo inactivo acumulado (desgaste pasivo) de una dirección lógica."""
        self._validate_address(logical_address)
        return self._idle_times[logical_address]

    def get_coherence_budget_remaining(self, logical_address: int) -> int:
        """Retorna el presupuesto restante antes de decoherencia (desgaste activo)."""
        self._validate_address(logical_address)
        return max(0, self.c_max - self._costs[logical_address])

    def get_time_coherence_remaining(self, logical_address: int) -> float:
        """Retorna el presupuesto de tiempo restante antes de decoherencia (desgaste pasivo)."""
        self._validate_address(logical_address)
        return max(0.0, self.t_max - self._idle_times[logical_address])
