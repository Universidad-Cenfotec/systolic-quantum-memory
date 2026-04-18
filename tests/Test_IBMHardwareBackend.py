from qiskit_ibm_runtime import QiskitRuntimeService

print("--- Test de Capacidades de Hardware Heron (Qiskit 1.0+) ---")

# Usamos explícitamente el canal nuevo para evitar problemas
service = QiskitRuntimeService(channel="ibm_quantum_platform")

backend_name = "ibm_kingston"
print(f"\nConectando a {backend_name}...")
backend = service.backend(backend_name)

# 1. En la arquitectura V2, las operaciones soportadas se consultan así:
instrucciones = backend.operation_names
print(f"\n✅ Instrucciones soportadas ({len(instrucciones)}):")
print(instrucciones)

# 2. Verificaciones CRÍTICAS para tu Tesis (SQM)
print("\n--- Checklist para Memoria Sistólica ---")
print(f"👉 ¿Soporta 'delay' nativo?: {'delay' in instrucciones}")
print(f"👉 ¿Soporta 'measure' (medición a mitad de circuito)?: {'measure' in instrucciones}")
# Heron usa 'if_else' o 'switch_case' para el feed-forward clásico de SamplerV2
print(f"👉 ¿Soporta flujo clásico dinámico?: {'if_else' in instrucciones or 'switch_case' in instrucciones}")

# 3. Consultar propiedades de un qubit específico (Ejemplo: Qubit 0)
# En BackendV2, las propiedades se acceden a través de backend.target
print("\n--- Propiedades Térmicas del Qubit 0 ---")
try:
    # BackendV2 usa backend.target para acceder a propiedades
    if hasattr(backend, 'target') and backend.target is not None:
        # Acceder a las propiedades del qubit 0 desde target
        # qubit_properties es una propiedad que devuelve una lista, no un método
        qubit_props = backend.target.qubit_properties[0] if backend.target.qubit_properties else None
        if qubit_props:
            if hasattr(qubit_props, 't1') and qubit_props.t1:
                print(f"T1 (Decoherencia de amplitud): {qubit_props.t1 * 1e9:.2f} ns")
            else:
                print("T1 no disponible")
            
            if hasattr(qubit_props, 't2') and qubit_props.t2:
                print(f"T2 (Desfase): {qubit_props.t2 * 1e9:.2f} ns")
            else:
                print("T2 no disponible")
        else:
            print("⚠️  Las propiedades del Qubit 0 no están disponibles")
    else:
        print("⚠️  backend.target no disponible en este backend")
        print("   Información general del backend:")
        print(f"   - Qubits: {backend.num_qubits}")
except AttributeError as e:
    print(f"⚠️  AttributeError: {e}")
    print("   BackendV2 podría no soportar acceso directo a propiedades T1/T2")
except Exception as e:
    print(f"⚠️  Error inesperado: {e}")
    print(f"   Tipo: {type(e).__name__}")