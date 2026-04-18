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
# En V2, las propiedades se acceden a través de backend.properties() o backend.target
print("\n--- Propiedades Térmicas del Qubit 0 ---")
try:
    # Intentar acceso directo a backend.target
    if hasattr(backend, 'target') and backend.target is not None:
        if hasattr(backend.target, 'qubit_properties') and backend.target.qubit_properties:
            propiedades_q0 = backend.target.qubit_properties[0]
            if propiedades_q0 and hasattr(propiedades_q0, 't1') and hasattr(propiedades_q0, 't2'):
                print(f"T1 (Decoherencia de amplitud): {propiedades_q0.t1 * 1e9:.2f} ns")
                print(f"T2 (Desfase): {propiedades_q0.t2 * 1e9:.2f} ns")
            else:
                print("Las propiedades T1/T2 no están disponibles en backend.target")
        else:
            print("backend.target.qubit_properties no disponible")
    else:
        print("backend.target no disponible, intentando backend.properties()...")
        # Alternativa: usar backend.properties()
        props = backend.properties()
        if props:
            t1_q0 = props.t1(0)
            t2_q0 = props.t2(0)
            print(f"T1 (Decoherencia de amplitud): {t1_q0 * 1e9:.2f} ns")
            print(f"T2 (Desfase): {t2_q0 * 1e9:.2f} ns")
        else:
            print("backend.properties() no disponible")
except Exception as e:
    print(f"⚠️  No se pudieron extraer las propiedades del Qubit 0: {e}")
    print(f"   Tipo de error: {type(e).__name__}")