"""
========================================================================
run_all.py  --  Ejecuta toda la reproducción del paper Creutz-Freedman
========================================================================

Corre secuencialmente los cinco scripts numerados.
Tiempo total estimado: ~5-15 minutos (depende de la CPU).

Uso:
    python run_all.py [--rapido]

    --rapido : reduce N_MEAS para una ejecución de prueba más corta
"""

import subprocess
import sys
import time
import os

SCRIPTS = [
    ('01_armonico.py',           'Oscilador armónico (validación)'),
    ('02_anarmonico.py',         'Oscilador anarmónico (niveles)'),
    ('03_pozo_doble.py',         'Doble pozo (instantones/kinks)'),
    ('04_fuente_externa.py',     'Respuesta a fuente externa J'),
    ('05_potencial_efectivo.py', 'Potencial efectivo V_R'),
]

def main():
    aqui = os.path.dirname(os.path.abspath(__file__))
    os.chdir(aqui)

    print("\n" + "█" * 64)
    print("  REPRODUCCIÓN: Creutz & Freedman (1981)")
    print("  'A Statistical Approach to Quantum Mechanics'")
    print("█" * 64 + "\n")

    t_global = time.time()
    resultados = []

    for script, descripcion in SCRIPTS:
        print(f"\n{'▶' * 3}  {script} — {descripcion}")
        print("-" * 64)
        t0 = time.time()
        try:
            res = subprocess.run([sys.executable, script],
                                  capture_output=False, check=False)
            dt = time.time() - t0
            status = '✓ OK' if res.returncode == 0 else '✗ ERROR'
            resultados.append((script, dt, status))
        except Exception as e:
            print(f"   Excepción: {e}")
            resultados.append((script, time.time() - t0, '✗ EXCEPCIÓN'))

    # Resumen final
    print("\n" + "█" * 64)
    print("  RESUMEN")
    print("█" * 64)
    print(f"\n  {'Script':<32} {'Tiempo':>10} {'Estado':>12}")
    print("  " + "─" * 56)
    for script, dt, status in resultados:
        print(f"  {script:<32} {dt:>8.1f}s {status:>12}")
    print(f"\n  Tiempo total: {time.time() - t_global:.1f} s")
    print("\n  Figuras generadas en este directorio (.png)")


if __name__ == '__main__':
    main()
