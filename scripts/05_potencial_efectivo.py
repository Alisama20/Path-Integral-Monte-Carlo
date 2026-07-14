"""
========================================================================
05_potencial_efectivo.py  --  Potencial efectivo V_R(x_J)
========================================================================

Reproduce las Figs. 13 y 14 del paper de Creutz & Freedman:
    - Fig. 13: V_R(x_J) para el oscilador armónico  →  parabólico
    - Fig. 14: V_R(x_J) para el doble pozo  →  un solo mínimo en x_J = 0

El potencial efectivo se obtiene integrando la función J(x_J) (la inversa
de la curva ⟨x⟩_J vs J calculada en el script anterior):

    V_R(x_J) = ∫₀^{x_J} dx'  J(x')                       (ec. 4.22)

Lo notable es que para el doble pozo, V_R tiene UN SOLO MÍNIMO (en x_J=0),
porque la mecánica cuántica no rompe espontáneamente la simetría
reflexión: el efecto túnel mezcla los dos pozos.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config_plot

config_plot()


def integrar_potencial_efectivo(x, J, x_ref=0.0):
    """
    Potencial efectivo renormalizado:
        V_R(x_J) = -∫_{x_ref}^{x_J} J(x') dx'

    El signo viene de la transformada de Legendre: ∂V_R/∂x_J = -J(x_J),
    consistente con la convención del paper S' = S + J·Σxᵢ.
    """
    idx = np.argsort(x)
    x_sorted = x[idx]
    J_sorted = J[idx]

    V_R = -cumulative_trapezoid(J_sorted, x_sorted, initial=0)

    interp = interp1d(x_sorted, V_R, kind='cubic', fill_value='extrapolate')
    V_R = V_R - interp(x_ref)

    return x_sorted, V_R


def integrar_potencial_efectivo_simetrico(x, J):
    """
    Potencial efectivo del doble pozo aprovechando la simetría V_R(x)=V_R(-x).

    Los datos del doble pozo NO cubren la región |x|<x_c (el sistema
    cuántico no se sienta en la barrera). Integrar trapezoidalmente
    a través de un punto espurio (0,0) genera un tramo plano artificial.
    En su lugar:
      1. Tomamos sólo los puntos con x>0 (rama derecha), monotónicos.
      2. Integramos V_R(x>0) = -∫ J(x') dx' desde el mínimo accesible.
      3. Por simetría, V_R(x<0) = V_R(-x).
      4. Anclamos V_R(x_min) = 0 (el "fondo" del pozo cuántico accesible).
    """
    mask_pos = x > 0
    x_pos = x[mask_pos]
    J_pos = J[mask_pos]
    idx = np.argsort(x_pos)
    x_pos = x_pos[idx]
    J_pos = J_pos[idx]

    V_pos = -cumulative_trapezoid(J_pos, x_pos, initial=0)
    V_pos = V_pos - V_pos.min()   # anclar el mínimo a cero

    # Simetrizar: V_R(-x) = V_R(x). Insertamos NaN en x=0 para que la
    # línea de unión NO atraviese la región no muestreada |x|<x_min.
    x_full = np.concatenate([-x_pos[::-1], [0.0], x_pos])
    V_full = np.concatenate([V_pos[::-1], [np.nan], V_pos])
    return x_full, V_full


def main():
    print("=" * 64)
    print("  POTENCIAL EFECTIVO V_R(x_J) — Creutz-Freedman, Figs. 13 y 14")
    print("=" * 64)

    # Cargar resultados del script 04
    try:
        datos = np.load('datos_fuente_externa.npz')
    except FileNotFoundError:
        print("\n  ERROR: ejecuta primero '04_fuente_externa.py' para")
        print("         generar los datos de J vs ⟨x⟩.")
        return

    # ─── 1. FIG. 13: V_R del oscilador armónico ────────────────────
    print("\n→ FIG. 13: potencial efectivo del oscilador armónico")
    J_arm = datos['J_arm']
    x_arm = datos['x_arm']
    mu2_arm = float(datos['mu2_arm'])

    x_sorted_arm, V_R_arm = integrar_potencial_efectivo(x_arm, J_arm)

    # Teoría: para el armónico, V_R debe ser exactamente (1/2)μ²x²
    x_th = np.linspace(x_sorted_arm.min(), x_sorted_arm.max(), 100)
    V_R_th = 0.5 * mu2_arm * x_th ** 2

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(x_sorted_arm, V_R_arm, 'ko', ms=7,
            label='$V_R$ (de los datos MC)')
    ax.plot(x_th, V_R_th, 'r-', lw=1.5,
            label=f'$V(x) = \\frac{{1}}{{2}}\\mu^2 x^2 = {mu2_arm/2:.2f}\\,x^2$')
    ax.set_xlabel('$x_J$')
    ax.set_ylabel('$V_R(x_J)$')
    ax.set_title('Potencial renormalizado: oscilador armónico')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    plt.tight_layout()
    plt.savefig('fig_13_VR_armonico.png')
    print("   guardado: fig_13_VR_armonico.png")
    plt.close()

    # ─── 2. FIG. 14: V_R del doble pozo ────────────────────────────
    print("\n→ FIG. 14: potencial efectivo del doble pozo")
    J_dp = datos['J_dp']
    x_dp = datos['x_dp']
    lam_dp = float(datos['lam_dp'])
    f2_dp = float(datos['f2_dp'])

    # Integración continua: ahora que el muestreo incluye J=0 (con
    # ⟨x⟩≈0 por efecto túnel cuántico) y muchos puntos densos en la
    # región central, podemos integrar a través de todo el rango y
    # anclar V_R(0)=0. Esto da una curva CONVEXA única con un solo
    # mínimo en x_J=0, como en la figura 14 de Creutz & Freedman.
    x_sorted_dp, V_R_dp = integrar_potencial_efectivo(x_dp, J_dp, x_ref=0.0)

    # Potencial clásico V(x) = λ(x²-f²)²
    xc = np.sqrt(f2_dp)
    x_cl_plot = np.linspace(-2.2, 2.2, 200)
    V_cl = lam_dp * (x_cl_plot ** 2 - f2_dp) ** 2
    # Trasladar para que V_cl(0) = 0  → restar λf⁴
    V_cl = V_cl - lam_dp * f2_dp ** 2

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_sorted_dp, V_R_dp, 'ko-', ms=6, lw=1.2,
            label='$V_R$ (de los datos MC)')
    ax.plot(x_cl_plot, V_cl, 'r--', lw=1.2, alpha=0.7,
            label=f'$V_{{\\rm clas}}=\\lambda(x^2-f^2)^2$ (referencia)')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.axvline(+xc, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.axvline(-xc, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('$x_J$')
    ax.set_ylabel('$V_R(x_J)$')
    ax.set_title('Potencial renormalizado: doble pozo\n'
                 '(¡un solo mínimo en $x_J=0$: no hay rotura espontánea cuántica!)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(-2.2, 2.2)
    plt.tight_layout()
    plt.savefig('fig_14_VR_doble_pozo.png')
    print("   guardado: fig_14_VR_doble_pozo.png")
    plt.close()

    # ─── 3. Comparación lado a lado ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(x_sorted_arm, V_R_arm, 'ko', ms=6)
    ax.plot(x_th, V_R_th, 'r-', lw=1.5)
    ax.set_xlabel('$x_J$');  ax.set_ylabel('$V_R$')
    ax.set_title('Armónico: $V_R(x_J)$')
    ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', lw=0.4)

    ax = axes[1]
    ax.plot(x_sorted_dp, V_R_dp, 'ko-', ms=6, lw=1)
    ax.plot(x_cl_plot, V_cl, 'r--', lw=1, alpha=0.6)
    ax.axvline(+xc, color='b', ls=':', alpha=0.5, label=f'$\\pm x_c=\\pm{xc:.2f}$')
    ax.axvline(-xc, color='b', ls=':', alpha=0.5)
    ax.set_xlabel('$x_J$');  ax.set_ylabel('$V_R$')
    ax.set_title('Doble pozo: $V_R(x_J)$ tiene un solo mínimo')
    ax.set_xlim(-2.2, 2.2)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.axhline(0, color='gray', lw=0.4)

    plt.tight_layout()
    plt.savefig('fig_13_14_comparacion_VR.png')
    print("\n   guardado: fig_13_14_comparacion_VR.png")
    plt.close()

    print("\n  ★ Observación clave (Sección IV del paper):")
    print("    El doble pozo CLÁSICO tiene dos mínimos en ±x_c, pero el")
    print("    potencial efectivo cuántico V_R tiene un único mínimo en")
    print("    x_J = 0. La mecánica cuántica no rompe espontáneamente")
    print("    la simetría de reflexión.")

    print("\n" + "=" * 64)
    print("  COMPLETADO")
    print("=" * 64)


if __name__ == "__main__":
    main()
