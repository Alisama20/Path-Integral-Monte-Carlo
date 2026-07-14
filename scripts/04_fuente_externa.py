"""
========================================================================
04_fuente_externa.py  --  Respuesta a una fuente externa J
========================================================================

Reproduce las Figs. 11 y 12 del paper de Creutz & Freedman:
    - Fig. 11: ⟨x⟩_J vs J para el oscilador armónico
    - Fig. 12: ⟨x⟩_J vs J para el oscilador anarmónico / doble pozo

La acción modificada es:
    S' = S + J · Σᵢ xᵢ

El acoplamiento a una fuente J rompe la simetría de reflexión y produce
⟨x⟩_J ≠ 0. Para el oscilador armónico la relación es exactamente lineal:
    J = -μ² · ⟨x⟩_J     (paper, ec. 4.20)

Para el doble pozo, la relación es no lineal y refleja la estructura del
potencial efectivo.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    metropolis_sweep_armonico_J,
    metropolis_sweep_doble_pozo,
    medir_x, config_plot,
)

config_plot()

# ─────────────────────────────────────────────────────────────────────
#  Parámetros
# ─────────────────────────────────────────────────────────────────────

DELTA   = 0.6
N_THERM = 3000
N_MEAS  = 15000    # más estadística para reducir scatter de ⟨x⟩_J
N_SKIP  = 5
N_CHAINS = 8


# ─────────────────────────────────────────────────────────────────────
#  Runners
# ─────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True, parallel=True)
def barrido_J_armonico(J_list, N, a, m0, mu2, delta,
                        n_therm, n_meas, n_skip, n_chains):
    """Para cada J, mide ⟨x⟩ promediando sobre cadenas"""
    n_J = J_list.shape[0]
    x_arr = np.zeros((n_J, n_chains))

    total = n_J * n_chains
    for idx in prange(total):
        i_J = idx // n_chains
        c = idx % n_chains
        J = J_list[i_J]

        # Inicializar cerca de la solución clásica J = -μ²x → x_cl = -J/μ²
        x0 = -J / mu2 if mu2 > 0 else 0.0
        x = np.ones(N) * x0 + 0.2 * (np.random.random(N) - 0.5)

        for _ in range(n_therm):
            metropolis_sweep_armonico_J(x, a, m0, mu2, delta, J)

        x_acc = 0.0
        for _ in range(n_meas):
            for __ in range(n_skip):
                metropolis_sweep_armonico_J(x, a, m0, mu2, delta, J)
            x_acc += medir_x(x)

        x_arr[i_J, c] = x_acc / n_meas

    return x_arr


@njit(cache=True, fastmath=True, parallel=True)
def barrido_J_doble_pozo(J_list, N, a, m0, lam, f2, delta,
                          n_therm, n_meas, n_skip, n_chains):
    """Para cada J en doble pozo, mide ⟨x⟩"""
    n_J = J_list.shape[0]
    x_arr = np.zeros((n_J, n_chains))
    xc = np.sqrt(f2)

    total = n_J * n_chains
    for idx in prange(total):
        i_J = idx // n_chains
        c = idx % n_chains
        J = J_list[i_J]

        # Inicializar en el mínimo más favorable
        # V_eff = V - J·x, mínimos cerca de ±x_c si J pequeño
        if J > 0:
            x = np.ones(N) * (-xc)  # J>0 prefiere x<0 (porque V' = J)
        elif J < 0:
            x = np.ones(N) * (+xc)
        else:
            # Mitad y mitad para tener ambos pozos
            x = np.ones(N) * (xc if c % 2 == 0 else -xc)
        x = x + 0.1 * (np.random.random(N) - 0.5)

        # Termalización más larga (el doble pozo es difícil)
        for _ in range(n_therm * 2):
            metropolis_sweep_doble_pozo(x, a, m0, lam, f2, delta, J)

        x_acc = 0.0
        for _ in range(n_meas):
            for __ in range(n_skip):
                metropolis_sweep_doble_pozo(x, a, m0, lam, f2, delta, J)
            x_acc += medir_x(x)

        x_arr[i_J, c] = x_acc / n_meas

    return x_arr


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  RESPUESTA A FUENTE EXTERNA — Creutz-Freedman, Figs. 11 y 12")
    print("=" * 64)

    # ─── 1. FIG. 11: oscilador armónico ────────────────────────────
    print("\n→ FIG. 11: ⟨x⟩_J vs J  (oscilador armónico)")
    N_arm  = 200
    a_arm  = 0.25
    M0_arm = 0.5
    mu2_arm = 2.0   # μ²

    J_list = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])

    t0 = time.time()
    x_arr = barrido_J_armonico(J_list, N_arm, a_arm, M0_arm, mu2_arm, DELTA,
                                N_THERM, N_MEAS, N_SKIP, N_CHAINS)
    print(f"   tiempo: {time.time()-t0:.1f} s")

    x_mean = x_arr.mean(axis=1)
    x_err = x_arr.std(axis=1) / np.sqrt(N_CHAINS)

    # Teoría: J = -μ²·⟨x⟩  →  ⟨x⟩ = -J/μ²
    x_th = -J_list / mu2_arm

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(x_mean, J_list, xerr=x_err, fmt='ko', ms=7, capsize=3,
                label='Monte Carlo')
    ax.plot(x_th, J_list, 'r-', lw=1.5, label=f'$J = -\\mu^2\\langle x\\rangle$')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel(r'$\langle x\rangle_J$')
    ax.set_ylabel('$J$')
    ax.set_title(f'Oscilador armónico: $J$ vs $\\langle x\\rangle$ '
                 f'($\\mu^2={mu2_arm}$)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_11_J_armonico.png')
    print("   guardado: fig_11_J_armonico.png")
    plt.close()

    # Tabla
    print("\n  Resultados:")
    print(f"  {'J':>8} {'⟨x⟩ MC':>12} {'⟨x⟩ teor':>12}")
    print("  " + "─" * 38)
    for i, J in enumerate(J_list):
        print(f"  {J:>8.3f} {x_mean[i]:>10.4f}±{x_err[i]:.4f} "
              f"{x_th[i]:>12.4f}")

    # ─── 2. FIG. 12: anarmónico / doble pozo ───────────────────────
    print("\n→ FIG. 12: ⟨x⟩_J vs J  (doble pozo)")
    N_dp  = 400       # más sitios para mantener T = N·a = 20
    a_dp  = 0.05      # red aún más fina: ω_local·a ~ 0.28, error de
                      # discretización (ω·a)²/12 ~ 0.7% (vs 2.7% a a=0.1).
                      # Necesario para que el ajuste a la cúbica clásica
                      # sea visible en las ramas externas.
    M0_dp = 0.5
    LAM_dp = 1.0
    F2_dp  = 2.0

    # Barrido denso, INCLUYENDO J=0 (el punto central ⟨x⟩=0 representa
    # la promediación entre las dos ramas, característico del efecto
    # túnel cuántico). Se densifica además la zona |J|<2 para ver bien
    # la transición entre la rama cuántica y el régimen clásico.
    J_list2 = np.unique(np.concatenate([
        np.linspace(-12, -5, 5),      # zona clásica negativa
        np.linspace(-5, -2, 7),       # transición negativa
        np.linspace(-2, -0.1, 15),    # zona cuántica negativa (denso)
        np.array([0.0]),              # PUNTO CENTRAL
        np.linspace(0.1, 2, 15),      # zona cuántica positiva (denso)
        np.linspace(2, 5, 7),         # transición positiva
        np.linspace(5, 12, 5),        # zona clásica positiva
    ]))

    t0 = time.time()
    x_arr2 = barrido_J_doble_pozo(J_list2, N_dp, a_dp, M0_dp, LAM_dp, F2_dp,
                                   DELTA, N_THERM, N_MEAS, N_SKIP, N_CHAINS)
    print(f"   tiempo: {time.time()-t0:.1f} s")

    x_mean2 = x_arr2.mean(axis=1)
    x_err2 = x_arr2.std(axis=1) / np.sqrt(N_CHAINS)

    # Print del punto central J=0 (y vecinos)
    idx_centrales = np.argsort(np.abs(J_list2))[:5]
    print("\n   Puntos centrales (|J| más pequeño):")
    print(f"   {'J':>8} {'⟨x⟩ MC':>20}")
    for idx in sorted(idx_centrales, key=lambda i: J_list2[i]):
        print(f"   {J_list2[idx]:>+8.3f} {x_mean2[idx]:>+12.4f} ± {x_err2[idx]:.4f}")

    # Teoría clásica: J = -dV/dx = -4λx(x²-f²)
    x_cl = np.linspace(-2.0, 2.0, 200)
    J_cl = -4 * LAM_dp * x_cl * (x_cl ** 2 - F2_dp)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Curva clásica (referencia)
    ax.plot(x_cl, J_cl, 'r-', lw=1.5,
            label=r'Clásico: $J = -dV/dx$ (referencia)')

    # Curva-guía cuántica: spline a través de los puntos MC, separado por rama
    # (la curva quántica es MULTIVALUADA: dos ramas inaccesibles entre sí)
    mask_R = x_mean2 > 0.5
    mask_L = x_mean2 < -0.5
    idx_R = np.argsort(x_mean2[mask_R])
    idx_L = np.argsort(x_mean2[mask_L])
    if mask_R.sum() >= 4:
        ax.plot(x_mean2[mask_R][idx_R], J_list2[mask_R][idx_R],
                'k--', lw=1, alpha=0.5)
    if mask_L.sum() >= 4:
        ax.plot(x_mean2[mask_L][idx_L], J_list2[mask_L][idx_L],
                'k--', lw=1, alpha=0.5,
                label='Cuántico: guía a través de los puntos MC')

    # Puntos MC encima
    ax.errorbar(x_mean2, J_list2, xerr=x_err2, fmt='ko', ms=6, capsize=3,
                label='Monte Carlo')

    # Resaltar el punto central J=0
    idx_J0 = int(np.argmin(np.abs(J_list2)))
    ax.errorbar([x_mean2[idx_J0]], [J_list2[idx_J0]], xerr=[x_err2[idx_J0]],
                fmt='s', ms=10, mfc='gold', mec='black', mew=1.5,
                capsize=4, ecolor='black', zorder=5,
                label=f'$J=0$  ($\\langle x\\rangle = {x_mean2[idx_J0]:+.3f}$)')

    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.axvline(+np.sqrt(F2_dp), color='gray', ls=':', lw=0.8)
    ax.axvline(-np.sqrt(F2_dp), color='gray', ls=':', lw=0.8)
    ax.set_xlabel(r'$\langle x\rangle_J$')
    ax.set_ylabel('$J$')
    ax.set_title(f'Doble pozo: $J$ vs $\\langle x\\rangle$ '
                 f'($\\lambda={LAM_dp},\\,f^2={F2_dp}$)')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-14, 14)
    plt.tight_layout()
    plt.savefig('fig_12_J_doble_pozo.png')
    print("   guardado: fig_12_J_doble_pozo.png")
    plt.close()

    # Guardar datos para el script 05 (potencial efectivo)
    np.savez('datos_fuente_externa.npz',
             J_arm=J_list, x_arm=x_mean, x_arm_err=x_err,
             mu2_arm=mu2_arm,
             J_dp=J_list2, x_dp=x_mean2, x_dp_err=x_err2,
             lam_dp=LAM_dp, f2_dp=F2_dp)
    print("\n   datos guardados: datos_fuente_externa.npz")

    print("\n" + "=" * 64)
    print("  COMPLETADO")
    print("=" * 64)


if __name__ == "__main__":
    main()
