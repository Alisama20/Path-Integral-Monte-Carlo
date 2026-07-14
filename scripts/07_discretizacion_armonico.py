"""
========================================================================
07_discretizacion_armonico.py  --  Errores sistemáticos por el
espaciado finito a (oscilador armónico)
========================================================================

El oscilador armónico es exactamente soluble TAMBIÉN en la red, lo que
permite separar el error sistemático de discretización del estadístico.

Con M0 = 1 y V = (1/2)μ²x², los resultados exactos en la red (T→∞) son:

    ⟨x²⟩_red   = 1 / [ 2μ·√(1 + (μa/2)²) ]
    E0(virial) = μ²⟨x²⟩ = (μ/2) / √(1 + (μa/2)²)
    E1 - E0    = (1/a)·arccosh(1 + (μa)²/2)

Todas tienden al continuo (μ/2, μ/2, μ) cuando a→0, con correcciones
O((μa)²). Aquí se duplica a (0.5 → 1.0 → 2.0) para ver crecer el sesgo.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import run_chains_armonico, run_chains_correlador, config_plot

config_plot()

# ─────────────────────────────────────────────────────────────────────
#  Parámetros
# ─────────────────────────────────────────────────────────────────────

M0   = 1.0
MU2  = 1.0
mu   = np.sqrt(MU2)

# Espaciados a estudiar (duplicando) y N para mantener T = N·a = 100 fijo
A_LIST = [0.5, 1.0, 2.0]
T_FIS  = 100.0
# δ ajustado por a para mantener aceptación ~60-70 %
DELTA_LIST = {0.5: 0.8, 1.0: 1.1, 2.0: 1.5}

N_THERM  = 1500
N_MEAS   = 8000
N_MEAS_G = 30000
N_SKIP   = 5
N_CHAINS = 8

X_MAX  = 3.5
N_BINS = 80
bins = np.linspace(-X_MAX, X_MAX, N_BINS + 1)


# ─────────────────────────────────────────────────────────────────────
#  Predicciones exactas en la red
# ─────────────────────────────────────────────────────────────────────

def x2_red(a):
    return 1.0 / (2.0 * mu * np.sqrt(1.0 + (mu * a / 2.0) ** 2))

def E0_red(a):
    return MU2 * x2_red(a)

def gap_red(a):
    return (1.0 / a) * np.arccosh(1.0 + (mu * a) ** 2 / 2.0)


def extraer_gap(G, G_err, a):
    """E1-E0 con el estimador log local (1/a)ln[G(τ)/G(τ+a)] promediado
    sobre los primeros τ con buena relación señal/ruido."""
    vals = []
    for t in range(1, len(G) - 1):
        if G[t] > 5 * G_err[t] and G[t + 1] > 5 * G_err[t + 1] and G[t] > G[t + 1]:
            vals.append((1.0 / a) * np.log(G[t] / G[t + 1]))
        if len(vals) >= 2:   # primeros dos puntos fiables
            break
    if not vals:
        return np.nan
    return float(np.mean(vals))


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  ERRORES DE DISCRETIZACIÓN — Oscilador armónico")
    print("=" * 64)
    print(f"  μ²={MU2}, M0={M0}, T=N·a={T_FIS}")
    print(f"  Continuo: E0=μ/2={mu/2:.3f}, E1-E0=μ={mu:.3f}, ⟨x²⟩=1/2μ={1/(2*mu):.3f}\n")

    resultados = []
    hists = {}

    for a in A_LIST:
        N = int(round(T_FIS / a))
        delta = DELTA_LIST[a]
        print(f"→ a = {a}  (N = {N}, μa = {mu*a:.2f}, δ = {delta})")

        # Observables + histograma
        t0 = time.time()
        hist, x2, x4, acc = run_chains_armonico(
            N_CHAINS, N, a, M0, MU2, delta,
            N_THERM, N_MEAS, N_SKIP, bins, X_MAX)
        x2_mean = x2.mean(); x2_std = x2.std() / np.sqrt(N_CHAINS)
        E0_mc = MU2 * x2_mean; E0_err = MU2 * x2_std

        # Correlador → E1-E0
        t_max = max(6, N // 3)
        G_chains, accG = run_chains_correlador(
            N_CHAINS, N, a, M0, MU2, delta,
            N_THERM, N_MEAS_G, N_SKIP, t_max)
        G = G_chains.mean(axis=0); G_err = G_chains.std(axis=0) / np.sqrt(N_CHAINS)
        gap_mc = extraer_gap(G, G_err, a)
        dt = time.time() - t0

        # Histograma normalizado
        hist_total = hist.sum(axis=0)
        dx = bins[1] - bins[0]
        hists[a] = hist_total / (hist_total.sum() * dx)

        print(f"   aceptación {acc.mean():.0%}  |  tiempo {dt:.1f}s")
        print(f"   E0:    MC {E0_mc:.4f}±{E0_err:.4f}   red {E0_red(a):.4f}   cont {mu/2:.4f}")
        print(f"   E1-E0: MC {gap_mc:.4f}            red {gap_red(a):.4f}   cont {mu:.4f}")
        print(f"   ⟨x²⟩:  MC {x2_mean:.4f}±{x2_std:.4f}   red {x2_red(a):.4f}   cont {1/(2*mu):.4f}\n")

        resultados.append(dict(a=a, N=N, mua=mu*a,
                               E0_mc=E0_mc, E0_err=E0_err,
                               gap_mc=gap_mc,
                               x2_mc=x2_mean, x2_err=x2_std))

    # ─── Figura: 2 paneles ─────────────────────────────────────────
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel izquierdo: |ψ0|² para los 3 espaciados ---
    centers = 0.5 * (bins[:-1] + bins[1:])
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(A_LIST)))
    for a, color in zip(A_LIST, colors):
        axL.step(centers, hists[a], where='mid', color=color, lw=1.6,
                 label=f'MC, $a={a}$  ($\\mu a={mu*a:.1f}$)')
    # Continuo exacto
    x_th = np.linspace(-X_MAX, X_MAX, 400)
    psi2_cont = np.sqrt(mu / np.pi) * np.exp(-mu * x_th ** 2)
    axL.plot(x_th, psi2_cont, 'r--', lw=2, label=r'continuo $\sqrt{\mu/\pi}\,e^{-\mu x^2}$')
    axL.set_xlabel('$x$'); axL.set_ylabel('$|\\psi_0(x)|^2$')
    axL.set_title('Función de onda: estrechamiento con $a$')
    axL.set_xlim(-3.2, 3.2)
    axL.legend(fontsize=8.5); axL.grid(alpha=0.3)

    # --- Panel derecho: E0 y E1-E0 vs μa ---
    a_fine = np.linspace(0.01, 2.3, 200)
    mua_fine = mu * a_fine

    # Curvas exactas en la red
    axR.plot(mua_fine, [E0_red(a) for a in a_fine], '-', color='C0', lw=1.8,
             label=r'$E_0$ red (exacto)')
    axR.plot(mua_fine, [gap_red(a) for a in a_fine], '-', color='C3', lw=1.8,
             label=r'$E_1-E_0$ red (exacto)')
    # Continuo (líneas horizontales)
    axR.axhline(mu / 2, color='C0', ls=':', lw=1.3, label=r'$E_0$ continuo $=\mu/2$')
    axR.axhline(mu, color='C3', ls=':', lw=1.3, label=r'$E_1-E_0$ continuo $=\mu$')
    # Puntos MC
    mua_pts = [r['mua'] for r in resultados]
    axR.errorbar(mua_pts, [r['E0_mc'] for r in resultados],
                 yerr=[r['E0_err'] for r in resultados],
                 fmt='o', ms=9, color='C0', mec='k', mew=1, capsize=4, zorder=5)
    axR.plot(mua_pts, [r['gap_mc'] for r in resultados],
             's', ms=9, color='C3', mec='k', mew=1, zorder=5)
    axR.set_xlabel(r'$\mu\, a$  (parámetro de discretización)')
    axR.set_ylabel('Energía')
    axR.set_title('Sesgo sistemático de $E_0$ y $E_1-E_0$')
    axR.set_xlim(0, 2.3); axR.set_ylim(0.3, 1.05)
    axR.legend(fontsize=8.5, loc='lower left'); axR.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig_18_discretizacion_armonico.png', dpi=150)
    print("   guardado: fig_18_discretizacion_armonico.png")
    plt.close()

    # Tabla resumen
    print("  " + "─" * 60)
    print(f"  {'a':>5} {'μa':>5} {'E0 MC':>14} {'E0 red':>8} {'gap MC':>9} {'gap red':>8}")
    print("  " + "─" * 60)
    for r in resultados:
        print(f"  {r['a']:>5} {r['mua']:>5.2f} "
              f"{r['E0_mc']:>7.4f}±{r['E0_err']:.4f} {E0_red(r['a']):>8.4f} "
              f"{r['gap_mc']:>9.4f} {gap_red(r['a']):>8.4f}")

    print("\n" + "=" * 64)
    print("  COMPLETADO")
    print("=" * 64)


if __name__ == "__main__":
    main()
