"""
========================================================================
01_armonico.py  --  Oscilador armónico (validación del método)
========================================================================

Reproduce la Fig. 3 (caminos típicos) y la Fig. 5 (|ψ₀(x)|²) del paper
de Creutz & Freedman.

Potencial:   V(x) = (1/2) μ² x²

Resultados teóricos esperados (en unidades con ħ = M₀ = 1, μ = ω):
    E₀ = ω/2 = μ/2
    ψ₀(x) = (μ/π)^(1/4) · exp(-μ x²/2)
    |ψ₀(x)|² = √(μ/π) · exp(-μ x²)
    ⟨x²⟩ = 1/(2μ)
    ⟨x⁴⟩ = 3/(4μ²)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    metropolis_sweep_armonico,
    run_chains_armonico, run_chains_correlador,
    config_plot,
)

config_plot()

# ─────────────────────────────────────────────────────────────────────
#  Parámetros (similares al paper)
# ─────────────────────────────────────────────────────────────────────

N        = 200      # sitios en la red temporal
a        = 0.5      # espaciado temporal
M0       = 1.0      # masa
MU2      = 1.0      # μ² (=ω²)
DELTA    = 0.8      # paso de Metropolis (ajustado para ~50% aceptación)
N_THERM  = 1500     # sweeps de termalización
N_MEAS   = 8000     # número de medidas (psi0)
N_MEAS_G = 30000    # número de medidas para el correlador (necesita más)
N_SKIP   = 5        # sweeps entre medidas (decorrelación)
N_CHAINS = 8        # cadenas independientes (paralelo)

# Para el histograma de |ψ₀|²
X_MAX  = 3.5
N_BINS = 80
bins = np.linspace(-X_MAX, X_MAX, N_BINS + 1)


# ─────────────────────────────────────────────────────────────────────
#  Ejecución principal
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  OSCILADOR ARMÓNICO — Creutz-Freedman (1981), Fig. 3 y 5")
    print("=" * 64)
    print(f"  Red: N={N}, a={a},  M₀={M0}, μ²={MU2}")
    print(f"  MC : {N_CHAINS} cadenas × {N_MEAS} medidas (skip={N_SKIP})")
    print(f"       termalización: {N_THERM} sweeps")
    print()

    # ─── 1. Mostrar evolución de un camino (Fig. 3) ────────────────
    print("→ Generando caminos típicos (Fig. 3)...")
    np.random.seed(42)
    x = (np.random.random(N) - 0.5) * 2.0
    snapshots = []
    snap_iters = [0, 50, 200, 1000, 5000]

    for it in range(snap_iters[-1] + 1):
        if it in snap_iters:
            snapshots.append(x.copy())
        if it < snap_iters[-1]:
            metropolis_sweep_armonico(x, a, M0, MU2, DELTA)

    # Figura 1: 5 subplots apilados con valores REALES (no desplazados)
    # Estilo matplotlib por defecto (no el "paper")
    import matplotlib as mpl
    saved_rc = dict(mpl.rcParams)
    mpl.rcdefaults()

    fig, axes = plt.subplots(len(snap_iters), 1, figsize=(8, 9),
                             sharex=True, sharey=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(snap_iters)))
    for ax, s, it, color in zip(axes, snapshots, snap_iters, colors):
        ax.plot(s, color=color, lw=1.0)
        ax.axhline(0, color='gray', lw=0.4, alpha=0.5)
        ax.set_ylabel(r'$x(\tau_n)$')
        ax.legend([f'$m = {it}$'], loc='upper right', framealpha=0.9,
                  handlelength=0, handletextpad=0)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(r'posición temporal $n$')
    axes[-1].set_xlim(-1, N + 2)
    fig.suptitle('Evolución Monte Carlo de un camino — oscilador armónico',
                 y=0.995)
    plt.tight_layout()
    plt.savefig('fig_01_caminos_armonico.png', dpi=150)
    print("   guardado: fig_01_caminos_armonico.png")
    plt.close()

    # Restaurar estilo paper para las siguientes figuras
    mpl.rcParams.update(saved_rc)

    # ─── 2. Histograma de |ψ₀(x)|² (Fig. 5) ────────────────────────
    print("\n→ Calculando |ψ₀(x)|²...")
    t0 = time.time()
    hist, x2, x4, acc = run_chains_armonico(
        N_CHAINS, N, a, M0, MU2, DELTA,
        N_THERM, N_MEAS, N_SKIP, bins, X_MAX
    )
    dt = time.time() - t0
    print(f"   tiempo: {dt:.1f} s,  aceptación: {acc.mean():.1%}")

    # Combinar histogramas y normalizar
    hist_total = hist.sum(axis=0)
    dx = bins[1] - bins[0]
    hist_norm = hist_total / (hist_total.sum() * dx)

    # Valor teórico
    mu = np.sqrt(MU2)
    x_th = np.linspace(-X_MAX, X_MAX, 400)
    psi2_th = np.sqrt(mu / np.pi) * np.exp(-mu * x_th ** 2)

    # Observables
    x2_mean = x2.mean();  x2_std = x2.std() / np.sqrt(N_CHAINS)
    x4_mean = x4.mean();  x4_std = x4.std() / np.sqrt(N_CHAINS)
    x2_th = 1.0 / (2.0 * mu)
    x4_th = 3.0 / (4.0 * mu ** 2)

    print(f"\n   ⟨x²⟩ MC    = {x2_mean:.4f} ± {x2_std:.4f}")
    print(f"   ⟨x²⟩ teor. = {x2_th:.4f}")
    print(f"   ⟨x⁴⟩ MC    = {x4_mean:.4f} ± {x4_std:.4f}")
    print(f"   ⟨x⁴⟩ teor. = {x4_th:.4f}")

    # Energía via teorema del virial: E₀ = μ²⟨x²⟩ (para HO)
    # Más general (paper, ec. 2.30 con λ=0): E₀ = μ²⟨x²⟩
    E0_virial = MU2 * x2_mean
    E0_th = 0.5 * mu
    print(f"\n   E₀ (virial)  = {E0_virial:.4f}")
    print(f"   E₀ (teórico) = {E0_th:.4f}  (= μ/2)")

    # Plot histograma vs teoría
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax.bar(centers, hist_norm, width=dx, alpha=0.5, color='steelblue',
           edgecolor='navy', label='Monte Carlo')
    ax.plot(x_th, psi2_th, 'r-', lw=2, label=r'$|\psi_0(x)|^2$ teórico')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|\\psi_0(x)|^2$')
    ax.set_title(f'Función de onda del estado base (osc. armónico, $\\mu^2={MU2}$)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_05_psi0_armonico.png')
    print("\n   guardado: fig_05_psi0_armonico.png")
    plt.close()

    # ─── 3. Correlador y energía E₁ - E₀ ───────────────────────────
    print("\n→ Calculando correlador <x(0)x(τ)>...")
    t_max = N // 3  # solo hasta T/3 para evitar wraparound
    t0 = time.time()
    G_chains, acc_G = run_chains_correlador(
        N_CHAINS, N, a, M0, MU2, DELTA,
        N_THERM, N_MEAS_G, N_SKIP, t_max
    )
    print(f"   tiempo: {time.time()-t0:.1f} s,  aceptación: {acc_G.mean():.1%}")

    G = G_chains.mean(axis=0)
    G_err = G_chains.std(axis=0) / np.sqrt(N_CHAINS)

    # Predicción teórica: G_th(τ) = (1/2μ) cosh[μ(T/2-τ)]/sinh(μT/2)
    T_total = N * a
    taus = np.arange(t_max) * a
    G_th = 1.0 / (2 * mu) * np.cosh(mu * (T_total / 2 - taus)) / np.sinh(mu * T_total / 2)

    # E_eff(τ) = (1/a) ln[G(τ)/G(τ+a)]    (sólo donde G es positivo y > error)
    E_eff = np.full(t_max - 1, np.nan)
    for t in range(t_max - 1):
        if G[t] > 3 * G_err[t] and G[t + 1] > 3 * G_err[t + 1] and G[t] > G[t + 1]:
            E_eff[t] = (1.0 / a) * np.log(G[t] / G[t + 1])

    # Efectiva teórica derivada del cosh periódico (NO es una recta horizontal):
    #   E_eff_th(τ) = (1/a) ln[cosh(μ(T/2-τ))/cosh(μ(T/2-τ-a))]
    taus_eff = (np.arange(t_max - 1) + 0.5) * a
    E_eff_th = (1.0 / a) * np.log(
        np.cosh(mu * (T_total / 2 - np.arange(t_max - 1) * a)) /
        np.cosh(mu * (T_total / 2 - (np.arange(t_max - 1) + 1) * a))
    )

    # Plateau con la efectiva teórica (limitando al rango fiable)
    mask_plat = ~np.isnan(E_eff)
    plateau = E_eff[mask_plat][:t_max // 4]
    E1m0_MC = plateau.mean() if len(plateau) else np.nan
    E1m0_th = mu

    print(f"\n   E₁-E₀ (MC plateau)  = {E1m0_MC:.4f}")
    print(f"   E₁-E₀ (teórico)     = {E1m0_th:.4f}  (= μ)")

    # Recortar la visualización al rango donde el correlador tiene señal/ruido
    # razonable: G(τ) > 3·error.  Más allá los puntos son ruido puro.
    snr_mask = G > 3 * G_err
    if snr_mask.sum() > 4:
        t_show = int(np.where(snr_mask)[0].max()) + 1
    else:
        t_show = t_max
    t_show = min(t_show, int(20 / a))   # nunca más allá de τ = 20

    # Para la masa efectiva: rango más estrecho (S/N > 10 para evitar
    # que la log-ratio explote). E_eff es mucho más sensible al ruido
    # que el propio G porque toma una diferencia logarítmica.
    snr_strict = G > 10 * G_err
    if snr_strict.sum() > 2:
        t_show_eff_max = int(np.where(snr_strict)[0].max())
    else:
        t_show_eff_max = t_show

    # Plot correlador y energía efectiva
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.errorbar(taus[:t_show], G[:t_show], yerr=G_err[:t_show], fmt='o', ms=4,
                 color='steelblue', capsize=2, label='Monte Carlo')
    ax1.plot(taus[:t_show], G_th[:t_show], 'r-', lw=1.5,
             label=r'teórico (cosh periódico)')
    ax1.set_xlabel(r'$\tau$')
    ax1.set_ylabel(r'$G(\tau) = \langle x(0)x(\tau)\rangle$')
    ax1.set_yscale('log')
    ax1.legend(); ax1.grid(alpha=0.3, which='both')
    ax1.set_title(r'Correlador $G(\tau)$  (rango con S/N$>3$)')

    # En la efectiva mostrar sólo los puntos válidos (S/N>10) y la curva teórica
    t_show_eff = min(t_show_eff_max, t_max - 1)
    ax2.plot(taus_eff[:t_show_eff], E_eff_th[:t_show_eff], 'r-', lw=1.5,
             label=r'teórico (cosh)')
    ax2.axhline(E1m0_th, color='gray', ls=':', alpha=0.7,
                label=fr'asíntota $\mu = {mu:.2f}$')
    valid = mask_plat[:t_show_eff]
    ax2.plot(taus_eff[:t_show_eff][valid], E_eff[:t_show_eff][valid],
             'o', ms=4, color='darkgreen', label=r'Monte Carlo')
    ax2.set_xlabel(r'$\tau$')
    ax2.set_ylabel(r'$E_{\rm eff}(\tau) = a^{-1}\ln[G(\tau)/G(\tau+a)]$')
    ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_title('Masa efectiva')
    ax2.set_ylim(0, 1.6 * mu)

    plt.tight_layout()
    plt.savefig('fig_03_correlador_armonico.png')
    print("   guardado: fig_03_correlador_armonico.png")
    plt.close()

    print("\n" + "=" * 64)
    print("  COMPLETADO")
    print("=" * 64)


if __name__ == "__main__":
    main()
