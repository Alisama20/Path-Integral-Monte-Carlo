"""
========================================================================
06_degrand_comparacion.py  --  Oscilador anarmónico V = λ(x²-f²)²
                                 comparación con Blankenbecler-DeGrand-Sugar
========================================================================

Compara los niveles E₀ y E₁ del oscilador anarmónico
    H = p² + λ(x² - f²)²
con los valores de referencia obtenidos por Blankenbecler, DeGrand y
Sugar (Phys. Rev. D 21, 1055, 1980) usando el método de momentos.

Convenciones (paper de DeGrand):
    - H = p² + V        →  kinetic term coefficient = 1
    - Comparando con H = p²/(2M₀) + V → M₀ = 1/2
    - λ se fija a λ = 1 sin pérdida de generalidad (escalado E∝λ^{1/3})

Se barre f² entre -1 y 5 y se calcula E₀ vía teorema del virial. La
energía exacta para el primer excitado se obtiene del decaimiento del
correlador.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    V_doble_pozo, medir_x2, medir_x4,
    correlador_2puntos, config_plot,
)

config_plot()


# ─────────────────────────────────────────────────────────────────────
#  Valores de referencia (Tabla II de Blankenbecler-DeGrand-Sugar 1980)
# ─────────────────────────────────────────────────────────────────────

DEGRAND_TABLE_II = {
    # f² : (E0, E1)
    -1.0: (2.6778265,  6.4098280),
     0.0: (1.0603621,  3.7996730),
     1.0: (1.1377858,  2.7130279),
     2.0: (2.2896495,  2.7520771),
     3.0: (3.2518095,  3.2932079),
     4.0: (3.8636669,  3.8651857),
     5.0: (4.3664229,  4.3664531),
}


# ─────────────────────────────────────────────────────────────────────
#  Sweep de Metropolis para V = λ(x²-f²)²
# ─────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def sweep_dp_pbc(x, a, m0, lam, f2, delta):
    """Sweep con condiciones de frontera periódicas para V = λ(x²-f²)²"""
    N = x.shape[0]
    n_accept = 0
    kin_coef = m0 / a

    for j in range(N):
        jm = (j - 1) % N
        jp = (j + 1) % N
        xm = x[jm]
        xp = x[jp]
        x_old = x[j]

        x_new = x_old + delta * (2.0 * np.random.random() - 1.0)

        dS_kin = 0.5 * kin_coef * (
            (x_new - xm) ** 2 + (xp - x_new) ** 2
            - (x_old - xm) ** 2 - (xp - x_old) ** 2
        )
        dS_pot = a * (V_doble_pozo(x_new, lam, f2) -
                      V_doble_pozo(x_old, lam, f2))
        dS = dS_kin + dS_pot

        if dS <= 0.0 or np.random.random() < np.exp(-dS):
            x[j] = x_new
            n_accept += 1

    return n_accept


# ─────────────────────────────────────────────────────────────────────
#  Runner paralelo: barrido sobre (f², cadena)
# ─────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True, parallel=True)
def barrido_f2_degrand(f2_list, n_chains, N, a, m0, lam, delta,
                        n_therm, n_meas, n_skip, t_max):
    """
    Para cada f² y cada cadena calcula ⟨x²⟩, ⟨x⁴⟩ y el correlador G(τ).
    """
    n_f = f2_list.shape[0]
    x2_arr = np.zeros((n_f, n_chains))
    x4_arr = np.zeros((n_f, n_chains))
    G_arr = np.zeros((n_f, n_chains, t_max))
    accept_arr = np.zeros((n_f, n_chains))
    xc_arr = np.zeros(n_chains)  # auxiliar

    total = n_f * n_chains
    for idx in prange(total):
        i_f = idx // n_chains
        c = idx % n_chains
        f2 = f2_list[i_f]

        # Inicialización: mitad cadenas en +x_c, mitad en -x_c si f²>0
        if f2 > 0:
            xc = np.sqrt(f2)
            if c % 2 == 0:
                x = np.ones(N) * xc
            else:
                x = np.ones(N) * (-xc)
            x = x + 0.2 * (np.random.random(N) - 0.5)
        else:
            x = 0.5 * (np.random.random(N) - 0.5)

        # Termalización
        for _ in range(n_therm):
            sweep_dp_pbc(x, a, m0, lam, f2, delta)

        # Medidas
        x2_acc = 0.0
        x4_acc = 0.0
        G_acc = np.zeros(t_max)
        n_acc_tot = 0
        for m in range(n_meas):
            for _ in range(n_skip):
                n_acc_tot += sweep_dp_pbc(x, a, m0, lam, f2, delta)
            x2_acc += medir_x2(x)
            x4_acc += medir_x4(x)
            G_acc += correlador_2puntos(x, t_max)

        x2_arr[i_f, c] = x2_acc / n_meas
        x4_arr[i_f, c] = x4_acc / n_meas
        G_arr[i_f, c] = G_acc / n_meas
        accept_arr[i_f, c] = n_acc_tot / (n_meas * n_skip * N)

    return x2_arr, x4_arr, G_arr, accept_arr


def E0_virial(x2, x4, f2, lam=1.0):
    """
    Energía del estado base vía teorema del virial para V = λ(x²-f²)².
    Derivación:
        dV/dx = 4λx(x²-f²)
        ⟨T⟩ = (1/2)⟨x·dV/dx⟩ = 2λ⟨x⁴⟩ - 2λf²⟨x²⟩
        ⟨V⟩ = λ⟨x⁴⟩ - 2λf²⟨x²⟩ + λf⁴
        E₀ = ⟨T⟩+⟨V⟩ = 3λ⟨x⁴⟩ - 4λf²⟨x²⟩ + λf⁴
    """
    return lam * (3.0 * x4 - 4.0 * f2 * x2 + f2 ** 2)


def E1_log(G, a, tau_target=1.0):
    """
    Estimador puntual de Creutz-Freedman (ec. 4.14): E₁-E₀ ≈ (1/a) ln[G(τ)/G(τ+a)]
    evaluado en un único τ ≈ tau_target. Devuelve (ΔE, 0). El error se debe
    obtener fuera, por dispersión entre cadenas.
    """
    N = len(G)
    C_plateau = float(np.min(G[max(2, N // 3):]))
    G_conn = G - C_plateau
    t = max(1, int(round(tau_target / a)))
    if t + 1 >= N:
        return np.nan, np.nan
    if G_conn[t] <= 0 or G_conn[t + 1] <= 0:
        return np.nan, np.nan
    if G_conn[t] <= G_conn[t + 1]:
        return np.nan, np.nan
    dE = (1.0 / a) * np.log(G_conn[t] / G_conn[t + 1])
    return float(dE), 0.0


def _legacy_coshfit_unused(G, a, T_total, t_min=2, t_max_frac=0.45):
    """[obsoleto, conservado por historial] Ajuste cosh — no se usa."""
    from scipy.optimize import curve_fit
    N = len(G)
    C_plateau = float(np.min(G[max(2, N // 3):]))

    t_max = int(t_max_frac * N)
    taus = np.arange(t_min, t_max) * a
    G_conn = G[t_min:t_max] - C_plateau

    mask = G_conn > 0
    if mask.sum() < 4:
        return np.nan, np.nan

    taus_fit = taus[mask]
    G_fit = G_conn[mask]

    def cosh_form(tau, A, dE):
        return A * np.cosh(dE * (T_total / 2 - tau))

    try:
        popt, pcov = curve_fit(cosh_form, taus_fit, G_fit,
                               p0=[G_fit[0], 1.0],
                               bounds=([0.0, 0.05], [np.inf, 25.0]),
                               maxfev=5000)
        dE = float(popt[1])
        dE_err = float(np.sqrt(max(pcov[1, 1], 0.0)))
        if not np.isfinite(dE) or not np.isfinite(dE_err):
            return np.nan, np.nan
        return dE, dE_err
    except Exception:
        return np.nan, np.nan


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  OSCILADOR ANARMÓNICO  V = λ(x²-f²)²")
    print("  Comparación con Blankenbecler-DeGrand-Sugar (1980)")
    print("=" * 64)
    print("  Hamiltoniano: H = p² + λ(x²-f²)²    (M₀ = 0.5, λ = 1)")
    print()

    # Parámetros físicos (convención DeGrand: p² → M₀ = 0.5)
    M0  = 0.5
    LAM = 1.0
    a   = 0.1       # espaciado fino para reducir artefactos de red
    N   = 200

    # Parámetros MC
    DELTA   = 0.4
    N_THERM = 3000
    N_MEAS  = 12000
    N_SKIP  = 8
    N_CHAINS = 8
    t_max = N // 3

    # Valores de f² (mismos que DeGrand para comparación directa)
    f2_list = np.array(sorted(DEGRAND_TABLE_II.keys()))

    print(f"  Red: N = {N}, a = {a}, M₀ = {M0}, λ = {LAM}")
    print(f"  MC : {N_CHAINS} cadenas × {N_MEAS} medidas")
    print(f"  Termalización: {N_THERM} barridos, skip: {N_SKIP}")
    print()
    print("  Ejecutando barrido en paralelo...")

    t0 = time.time()
    x2_arr, x4_arr, G_arr, acc_arr = barrido_f2_degrand(
        f2_list, N_CHAINS, N, a, M0, LAM, DELTA,
        N_THERM, N_MEAS, N_SKIP, t_max
    )
    dt = time.time() - t0
    print(f"  Tiempo total: {dt:.1f} s,  aceptación media: {acc_arr.mean():.1%}")
    print()

    # Calcular E₀ y E₁
    E0_arr = np.zeros((len(f2_list), N_CHAINS))
    for i, f2 in enumerate(f2_list):
        for c in range(N_CHAINS):
            E0_arr[i, c] = E0_virial(x2_arr[i, c], x4_arr[i, c], f2, LAM)

    E0_mean = E0_arr.mean(axis=1)
    E0_err = E0_arr.std(axis=1) / np.sqrt(N_CHAINS)

    T_total = N * a
    E1_mean = np.full(len(f2_list), np.nan)
    E1_err  = np.full(len(f2_list), np.nan)
    for i in range(len(f2_list)):
        f2 = f2_list[i]
        # Para f² ≥ 3 la brecha E₁-E₀ es exponencialmente pequeña
        # (∼ exp(-S_inst) ∼ exp(-(2√2/3)f³)) y resulta ininfluyente
        # frente a estados de mayor energía: se descarta.
        if f2 >= 3.0:
            continue  # deja NaN

        # Estimador puntual de Creutz (ec. 4.14): log evaluado a τ ≈ 1,
        # promediando por cadena para obtener error como dispersión.
        dE_chains = []
        for c in range(N_CHAINS):
            dE, _ = E1_log(G_arr[i, c], a, tau_target=1.0)
            if not np.isnan(dE) and 0 < dE < 20:
                dE_chains.append(dE)
        if len(dE_chains) >= 2:
            arr = np.array(dE_chains)
            E1m0 = arr.mean()
            E1m0_err = arr.std() / np.sqrt(len(arr))
            E1_mean[i] = E0_mean[i] + E1m0
            E1_err[i]  = np.sqrt(E0_err[i]**2 + E1m0_err**2)

    # ─── Tabla comparativa ────────────────────────────────────────
    print("  " + "─" * 78)
    print(f"  {'f²':>5} | {'E₀ MC':>14} | {'E₀ ref':>10} | "
          f"{'E₁ MC':>14} | {'E₁ ref':>10}")
    print("  " + "─" * 78)

    E0_ref_arr = np.array([DEGRAND_TABLE_II[f2][0] for f2 in f2_list])
    E1_ref_arr = np.array([DEGRAND_TABLE_II[f2][1] for f2 in f2_list])

    for i, f2 in enumerate(f2_list):
        E0_ref = DEGRAND_TABLE_II[f2][0]
        E1_ref = DEGRAND_TABLE_II[f2][1]
        e1_str = (f"{E1_mean[i]:8.4f} ± {E1_err[i]:.4f}"
                  if not np.isnan(E1_mean[i]) else "      -      ")
        print(f"  {f2:>5.1f} | {E0_mean[i]:8.4f} ± {E0_err[i]:.4f} | "
              f"{E0_ref:>10.4f} | {e1_str} | {E1_ref:>10.4f}")
    print("  " + "─" * 78)

    # ─── Gráfica comparativa ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    # Curvas de referencia (Tabla II DeGrand)
    f2_ref_smooth = np.linspace(f2_list.min(), f2_list.max(), 200)
    from scipy.interpolate import interp1d
    interp_E0_ref = interp1d(f2_list, E0_ref_arr, kind='cubic')
    interp_E1_ref = interp1d(f2_list, E1_ref_arr, kind='cubic')

    ax.plot(f2_ref_smooth, interp_E0_ref(f2_ref_smooth),
            '-', color='steelblue', lw=1.2, alpha=0.7,
            label='$E_0$ (DeGrand)')
    ax.plot(f2_ref_smooth, interp_E1_ref(f2_ref_smooth),
            '-', color='darkred', lw=1.2, alpha=0.7,
            label='$E_1$ (DeGrand)')

    # Puntos Monte Carlo
    ax.errorbar(f2_list, E0_mean, yerr=E0_err, fmt='o',
                color='steelblue', ms=8, capsize=4,
                markeredgecolor='black', markeredgewidth=0.8,
                label='$E_0$ (Monte Carlo)')
    mask = ~np.isnan(E1_mean)
    ax.errorbar(f2_list[mask], E1_mean[mask], yerr=E1_err[mask], fmt='s',
                color='darkred', ms=8, capsize=4,
                markeredgecolor='black', markeredgewidth=0.8,
                label='$E_1$ (Monte Carlo)')

    ax.set_xlabel('$f^2$')
    ax.set_ylabel('Energía')
    ax.set_title(r'Niveles del oscilador $V=(x^2-f^2)^2$:'
                 ' Monte Carlo vs DeGrand')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    # Eje X invertido como en el paper de Creutz-Freedman
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig('fig_18_degrand_comparacion.png')
    print("\n   guardado: fig_18_degrand_comparacion.png")
    plt.close()

    # ─── Gráfica adicional: error relativo en E₀ ─────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    err_rel = 100 * np.abs(E0_mean - E0_ref_arr) / E0_ref_arr
    ax.bar(np.arange(len(f2_list)), err_rel, color='steelblue',
           edgecolor='navy', alpha=0.7)
    ax.set_xticks(np.arange(len(f2_list)))
    ax.set_xticklabels([f'{f2:.1f}' for f2 in f2_list])
    ax.set_xlabel('$f^2$')
    ax.set_ylabel(r'$|E_0^{\rm MC} - E_0^{\rm ref}|/E_0^{\rm ref}$ (\%)')
    ax.set_title('Desviación relativa del Monte Carlo respecto a DeGrand')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('fig_19_error_degrand.png')
    print("   guardado: fig_19_error_degrand.png")
    plt.close()

    # Guardar datos para potencial uso posterior
    np.savez('datos_degrand.npz',
             f2=f2_list,
             E0_mc=E0_mean, E0_err=E0_err,
             E1_mc=E1_mean, E1_err=E1_err,
             E0_ref=E0_ref_arr, E1_ref=E1_ref_arr,
             x2=x2_arr.mean(axis=1), x4=x4_arr.mean(axis=1))
    print("   datos guardados: datos_degrand.npz")

    print("\n" + "=" * 64)
    print("  COMPLETADO")
    print("=" * 64)


if __name__ == "__main__":
    main()
