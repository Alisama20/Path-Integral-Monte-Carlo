"""
========================================================================
02_anarmonico.py  --  Oscilador anarmónico
========================================================================

Reproduce las Figs. 8, 9 y 10 del paper de Creutz & Freedman:
    - Fig. 8: |ψ₀(x)|² para el oscilador anarmónico (un solo mínimo)
    - Fig. 9: |ψ₀(x)|² para el doble pozo (dos mínimos)
    - Fig. 10: niveles de energía E₀, E₁ vs f²

Notación del paper:  V(x) = (1/4) μ²(x² - f²)² + const
                          = (1/4) μ² x⁴ - (1/2) μ² f² x² + ...

Aquí usamos la forma equivalente:
    V(x) = (1/2) μ² x² + λ x⁴

con μ² < 0 para producir doble pozo.

Si V = λ(x²-f²)² = λx⁴ - 2λf²x² + λf⁴ podemos identificar:
    λ_eff = λ,   μ²_eff = -2λf² (negativo)

Vamos a usar V = μ²/2 · x² + λ x⁴ y barremos μ² desde positivo (un solo
mínimo) hasta negativo (doble pozo).
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    metropolis_sweep_anarmonico, metropolis_sweep_doble_pozo,
    V_doble_pozo, medir_x2, medir_x4,
    correlador_2puntos, config_plot,
)

config_plot()

# ─────────────────────────────────────────────────────────────────────
#  Parámetros
# ─────────────────────────────────────────────────────────────────────

N        = 200
a        = 0.5
M0       = 1.0
LAM      = 1.0      # λ del término x⁴
DELTA    = 2*np.sqrt(a)
N_THERM  = 2000
N_MEAS   = 15000    # más estadística para reducir scatter en la curva E(f²)
N_SKIP   = 5
N_CHAINS = 1

X_MAX  = 3.5
N_BINS = 80


# ─────────────────────────────────────────────────────────────────────
#  Runner para anarmónico (varios μ² en paralelo)
# ─────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True, parallel=True)
def run_anarmonico_barrido(mu2_list, n_chains, N, a, m0, lam, delta,
                            n_therm, n_meas, n_skip, t_max):
    """
    Barre valores de μ² y para cada uno corre n_chains cadenas.
    Devuelve: <x²>, <x⁴>, G(τ) para extraer E₀ y E₁.
    """
    n_mu = mu2_list.shape[0]
    x2_arr = np.zeros((n_mu, n_chains))
    x4_arr = np.zeros((n_mu, n_chains))
    G_arr = np.zeros((n_mu, n_chains, t_max))

    # Paralelizar sobre (μ², cadena)
    total = n_mu * n_chains
    for idx in prange(total):
        i_mu = idx // n_chains
        c = idx % n_chains
        mu2 = mu2_list[i_mu]

        x = (np.random.random(N) - 0.5) * 2.0
        # Si μ² < 0, inicializar cerca de uno de los mínimos
        if mu2 < 0:
            f = np.sqrt(-mu2 / (2 * lam))
            if c % 2 == 0:
                x = x + f
            else:
                x = x - f

        for _ in range(n_therm):
            metropolis_sweep_anarmonico(x, a, m0, mu2, lam, delta)

        x2_acc = 0.0
        x4_acc = 0.0
        G_acc = np.zeros(t_max)
        for _ in range(n_meas):
            for __ in range(n_skip):
                metropolis_sweep_anarmonico(x, a, m0, mu2, lam, delta)
            x2_acc += medir_x2(x)
            x4_acc += medir_x4(x)
            G_acc += correlador_2puntos(x, t_max)

        x2_arr[i_mu, c] = x2_acc / n_meas
        x4_arr[i_mu, c] = x4_acc / n_meas
        G_arr[i_mu, c] = G_acc / n_meas

    return x2_arr, x4_arr, G_arr


@njit(cache=True, fastmath=True, parallel=True)
def run_histograma(mu2, lam, n_chains, N, a, m0, delta,
                    n_therm, n_meas, n_skip, x_bins, x_max):
    """Histograma de x para |ψ₀|² con V = ½μ²x² + λx⁴"""
    n_bins = x_bins.shape[0] - 1
    hist = np.zeros((n_chains, n_bins))

    for c in prange(n_chains):
        x = (np.random.random(N) - 0.5) * 2.0
        if mu2 < 0:
            # x_c² = -μ²/(4λ)  (mínimo de V = ½μ²x² + λx⁴)
            f = np.sqrt(-mu2 / (4 * lam))
            if c % 2 == 0:
                x = x + f
            else:
                x = x - f

        for _ in range(n_therm):
            metropolis_sweep_anarmonico(x, a, m0, mu2, lam, delta)

        for _ in range(n_meas):
            for __ in range(n_skip):
                metropolis_sweep_anarmonico(x, a, m0, mu2, lam, delta)
            for v in x:
                if -x_max < v < x_max:
                    b = int((v + x_max) / (2 * x_max) * n_bins)
                    if 0 <= b < n_bins:
                        hist[c, b] += 1.0

    return hist


@njit(cache=True, fastmath=True, parallel=True)
def barrido_f2_creutz(f2_list, n_chains, N, a, m0, lam, delta,
                       n_therm, n_meas, n_skip, t_max):
    """
    Barre f² para V = λ(x²-f²)² (forma Creutz). Devuelve ⟨x²⟩, ⟨x⁴⟩ y G(τ).
    """
    n_f = f2_list.shape[0]
    x2_arr = np.zeros((n_f, n_chains))
    x4_arr = np.zeros((n_f, n_chains))
    G_arr  = np.zeros((n_f, n_chains, t_max))
    accept_arr = np.zeros((n_f, n_chains))

    total = n_f * n_chains
    for idx in prange(total):
        i_f = idx // n_chains
        c   = idx % n_chains
        f2  = f2_list[i_f]

        # Inicialización: mitad de cadenas en +x_c, mitad en -x_c si f²>0
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
            metropolis_sweep_doble_pozo(x, a, m0, lam, f2, delta, 0.0)

        # Medidas
        x2_acc = 0.0
        x4_acc = 0.0
        G_acc  = np.zeros(t_max)
        n_acc_tot = 0
        for _m in range(n_meas):
            for __ in range(n_skip):
                n_acc_tot += metropolis_sweep_doble_pozo(
                    x, a, m0, lam, f2, delta, 0.0)
            x2_acc += medir_x2(x)
            x4_acc += medir_x4(x)
            G_acc  += correlador_2puntos(x, t_max)

        x2_arr[i_f, c] = x2_acc / n_meas
        x4_arr[i_f, c] = x4_acc / n_meas
        G_arr[i_f, c]  = G_acc  / n_meas
        accept_arr[i_f, c] = n_acc_tot / (n_meas * n_skip * N)

    return x2_arr, x4_arr, G_arr, accept_arr


@njit(cache=True, fastmath=True, parallel=True)
def barrido_f2_single_chain(f2_list, n_blocks, N, a, m0, lam, delta,
                              n_therm, n_meas, n_skip, t_max):
    """
    Método de bloques: 1 cadena por f² que termaliza una vez y luego
    hace n_blocks medidas separadas por n_therm sweeps de re-relajación.
    El error se estima como std/√n_blocks entre bloques.

    La paralelización es sobre f² (cada hilo corre su propia cadena).
    """
    n_f = f2_list.shape[0]
    x2_arr = np.zeros((n_f, n_blocks))
    x4_arr = np.zeros((n_f, n_blocks))
    G_arr  = np.zeros((n_f, n_blocks, t_max))

    for i_f in prange(n_f):
        f2 = f2_list[i_f]
        # Inicialización en un pozo
        if f2 > 0:
            xc = np.sqrt(f2)
            x = np.ones(N) * xc + 0.2 * (np.random.random(N) - 0.5)
        else:
            x = 0.5 * (np.random.random(N) - 0.5)

        # Termalización inicial
        for _ in range(n_therm):
            metropolis_sweep_doble_pozo(x, a, m0, lam, f2, delta, 0.0)

        # n_blocks bloques de medida
        for b in range(n_blocks):
            # Re-relajación entre bloques (= n_therm sweeps)
            for _ in range(n_therm):
                metropolis_sweep_doble_pozo(x, a, m0, lam, f2, delta, 0.0)

            # Medidas dentro del bloque
            x2_acc = 0.0
            x4_acc = 0.0
            G_acc  = np.zeros(t_max)
            for _m in range(n_meas):
                for __ in range(n_skip):
                    metropolis_sweep_doble_pozo(x, a, m0, lam, f2, delta, 0.0)
                x2_acc += medir_x2(x)
                x4_acc += medir_x4(x)
                G_acc  += correlador_2puntos(x, t_max)
            x2_arr[i_f, b] = x2_acc / n_meas
            x4_arr[i_f, b] = x4_acc / n_meas
            G_arr[i_f, b]  = G_acc  / n_meas

    return x2_arr, x4_arr, G_arr


@njit(cache=True, fastmath=True, parallel=True)
def run_histograma_creutz(f2, lam, n_chains, N, a, m0, delta,
                            n_therm, n_meas, n_skip, x_bins, x_max):
    """Histograma de x para |ψ₀|² con V = λ(x²-f²)² (forma de Creutz)"""
    n_bins = x_bins.shape[0] - 1
    hist = np.zeros((n_chains, n_bins))
    xc = np.sqrt(f2)

    for c in prange(n_chains):
        x = (np.random.random(N) - 0.5) * 2.0
        if c % 2 == 0:
            x = x + xc
        else:
            x = x - xc

        for _ in range(n_therm):
            metropolis_sweep_doble_pozo(x, a, m0, lam, f2, delta, 0.0)

        for _ in range(n_meas):
            for __ in range(n_skip):
                metropolis_sweep_doble_pozo(x, a, m0, lam, f2, delta, 0.0)
            for v in x:
                if -x_max < v < x_max:
                    b = int((v + x_max) / (2 * x_max) * n_bins)
                    if 0 <= b < n_bins:
                        hist[c, b] += 1.0

    return hist


# ─────────────────────────────────────────────────────────────────────
#  Análisis: extraer E₀ y E₁
# ─────────────────────────────────────────────────────────────────────

def extraer_E0(x2, x4, mu2, lam):
    """E₀ via teorema del virial (paper, ec. 2.30):
       E₀ = μ²⟨x²⟩ + 3λ⟨x⁴⟩
    Nota: válido cuando V = (1/2)μ²x² + λx⁴
    """
    return mu2 * x2 + 3 * lam * x4


def extraer_E1(G, a, t_min=2, t_max=None):
    """E₁-E₀ del decaimiento del correlador (estimador efectivo, no usado).
    Mantenido por compatibilidad. La función fiable es extraer_E1_coshfit.
    """
    if t_max is None:
        t_max = len(G) // 2
    E_eff = []
    for t in range(t_min, t_max):
        if G[t + 1] > 0 and G[t] > G[t + 1]:
            E_eff.append((1.0 / a) * np.log(G[t] / G[t + 1]))
    return np.mean(E_eff) if E_eff else np.nan


def extraer_E1_log(G, a, tau_target=1.0):
    """
    Estimador puntual de Creutz-Freedman (ec. 4.14):
        E₁ - E₀ ≈ (1/a) ln[G(τ)/G(τ+a)]
    evaluado en un único τ ≈ tau_target.
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


def extraer_E1_log_adaptativo(G, a, T_total):
    """
    Estimador log con τ adaptativo. El τ óptimo del estimador puntual
    depende de la brecha E₁-E₀:
      - τ ≪ 1/(E₁-E₀): contaminación por estados excitados → sobrestima
      - τ ≫ 1/(E₁-E₀): G es ruido → log explota
      - τ óptimo: ~ 1 a 2 tiempos de decaimiento, 1.5/(E₁-E₀)

    Estrategia:
      1. Usa los primeros dos puntos para una estimación inicial gap_0.
      2. Toma τ_opt = clamp(1.5/gap_0, 2a, T/4).
      3. Promedia (1/a) ln[G(t)/G(t+a)] en una ventana pequeña alrededor de t_opt.
    """
    N = len(G)
    C_plateau = float(np.min(G[max(2, N // 3):]))
    G_conn = G - C_plateau
    # Estimación inicial de la brecha con t=1 → t=2 (descartando t=0
    # que está contaminado por todos los estados excitados)
    if G_conn[1] <= 0 or G_conn[2] <= 0 or G_conn[1] <= G_conn[2]:
        return np.nan, np.nan
    gap0 = (1.0 / a) * np.log(G_conn[1] / G_conn[2])
    if gap0 <= 0.05 or gap0 > 30:
        return np.nan, np.nan
    # τ_opt: 1.5 tiempos de decaimiento, acotado entre 2a y T/4
    tau_opt = max(2 * a, min(1.5 / gap0, T_total / 4))
    t_opt = max(1, min(int(round(tau_opt / a)), N - 2))
    # Promediar el m_eff en una ventana pequeña alrededor de t_opt
    vals = []
    for t in range(max(1, t_opt - 1), min(N - 1, t_opt + 2)):
        if G_conn[t] > 0 and G_conn[t + 1] > 0 and G_conn[t] > G_conn[t + 1]:
            vals.append((1.0 / a) * np.log(G_conn[t] / G_conn[t + 1]))
    if not vals:
        return np.nan, np.nan
    arr = np.array(vals)
    err = float(arr.std() / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return float(arr.mean()), err


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  OSCILADOR ANARMÓNICO — Creutz-Freedman, Figs. 8, 9, 10")
    print("=" * 64)

    # ─── 1. Histogramas para diferentes μ² ─────────────────────────
    print("\n→ Calculando |ψ₀(x)|² para distintos μ²...")
    bins = np.linspace(-X_MAX, X_MAX, N_BINS + 1)
    casos = [
        (+1.0, "Anarmónico (un mínimo)"),
        (-1.0, "Doble pozo suave"),
        (-3.0, "Doble pozo profundo"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, (mu2, label) in zip(axes[:3], casos):
        print(f"   μ² = {mu2:.1f}   ({label})")
        t0 = time.time()
        hist = run_histograma(mu2, LAM, N_CHAINS, N, a, M0, DELTA,
                              N_THERM, N_MEAS, N_SKIP, bins, X_MAX)
        print(f"      tiempo: {time.time()-t0:.1f} s")

        hist_total = hist.sum(axis=0)
        dx = bins[1] - bins[0]
        hist_norm = hist_total / (hist_total.sum() * dx)

        centers = 0.5 * (bins[:-1] + bins[1:])
        ax.bar(centers, hist_norm, width=dx, alpha=0.5,
               color='steelblue', edgecolor='navy', label=r'$|\psi_0|^2$ MC')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$|\\psi_0(x)|^2$', color='navy')
        ax.tick_params(axis='y', labelcolor='navy')
        ax.set_title(f'$\\mu^2 = {mu2:+.1f},\\;\\lambda = {LAM}$\n{label}')
        ax.grid(alpha=0.3)

        # Eje Y secundario para el potencial en su escala REAL: así
        # los pozos del doble pozo son visibles, no aplastados por x^4
        ax2 = ax.twinx()
        # Recortar el rango de x donde dibujamos V, para no estirar
        # la escala por culpa del x^4 en los bordes
        x_pot = np.linspace(-2.0, 2.0, 400)
        V = 0.5 * mu2 * x_pot ** 2 + LAM * x_pot ** 4
        ax2.plot(x_pot, V, 'r--', lw=1.4, label='$V(x)$')
        ax2.axhline(0, color='gray', lw=0.4, alpha=0.5)
        ax2.set_ylabel('$V(x)$', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        # Marcar los mínimos clásicos si mu²<0
        # V'(x) = μ²x + 4λx³ = 0  →  x_c² = -μ²/(4λ)
        if mu2 < 0:
            xc = np.sqrt(-mu2 / (4 * LAM))
            ax2.axvline(+xc, color='red', ls=':', lw=0.8, alpha=0.6)
            ax2.axvline(-xc, color='red', ls=':', lw=0.8, alpha=0.6)
        # Ajustar el ylim de V para que se vean los pozos pero no
        # los valores enormes en los bordes
        ax2.set_ylim(V.min() - 0.5, max(2.0, abs(V.min()) * 4))

    # ── Panel 4: reproducción exacta de la figura 8 de Creutz-Freedman ──
    print("   Creutz fig 8: V=(x²-f²)², f²=2, M₀=0.5, a=0.25, N=200")
    ax = axes[3]
    F2_CR  = 2.0
    M0_CR  = 0.5
    a_CR   = 0.25
    N_CR   = 200
    bins_CR = np.linspace(-X_MAX, X_MAX, N_BINS + 1)
    t0 = time.time()
    hist_CR = run_histograma_creutz(F2_CR, LAM, N_CHAINS, N_CR, a_CR,
                                     M0_CR, DELTA, N_THERM, N_MEAS, N_SKIP,
                                     bins_CR, X_MAX)
    print(f"      tiempo: {time.time()-t0:.1f} s")
    hist_CR_total = hist_CR.sum(axis=0)
    dx_CR = bins_CR[1] - bins_CR[0]
    hist_CR_norm = hist_CR_total / (hist_CR_total.sum() * dx_CR)
    centers_CR = 0.5 * (bins_CR[:-1] + bins_CR[1:])

    ax.bar(centers_CR, hist_CR_norm, width=dx_CR, alpha=0.5,
           color='steelblue', edgecolor='navy', label=r'$|\psi_0|^2$ MC')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|\\psi_0(x)|^2$', color='navy')
    ax.tick_params(axis='y', labelcolor='navy')
    ax.set_title(f'$V=\\lambda(x^2-f^2)^2,\\;f^2={F2_CR}$\n'
                 'Creutz fig 8 ($M_0=0.5$, $a=0.25$)')
    ax.grid(alpha=0.3)

    # Eje Y derecho con el potencial en su escala real
    ax2 = ax.twinx()
    x_pot = np.linspace(-2.0, 2.0, 400)
    V_CR = LAM * (x_pot ** 2 - F2_CR) ** 2
    ax2.plot(x_pot, V_CR, 'r--', lw=1.4, label='$V(x)$')
    xc_CR = np.sqrt(F2_CR)
    ax2.axvline(+xc_CR, color='red', ls=':', lw=0.8, alpha=0.6)
    ax2.axvline(-xc_CR, color='red', ls=':', lw=0.8, alpha=0.6)
    ax2.set_ylabel('$V(x)$', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(-0.5, 5.5)

    plt.tight_layout()
    plt.savefig('fig_08_09_psi_anarmonico.png')
    print("   guardado: fig_08_09_psi_anarmonico.png")
    plt.close()

    # ─── 2. Niveles E₀ y E₁ vs f² (reproducción de la Fig. 10 de Creutz) ──
    # Forma de Creutz: V = λ(x²-f²)² con M₀ = 0.5, a = 0.25, N = 200.
    # Esto NO es lo mismo que el panel anterior con V = ½μ²x² + λx⁴:
    # son la misma física pero con masa y espaciado distintos.
    print("\n→ Barrido de f² (Creutz fig 10): V=λ(x²-f²)², M₀=0.5, a=0.25, N=200")

    F2_LIST = np.linspace(-1.0, 3.0, 13)   # rango y resolución como Creutz
    M0_F10  = 0.5
    a_F10   = 0.25
    N_F10   = 200
    DELTA_F10 = 0.5

    t_max_F10 = N_F10 // 3
    t0 = time.time()
    x2_arr, x4_arr, G_arr, _ = barrido_f2_creutz(
        F2_LIST, N_CHAINS, N_F10, a_F10, M0_F10, LAM, DELTA_F10,
        N_THERM, N_MEAS, N_SKIP, t_max_F10
    )
    print(f"   tiempo total: {time.time()-t0:.1f} s")

    E0_list = np.zeros(len(F2_LIST))
    E0_err = np.zeros(len(F2_LIST))
    E1_list = np.full(len(F2_LIST), np.nan)
    E1_err = np.full(len(F2_LIST), np.nan)
    T_total_F10 = N_F10 * a_F10

    for i, f2 in enumerate(F2_LIST):
        # E₀ via virial de Creutz: E₀ = λ(3⟨x⁴⟩ - 4f²⟨x²⟩ + f⁴)
        E0_chains = LAM * (3 * x4_arr[i] - 4 * f2 * x2_arr[i] + f2 ** 2)
        E0_list[i] = E0_chains.mean()
        E0_err[i] = E0_chains.std() / np.sqrt(N_CHAINS)

        # E₁ por estimador puntual log en τ ≈ 1 (Creutz ec. 4.14).
        # Aplicamos por cadena para obtener el error como desviación
        # estándar de la media entre cadenas.
        dE_chains = []
        for c in range(N_CHAINS):
            dE, _ = extraer_E1_log(G_arr[i, c], a_F10, tau_target=1.0)
            if not np.isnan(dE) and 0 < dE < 20:
                dE_chains.append(dE)
        if len(dE_chains) >= 2:
            dE_arr = np.array(dE_chains)
            E1m0 = dE_arr.mean()
            E1m0_err = dE_arr.std() / np.sqrt(len(dE_arr))
            E1_list[i] = E0_list[i] + E1m0
            E1_err[i] = np.sqrt(E0_err[i] ** 2 + E1m0_err ** 2)

    # Valores de referencia "continuum theory" (Tabla II Blankenbecler-DeGrand-
    # Sugar 1980, los mismos que cita Creutz en su ref. [9])
    f2_ref = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    E0_ref = np.array([2.6778, 1.0604, 1.1378, 2.2896, 3.2518])
    E1_ref = np.array([6.4098, 3.7997, 2.7130, 2.7521, 3.2932])

    # ── BARRIDO MULTI-A: convergencia E₀ y E₁ → continuo cuando a → 0 ──
    # Método de bloques: 1 cadena por f², termaliza una vez, luego
    # n_blocks medidas separadas por n_therm sweeps de re-relajación.
    # Estimador log con τ adaptativo: cada f² escoge su τ ~ 1.5/(E₁-E₀).
    N_BLOCKS    = 100
    N_THERM_MB  = 1000   # termalización y re-relajación entre bloques
    N_MEAS_MB   = 1000   # medidas dentro de cada bloque
    print(f"\n→ Convergencia con el espaciado: barrido multi-a (T = 50 fijo, "
          f"método de bloques: {N_BLOCKS} bloques)", flush=True)
    a_values = [0.5, 0.25, 0.1, 0.05]
    T_fixed  = 50.0
    F2_short = np.linspace(-1.0, 3.0, 13)
    E0_by_a, E0err_by_a = [], []
    E1_by_a, E1err_by_a = [], []
    t_start_all = time.time()
    for ia, a_test in enumerate(a_values):
        N_test = int(round(T_fixed / a_test))
        t_max_t = max(8, N_test // 3)
        print(f"   [{ia+1}/{len(a_values)}] a = {a_test:.2f},  N = {N_test} ... ",
              end='', flush=True)
        t0 = time.time()
        x2_t, x4_t, G_t = barrido_f2_single_chain(
            F2_short, N_BLOCKS, N_test, a_test, M0_F10, LAM,
            DELTA_F10, N_THERM_MB, N_MEAS_MB, N_SKIP, t_max_t)
        # E₀ (virial Creutz, promediado sobre bloques)
        E0_t = LAM * (3 * x4_t - 4 * F2_short[:, None] * x2_t + F2_short[:, None] ** 2)
        E0_by_a.append(E0_t.mean(axis=1))
        E0err_by_a.append(E0_t.std(axis=1) / np.sqrt(N_BLOCKS))
        # E₁ (estimador log adaptativo aplicado a cada bloque)
        T_total_test = N_test * a_test
        E1_t = np.full(len(F2_short), np.nan)
        E1err_t = np.full(len(F2_short), np.nan)
        for k, f2 in enumerate(F2_short):
            dE_blocks = []
            for b in range(N_BLOCKS):
                dE, _ = extraer_E1_log_adaptativo(G_t[k, b], a_test, T_total_test)
                if not np.isnan(dE) and 0 < dE < 25:
                    dE_blocks.append(dE)
            if len(dE_blocks) >= 2:
                arr = np.array(dE_blocks)
                E1_t[k]   = E0_by_a[-1][k] + arr.mean()
                E1err_t[k] = np.sqrt(E0err_by_a[-1][k]**2 +
                                     (arr.std() / np.sqrt(len(arr)))**2)
        E1_by_a.append(E1_t)
        E1err_by_a.append(E1err_t)
        print(f"hecho en {time.time()-t0:.1f}s  "
              f"(acumulado {time.time()-t_start_all:.1f}s)", flush=True)

    # ── Plot: 2 paneles, convergencia de E₀ y E₁ con a ──────────────
    from scipy.interpolate import interp1d
    fig, (ax_L, ax_R) = plt.subplots(1, 2, figsize=(13, 5))

    f_smooth = np.linspace(f2_ref.min(), f2_ref.max(), 200)
    colors_a = plt.cm.viridis(np.linspace(0.15, 0.85, len(a_values)))
    markers_a = ['s', 'o', '^', 'D']

    # PANEL IZQUIERDO: E₀
    ax_L.plot(f_smooth, interp1d(f2_ref, E0_ref, kind='cubic')(f_smooth),
              'k-', lw=2, alpha=0.7, label='$E_0$ (continuo)')
    for a_test, E0_t, E0err_t, color, marker in zip(
            a_values, E0_by_a, E0err_by_a, colors_a, markers_a):
        ax_L.errorbar(F2_short, E0_t, yerr=E0err_t, fmt=marker, color=color,
                      ms=6, capsize=2, markeredgecolor='black',
                      markeredgewidth=0.4, label=f'$a = {a_test:.2f}$')
    ax_L.set_xlabel('$f^2$');  ax_L.set_ylabel('$E_0(f^2)$')
    ax_L.set_title(r'Convergencia de $E_0$ con $a\to 0$  ($T = Na = 50$ fijo)')
    ax_L.legend(loc='upper left', fontsize=9)
    ax_L.grid(alpha=0.3)
    ax_L.set_ylim(0, 4.0)
    ax_L.invert_xaxis()

    # PANEL DERECHO: E₁
    ax_R.plot(f_smooth, interp1d(f2_ref, E1_ref, kind='cubic')(f_smooth),
              'k-', lw=2, alpha=0.7, label='$E_1$ (continuo)')
    for a_test, E1_t, E1err_t, color, marker in zip(
            a_values, E1_by_a, E1err_by_a, colors_a, markers_a):
        mask = ~np.isnan(E1_t)
        if mask.any():
            ax_R.errorbar(F2_short[mask], E1_t[mask], yerr=E1err_t[mask],
                          fmt=marker, color=color, ms=6, capsize=2,
                          markeredgecolor='black', markeredgewidth=0.4,
                          label=f'$a = {a_test:.2f}$')
    ax_R.set_xlabel('$f^2$');  ax_R.set_ylabel('$E_1(f^2)$')
    ax_R.set_title(r'Convergencia de $E_1$ con $a\to 0$  ($T = Na = 50$ fijo)')
    ax_R.legend(loc='upper left', fontsize=9)
    ax_R.grid(alpha=0.3)
    ax_R.set_ylim(0, 7.0)
    ax_R.invert_xaxis()

    plt.tight_layout()
    plt.savefig('fig_10_niveles_anarmonico.png')
    print("\n   guardado: fig_10_niveles_anarmonico.png")
    plt.close()

    # Tabla de resultados
    print("\n  Resultados numéricos (V=λ(x²-f²)², M₀=0.5):")
    print("  " + "─" * 62)
    print(f"  {'f²':>6} {'E₀ MC':>14} {'E₀ ref':>10} {'E₁ MC':>14} {'E₁ ref':>10}")
    print("  " + "─" * 62)
    for i, f2 in enumerate(F2_LIST):
        # interpolar la ref para este f²
        e0_r = interp1d(f2_ref, E0_ref, kind='cubic',
                        fill_value='extrapolate')(f2).item()
        e1_r = interp1d(f2_ref, E1_ref, kind='cubic',
                        fill_value='extrapolate')(f2).item()
        e1_str = (f"{E1_list[i]:8.3f}±{E1_err[i]:.3f}"
                  if not np.isnan(E1_list[i]) else "      ---     ")
        print(f"  {f2:>6.2f} {E0_list[i]:8.3f}±{E0_err[i]:.3f} "
              f"{e0_r:>10.3f} {e1_str} {e1_r:>10.3f}")

    print("\n" + "=" * 64)
    print("  COMPLETADO")
    print("=" * 64)


if __name__ == "__main__":
    main()
