"""
========================================================================
utils.py  --  Funciones comunes para reproducir Creutz & Freedman (1981)
========================================================================

Mecánica cuántica en la red por Monte Carlo (algoritmo Metropolis) en
tiempo euclídeo.

Convenciones:
    - Tiempo euclídeo, ħ = 1
    - Acción: S_E = Σᵢ a · [M₀/2 · ((x_{i+1}-x_i)/a)² + V(x_i)]
    - Condiciones de frontera periódicas: x_N = x_0
    - Peso de Boltzmann: exp(-S_E)

El cambio ΔS al mover x_j → x_j' es completamente local:
    ΔS = (M₀/2a)·[(x_j'-x_{j-1})² + (x_{j+1}-x_j')² - los anteriores]
         + a·[V(x_j') - V(x_j)]
"""

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────
#  Potenciales (compilados con numba)
# ─────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True, inline='always')
def V_armonico(x, mu2):
    """Oscilador armónico: V(x) = (1/2) μ² x²"""
    return 0.5 * mu2 * x * x


@njit(cache=True, fastmath=True, inline='always')
def V_anarmonico(x, mu2, lam):
    """Anarmónico: V(x) = (1/2) μ² x² + λ x⁴"""
    x2 = x * x
    return 0.5 * mu2 * x2 + lam * x2 * x2


@njit(cache=True, fastmath=True, inline='always')
def V_doble_pozo(x, lam, f2):
    """Doble pozo: V(x) = λ (x² - f²)²   con mínimos en ±√(f²)"""
    d = x * x - f2
    return lam * d * d


# ─────────────────────────────────────────────────────────────────────
#  Sweep de Metropolis (núcleo computacional)
# ─────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def metropolis_sweep_armonico(x, a, m0, mu2, delta):
    """
    Un sweep completo del Metropolis sobre la red.
    Recorre los N sitios y propone x_j → x_j + δ·U(-1,1).
    Retorna el número de cambios aceptados.
    """
    N = x.shape[0]
    n_accept = 0
    inv_a = 1.0 / a
    kin_coef = m0 * inv_a  # = M₀/a (factor que aparece en ΔS_kin)

    for j in range(N):
        # Vecinos con condiciones periódicas
        jm = (j - 1) % N
        jp = (j + 1) % N
        xm = x[jm]
        xp = x[jp]
        x_old = x[j]

        # Propuesta
        x_new = x_old + delta * (2.0 * np.random.random() - 1.0)

        # ΔS = ΔS_kin + ΔS_pot
        # ΔS_kin = (M₀/2a)·[(x'-xm)² + (xp-x')² - (x-xm)² - (xp-x)²]
        #        = (M₀/a)·[(x'-x)(xm+xp) ...]
        # Pero más limpio computarlo directo:
        dS_kin = 0.5 * kin_coef * (
            (x_new - xm) * (x_new - xm)
            + (xp - x_new) * (xp - x_new)
            - (x_old - xm) * (x_old - xm)
            - (xp - x_old) * (xp - x_old)
        )
        dS_pot = a * (V_armonico(x_new, mu2) - V_armonico(x_old, mu2))
        dS = dS_kin + dS_pot

        # Criterio de Metropolis
        if dS <= 0.0 or np.random.random() < np.exp(-dS):
            x[j] = x_new
            n_accept += 1

    return n_accept


@njit(cache=True, fastmath=True)
def metropolis_sweep_anarmonico(x, a, m0, mu2, lam, delta):
    """Sweep para potencial anarmónico V = (1/2)μ²x² + λx⁴"""
    N = x.shape[0]
    n_accept = 0
    inv_a = 1.0 / a
    kin_coef = m0 * inv_a

    for j in range(N):
        jm = (j - 1) % N
        jp = (j + 1) % N
        xm = x[jm]
        xp = x[jp]
        x_old = x[j]

        x_new = x_old + delta * (2.0 * np.random.random() - 1.0)

        dS_kin = 0.5 * kin_coef * (
            (x_new - xm) * (x_new - xm)
            + (xp - x_new) * (xp - x_new)
            - (x_old - xm) * (x_old - xm)
            - (xp - x_old) * (xp - x_old)
        )
        dS_pot = a * (V_anarmonico(x_new, mu2, lam) - V_anarmonico(x_old, mu2, lam))
        dS = dS_kin + dS_pot

        if dS <= 0.0 or np.random.random() < np.exp(-dS):
            x[j] = x_new
            n_accept += 1

    return n_accept


@njit(cache=True, fastmath=True)
def metropolis_sweep_doble_pozo(x, a, m0, lam, f2, delta, J=0.0):
    """
    Sweep para doble pozo V = λ(x²-f²)² con fuente externa.
    Convención del paper: S' = S + J·Σxᵢ
    → Para minimizar V'(x) = -J, es decir, ⟨x⟩ = -J/(...).
    """
    N = x.shape[0]
    n_accept = 0
    inv_a = 1.0 / a
    kin_coef = m0 * inv_a

    for j in range(N):
        jm = (j - 1) % N
        jp = (j + 1) % N
        xm = x[jm]
        xp = x[jp]
        x_old = x[j]

        x_new = x_old + delta * (2.0 * np.random.random() - 1.0)

        dS_kin = 0.5 * kin_coef * (
            (x_new - xm) * (x_new - xm)
            + (xp - x_new) * (xp - x_new)
            - (x_old - xm) * (x_old - xm)
            - (xp - x_old) * (xp - x_old)
        )
        dV = V_doble_pozo(x_new, lam, f2) - V_doble_pozo(x_old, lam, f2)
        dS_pot = a * (dV + J * (x_new - x_old))  # +J·x convention
        dS = dS_kin + dS_pot

        if dS <= 0.0 or np.random.random() < np.exp(-dS):
            x[j] = x_new
            n_accept += 1

    return n_accept


@njit(cache=True, fastmath=True)
def metropolis_sweep_armonico_J(x, a, m0, mu2, delta, J):
    """Sweep para oscilador armónico con fuente externa.
    Convención del paper: S' = S + J·Σxᵢ → J = -μ²·⟨x⟩
    """
    N = x.shape[0]
    n_accept = 0
    inv_a = 1.0 / a
    kin_coef = m0 * inv_a

    for j in range(N):
        jm = (j - 1) % N
        jp = (j + 1) % N
        xm = x[jm]
        xp = x[jp]
        x_old = x[j]

        x_new = x_old + delta * (2.0 * np.random.random() - 1.0)

        dS_kin = 0.5 * kin_coef * (
            (x_new - xm) * (x_new - xm)
            + (xp - x_new) * (xp - x_new)
            - (x_old - xm) * (x_old - xm)
            - (xp - x_old) * (xp - x_old)
        )
        dS_pot = a * (V_armonico(x_new, mu2) - V_armonico(x_old, mu2)
                      + J * (x_new - x_old))  # +J·x convention
        dS = dS_kin + dS_pot

        if dS <= 0.0 or np.random.random() < np.exp(-dS):
            x[j] = x_new
            n_accept += 1

    return n_accept


# ─────────────────────────────────────────────────────────────────────
#  Observables
# ─────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def medir_x2(x):
    """⟨x²⟩ sobre la red"""
    s = 0.0
    for v in x:
        s += v * v
    return s / x.shape[0]


@njit(cache=True, fastmath=True)
def medir_x4(x):
    """⟨x⁴⟩ sobre la red"""
    s = 0.0
    for v in x:
        v2 = v * v
        s += v2 * v2
    return s / x.shape[0]


@njit(cache=True, fastmath=True)
def medir_x(x):
    """⟨x⟩ sobre la red"""
    s = 0.0
    for v in x:
        s += v
    return s / x.shape[0]


@njit(cache=True, fastmath=True)
def correlador_2puntos(x, t_max):
    """
    Función de correlación de dos puntos: G(τ) = ⟨x(t)·x(t+τ)⟩
    para τ = 0, 1, ..., t_max-1
    """
    N = x.shape[0]
    G = np.zeros(t_max)
    for tau in range(t_max):
        s = 0.0
        for i in range(N):
            s += x[i] * x[(i + tau) % N]
        G[tau] = s / N
    return G


# ─────────────────────────────────────────────────────────────────────
#  Runners de simulación (con paralelismo entre cadenas)
# ─────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True, parallel=True)
def run_chains_armonico(n_chains, N, a, m0, mu2, delta,
                         n_therm, n_meas, n_skip, x_bins, x_max):
    """
    Ejecuta n_chains cadenas independientes en paralelo.
    Devuelve:
        histograma global de x (para |ψ₀|²)
        ⟨x²⟩ promedio por cadena
        ⟨x⁴⟩ promedio por cadena
        tasa de aceptación promedio
    """
    n_bins = x_bins.shape[0] - 1
    hist_total = np.zeros((n_chains, n_bins), dtype=np.float64)
    x2_chain = np.zeros(n_chains)
    x4_chain = np.zeros(n_chains)
    accept_chain = np.zeros(n_chains)

    for c in prange(n_chains):
        # Configuración inicial aleatoria
        x = (np.random.random(N) - 0.5) * 2.0

        # Termalización
        for _ in range(n_therm):
            metropolis_sweep_armonico(x, a, m0, mu2, delta)

        # Medidas
        n_acc_tot = 0
        for m in range(n_meas):
            for _ in range(n_skip):
                n_acc_tot += metropolis_sweep_armonico(x, a, m0, mu2, delta)
            x2_chain[c] += medir_x2(x)
            x4_chain[c] += medir_x4(x)
            # Histograma
            for v in x:
                if -x_max < v < x_max:
                    b = int((v + x_max) / (2 * x_max) * n_bins)
                    if 0 <= b < n_bins:
                        hist_total[c, b] += 1.0

        x2_chain[c] /= n_meas
        x4_chain[c] /= n_meas
        accept_chain[c] = n_acc_tot / (n_meas * n_skip * N)

    return hist_total, x2_chain, x4_chain, accept_chain


@njit(cache=True, fastmath=True, parallel=True)
def run_chains_correlador(n_chains, N, a, m0, mu2, delta,
                           n_therm, n_meas, n_skip, t_max):
    """Ejecuta cadenas y mide correlador <x(0)x(t)>"""
    G_chains = np.zeros((n_chains, t_max))
    accept_chain = np.zeros(n_chains)

    for c in prange(n_chains):
        x = (np.random.random(N) - 0.5) * 2.0

        for _ in range(n_therm):
            metropolis_sweep_armonico(x, a, m0, mu2, delta)

        n_acc_tot = 0
        for m in range(n_meas):
            for _ in range(n_skip):
                n_acc_tot += metropolis_sweep_armonico(x, a, m0, mu2, delta)
            G_chains[c] += correlador_2puntos(x, t_max)

        G_chains[c] /= n_meas
        accept_chain[c] = n_acc_tot / (n_meas * n_skip * N)

    return G_chains, accept_chain


# ─────────────────────────────────────────────────────────────────────
#  Configuración de matplotlib estilo "paper"
# ─────────────────────────────────────────────────────────────────────

def config_plot():
    """Estilo de gráficas similar al paper original"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.linewidth': 0.8,
        'axes.labelsize': 12,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'figure.dpi': 110,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })


# ─────────────────────────────────────────────────────────────────────
#  Información del módulo al ejecutar directamente
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(__doc__)
    print("\n→ Este módulo provee funciones. Ejecuta los scripts 01–05 para")
    print("  reproducir las figuras del paper.")
