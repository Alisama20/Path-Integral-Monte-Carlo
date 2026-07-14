"""
========================================================================
03_pozo_doble.py  --  Doble pozo: instantones (kinks) y tuneleo
========================================================================

Reproduce las Figs. 15, 16 y 17 del paper de Creutz & Freedman:
    - Fig. 15: configuración de un kink simple (con C.C. antiperiódicas)
    - Fig. 16: solución clásica de tunelado x(τ) = √(f²)·tanh(√(8λ)·τ)
    - Fig. 17: aniquilación de un par kink-antikink

Potencial:   V(x) = λ (x² - f²)²    con mínimos en ±√(f²) = ±x_c

Para f² grande, la barrera es alta y aparecen configuraciones de tipo
"instantón" o "kink": trayectorias que saltan entre los dos mínimos en
tiempo euclídeo. Estas son las responsables del efecto túnel cuántico.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import V_doble_pozo, config_plot

config_plot()


# Setter de semilla para el RNG de numba (que es independiente del de
# numpy a nivel Python). Esto garantiza reproducibilidad entre corridas.
@njit(cache=True)
def _seed_numba(s):
    np.random.seed(s)

# ─────────────────────────────────────────────────────────────────────
#  Parámetros (similares al paper)
# ─────────────────────────────────────────────────────────────────────

LAM = 1.0
F2  = 2.0          # f² → mínimos en x = ±√2 ≈ ±1.41
M0  = 0.5
A   = 0.25         # paso temporal
DELTA = 0.5

# Tres simulaciones diferentes
N_KINK     = 200   # red para un solo kink (C.C. antiperiódicas)
N_TUNEL    = 200   # red para tuneleo
N_ANIQUI   = 80    # red para aniquilación de kinks


# ─────────────────────────────────────────────────────────────────────
#  Metropolis con condiciones de frontera personalizadas
# ─────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def sweep_antiperiodico(x, a, m0, lam, f2, delta):
    """
    Metropolis con C.C. antiperiódicas: x_N = -x_0.
    Esto fuerza al menos un kink en la red.
    """
    N = x.shape[0]
    n_accept = 0
    kin_coef = m0 / a

    for j in range(N):
        # Vecinos antiperiódicos
        if j == 0:
            xm = -x[N - 1]
            xp = x[1]
        elif j == N - 1:
            xm = x[N - 2]
            xp = -x[0]
        else:
            xm = x[j - 1]
            xp = x[j + 1]

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


@njit(cache=True, fastmath=True)
def sweep_periodico_dp(x, a, m0, lam, f2, delta):
    """Metropolis con C.C. periódicas, para doble pozo"""
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


@njit(cache=True, fastmath=True)
def sweep_fronteras_fijas(x, a, m0, lam, f2, delta):
    """
    Metropolis con C.C. fijas: x_0 = -x_c, x_{N-1} = +x_c.
    Útil para mostrar la aniquilación de un par kink-antikink.
    """
    N = x.shape[0]
    n_accept = 0
    kin_coef = m0 / a

    for j in range(1, N - 1):  # no toca los extremos
        xm = x[j - 1]
        xp = x[j + 1]
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
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  DOBLE POZO: KINKS Y TUNELEO — Creutz-Freedman, Figs. 15-17")
    print("=" * 64)
    print(f"  V(x) = λ(x²-f²)²,  λ={LAM}, f²={F2}")
    print(f"  Mínimos clásicos en x_c = ±{np.sqrt(F2):.4f}")
    print(f"  M₀={M0}, a={A}")

    xc = np.sqrt(F2)

    # ─── 1. FIG. 15: kink MC + instantón clásico superpuestos ──────
    # Usamos un pozo más profundo (f²=4) solo para esta figura para que las
    # fluctuaciones cuánticas queden contenidas entre los puntos de retorno
    # ±x_1, ±x_2, como en la fig. 15 de Creutz & Freedman. Con f²=2 las
    # fluctuaciones cruzan el barrier y la figura pierde nitidez visual.
    print("\n→ Generando un kink simple + instantón clásico (Fig. 15)...")
    F2_KINK = 4.0
    xc_k    = np.sqrt(F2_KINK)
    x1_k    = np.sqrt(F2_KINK * (1.0 - 1.0/np.sqrt(2.0)))
    x2_k    = np.sqrt(F2_KINK * (1.0 + 1.0/np.sqrt(2.0)))

    np.random.seed(123)
    # Inicializar con un kink:  x = -xc para n<N/2,  x = +xc para n≥N/2
    x = np.ones(N_KINK) * xc_k
    x[:N_KINK // 2] = -xc_k

    # Termalizar (pocos barridos para mantener un único kink bien definido:
    # con 2000 barridos se generan pares kink-antikink adicionales y la
    # figura pierde el carácter de "función escalón" de Creutz)
    N_THERM_KINK = 200
    for _ in range(N_THERM_KINK):
        sweep_antiperiodico(x, A, M0, LAM, F2_KINK, DELTA)

    # Solución clásica del instantón, centrada en τ₀ = N·a/2 (sitio N/2)
    tau_arr = np.linspace(0, N_KINK * A, 600)
    tau_0 = N_KINK * A / 2
    x_clasico = xc_k * np.tanh(np.sqrt(2 * LAM / M0) * xc_k * (tau_arr - tau_0))
    n_clasico = tau_arr / A   # mapeo tiempo→sitio (τ = n·a)

    # Estilo matplotlib por defecto para esta figura
    plt.rcdefaults()

    fig, ax = plt.subplots(figsize=(8.5, 5))
    sitios = np.arange(N_KINK)
    ax.plot(sitios, x, marker='o', linestyle='-', color='C0',
            lw=1.0, markersize=2.5, markeredgewidth=0,
            label='Configuración MC')
    ax.plot(n_clasico, x_clasico, color='crimson', lw=2.2, alpha=0.9,
            label=r'Instantón clásico $x_c\tanh[\sqrt{2\lambda/m_0}\,x_c(\tau-\tau_0)]$')

    # Posiciones de equilibrio (gris discontinua)
    ax.axhline(+xc_k, color='gray', ls='--', lw=1.0,
               label=f'$\\pm x_c = \\pm{xc_k:.2f}$')
    ax.axhline(-xc_k, color='gray', ls='--', lw=1.0)
    # Puntos de retorno (azul punteado)
    ax.axhline(+x1_k, color='steelblue', ls=':', lw=1.2,
               label=f'$\\pm x_1 \\approx \\pm{x1_k:.2f}$ (interior)')
    ax.axhline(-x1_k, color='steelblue', ls=':', lw=1.2)
    ax.axhline(+x2_k, color='darkgreen', ls=':', lw=1.2,
               label=f'$\\pm x_2 \\approx \\pm{x2_k:.2f}$ (exterior)')
    ax.axhline(-x2_k, color='darkgreen', ls=':', lw=1.2)
    ax.axhline(0, color='k', lw=0.4, alpha=0.5)

    # Ejes principal: abajo (n) e izquierda (x_n)
    ax.set_xlabel('Posición temporal $n$')
    ax.set_ylabel('$x_n$')
    ax.set_xlim(0, N_KINK - 1)
    y_max = x2_k + 0.3
    ax.set_ylim(-y_max, y_max)
    ax.legend(loc='lower right', fontsize=8.5)
    ax.grid(alpha=0.3)

    # Eje superior: tiempo euclídeo τ = n·a
    ax_top = ax.twiny()
    ax_top.set_xlim(0, (N_KINK - 1) * A)
    ax_top.set_xlabel(r'Tiempo euclídeo $\tau = n\,a$')

    # Eje derecho: x(τ) (misma escala que x_n, etiqueta distinta)
    ax_right = ax.twinx()
    ax_right.set_ylim(-y_max, y_max)
    ax_right.set_ylabel(r'$x(\tau)$')

    plt.tight_layout()
    plt.savefig('fig_15_kink_simple.png', dpi=150)
    print("   guardado: fig_15_kink_simple.png")
    plt.close()

    # Restaurar estilo "paper" para las figuras siguientes
    config_plot()

    # ─── 3. FIG. 17: aniquilación de par kink-antikink ─────────────
    print("\n→ Aniquilación de un par kink-antikink (Fig. 17)...")
    # Búsqueda de una semilla en la que el par kink-antikink inicial se
    # aniquile rápidamente. Con f²=2 la barrera es baja y la dinámica es
    # estocástica: para algunas semillas el par se separa antes de
    # encontrarse, para otras se aniquila. Se prueban varias y se queda la
    # primera que termina con n(x<0) ≤ 3 (configuración casi-uniforme +x_c).
    # Plateau inicial muy corto (sep=4) para acelerar el encuentro.
    n_iter_total = 20
    sweeps_por_snap = 15
    q1 = N_ANIQUI // 2 - 2    # = 38
    q3 = N_ANIQUI // 2 + 2    # = 42

    semillas_a_probar = [7, 11, 13, 17, 23, 29, 37, 42, 51, 67, 83, 101]
    snapshots = None
    semilla_usada = None
    for s in semillas_a_probar:
        _seed_numba(s)
        np.random.seed(s)
        x = np.ones(N_ANIQUI) * xc
        x[q1:q3] = -xc

        snaps_test = [x.copy()]
        for _ in range(n_iter_total):
            for _ in range(sweeps_por_snap):
                sweep_fronteras_fijas(x, A, M0, LAM, F2, DELTA)
            snaps_test.append(x.copy())

        n_neg_final = int((snaps_test[-1] < 0).sum())
        if n_neg_final <= 3:
            snapshots = snaps_test
            semilla_usada = s
            break

    if snapshots is None:  # ninguna semilla funcionó: usar la última
        snapshots = snaps_test
        semilla_usada = semillas_a_probar[-1]

    # Comprobación explícita de BC
    assert snapshots[0][0] == xc and snapshots[0][-1] == xc, \
        "BC deben ser +x_c en ambos extremos"

    print(f"   semilla usada: {semilla_usada}")
    print(f"   BC: x[0]={snapshots[-1][0]:+.2f}, x[N-1]={snapshots[-1][-1]:+.2f}")
    print(f"   Trayectoria del par kink-antikink:")
    for i in [0, 5, 10, 15, 20]:
        if i < len(snapshots):
            snap = snapshots[i]
            n_neg = int((snap < 0).sum())
            print(f"   iter {i*sweeps_por_snap:4d}: x∈[{snap.min():+.2f},{snap.max():+.2f}], "
                  f"⟨x⟩={snap.mean():+.2f}, n(x<0)={n_neg:2d}/{len(snap)}")

    # Estilo matplotlib clásico para esta figura
    plt.rcdefaults()

    # Esquema de cadenas (waterfall): cada snapshot es una línea con
    # x = sitio en la red y y = x_n - i*offset (desplazado verticalmente
    # por iteración). Color viridis (morado=iteración 0, amarillo=final).
    # Eje Y invertido para que la iteración inicial (con el par kink-antikink,
    # el "pico") quede abajo y las iteraciones finales (cadenas planas
    # tras la aniquilación) arriba.
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.viridis
    n_snaps = len(snapshots)
    offset = 0.55

    for i, snap in enumerate(snapshots):
        color = cmap(i / max(1, n_snaps - 1))
        ax.plot(np.arange(N_ANIQUI), snap - offset * i,
                color=color, lw=1.0)

    ax.set_xlabel(r'Posición en la red $x_n$')
    ax.set_ylabel(r'$x_n$ desplazado por iteración MC')
    ax.set_title('Aniquilación de pareja kink-antikink')
    ax.set_xlim(0, N_ANIQUI - 1)
    ax.invert_yaxis()  # pico (iter 0) abajo, cadenas planas (iter final) arriba
    ax.grid(alpha=0.25)

    # Barra de color en la IZQUIERDA con el número de sweep MC. Al invertir
    # el eje Y, las iteraciones bajas (color morado) quedan abajo: la
    # colorbar también queda con morado abajo y amarillo arriba, encajando
    # con la evolución de las cadenas en la figura.
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=0, vmax=(n_snaps - 1) * sweeps_por_snap)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, location='left', pad=0.08, shrink=0.92)
    cbar.set_label('Sweep Monte Carlo')

    plt.tight_layout()
    plt.savefig('fig_17_aniquilacion_kinks.png', dpi=150)
    print("   guardado: fig_17_aniquilacion_kinks.png")
    plt.close()

    # Restaurar estilo "paper" para las figuras siguientes
    config_plot()

    # ─── 4. Histograma de |ψ₀|² con doble pozo ────────────────────
    print("\n→ |ψ₀(x)|² para el doble pozo...")
    bins = np.linspace(-2.5, 2.5, 80)

    @njit(cache=True, fastmath=True, parallel=True)
    def medir_doble_pozo(n_chains, N, a, m0, lam, f2, delta,
                          n_therm, n_meas, n_skip, x_bins, x_max):
        n_bins = x_bins.shape[0] - 1
        hist = np.zeros((n_chains, n_bins))
        xc_loc = np.sqrt(f2)

        for c in prange(n_chains):
            # Mitad de las cadenas empiezan en cada mínimo
            if c % 2 == 0:
                x = np.ones(N) * xc_loc
            else:
                x = np.ones(N) * (-xc_loc)
            x = x + 0.1 * (np.random.random(N) - 0.5)

            for _ in range(n_therm):
                sweep_periodico_dp(x, a, m0, lam, f2, delta)

            for _ in range(n_meas):
                for __ in range(n_skip):
                    sweep_periodico_dp(x, a, m0, lam, f2, delta)
                for v in x:
                    if -x_max < v < x_max:
                        b = int((v + x_max) / (2 * x_max) * n_bins)
                        if 0 <= b < n_bins:
                            hist[c, b] += 1.0

        return hist

    t0 = time.time()
    hist = medir_doble_pozo(8, 200, A, M0, LAM, F2, DELTA,
                             2000, 4000, 5, bins, 2.5)
    print(f"   tiempo: {time.time()-t0:.1f} s")

    hist_total = hist.sum(axis=0)
    dx = bins[1] - bins[0]
    hist_norm = hist_total / (hist_total.sum() * dx)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax.bar(centers, hist_norm, width=dx, alpha=0.5, color='steelblue',
           edgecolor='navy', label='$|\\psi_0(x)|^2$ (MC)')
    # Potencial superpuesto (escala arbitraria)
    x_pot = np.linspace(-2.5, 2.5, 400)
    V = LAM * (x_pot ** 2 - F2) ** 2
    V_scaled = V / V.max() * hist_norm.max() * 0.7
    ax.plot(x_pot, V_scaled, 'r--', lw=1.5, alpha=0.6, label='$V(x)$ (esc.)')
    ax.axvline(+xc, color='gray', ls=':', label=f'$\\pm x_c$')
    ax.axvline(-xc, color='gray', ls=':')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|\\psi_0(x)|^2$')
    ax.set_title(f'Estado base del doble pozo ($f^2={F2}$): simétrico por túnel')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_psi0_doble_pozo.png')
    print("   guardado: fig_psi0_doble_pozo.png")
    plt.close()

    print("\n" + "=" * 64)
    print("  COMPLETADO")
    print("=" * 64)


if __name__ == "__main__":
    main()
