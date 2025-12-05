import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
g = 9.81       # m/s^2
cp = 1005.0    # J/kg/K
Rd = 287.0     # J/kg/K
Rv = 461.0     # J/kg/K
L = 2.5e6      # J/kg
eps = 0.622    # Rd/Rv
T0 = 273.15    # K
p0 = 1e5       # Pa
epsilon_drag = 0.05   # 1/s
mu_entrain = 1e-4     # 1/m

# --- Thermodynamic helper functions ---

def sat_mix_ratio(T, p):
    """Saturation mixing ratio q_s(T,p)."""
    es = 611.2 * np.exp(17.67 * (T - T0) / (T - 29.65))   # Pa
    e = es * p / (p + 0.378 * es)
    return eps * e / (p - e)

def moist_lapse_rate(T, p):
    """Moist adiabatic lapse rate Γ_m(T,p) in K/m."""
    q_s = sat_mix_ratio(T, p)
    Gamma_d = g / cp
    num = 1.0 + L * q_s / (Rd * T)
    den = 1.0 + (L**2 * q_s * eps) / (cp * Rv * T**2)
    return Gamma_d * num / den

def env_T(z):
    """Environmental temperature profile T_e(z)."""
    z = np.asarray(z)
    T = np.empty_like(z, dtype=float)
    mask = z < 15000.0
    T[mask] = 301.0 - 6.5e-3 * z[mask]
    T[~mask] = 216.65
    return T

def env_q(z):
    """Environmental humidity profile q_e(z)."""
    return 0.018 * np.exp(-z / 2000.0)

# --- Simple 2D secondary circulation (front view) ---

def radial_inflow_velocity(x, z, Lx=200e3, Vmax=10.0):
    """
    Prescribed horizontal velocity u(x,z): inflow toward x=0 near the surface,
    decaying with height, zero at domain edges. Mimics hurricane boundary-layer
    inflow seen in 2D cross‑sections. [file:1][file:86]
    """
    # non-dimensional radius, limited to [-1,1]
    r_hat = np.clip(x / (Lx / 2.0), -1.0, 1.0)
    u0 = -Vmax * r_hat          # toward center
    return u0 * np.exp(-z / 5000.0)  # weaker inflow aloft

# --- Main simulation function ---

def simulate_air_particle(
        x0=100e3,      # initial x (m), >0 means right of center
        z0=50.0,       # initial height (m)
        T_init=302.0,  # initial parcel temperature (K)
        q_init=0.018,  # initial parcel humidity (kg/kg)
        t_max=3600.0,  # total time (s)
        dt=1.0,        # time step (s)
        Lx=200e3       # domain half-width for inflow profile
    ):
    """
    Simulate a single moist air parcel in 2D (x,z) with:
    - Horizontal motion from prescribed radial inflow u(x,z).
    - Vertical motion from buoyancy with linear drag.
    - Dry/moist thermodynamics + simple condensation + entrainment.
    Returns time array and trajectories for x, z, w, T_p, q_p, p_p.
    """

    N = int(t_max / dt) + 1
    t = np.linspace(0.0, t_max, N)

    # Allocate arrays
    x = np.zeros(N)
    z = np.zeros(N)
    w = np.zeros(N)
    T_p = np.zeros(N)
    q_p = np.zeros(N)
    p_p = np.zeros(N)

    # Initial conditions
    x[0] = x0
    z[0] = z0
    w[0] = 0.1
    T_p[0] = T_init
    q_p[0] = q_init
    p_p[0] = p0

    for i in range(1, N):
        # Environment / pressure
        p_env = p0 * np.exp(-z[i-1] / 8000.0)
        p_p[i] = p_env

        T_e = env_T(z[i-1])
        q_e = env_q(z[i-1])
        q_s = sat_mix_ratio(T_p[i-1], p_p[i-1])

        # Virtual temperatures and buoyancy
        Tv_p = T_p[i-1] * (1.0 + 0.608 * q_p[i-1])
        Tv_e = T_e * (1.0 + 0.608 * q_e)
        B = g * (Tv_p - Tv_e) / Tv_e

        # Vertical momentum with drag
        dwdt = B - epsilon_drag * w[i-1]
        w[i] = w[i-1] + dwdt * dt
        if w[i] < 0.0:
            w[i] = 0.0  # no descent in this simple toy

        # Horizontal velocity and position
        u = radial_inflow_velocity(x[i-1], z[i-1], Lx=Lx)
        x[i] = x[i-1] + u * dt

        # Vertical position
        z[i] = max(z[i-1] + w[i] * dt, 0.0)

        # Thermodynamics
        saturated = q_p[i-1] >= 0.99 * q_s
        if saturated:
            Gamma_m = moist_lapse_rate(T_p[i-1], p_p[i-1])
            dTdt = -Gamma_m * w[i]
            dqdt_cond = - (L / cp) * (q_p[i-1] - q_s) * w[i] / 8000.0
        else:
            dTdt = - (g / cp) * w[i]
            dqdt_cond = 0.0

        # Entrainment (mix with environment)
        dqdt_entr = mu_entrain * (q_e - q_p[i-1]) * abs(w[i]) / max(z[i], 100.0)

        # Update T and q
        T_p[i] = T_p[i-1] + dTdt * dt
        q_p[i] = q_p[i-1] + (dqdt_cond + dqdt_entr) * dt

    return t, x, z, w, T_p, q_p, p_p

# --- Example run and plots ---

t, x, z, w, T_p, q_p, p_p = simulate_air_particle()

# Front-view trajectory
plt.figure(figsize=(6, 6))
plt.plot(x/1000.0, z/1000.0, '-k')
plt.scatter(x[0]/1000.0, z[0]/1000.0, c='green', label='start')
plt.scatter(x[-1]/1000.0, z[-1]/1000.0, c='red', label='end')
plt.axvline(0.0, color='gray', lw=0.5)
plt.xlabel('x (km)')
plt.ylabel('z (km)')
plt.title('Air parcel trajectory (front view)')
plt.grid(True)
plt.legend()
plt.show()

