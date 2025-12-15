import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(26)

# ----------------- CONSTANTS -----------------
g = 9.81
cp = 1005.0
Rd = 287.0
Rv = 461.0
L = 2.5e6
eps = 0.622
T0 = 273.15
p0 = 1e5
epsilon_drag = 0.05
mu_entrain = 1e-4

temperatures = []
pressures = []
qs = []
Z = []
v_p_list = []
total_latent = []
Q = []
total_work = []
W = []
V_vert = []
V_hor = []

# ----------------- HURRICANE RADII (m) -----------------
r_eye = 20e3
r_ew = 25e3
r_rb = 600e3
r_max = 600e3
r_ew2 = 100e3  # second eyewall / rainband radius (100 km)

# ----------------- THERMODYNAMICS -----------------
def sat_mix_ratio(T, p):
    es = 1710 * np.exp((T - 288) / 16.5)
    e = es * p / (p + 0.378 * es)
    return eps * e / (p - e)

def moist_lapse_rate(T, p):
    r = sat_mix_ratio(T, p)
    numerator = 1 + (L * r) / (Rd * T)
    denominator = cp + (L**2 * r) / (Rv * T**2)
    return g * numerator / denominator

def env_T(z):
    z = np.asarray(z)
    T = np.empty_like(z, dtype=float)
    mask = z < 7000.0
    T[mask] = 300.0 - g/cp * z[mask]
    T[~mask] = 216.65
    return T

def env_q(z):
    return 0.15 * np.exp(-z / 8000)

def parcels_volume(initial_volume, initial_pressure, final_pressure):
    return initial_volume * (initial_pressure / final_pressure)

# ----------------- SECONDARY CIRCULATION -----------------
def inflow_radial_velocity(r, z, Vmax_in=15.0):
    fr = np.exp(-((r - 2.0*r_eye)/(0.7*r_max))**2)
    fr *= (r > r_eye)
    fz = np.exp(-z / 2000.0)
    return -Vmax_in * fr * fz

def outflow_radial_velocity(r, z, Vmax_out=10.0):
    fz = np.exp(-((z - 12000.0)/3000.0)**2)
    fr = np.exp(-(r/(0.6*r_max))**2)
    return Vmax_out * fr * fz

def eyewall_updraft_forcing(r, z, W0=1.0, sigma_r=5e3):
    fr = np.exp((-0.5*((r - r_ew)/sigma_r)**2)/2)
    fz = np.exp(-z/1000)
    return W0 * (fr + fz)

def eye_subsidence_forcing(r, z, W0=1.0):
    fr = np.exp(-0.5*(r/(0.6*r_eye))**2)
    fz = np.exp(-((z - 6000.0)/4000.0)**2)
    return -W0 * (fr + fz)

def outer_rainband_updraft_forcing(r, z, W0=0.5, sigma_r=10e3):
    """
    Weaker, broader updraft centered near r_ew2 to mimic a secondary rainband.
    """
    fr = np.exp(-0.5 * ((r - r_ew2)/sigma_r)**2)
    fz = np.exp(-z / 4000.0)
    return W0 * (fr + fz)

def simulate_hurricane_parcels(
    Npar=10000,
    t_max=18000.0,
    dt=10.0,
    plot_interval=10
):
    """
    Lagrangian parcel simulation with condensation and precipitation.
    Returns animation frames for visualization.
    """
    N = int(t_max / dt) + 1
    
    # Initialize parcels inside eyewall region for stronger updrafts
    r = np.random.uniform(r_ew+10e3, r_ew + 500e3, size=Npar)
    selected_index = np.argmin(r)
    z = np.full(Npar, 0.0)
    w = np.full(Npar, 0.1)
    T_p = np.full(Npar, 302.0) + np.random.normal(0, 0.3, size=Npar)
    q_p = np.full(Npar, 0.015)
    condensate = np.zeros(Npar)
    
    # Track when each parcel first reaches rain conditions
    rain_onset_time = np.full(Npar, np.inf)
    rain_onset_z = np.zeros(Npar)
    rain_onset_r = np.zeros(Npar)
    
    # Storage for animation frames
    frames = []
    
    rain_parcels_count = 0
    
    for n in range(1, N):
        t = n * dt
        
        # Environment
        p_env = p0 * np.exp(-z / 8000.0)
        p_p = p_env.copy()
        v_p = parcels_volume(1.0, p0, p_p)
        T_e = env_T(z)
        q_e = env_q(z)
        q_s = sat_mix_ratio(T_p, p_p)
        
        # Buoyancy
        Tv_p = T_p * (1.0 + 0.608 * q_p)
        Tv_e = T_e * (1.0 + 0.608 * q_e)
        B = g * (Tv_p - Tv_e) / Tv_e
        
        # IF A PARTICLE HAS REACHED 100 RADIUS:
        parcels_above = z >= 10000
        parcels_far = r >= 100000
        parcels_above_and_far = parcels_above * parcels_far
        
        if np.sum(parcels_above_and_far) >= 1:
            parcels_below_and_far = (1 - parcels_above) * parcels_far
            parcels_above_or_close = 1 - parcels_below_and_far
        else:
            parcels_below_and_far = np.zeros(Npar)
            parcels_above_or_close = np.ones(Npar)
        
        w_forcing = (eyewall_updraft_forcing(r, z) + 
                    eye_subsidence_forcing(r, z) + 
                    outer_rainband_updraft_forcing(r, z)) * parcels_below_and_far
        w_forcing += (eyewall_updraft_forcing(r, z) + 
                     eye_subsidence_forcing(r, z)) * parcels_above_or_close
        
        u_r = inflow_radial_velocity(r, z) + outflow_radial_velocity(r, z)
        
        # Thermodynamics & condensation
        saturated = q_p >= 0.99 * q_s
        dTdt = np.zeros_like(T_p)
        dqdt_cond = np.zeros_like(q_p)
        dqdt_entr = np.zeros_like(q_p)
        
        # Moist adiabatic cooling when saturated
        if np.any(saturated):
            Gamma_m = moist_lapse_rate(T_p[saturated], p_p[saturated])
            dTdt[saturated] = -Gamma_m * w[saturated]
            dq_excess = np.maximum(q_p[saturated] - q_s[saturated], 0.0)
            dqdt_cond[saturated] = -dq_excess * np.maximum(w[saturated], 0.0) / 8000.0
            condensate[saturated] += -dqdt_cond[saturated] * dt
        
        # Dry adiabatic cooling when unsaturated
        dry = ~saturated
        dTdt[dry] = - (g / cp) * w[dry]
        
        cond_rate = -dqdt_cond
        
        # Rain-induced downdrafts
        thresh = 1e-6
        heavy = cond_rate > thresh
        Wd = 3.0
        
        # Vertical momentum
        dwdt = B - epsilon_drag * w + w_forcing
        w = np.clip(w + dwdt * dt, -15.0, 50.0)
        
        # Position update
        r = np.clip(r + u_r * dt, 0.0, r_max)
        z = np.maximum(z + w * dt, 0.0)
        
        V_hor.append(u_r[selected_index])
        V_vert.append(w[selected_index])
        
        # Entrainment
        dqdt_entr = mu_entrain * (q_e - q_p) * np.abs(w) / np.maximum(z, 100.0)
        
        # Update thermodynamic variables
        T_p = T_p + dTdt * dt
        temperatures.append(T_p[selected_index])
        q_p = q_p + (dqdt_cond + dqdt_entr) * dt
        pressures.append(p_env[selected_index])
        qs.append(q_s[selected_index])
        Z.append(z[selected_index])
        v_p_list.append(v_p[selected_index])
        
        # ===== RAIN TRACKING =====
        raining = saturated & (cond_rate > 1e-7)
        newly_raining = raining & (rain_onset_time == np.inf)
        
        if np.any(newly_raining):
            indices = np.where(newly_raining)[0]
            for idx in indices:
                rain_onset_time[idx] = t
                rain_onset_z[idx] = z[idx]
                rain_onset_r[idx] = r[idx]
        
        rain_parcels_count = np.sum(rain_onset_time != np.inf)
        
        # Store frame data every plot_interval timesteps
        if n % plot_interval == 0:
            raining_now = saturated & (cond_rate > 1e-7)
            frame_data = {
                'r': r.copy(),
                'z': z.copy(),
                'condensate': condensate.copy(),
                'raining_now': raining_now.copy(),
                't': t,
                'rain_parcels_count': rain_parcels_count
            }
            frames.append(frame_data)
        
        if np.any(saturated):
            # Step 1: Latent heat released (J/kg of air)
            latent_heat_released = -dqdt_cond[saturated] * L * dt
            
            # Step 2: Temperature of hot and cold reservoirs
            T_hot_local = env_T(0)
            T_cold_local = env_T(8000)
            
            # Step 3: Carnot efficiency
            eta_carnot = 1.0 - T_cold_local / np.maximum(T_hot_local, 1.0)
            eta_carnot = np.clip(eta_carnot, 0, 1)
            
            # Step 4: Maximum work available
            work_available = latent_heat_released * eta_carnot
            
            # Step 5: Actual work done
            total_latent.append(latent_heat_released)
            total_work.append(work_available)
            Q.append(np.concatenate(total_latent).sum())
            W.append(np.concatenate(total_work).sum())
        
        if n == N-1:
            hurricanes_latent = np.concatenate(total_latent).sum()
            hurricanes_work = np.concatenate(total_work).sum()
            print(f"Total latent heat: {hurricanes_latent:.2e} J/kg")
            print(f"Total work: {hurricanes_work:.2e} J/kg")
    
    return frames

def create_animation(frames, save_filename=None, vmin=None, vmax=None):
    """
    Create an animated plot from the simulation frames.
    
    Parameters:
    -----------
    frames : list
        List of frame dictionaries from simulate_hurricane_parcels
    save_filename : str, optional
        If provided, save animation to this filename (e.g., 'hurricane.mp4' or 'hurricane.gif')
    vmin : float, optional
        Minimum value for colormap scale. If None, uses 0.
    vmax : float, optional
        Maximum value for colormap scale. If None, auto-scales to data max.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determine colormap limits
    if vmax is None:
        # Calculate max condensate across all frames
        vmax = max(frame['condensate'].max() for frame in frames)
    if vmin is None:
        vmin = 0
    
    print(f"Colormap range: {vmin:.2e} to {vmax:.2e}")
    
    # Initialize plot elements
    scatter = ax.scatter([], [], c=[], cmap='tab10', s=6, alpha=0.7, 
                        vmin=vmin, vmax=vmax)
    rain_scatter = ax.scatter([], [], marker='x', c='yellowgreen', s=30, 
                             linewidth=1.5, label='Raining')
    
    ax.axvline(r_eye/1000.0, color='gray', ls='--', label='eye', linewidth=2)
    ax.axvline(r_ew/1000.0, color='tomato', ls='--', label='eyewall', linewidth=2)
    ax.axvline(r_ew2/1000.0, color='green', ls='--', label='rain band', linewidth=2)
    
    ax.set_xlabel('radius r (km)', fontsize=11)
    ax.set_ylabel('z (km)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim((0, 18))
    ax.set_xlim((0, 150))
    
    title = ax.set_title('', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='total condensate (kg/kg)')
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        rain_scatter.set_offsets(np.empty((0, 2)))
        return scatter, rain_scatter, title
    
    def update(frame_idx):
        frame = frames[frame_idx]
        
        # Update main scatter plot
        positions = np.column_stack((frame['r']/1000.0, frame['z']/1000.0))
        scatter.set_offsets(positions)
        scatter.set_array(frame['condensate'])
        
        # Update raining parcels
        raining_idx = frame['raining_now']
        if np.any(raining_idx):
            rain_positions = np.column_stack((
                frame['r'][raining_idx]/1000.0, 
                frame['z'][raining_idx]/1000.0
            ))
            rain_scatter.set_offsets(rain_positions)
        else:
            rain_scatter.set_offsets(np.empty((0, 2)))
        
        # Update title
        title.set_text(
            f"t = {frame['t']/60:.1f} min"
        )
        
        return scatter, rain_scatter, title
    
    anim = FuncAnimation(
        fig, update, init_func=init, frames=len(frames),
        interval=50, blit=True, repeat=True
    )
    
    if save_filename:
        print(f"Saving animation to {save_filename}...")
        anim.save(save_filename, writer='pillow', fps=20)
        print("Animation saved!")
    
    plt.tight_layout()
    plt.show()
    
    return anim

# Run simulation and create animation
print("Running simulation...")
frames = simulate_hurricane_parcels(Npar=6000, t_max=18000.0, dt=10.0, plot_interval=10)

print(f"Creating animation with {len(frames)} frames...")
#anim = create_animation(frames)

# For better visualization, you can set custom colormap limits:
anim = create_animation(frames, vmin=0, vmax=5e-3)  # Focus on lower values
# anim = create_animation(frames, vmin=0, vmax=5e-4)  # Even tighter range

# To save the animation, uncomment one of these lines:
anim.save('hurricane_animation.gif', writer='pillow', fps=20)
#anim.save('hurricane_animation.mp4', writer='pillow', fps=20)