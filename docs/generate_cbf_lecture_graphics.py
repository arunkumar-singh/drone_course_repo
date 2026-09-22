import matplotlib.pyplot as plt
import numpy as np
import os

# Set global styles
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

out_dir = "/home/aks1/drone_course_repo/docs"

# =========================================================================
# Figure 1: Safe Set & Control Barrier Function Concept
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# Background safe domain
ax.set_facecolor('#f4f8fb')

# Obstacle and buffer
p_obs = np.array([3.0, 3.0])
r_obs = 1.2
D_obs = 1.8

# Danger zone (obstacle)
circle_obs = plt.Circle(p_obs, r_obs, color='#e63946', alpha=0.85, label='Physical Obstacle')
ax.add_patch(circle_obs)

# Buffer boundary (h = 0)
circle_buffer = plt.Circle(p_obs, D_obs, color='#f4a261', fill=False, lw=2.5, linestyle='--', label=r'Safety Boundary $\partial\mathcal{S}$ ($h(\mathbf{p})=0$)')
ax.add_patch(circle_buffer)

# Shaded unsafe region
circle_buffer_fill = plt.Circle(p_obs, D_obs, color='#f4a261', alpha=0.2, label=r'Unsafe Set ($h(\mathbf{p}) < 0$)')
ax.add_patch(circle_buffer_fill)

# Drone position
p_drone = np.array([1.2, 4.2])
ax.plot(p_drone[0], p_drone[1], 'o', color='#1d3557', markersize=12, label=r'Drone Position $\mathbf{p}$')

# Vector from obstacle to drone (gradient normal)
diff = p_drone - p_obs
dist = np.linalg.norm(diff)
normal = diff / dist

# Draw line from obstacle center to drone
ax.plot([p_obs[0], p_drone[0]], [p_obs[1], p_drone[1]], 'k:', lw=1.5)
ax.text(2.0, 3.4, r'$\|\mathbf{p} - \mathbf{p}_{\mathrm{obs}}\|$', fontsize=12, color='#333333', rotation=34)

# Normal vector at drone
ax.quiver(p_drone[0], p_drone[1], normal[0], normal[1], color='#2a9d8f', angles='xy', scale_units='xy', scale=1.2, width=0.012, label=r'Gradient $\nabla h(\mathbf{p})$ (Outward Normal)')

# Allowed half-space line tangent to barrier
tangent = np.array([-normal[1], normal[0]])
line_t = np.linspace(-1.5, 1.5, 50)
halfspace_x = p_drone[0] + line_t * tangent[0]
halfspace_y = p_drone[1] + line_t * tangent[1]
ax.plot(halfspace_x, halfspace_y, color='#2a9d8f', lw=1.8, linestyle='-', label=r'Boundary Plane $\nabla h^T \mathbf{v} = -\alpha h$')

# Annotate sets
ax.text(4.5, 5.0, r'$\mathbf{Safe\ Set\ }\mathcal{S}$' + '\n' + r'$h(\mathbf{p}) = \|\mathbf{p} - \mathbf{p}_{\mathrm{obs}}\| - D_{\mathrm{obs}} \geq 0$',
        fontsize=13, fontweight='bold', color='#1d3557', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#1d3557'))

ax.text(p_obs[0], p_obs[1], r'$\mathbf{p}_{\mathrm{obs}}$', fontsize=14, color='white', ha='center', va='center', fontweight='bold')
ax.text(p_drone[0]-0.3, p_drone[1]+0.3, r'$\mathbf{p}$ (Drone)', fontsize=13, color='#1d3557', fontweight='bold')

# Allowed velocity vector examples
v_safe = np.array([0.8, 1.2]) # points away / tangent
ax.quiver(p_drone[0], p_drone[1], v_safe[0], v_safe[1], color='#2a9d8f', angles='xy', scale_units='xy', scale=1.5, width=0.01, label=r'Permitted Velocity $\mathbf{v}$ ($\dot{h} \geq -\alpha h$)')

v_unsafe = np.array([1.2, -1.0]) # points directly towards obstacle
ax.quiver(p_drone[0], p_drone[1], v_unsafe[0], v_unsafe[1], color='#e63946', angles='xy', scale_units='xy', scale=1.5, width=0.01, label=r'Prohibited Velocity ($\dot{h} < -\alpha h$)')


ax.set_xlim(-0.5, 6.0)
ax.set_ylim(0.5, 6.0)
ax.set_aspect('equal')
ax.set_title(r'Control Barrier Function (CBF) Geometry & Forward Invariance', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('X Position (meters)', fontsize=11)
ax.set_ylabel('Y Position (meters)', fontsize=11)
ax.grid(True, linestyle=':', alpha=0.4)
ax.legend(loc='lower left', fontsize=9, framealpha=0.92)

plt.tight_layout()
fig1_path = os.path.join(out_dir, "cbf_concept_diagram.png")
plt.savefig(fig1_path, dpi=300)
plt.close()
print("Saved:", fig1_path)

# =========================================================================
# Figure 2: The Projection Geometry of the Safety Filter
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
ax.set_facecolor('#ffffff')

# Draw origin of velocity space
ax.plot(0, 0, 'ko', markersize=6)
ax.text(-0.15, -0.15, '0 (Rest)', fontsize=11, color='#333333')

# Normal vector in velocity space
n = np.array([0.6, 0.8])
n = n / np.linalg.norm(n)

# Hyperplane: n^T v = b (where b = -alpha * h < 0 if approaching boundary)
b = -0.6
t = np.array([-n[1], n[0]])

# Draw hyperplane
s_vals = np.linspace(-2.5, 2.5, 100)
plane_pts = b * n[:, None] + s_vals[None, :] * t[:, None]
ax.plot(plane_pts[0], plane_pts[1], color='#2a9d8f', lw=2.5, label=r'Safe Half-Space Boundary: $\nabla h^T \mathbf{v} = -\alpha h(\mathbf{p})$')

# Shade allowed safe half space
ax.fill_between(plane_pts[0], plane_pts[1], plane_pts[1] + 3.0*n[1], color='#2a9d8f', alpha=0.15, label=r'Allowed Safe Velocities: $\nabla h^T \mathbf{v} \geq -\alpha h(\mathbf{p})$')

# Nominal desired velocity that violates safety
v_des = np.array([0.2, -1.8]) # points strongly in -n direction
dot_val = np.dot(n, v_des)
psi = dot_val - b # Psi < 0
v_corr = -psi * n
v_safe = v_des + v_corr

# Draw vectors
ax.quiver(0, 0, v_des[0], v_des[1], color='#e63946', angles='xy', scale_units='xy', scale=1.0, width=0.012, label=r'Nominal Command $\mathbf{v}_{\mathrm{des}}$ (PID, Unsafe!)')
ax.quiver(v_des[0], v_des[1], v_corr[0], v_corr[1], color='#e76f51', angles='xy', scale_units='xy', scale=1.0, width=0.012, label=r'Safety Correction $\mathbf{v}_{\mathrm{safe}} = -\Psi \nabla h$')

ax.quiver(0, 0, v_safe[0], v_safe[1], color='#1d3557', angles='xy', scale_units='xy', scale=1.0, width=0.014, label=r'Filtered Safe Velocity $\mathbf{v}^* = \mathbf{v}_{\mathrm{des}} + \mathbf{v}_{\mathrm{safe}}$')

# Outward normal
ax.quiver(b*n[0], b*n[1], n[0], n[1], color='#2a9d8f', angles='xy', scale_units='xy', scale=1.2, width=0.01, label=r'Outward Normal $\nabla h(\mathbf{p})$')

ax.text(v_des[0]+0.1, v_des[1]-0.15, r'$\mathbf{v}_{\mathrm{des}}$', fontsize=13, fontweight='bold', color='#e63946')
ax.text(v_safe[0]+0.15, v_safe[1]+0.05, r'$\mathbf{v}^*$ (Pointwise Optimal)', fontsize=13, fontweight='bold', color='#1d3557')
ax.text((v_des[0]+v_safe[0])/2 - 0.7, (v_des[1]+v_safe[1])/2, r'$\mathbf{v}_{\mathrm{safe}}$', fontsize=12, fontweight='bold', color='#e76f51')

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.0)
ax.set_aspect('equal')
ax.set_title(r'CBF-QP Safety Filter: Minimal-Invasive Projection Geometry', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel(r'Velocity Component $v_x$ (m/s)', fontsize=11)
ax.set_ylabel(r'Velocity Component $v_y$ (m/s)', fontsize=11)
ax.grid(True, linestyle=':', alpha=0.4)
ax.legend(loc='upper left', fontsize=9, framealpha=0.92)

plt.tight_layout()
fig2_path = os.path.join(out_dir, "cbf_projection_geometry.png")
plt.savefig(fig2_path, dpi=300)
plt.close()
print("Saved:", fig2_path)

# =========================================================================
# Figure 3: APF vs CBF Comparison
# =========================================================================
fig, (ax_apf, ax_cbf) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

for ax, title in zip([ax_apf, ax_cbf], ['Traditional Artificial Potential Field (APF)', 'Modern Control Barrier Function (CBF)']):
    ax.set_facecolor('#f8f9fa')
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, linestyle=':', alpha=0.4)

    # Goal and Obstacle
    ax.plot(0.5, 2.0, 'go', markersize=10, label='Start')
    ax.plot(4.5, 2.0, 'y*', markersize=15, markeredgecolor='black', label='Goal')
    obs = plt.Circle((2.5, 2.0), 0.8, color='#e63946', alpha=0.8, label='Obstacle')
    ax.add_patch(obs)

# Left: APF Behavior (local minima, oscillatory force vector fight)
theta = np.linspace(0, 2*np.pi, 100)
ax_apf.plot(2.5 + 1.6*np.cos(theta), 2.0 + 1.6*np.sin(theta), 'r--', lw=1.2, label='Repulsive Region (ρ0)')
ax_apf.plot(2.5 + 1.6*np.cos(theta), 2.0 + 1.6*np.sin(theta), 'r--', lw=1.2, label=r'Repulsive Region ($D_{\mathrm{obs}}$)')
# Drone gets stuck directly head-on or oscillates
t_stuck = np.linspace(0.5, 1.6, 20)
ax_apf.plot(t_stuck, np.full_like(t_stuck, 2.0), 'r-', lw=2.5, label='Path Trapped / Stalled')
ax_apf.plot(1.6, 2.0, 'X', color='black', markersize=12, label='Local Minimum (F_att = -F_rep)')
ax_apf.annotate('Repulsive & Attractive\nforces cancel out!', xy=(1.6, 2.0), xytext=(0.5, 3.2),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5), fontsize=10,
ax_apf.plot(1.6, 2.0, 'X', color='black', markersize=12, label=r'Deadlock ($\mathbf{v}_{\mathrm{att}} = -\mathbf{v}_{\mathrm{rep}}$)')

# Draw opposing velocity arrows at deadlock
ax_apf.quiver(1.6, 2.0, 0.65, 0, color='#2a9d8f', angles='xy', scale_units='xy', scale=1, width=0.015, zorder=5)
ax_apf.text(1.8, 2.15, r'$\mathbf{v}_{\mathrm{att}}$', color='#2a9d8f', fontweight='bold', fontsize=11)
ax_apf.quiver(1.6, 2.0, -0.65, 0, color='#e63946', angles='xy', scale_units='xy', scale=1, width=0.015, zorder=5)
ax_apf.text(1.0, 2.15, r'$\mathbf{v}_{\mathrm{rep}}$', color='#e63946', fontweight='bold', fontsize=11)

ax_apf.annotate(r'Opposing velocities cancel:' + '\n' + r'$\mathbf{v}_{\mathrm{cmd}} = \mathbf{v}_{\mathrm{att}} + \mathbf{v}_{\mathrm{rep}} = \mathbf{0}$',
                xy=(1.6, 2.0), xytext=(0.4, 3.2),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5), fontsize=9.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe5d9', edgecolor='#e63946'))
ax_apf.legend(loc='lower left', fontsize=8)

# Right: CBF Behavior (smooth circumnavigation, zero distortion when far)
circle_cbf = plt.Circle((2.5, 2.0), 1.1, color='#2a9d8f', fill=False, lw=2.0, linestyle='--', label='Safety Barrier (h=0)')
circle_cbf = plt.Circle((2.5, 2.0), 1.1, color='#2a9d8f', fill=False, lw=2.0, linestyle='--', label=r'Safety Barrier ($h(\mathbf{p})=0$)')
ax_cbf.add_patch(circle_cbf)
# Smooth tangent path
px = np.array([0.5, 1.4, 2.5, 3.6, 4.5])
py = np.array([2.0, 2.1, 3.4, 2.2, 2.0])
from scipy.interpolate import make_interp_spline
spl = make_interp_spline(np.linspace(0, 1, len(px)), np.c_[px, py], k=3)
dense_pts = spl(np.linspace(0, 1, 100))
ax_cbf.plot(dense_pts[:, 0], dense_pts[:, 1], color='#1d3557', lw=2.8, label='Smooth CBF Safe Path')
ax_cbf.annotate('Zero intervention\nwhen safe', xy=(0.8, 2.0), xytext=(0.2, 0.6),
                arrowprops=dict(facecolor='#1d3557', arrowstyle='->', lw=1.2), fontsize=9)
ax_cbf.annotate('Tangential safe deflection\n(no local minimum!)', xy=(2.5, 3.4), xytext=(2.0, 3.9),
                arrowprops=dict(facecolor='#2a9d8f', arrowstyle='->', lw=1.2), fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f8f5', edgecolor='#2a9d8f'))
ax_cbf.legend(loc='lower left', fontsize=8)

plt.tight_layout()
fig3_path = os.path.join(out_dir, "apf_vs_cbf_concept.png")
plt.savefig(fig3_path, dpi=300)
plt.close()
print("Saved:", fig3_path)

# =========================================================================
# Figure 3B: Four Classic Failure Modes of APF in Velocity Control
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=300)

titles = [
    '(a) Collinear Deadlock (Local Minimum)',
    '(b) Goal Non-Reachable with Obstacle Nearby (GNRON)',
    '(c) Chattering & Limit Cycles in Narrow Passages',
    '(d) Parasitic Deflection of Safe Nominal Paths'
]

for ax, t in zip(axes.flatten(), titles):
    ax.set_facecolor('#fdfefe')
    ax.set_aspect('equal')
    ax.set_title(t, fontsize=10.5, fontweight='bold', pad=8)
    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.4)

# --- Subplot 1: Collinear Deadlock ---
ax1 = axes[0, 0]
ax1.set_xlim(-0.5, 5.5)
ax1.set_ylim(-0.5, 4.0)
ax1.plot(0.5, 2.0, 'go', markersize=9, label='Start')
ax1.plot(4.8, 2.0, 'y*', markersize=14, markeredgecolor='black', label='Goal')
obs1 = plt.Circle((3.0, 2.0), 0.75, color='#e63946', alpha=0.8, label='Obstacle')
buf1 = plt.Circle((3.0, 2.0), 1.5, color='#e63946', fill=False, linestyle='--', lw=1.2, label=r'Buffer $D_{\mathrm{obs}}$')
ax1.add_patch(obs1)
ax1.add_patch(buf1)
ax1.plot([0.5, 1.8], [2.0, 2.0], 'r-', lw=2.5)
ax1.plot(1.8, 2.0, 'X', color='black', markersize=11)
ax1.quiver(1.8, 2.0, 0.7, 0, color='#2a9d8f', angles='xy', scale_units='xy', scale=1, width=0.018)
ax1.quiver(1.8, 2.0, -0.7, 0, color='#e63946', angles='xy', scale_units='xy', scale=1, width=0.018)
ax1.text(2.0, 2.2, r'$\mathbf{v}_{\mathrm{att}}$', color='#2a9d8f', fontweight='bold', fontsize=10)
ax1.text(1.2, 2.2, r'$\mathbf{v}_{\mathrm{rep}}$', color='#e63946', fontweight='bold', fontsize=10)
ax1.text(1.8, 1.2, r'$\mathbf{v}_{\mathrm{cmd}} = \mathbf{v}_{\mathrm{att}} + \mathbf{v}_{\mathrm{rep}} = \mathbf{0}$' + '\n' + 'Permanent Stall!',
         ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffe5d9', edgecolor='#e63946'))
ax1.legend(loc='lower left', fontsize=7.5)

# --- Subplot 2: GNRON ---
ax2 = axes[0, 1]
ax2.set_xlim(-0.5, 5.5)
ax2.set_ylim(-0.5, 4.0)
ax2.plot(0.5, 2.0, 'go', markersize=9, label='Start')
ax2.plot(3.8, 2.0, 'y*', markersize=14, markeredgecolor='black', label='Goal')
obs2 = plt.Circle((4.7, 2.0), 0.65, color='#e63946', alpha=0.8, label='Obstacle')
buf2 = plt.Circle((4.7, 2.0), 1.6, color='#e63946', fill=False, linestyle='--', lw=1.2, label=r'Buffer $D_{\mathrm{obs}}$')
ax2.add_patch(obs2)
ax2.add_patch(buf2)
ax2.plot([0.5, 2.9], [2.0, 2.0], 'r-', lw=2.5)
ax2.plot(2.9, 2.0, 's', color='#9d0208', markersize=9, label='False Equilibrium')
ax2.quiver(3.8, 2.0, -0.7, 0, color='#e63946', angles='xy', scale_units='xy', scale=1, width=0.018)
ax2.text(3.4, 2.2, r'$\mathbf{v}_{\mathrm{rep}} \neq \mathbf{0}$', color='#e63946', fontweight='bold', fontsize=9.5)
ax2.text(2.9, 1.2, r'At Goal: $\mathbf{v}_{\mathrm{att}}=\mathbf{0}$, but $\mathbf{v}_{\mathrm{rep}} \neq \mathbf{0}$' + '\n' + r'Pushed away $\to$ Goal unreachable!',
         ha='center', fontsize=8.5, bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffe5d9', edgecolor='#e63946'))
ax2.legend(loc='lower left', fontsize=7.5)

# --- Subplot 3: Chattering in Narrow Passage ---
ax3 = axes[1, 0]
ax3.set_xlim(-0.5, 5.5)
ax3.set_ylim(-0.5, 4.5)
ax3.plot(0.5, 2.0, 'go', markersize=9, label='Start')
ax3.plot(4.8, 2.0, 'y*', markersize=14, markeredgecolor='black', label='Goal')
obs3a = plt.Circle((2.5, 3.4), 0.7, color='#e63946', alpha=0.8)
obs3b = plt.Circle((2.5, 0.6), 0.7, color='#e63946', alpha=0.8)
ax3.add_patch(obs3a)
ax3.add_patch(obs3b)
ax3.add_patch(plt.Circle((2.5, 3.4), 1.2, color='#e63946', fill=False, linestyle='--', lw=1.0))
ax3.add_patch(plt.Circle((2.5, 0.6), 1.2, color='#e63946', fill=False, linestyle='--', lw=1.0))
# Zigzag chattering trajectory
chat_x = [0.5, 1.2, 1.6, 2.0, 2.3, 2.6, 2.9, 3.3]
chat_y = [2.0, 2.0, 2.4, 1.6, 2.5, 1.5, 2.3, 1.8]
ax3.plot(chat_x, chat_y, 'r-', lw=2.0, label='Chattering Trajectory')
ax3.quiver(2.0, 1.6, 0.4, 0.8, color='#e63946', angles='xy', scale_units='xy', scale=1, width=0.015)
ax3.quiver(2.3, 2.5, 0.4, -0.8, color='#e63946', angles='xy', scale_units='xy', scale=1, width=0.015)
ax3.text(2.6, 3.9, 'Opposing repulsive spikes\ncause violent limit cycles!',
         ha='center', fontsize=8.5, bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffe5d9', edgecolor='#e63946'))
ax3.legend(loc='lower left', fontsize=7.5)

# --- Subplot 4: Parasitic Deflection ---
ax4 = axes[1, 1]
ax4.set_xlim(-0.5, 5.5)
ax4.set_ylim(-0.5, 4.5)
ax4.plot(0.5, 1.5, 'go', markersize=9, label='Start')
ax4.plot(4.8, 1.5, 'y*', markersize=14, markeredgecolor='black', label='Goal')
obs4 = plt.Circle((2.5, 3.0), 0.7, color='#e63946', alpha=0.8, label='Obstacle')
buf4 = plt.Circle((2.5, 3.0), 1.9, color='#e63946', fill=False, linestyle='--', lw=1.2, label=r'Buffer $D_{\mathrm{obs}}$')
ax4.add_patch(obs4)
ax4.add_patch(buf4)
# Nominal straight line (already safe!)
ax4.plot([0.5, 4.8], [1.5, 1.5], 'k:', lw=2.0, label='Nominal Safe Path (Desired)')
# APF deflects it downward
t_defl = np.linspace(0.5, 4.8, 100)
y_defl = 1.5 - 0.75 * np.exp(-((t_defl - 2.5)/0.8)**2)
ax4.plot(t_defl, y_defl, 'r-', lw=2.2, label='APF Distorted Path')
ax4.annotate('Repulsive force pushes down\neven though path is safe!',
             xy=(2.5, 0.75), xytext=(1.2, 0.2),
             arrowprops=dict(facecolor='#e63946', arrowstyle='->', lw=1.2), fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffe5d9', edgecolor='#e63946'))
ax4.legend(loc='upper left', fontsize=7.5)

plt.tight_layout()
fig3b_path = os.path.join(out_dir, "apf_failure_modes.png")
plt.savefig(fig3b_path, dpi=300)
plt.close()
print("Saved:", fig3b_path)

# =========================================================================
# Figure 4: Quadrotor Two-Tier Architecture with CBF Safety Filter
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
ax.set_facecolor('#ffffff')
ax.axis('off')

from matplotlib.patches import FancyBboxPatch

boxes = {
    'goal': (0.04, 0.45, 0.17, 0.35, '#e9ecef', '#495057', 'Goal Waypoint\n' + r'$\mathbf{p}_{\mathrm{goal}}$'),
    'pid': (0.26, 0.45, 0.19, 0.35, '#dee2e6', '#212529', 'Tier 1 Guidance\n(Position PID)\n' + r'$\mathbf{v}_{\mathrm{des}} = \mathbf{K}_p \mathbf{e} + \dots$'),
    'cbf': (0.50, 0.38, 0.23, 0.48, '#d8f3dc', '#1b4332', 'CBF Safety Filter\n' + r'$\min \|\mathbf{v} - \mathbf{v}_{\mathrm{des}}\|^2$' + '\n' + r'$\nabla h^T \mathbf{v} \geq -\alpha h$' + '\n' + r'$\mathbf{v}^* = \mathbf{v}_{\mathrm{des}} + \mathbf{v}_{\mathrm{safe}}$'),
    'bbox': (0.78, 0.45, 0.18, 0.35, '#e0f2fe', '#0369a1', 'Tier 2 Black-Box\nLower Controller\n(Tilt & Allocation)')
}

for name, (x, y, w, h, bg, border, txt) in boxes.items():
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                           facecolor=bg, edgecolor=border, lw=2.0)
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, txt, ha='center', va='center', fontsize=9.2, fontweight='bold', color=border)


# Arrows
ax.annotate('', xy=(0.26, 0.625), xytext=(0.21, 0.625), arrowprops=dict(facecolor='#495057', arrowstyle='->', lw=2))
ax.annotate('', xy=(0.50, 0.625), xytext=(0.44, 0.625), arrowprops=dict(facecolor='#212529', arrowstyle='->', lw=2))
ax.text(0.47, 0.66, r'$\mathbf{v}_{\mathrm{des}}$', fontsize=11, fontweight='bold', color='#c1121f', ha='center')

ax.annotate('', xy=(0.78, 0.625), xytext=(0.72, 0.625), arrowprops=dict(facecolor='#1b4332', arrowstyle='->', lw=2))
ax.text(0.75, 0.66, r'$\mathbf{v}^*$', fontsize=11, fontweight='bold', color='#1b4332', ha='center')

# Obstacle sensing into CBF
ax.annotate('', xy=(0.61, 0.40), xytext=(0.61, 0.15), arrowprops=dict(facecolor='#c1121f', arrowstyle='->', lw=2))
ax.text(0.61, 0.08, 'Obstacle Coordinates & Radius\n' + r'$\mathbf{p}_{\mathrm{obs}}, D_{\mathrm{obs}}$ (LIDAR / Camera)', fontsize=9, fontweight='bold', color='#c1121f', ha='center')

# Motors out of Black-Box
ax.annotate('', xy=(1.02, 0.625), xytext=(0.96, 0.625), arrowprops=dict(facecolor='#0369a1', arrowstyle='->', lw=2))
ax.text(1.05, 0.625, '4 Motors\n' + r'$[-1, 1]^4$', fontsize=9.5, fontweight='bold', color='#0369a1', va='center')

ax.set_xlim(0.0, 1.15)
ax.set_ylim(0.0, 1.0)
ax.set_title('Hierarchical Drone Control with Control Barrier Function Safety Shield', fontsize=13, fontweight='bold', pad=10)

plt.tight_layout()
fig4_path = os.path.join(out_dir, "cbf_architecture_block.png")
plt.savefig(fig4_path, dpi=300)
plt.close()
print("Saved:", fig4_path)
