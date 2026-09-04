import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set high DPI and clean typography
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# Crisp white background for publication & PDF documents
fig = plt.figure(figsize=(18, 10), dpi=300, facecolor='#ffffff')
ax = fig.add_subplot(111, facecolor='#ffffff')
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis('off')

# Title & Subtitle in dark navy
ax.text(9, 9.45, "6-DOF Quadrotor Control Pipeline: Single-Integrator Augmented with Black-Box Controller", 
        fontsize=17.5, fontweight='bold', color='#0f172a', ha='center', va='center')
ax.text(9, 9.05, "Hierarchical Architecture bridging High-Level Velocity Guidance (ṗ = v_cmd) to Low-Level MuJoCo Motor Actuators", 
        fontsize=11.5, color='#475569', ha='center', va='center')

# Helper function for drawing rounded card boxes
def draw_card(ax, x, y, w, h, bg_color, border_color, border_width=1.5, radius=0.25):
    fancy = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=bg_color, edgecolor=border_color, linewidth=border_width, zorder=2
    )
    ax.add_patch(fancy)

# Helper function for arrows
def draw_arrow(ax, x1, y1, x2, y2, color='#0284c7', lw=2.4, label=None, label_y_offset=0.25, fontsize=10, style='-|>', rad=0.0):
    connectionstyle = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0"
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style, color=color, lw=lw,
            connectionstyle=connectionstyle, shrinkA=2, shrinkB=4
        ),
        zorder=5
    )
    if label:
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0 + label_y_offset
        ax.text(mx, my, label, fontsize=fontsize, fontweight='bold', color=color, ha='center', va='center', zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc='#ffffff', ec=color, lw=1.2, alpha=0.98))

# =========================================================================
# 1. OUTER WRAPPER CONTAINER: SingleIntegratorQuadrotorEnv (Dotted border)
# =========================================================================
wrapper_box = patches.FancyBboxPatch(
    (5.7, 1.4), 11.8, 7.1,
    boxstyle="round,pad=0.0,rounding_size=0.4",
    facecolor='#f8fafc', edgecolor='#0284c7', linewidth=2.0, linestyle='--', zorder=1
)
ax.add_patch(wrapper_box)
ax.text(5.9, 8.25, "Gymnasium Wrapper: SingleIntegratorQuadrotorEnv", 
        fontsize=13, fontweight='bold', color='#0284c7', ha='left', va='center', zorder=3)
ax.text(5.9, 7.95, "Exposes Action Space: Box(-v_max, v_max, shape=(3,))  |  Takes 3D Velocity Command v_cmd", 
        fontsize=9.5, color='#0369a1', ha='left', va='center', zorder=3)

# =========================================================================
# 2. BLOCK 1: HIGH-LEVEL SINGLE-INTEGRATOR PID CONTROLLER (Left, Emerald)
# =========================================================================
draw_card(ax, 0.6, 3.8, 4.4, 3.8, '#ecfdf5', '#059669', border_width=2.0)

ax.text(2.8, 7.25, "TIER 1: HIGH-LEVEL GUIDANCE", fontsize=11, fontweight='bold', color='#047857', ha='center')
ax.text(2.8, 6.85, "Single-Integrator Model", fontsize=13, fontweight='bold', color='#064e3b', ha='center')
ax.text(2.8, 6.45, r"$\mathbf{\dot{p}}(t) = \mathbf{v}_{\mathrm{cmd}}(t)$", fontsize=13, fontweight='bold', color='#047857', ha='center')

# Inner logic details
draw_card(ax, 0.9, 4.1, 3.8, 2.05, '#ffffff', '#10b981', border_width=1.2)
ax.text(2.8, 5.85, "3D Position Error:", fontsize=10, color='#065f46', ha='center', fontweight='bold')
ax.text(2.8, 5.50, r"$\mathbf{e}(t) = \mathbf{p}_{\mathrm{target}}(t) - \mathbf{p}(t)$", fontsize=11, color='#0f172a', ha='center')
ax.text(2.8, 4.95, "Velocity Command PID Law:", fontsize=10, color='#065f46', ha='center', fontweight='bold')
ax.text(2.8, 4.50, r"$\mathbf{v}_{\mathrm{cmd}} = \mathbf{K}_p \mathbf{e} + \mathbf{K}_i \int \mathbf{e}\,d\tau + \mathbf{K}_d \mathbf{\dot{e}}$", 
        fontsize=11.5, color='#047857', ha='center', fontweight='bold')
ax.text(2.8, 4.18, r"Saturated to $|\mathbf{v}_{\mathrm{cmd}}| \leq v_{\max}$ (1.8 m/s)", fontsize=8.5, color='#475569', ha='center')

# Input from Waypoint Goal
ax.text(0.4, 8.4, "Goal Waypoint\n" + r"$\mathbf{p}_{\mathrm{target}} = [x_d, y_d, z_d]^T$", 
        fontsize=10.5, fontweight='bold', color='#b45309', ha='left', va='center',
        bbox=dict(boxstyle="round,pad=0.3", fc='#fef3c7', ec='#f59e0b', lw=1.5))
draw_arrow(ax, 1.8, 7.9, 1.8, 7.6, color='#d97706', lw=2.2)

# =========================================================================
# 3. BLOCK 2: "BLACK-BOX" LOWER-LEVEL CONTROLLER (Middle, Indigo/Purple)
# =========================================================================
draw_card(ax, 6.1, 2.0, 5.3, 5.7, '#f5f3ff', '#6366f1', border_width=2.0)

ax.text(8.75, 7.40, "TIER 2: 'BLACK-BOX' CONTROLLER", fontsize=11, fontweight='bold', color='#4f46e5', ha='center')
ax.text(8.75, 7.00, "BlackBoxLowLevelController", fontsize=13, fontweight='bold', color='#1e1b4b', ha='center')
ax.text(8.75, 6.65, "Bridges Velocity Command → 4-Motor Thrusts", fontsize=9.5, color='#4338ca', ha='center')

# Internal steps
# Step 1: Accel tracking
draw_card(ax, 6.3, 5.5, 4.9, 0.95, '#ffffff', '#c7d2fe', border_width=1.0)
ax.text(6.5, 6.2, "1. Velocity Error to World Acceleration:", fontsize=8.5, color='#4338ca', fontweight='bold')
ax.text(8.75, 5.75, r"$\mathbf{a}_{\mathrm{cmd}} = \mathbf{K}_v (\mathbf{v}_{\mathrm{cmd}} - \mathbf{v}) \quad [a_x, a_y, a_z]^T$", fontsize=9.8, color='#0f172a', ha='center')

# Step 2: Heading & Tilt Kinematics
draw_card(ax, 6.3, 4.35, 4.9, 1.05, '#ffffff', '#c7d2fe', border_width=1.0)
ax.text(6.5, 5.15, "2. Heading Rotation & Tilt Kinematics:", fontsize=8.5, color='#4338ca', fontweight='bold')
ax.text(8.75, 4.80, r"$\mathbf{a}_{\mathrm{body}} = \mathbf{R}_z(-\psi)\,\mathbf{a}_{\mathrm{cmd}}$", fontsize=9.8, color='#0284c7', ha='center', fontweight='bold')
ax.text(8.75, 4.45, r"$\theta_{\mathrm{des}} = a_{x,\mathrm{body}}/g, \quad \phi_{\mathrm{des}} = -a_{y,\mathrm{body}}/g$", fontsize=9.8, color='#0f172a', ha='center')

# Step 3: Thrust & Tilt Comp
draw_card(ax, 6.3, 3.25, 4.9, 0.95, '#ffffff', '#c7d2fe', border_width=1.0)
ax.text(6.5, 3.95, "3. Tilt-Compensated Collective Thrust:", fontsize=8.5, color='#4338ca', fontweight='bold')
ax.text(8.75, 3.50, r"$T_{\mathrm{total}} = \frac{m (g + a_{z,\mathrm{cmd}})}{\cos\phi \cos\theta}, \quad \boldsymbol{\tau} = \mathbf{K}_p (\boldsymbol{\eta}_{\mathrm{des}} - \boldsymbol{\eta}) - \mathbf{K}_d \boldsymbol{\omega}$", fontsize=9.5, color='#0f172a', ha='center')

# Step 4: Analytical Mixer Inversion
draw_card(ax, 6.3, 2.15, 4.9, 0.95, '#ffffff', '#c7d2fe', border_width=1.0)
ax.text(6.5, 2.85, "4. Exact Analytical Mixer Inversion:", fontsize=8.5, color='#4338ca', fontweight='bold')
ax.text(8.75, 2.40, r"$[T_0, T_1, T_2, T_3]^T = \mathbf{M}^{-1} [T_{\mathrm{total}}, \tau_x, \tau_y, \tau_z]^T \in [0, T_{\max}]^4$", fontsize=9.5, color='#4338ca', ha='center', fontweight='bold')

# =========================================================================
# 4. BLOCK 3: MUJOCO GYMNASIUM ENVIRONMENT (Right, Amber/Orange)
# =========================================================================
draw_card(ax, 12.3, 2.0, 5.0, 5.7, '#fff7ed', '#ea580c', border_width=2.0)

ax.text(14.8, 7.40, "TIER 3: MUJOCO PHYSICS & GYM", fontsize=11, fontweight='bold', color='#c2410c', ha='center')
ax.text(14.8, 7.00, "QuadrotorEnv.step(action)", fontsize=13, fontweight='bold', color='#7c2d12', ha='center')
ax.text(14.8, 6.65, "6-DOF Rigid-Body Dynamics & Rendering", fontsize=9.5, color='#9a3412', ha='center')

# Inner components of MuJoCo
# Site Thrusters
draw_card(ax, 12.5, 5.4, 4.6, 1.05, '#ffffff', '#fed7aa', border_width=1.0)
ax.text(12.7, 6.15, "4 Site Thrusters (X-Frame):", fontsize=8.5, color='#c2410c', fontweight='bold')
ax.text(14.8, 5.70, r"$\mathrm{ctrl}[0:4] = (T_i / T_{\max}) \times 3.0\,\mathrm{N}$", fontsize=10, color='#0f172a', ha='center')

# Rigid Body Dynamics
draw_card(ax, 12.5, 3.8, 4.6, 1.45, '#ffffff', '#fed7aa', border_width=1.0)
ax.text(12.7, 4.95, "Newton-Euler Equations of Motion:", fontsize=8.5, color='#c2410c', fontweight='bold')
ax.text(14.8, 4.55, r"$m \mathbf{\ddot{p}} = \mathbf{R} \mathbf{f}_z - m g \mathbf{e}_3 - \mathbf{D} \mathbf{v}$", fontsize=10, color='#0f172a', ha='center')
ax.text(14.8, 4.05, r"$\mathbf{J} \mathbf{\dot{\omega}} + \boldsymbol{\omega} \times \mathbf{J} \boldsymbol{\omega} = \boldsymbol{\tau}_{\mathrm{rotor}}$", fontsize=10, color='#0f172a', ha='center')

# State Observation
draw_card(ax, 12.5, 2.15, 4.6, 1.45, '#ffffff', '#fed7aa', border_width=1.0)
ax.text(12.7, 3.30, "12-DOF State Observation:", fontsize=8.5, color='#c2410c', fontweight='bold')
ax.text(14.8, 2.90, r"$\mathbf{s} = [\mathbf{p}, \mathbf{v}, \boldsymbol{\eta}, \boldsymbol{\omega}]^T \in \mathbb{R}^{12}$", fontsize=10.2, color='#059669', ha='center', fontweight='bold')
ax.text(14.8, 2.45, "Offscreen RGB Video Rendering (track_cam)", fontsize=8.5, color='#64748b', ha='center')

# =========================================================================
# 5. FORWARD DATA FLOW ARROWS
# =========================================================================
# PID -> Black-Box (v_cmd)
draw_arrow(ax, 5.0, 5.6, 6.1, 5.6, color='#0284c7', lw=3.0, 
           label=r"$\mathbf{v}_{\mathrm{cmd}} = [v_x, v_y, v_z]^T$" + "\n(3D Velocity Command)", label_y_offset=0.45, fontsize=9.5)

# Black-Box -> MuJoCo (motor action)
draw_arrow(ax, 11.4, 5.6, 12.3, 5.6, color='#ea580c', lw=3.0, 
           label=r"$\mathbf{u} \in [-1.0, 1.0]^4$" + "\n(Motor Actions)", label_y_offset=0.45, fontsize=9.5)

# =========================================================================
# 6. FEEDBACK LOOPS (Bottom paths)
# =========================================================================
# Inner State Feedback from MuJoCo to Black-Box (v, R, omega)
ax.plot([14.8, 14.8, 8.75, 8.75], [2.15, 1.1, 1.1, 2.0], color='#6366f1', lw=2.2, linestyle='-', zorder=4)
ax.annotate("", xy=(8.75, 2.0), xytext=(8.75, 1.8),
            arrowprops=dict(arrowstyle="-|>", color='#6366f1', lw=2.2, shrinkA=0, shrinkB=0), zorder=5)
ax.text(11.8, 0.85, "Inner State Feedback: Velocities v, Euler Angles [roll, pitch, yaw], Gyro Rates [p, q, r]", 
        fontsize=8.5, color='#4338ca', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.25", fc='#ffffff', ec='#6366f1', lw=1.0))

# Outer Position Feedback from MuJoCo to High-Level PID (pos)
ax.plot([15.5, 15.5, 2.8, 2.8], [2.15, 0.35, 0.35, 3.8], color='#059669', lw=2.4, linestyle='-', zorder=4)
ax.annotate("", xy=(2.8, 3.8), xytext=(2.8, 3.6),
            arrowprops=dict(arrowstyle="-|>", color='#059669', lw=2.4, shrinkA=0, shrinkB=0), zorder=5)
ax.text(9.0, 0.35, "Outer Closed-Loop Position Feedback: Position p = [x, y, z] to High-Level PID", 
        fontsize=9.5, fontweight='bold', color='#047857', ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.25", fc='#ffffff', ec='#059669', lw=1.2))

plt.tight_layout()

# Save white-background diagram in docs, tutorials, and artifact directories
out_paths = [
    "/home/aks1/drone_course_repo/docs/single_integrator_pipeline.png",
    "/home/aks1/drone_course_repo/tutorials/single_integrator_pipeline.png",
    "/home/aks1/.gemini/antigravity/brain/f28fbdeb-7219-4450-b64b-a873c9487af9/single_integrator_pipeline.png"
]

for p in out_paths:
    fig.savefig(p, dpi=300, facecolor='#ffffff', edgecolor='none')
    print(f"Saved white-background pipeline image to: {p}")

plt.close(fig)
