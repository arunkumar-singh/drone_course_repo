import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

fig = plt.figure(figsize=(10, 10), dpi=300, facecolor='#ffffff')
ax = fig.add_subplot(111, facecolor='#ffffff')
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.axis('off')

# Title
ax.text(0, 1.65, "X-Configuration Quadrotor Free-Body Diagram", fontsize=15, fontweight='bold', color='#0f172a', ha='center')
ax.text(0, 1.50, "Coordinate Frames, Rotor Placements, Forces, and Reaction Torques", fontsize=10.5, color='#64748b', ha='center')

# Carbon fiber arms (thick diagonal cross)
d = 0.95
ax.plot([d, -d], [d, -d], color='#334155', lw=8, zorder=2, solid_capstyle='round')
ax.plot([-d, d], [d, -d], color='#334155', lw=8, zorder=2, solid_capstyle='round')

# Central chassis
center_circle = plt.Circle((0, 0), 0.28, facecolor='#1e293b', edgecolor='#0f172a', lw=2.5, zorder=4)
ax.add_patch(center_circle)

# Body axes in center
arrow_len = 0.55
# Body X axis (Forward / +X)
ax.annotate("", xy=(arrow_len, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color='#ef4444', lw=3, shrinkA=0, shrinkB=0), zorder=6)
ax.text(arrow_len + 0.08, 0, r"$\mathbf{x}_B$ (Forward)", fontsize=11, fontweight='bold', color='#ef4444', va='center')

# Body Y axis (Right / +Y)
ax.annotate("", xy=(0, arrow_len), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color='#10b981', lw=3, shrinkA=0, shrinkB=0), zorder=6)
ax.text(0, arrow_len + 0.08, r"$\mathbf{y}_B$ (Right)", fontsize=11, fontweight='bold', color='#10b981', ha='center')

# Body Z axis (Up / out of page)
ax.scatter([0], [0], s=80, color='#38bdf8', zorder=7)
ax.plot([0], [0], marker='o', markersize=8, color='#0284c7', zorder=8)
ax.text(0.06, -0.08, r"$\odot\ \mathbf{z}_B$ (Up)", fontsize=11, fontweight='bold', color='#0284c7')

# Dimension labels
ax.plot([0, d], [-0.05, -0.05], color='#94a3b8', lw=1, linestyle='--')
ax.plot([d, d], [-0.05, d], color='#94a3b8', lw=1, linestyle='--')
ax.text(d/2, -0.15, "d = 0.12 m", fontsize=9.5, color='#64748b', ha='center')
ax.text(d + 0.12, d/2, "d = 0.12 m", fontsize=9.5, color='#64748b', va='center', rotation=90)

# Motor configurations
rotors = [
    {"name": r"$M_0$", "pos": (d, d), "color": "#ef4444", "dir": "CCW", "sign": "+", "t_name": r"$T_0$", "tau_name": r"$+c_\tau T_0$"},
    {"name": r"$M_1$", "pos": (-d, d), "color": "#3b82f6", "dir": "CW", "sign": "-", "t_name": r"$T_1$", "tau_name": r"$-c_\tau T_1$"},
    {"name": r"$M_2$", "pos": (-d, -d), "color": "#3b82f6", "dir": "CCW", "sign": "+", "t_name": r"$T_2$", "tau_name": r"$+c_\tau T_2$"},
    {"name": r"$M_3$", "pos": (d, -d), "color": "#ef4444", "dir": "CW", "sign": "-", "t_name": r"$T_3$", "tau_name": r"$-c_\tau T_3$"}
]

r_radius = 0.38
for r in rotors:
    x, y = r["pos"]
    # Rotor disc
    disc = plt.Circle((x, y), r_radius, facecolor=r["color"] + '22', edgecolor=r["color"], lw=2.2, zorder=3)
    ax.add_patch(disc)
    
    # Motor hub
    hub = plt.Circle((x, y), 0.08, facecolor='#0f172a', edgecolor=r["color"], lw=1.5, zorder=5)
    ax.add_patch(hub)
    
    # Motor label
    ax.text(x, y + (0.48 if y > 0 else -0.52), f"{r['name']}: {r['dir']} ({r['sign']})", 
            fontsize=10.5, fontweight='bold', color=r["color"], ha='center',
            bbox=dict(boxstyle="round,pad=0.2", fc='#f8fafc', ec=r["color"], lw=1.2))
    
    # Thrust and torque annotations
    offset_x = 0.42 if x > 0 else -0.42
    ax.text(x + offset_x, y, f"Thrust: {r['t_name']}\nTorque: {r['tau_name']}",
            fontsize=8.5, color='#334155', ha='left' if x > 0 else 'right', va='center',
            bbox=dict(boxstyle="square,pad=0.2", fc='#f1f5f9', ec='#cbd5e1', lw=0.8))

# Rotation direction arrows around rotors
# M0 (CCW)
arc0 = patches.Arc((d, d), 0.52, 0.52, angle=0, theta1=20, theta2=160, color='#ef4444', lw=1.8, zorder=5)
ax.add_patch(arc0)
ax.annotate("", xy=(d - 0.25, d + 0.06), xytext=(d - 0.26, d + 0.12),
            arrowprops=dict(arrowstyle="-|>", color='#ef4444', lw=1.8))

# M1 (CW)
arc1 = patches.Arc((-d, d), 0.52, 0.52, angle=0, theta1=20, theta2=160, color='#3b82f6', lw=1.8, zorder=5)
ax.add_patch(arc1)
ax.annotate("", xy=(-d + 0.25, d + 0.06), xytext=(-d + 0.26, d + 0.12),
            arrowprops=dict(arrowstyle="-|>", color='#3b82f6', lw=1.8))

# M2 (CCW)
arc2 = patches.Arc((-d, -d), 0.52, 0.52, angle=0, theta1=200, theta2=340, color='#3b82f6', lw=1.8, zorder=5)
ax.add_patch(arc2)
ax.annotate("", xy=(-d + 0.25, -d - 0.06), xytext=(-d + 0.26, -d - 0.12),
            arrowprops=dict(arrowstyle="-|>", color='#3b82f6', lw=1.8))

# M3 (CW)
arc3 = patches.Arc((d, -d), 0.52, 0.52, angle=0, theta1=200, theta2=340, color='#ef4444', lw=1.8, zorder=5)
ax.add_patch(arc3)
ax.annotate("", xy=(d - 0.25, -d - 0.06), xytext=(d - 0.26, -d - 0.12),
            arrowprops=dict(arrowstyle="-|>", color='#ef4444', lw=1.8))

# Legend at bottom
ax.text(0, -1.65, "Red Rotors: Front (+X) | Blue Rotors: Rear (-X)\nFront-Right (M0) & Rear-Left (M2) spin CCW; Rear-Right (M1) & Front-Left (M3) spin CW", 
        fontsize=9.5, color='#475569', ha='center',
        bbox=dict(boxstyle="round,pad=0.3", fc='#f8fafc', ec='#94a3b8', lw=1))

plt.tight_layout()
out_fig = "/home/aks1/drone_course_repo/docs/quadrotor_free_body_diagram.png"
fig.savefig(out_fig, dpi=300, facecolor='#ffffff', edgecolor='none')
print(f"Saved free-body diagram to {out_fig}")
plt.close(fig)

