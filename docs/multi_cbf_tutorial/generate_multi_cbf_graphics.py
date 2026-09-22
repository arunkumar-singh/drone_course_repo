#!/usr/bin/env python3
"""
generate_multi_cbf_graphics.py
Generates high-resolution pedagogical graphics for the Multi-Obstacle CBF Tutorial:
1. workspace_pid_vs_cbf.png: Physical workspace diagram showing the quadrotor at p,
   multiple obstacles with safety bubbles, the nominal PID velocity vector v_PID pointing
   blindly toward p_goal into the collision zone, and the filtered CBF velocity vector v*
   safely tangent to the boundary.
2. velocity_space_polytope.png: Velocity space (vx, vy) showing the multiple linear barrier boundaries,
   the shaded safe polytope K(p), the nominal PID command v_des, and the minimal QP projection producing v*.
3. flight_trajectories_comparison.png: Flight trajectory simulation comparing Unfiltered PID vs. CBF Filter.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from cvxopt import matrix, solvers

# Output directory
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

# Styling
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0
})


def generate_workspace_pid_vs_cbf():
    """Generates a detailed 2D physical workspace diagram with PID vs CBF velocity vectors."""
    fig, ax = plt.subplots(figsize=(9.0, 7.5), dpi=300)

    # Quadrotor current position
    p = np.array([0.75, 0.70])

    # Goal position
    p_goal = np.array([2.50, 2.10])

    # Obstacles: center, physical radius, safety buffer radius
    obs = [
        {'center': np.array([1.35, 0.85]), 'R': 0.22, 'D': 0.52, 'name': 'Obstacle 1'},
        {'center': np.array([0.90, 1.50]), 'R': 0.22, 'D': 0.50, 'name': 'Obstacle 2'},
    ]

    # Draw Obstacle Safety Bubbles and Physical Cores
    for i, ob in enumerate(obs):
        c = ob['center']
        # Safety buffer zone (light red fill with dashed red boundary)
        safety_circle = Circle(c, ob['D'], facecolor='mistyrose', edgecolor='crimson',
                               linestyle='--', lw=2.0, alpha=0.6,
                               label='CBF Safety Boundary ($h_i = 0$)' if i == 0 else None)
        ax.add_patch(safety_circle)

        # Forbidden interior buffer
        buffer_strip = Circle(c, ob['D'], facecolor='none', edgecolor='red',
                              linestyle=':', lw=1.2, alpha=0.8)
        ax.add_patch(buffer_strip)

        # Physical solid obstacle (dark red core)
        core_circle = Circle(c, ob['R'], facecolor='firebrick', edgecolor='darkred', lw=2.2,
                             label='Physical Solid Obstacle' if i == 0 else None)
        ax.add_patch(core_circle)

        ax.text(c[0], c[1], f"Obs {i+1}\n($R={ob['R']}$m)", color='white',
                fontweight='bold', fontsize=9.5, ha='center', va='center')
        ax.text(c[0], c[1] + ob['D'] + 0.05, f"$D_{{\\mathrm{{obs}}, {i+1}}} = {ob['D']}$m",
                color='darkred', fontweight='bold', fontsize=9.5, ha='center')

    # Draw Outward Normal Gradients from obstacles to drone
    for i, ob in enumerate(obs):
        c = ob['center']
        diff = p - c
        dist = np.linalg.norm(diff)
        normal = diff / dist
        scale_n = 0.35
        ax.annotate('', xy=(p[0] + scale_n * normal[0], p[1] + scale_n * normal[1]), xytext=(p[0], p[1]),
                    arrowprops=dict(facecolor='darkgreen', edgecolor='black', width=1.8, headwidth=7, shrink=0))
        ax.text(p[0] + scale_n * normal[0] + (0.04 if i == 0 else -0.22),
                p[1] + scale_n * normal[1] + (-0.06 if i == 0 else 0.04),
                f"$\\nabla h_{i+1}(\\mathbf{{p}})$", color='darkgreen', fontweight='bold', fontsize=10.5)

    # Compute Nominal Position PID Velocity Vector: v_PID = Kp * (p_goal - p)
    Kp = 1.0
    v_pid = Kp * (p_goal - p)
    # Speed saturate to v_max = 1.8 m/s
    v_max = 1.8
    if np.linalg.norm(v_pid) > v_max:
        v_pid = v_pid * (v_max / np.linalg.norm(v_pid))

    # Compute CBF Filtered Safe Velocity via QP:
    # min 0.5 * ||v - v_pid||^2
    # s.t. -normal_i^T v <= alpha * h_i
    alpha = 1.0
    G_rows = []
    h_rows = []
    for ob in obs:
        diff = p - ob['center']
        dist = np.linalg.norm(diff)
        normal = diff / dist
        h_i = dist - ob['D']
        G_rows.append(-normal)
        h_rows.append(alpha * h_i)

    # Add speed limits
    G_rows.extend([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    h_rows.extend([v_max, v_max, v_max, v_max])

    solvers.options['show_progress'] = False
    P_qp = matrix(np.eye(2, dtype=np.float64))
    q_qp = matrix(-v_pid.astype(np.float64))
    G_qp = matrix(np.array(G_rows, dtype=np.float64))
    h_qp = matrix(np.array(h_rows, dtype=np.float64))
    sol = solvers.qp(P_qp, q_qp, G_qp, h_qp)
    v_safe = np.array(sol['x']).flatten()

    # Draw Nominal PID Velocity Vector (RED ARROW)
    arrow_scale = 0.55
    ax.annotate('', xy=(p[0] + arrow_scale * v_pid[0], p[1] + arrow_scale * v_pid[1]), xytext=(p[0], p[1]),
                arrowprops=dict(facecolor='crimson', edgecolor='darkred', width=3.0, headwidth=11, shrink=0))
    ax.text(p[0] + arrow_scale * v_pid[0] + 0.05, p[1] + arrow_scale * v_pid[1] + 0.05,
            f"Nominal PID Velocity: $\\mathbf{{v}}_{{\\mathrm{{PID}}}}$\n(Collision Course! Directly into Obstacle 1)",
            color='darkred', fontweight='bold', fontsize=10)

    # Draw line-of-sight path to goal (dashed red line)
    ax.plot([p[0], p_goal[0]], [p[1], p_goal[1]], 'r--', lw=1.6, alpha=0.6, label='Blind PID Line-of-Sight')

    # Draw Filtered CBF Safe Velocity Vector (BLUE ARROW)
    ax.annotate('', xy=(p[0] + arrow_scale * v_safe[0], p[1] + arrow_scale * v_safe[1]), xytext=(p[0], p[1]),
                arrowprops=dict(facecolor='royalblue', edgecolor='darkblue', width=3.2, headwidth=11, shrink=0))
    ax.text(p[0] + arrow_scale * v_safe[0] - 0.55, p[1] + arrow_scale * v_safe[1] - 0.16,
            f"Filtered CBF Safe Velocity: $\\mathbf{{v}}^*$\n(Minimal Tangential Deflection)",
            color='darkblue', fontweight='bold', fontsize=10)

    # Draw Quadrotor Body
    ax.scatter(p[0], p[1], color='navy', s=180, zorder=6, marker='o', edgecolor='black', lw=1.5, label='Quadrotor $\\mathbf{p}(t)$')
    # Drone propeller arm cross
    arm_len = 0.10
    ax.plot([p[0] - arm_len, p[0] + arm_len], [p[1] - arm_len, p[1] + arm_len], color='black', lw=2.5, zorder=5)
    ax.plot([p[0] - arm_len, p[0] + arm_len], [p[1] + arm_len, p[1] - arm_len], color='black', lw=2.5, zorder=5)
    ax.text(p[0] - 0.22, p[1] - 0.10, "Drone $\\mathbf{p}$", color='navy', fontweight='bold', fontsize=11)

    # Draw Goal Waypoint
    ax.scatter(p_goal[0], p_goal[1], color='gold', s=260, zorder=6, marker='*', edgecolor='black', lw=1.5, label='Target Waypoint $\\mathbf{p}_{\\mathrm{goal}}$')
    ax.text(p_goal[0] - 0.10, p_goal[1] + 0.10, "Goal $\\mathbf{p}_{\\mathrm{goal}}$", color='darkgoldenrod', fontweight='bold', fontsize=11)

    ax.set_xlim(0.2, 2.9)
    ax.set_ylim(0.2, 2.5)
    ax.set_xlabel('Workspace $X$ Position (m)', fontsize=12)
    ax.set_ylabel('Workspace $Y$ Position (m)', fontsize=12)
    ax.set_title('Physical Workspace: Nominal PID Velocity vs. Filtered CBF Safe Velocity', fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', framealpha=0.92, fontsize=9.5)

    out_path = os.path.join(OUT_DIR, 'workspace_pid_vs_cbf.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def generate_velocity_space_polytope():
    """Generates the velocity space diagram showing multiple barrier half-spaces and the safe polytope."""
    fig, ax = plt.subplots(figsize=(8.8, 7.6), dpi=300)

    p = np.array([0.75, 0.70])
    p_goal = np.array([2.50, 2.10])
    v_pid = 1.0 * (p_goal - p)
    v_max = 1.8
    if np.linalg.norm(v_pid) > v_max:
        v_pid = v_pid * (v_max / np.linalg.norm(v_pid))

    obs = [
        {'center': np.array([1.35, 0.85]), 'D': 0.52},
        {'center': np.array([0.90, 1.50]), 'D': 0.50},
    ]

    alpha = 1.0
    normals = []
    b_vals = []
    for ob in obs:
        diff = p - ob['center']
        dist = np.linalg.norm(diff)
        n = diff / dist
        h_val = dist - ob['D']
        normals.append(n)
        b_vals.append(alpha * h_val)

    n1, n2 = normals[0], normals[1]
    b1, b2 = b_vals[0], b_vals[1]

    # Grid for velocity space
    vx = np.linspace(-0.5, 2.5, 400)
    vy = np.linspace(-0.5, 2.5, 400)
    VX, VY = np.meshgrid(vx, vy)

    # Cost: 0.5 * ||v - v_pid||^2
    Z = 0.5 * ((VX - v_pid[0])**2 + (VY - v_pid[1])**2)
    levels = [0.04, 0.15, 0.35, 0.65, 1.05, 1.55, 2.15]
    cs = ax.contour(VX, VY, Z, levels=levels, cmap='Blues_r', alpha=0.75, linewidths=1.3)
    ax.clabel(cs, inline=True, fontsize=8, fmt='f(v)=%.2f')

    # Barrier boundary lines:
    # -n[0]*vx - n[1]*vy = b -> vy = -(b + n[0]*vx)/n[1]
    vx_line = np.linspace(-0.5, 2.5, 500)
    vy_line1 = -(b1 + n1[0] * vx_line) / n1[1]
    vy_line2 = -(b2 + n2[0] * vx_line) / n2[1]

    # Plot barrier boundary hyperplanes
    ax.plot(vx_line, vy_line1, color='purple', lw=2.5, label='Barrier 1 Boundary: $-\\nabla h_1^T \\mathbf{v} = \\alpha_1 h_1$')
    ax.plot(vx_line, vy_line2, color='darkorange', lw=2.5, label='Barrier 2 Boundary: $-\\nabla h_2^T \\mathbf{v} = \\alpha_2 h_2$')

    # Box speed limits
    ax.axvline(v_max, color='gray', linestyle=':', lw=1.6, label=f'Speed Limit $v_x \\leq {v_max}$')
    ax.axhline(v_max, color='gray', linestyle=':', lw=1.6, label=f'Speed Limit $v_y \\leq {v_max}$')

    # Solve QP using CVXOPT
    G_all = [-n1, -n2, [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    h_all = [b1, b2, v_max, v_max, v_max, v_max]
    solvers.options['show_progress'] = False
    P_qp = matrix(np.eye(2, dtype=np.float64))
    q_qp = matrix(-v_pid.astype(np.float64))
    G_qp = matrix(np.array(G_all, dtype=np.float64))
    h_qp = matrix(np.array(h_all, dtype=np.float64))
    sol = solvers.qp(P_qp, q_qp, G_qp, h_qp)
    v_star = np.array(sol['x']).flatten()

    # Shade Admissible Velocity Polytope K(p)
    mask = (-n1[0] * VX - n1[1] * VY <= b1) & (-n2[0] * VX - n2[1] * VY <= b2) & (VX <= v_max) & (VY <= v_max) & (VX >= -0.5) & (VY >= -0.5)
    ax.contourf(VX, VY, mask.astype(float), levels=[0.5, 1.5], colors=['mediumseagreen'], alpha=0.25)
    ax.plot([], [], color='mediumseagreen', lw=8, alpha=0.4, label='Admissible Velocity Polytope $\\mathcal{K}(\\mathbf{p})$')

    # Draw Velocity Vector Arrows from Origin
    # 1. Nominal PID Vector (Red)
    ax.annotate('', xy=(v_pid[0], v_pid[1]), xytext=(0, 0),
                arrowprops=dict(facecolor='crimson', edgecolor='darkred', width=2.6, headwidth=9, shrink=0))
    ax.scatter(v_pid[0], v_pid[1], color='crimson', s=160, zorder=6, marker='X', edgecolor='black', lw=1.2, label='Nominal PID Vector $\\mathbf{v}_{\\mathrm{PID}}$ (Infeasible)')
    ax.text(v_pid[0] + 0.06, v_pid[1] + 0.06, f'$\\mathbf{{v}}_{{\\mathrm{{PID}}}} = [{v_pid[0]:.2f}, {v_pid[1]:.2f}]^T$', color='darkred', fontweight='bold', fontsize=10.5)

    # 2. Filtered CBF Safe Vector (Blue)
    ax.annotate('', xy=(v_star[0], v_star[1]), xytext=(0, 0),
                arrowprops=dict(facecolor='royalblue', edgecolor='darkblue', width=2.8, headwidth=10, shrink=0))
    ax.scatter(v_star[0], v_star[1], color='blue', s=180, zorder=6, marker='*', edgecolor='black', lw=1.2, label='Filtered CBF Vector $\\mathbf{v}^*$ (QP Optimum)')
    ax.text(v_star[0] - 0.72, v_star[1] - 0.16, f'$\\mathbf{{v}}^* = [{v_star[0]:.2f}, {v_star[1]:.2f}]^T$', color='darkblue', fontweight='bold', fontsize=10.5)

    # Draw minimal projection vector between v_pid and v_star
    ax.annotate('', xy=(v_star[0], v_star[1]), xytext=(v_pid[0], v_pid[1]),
                arrowprops=dict(facecolor='blue', edgecolor='darkblue', width=2.0, headwidth=8, linestyle='--'))
    ax.text(0.5 * (v_pid[0] + v_star[0]) + 0.08, 0.5 * (v_pid[1] + v_star[1]) + 0.05,
            'Minimal Projection\n$\\|\\mathbf{v}^* - \\mathbf{v}_{\\mathrm{PID}}\\|^2$',
            color='darkblue', fontweight='bold', fontsize=9.5)

    # Mark origin
    ax.scatter(0, 0, color='black', s=80, marker='o', zorder=5)
    ax.text(-0.15, -0.15, "Origin $(0,0)$", color='black', fontsize=9.5)

    # Normal vectors on boundary hyperplanes
    ax.annotate('', xy=(1.0 - 0.3 * n1[0], 0.8 - 0.3 * n1[1]), xytext=(1.0, 0.8),
                arrowprops=dict(facecolor='purple', edgecolor='black', width=1.5, headwidth=6))
    ax.text(1.0 - 0.3 * n1[0] - 0.35, 0.8 - 0.3 * n1[1] + 0.06, '$-\\nabla h_1$', color='purple', fontweight='bold', fontsize=9.5)

    ax.annotate('', xy=(1.1 - 0.3 * n2[0], 0.2 - 0.3 * n2[1]), xytext=(1.1, 0.2),
                arrowprops=dict(facecolor='darkorange', edgecolor='black', width=1.5, headwidth=6))
    ax.text(1.1 - 0.3 * n2[0] + 0.06, 0.2 - 0.3 * n2[1] - 0.05, '$-\\nabla h_2$', color='darkorange', fontweight='bold', fontsize=9.5)

    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-0.3, 2.3)
    ax.set_xlabel('Commanded Velocity $v_x$ (m/s)', fontsize=12)
    ax.set_ylabel('Commanded Velocity $v_y$ (m/s)', fontsize=12)
    ax.set_title('Velocity Space: Admissible Polytope $\\mathcal{K}(\\mathbf{p})$ and QP Safety Projection', fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower left', framealpha=0.92, fontsize=9.5)

    out_path = os.path.join(OUT_DIR, 'velocity_space_polytope.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def generate_flight_trajectories_comparison():
    """Simulates and plots comparative flight trajectories of Unfiltered PID vs CBF Safety Filter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 6.0), dpi=300)

    # Obstacles
    obs_list = [
        {'center': np.array([1.20, 0.90]), 'R': 0.22, 'D': 0.52},
        {'center': np.array([0.90, 1.65]), 'R': 0.22, 'D': 0.50},
        {'center': np.array([1.90, 1.80]), 'R': 0.22, 'D': 0.50},
    ]

    p_start = np.array([0.2, 0.2])
    p_goal = np.array([2.5, 2.4])

    dt = 0.02
    v_max = 1.6
    kp, ki, kd = 2.0, 0.2, 0.3

    # 1. Simulate Unfiltered PID
    pos_pid = p_start.copy()
    traj_pid = [pos_pid.copy()]
    e_int = np.zeros(2)
    e_prev = np.zeros(2)
    crashed_pid = False

    for _ in range(250):
        err = p_goal - pos_pid
        e_int = np.clip(e_int + err * dt, -0.5, 0.5)
        d_err = (err - e_prev) / dt
        e_prev = err.copy()

        v = kp * err + ki * e_int + kd * d_err
        if np.linalg.norm(v) > v_max:
            v = v * (v_max / np.linalg.norm(v))

        pos_pid = pos_pid + v * dt
        traj_pid.append(pos_pid.copy())

        # Check collision with physical core
        for ob in obs_list:
            if np.linalg.norm(pos_pid - ob['center']) <= ob['R'] + 0.17:
                crashed_pid = True
                break
        if crashed_pid or np.linalg.norm(pos_pid - p_goal) < 0.08:
            break

    # 2. Simulate CBF Filtered Flight
    pos_cbf = p_start.copy()
    traj_cbf = [pos_cbf.copy()]
    e_int_cbf = np.zeros(2)
    e_prev_cbf = np.zeros(2)
    h_hist = {i: [] for i in range(len(obs_list))}
    time_hist = []

    alpha = 0.9
    for step in range(350):
        t = step * dt
        time_hist.append(t)
        err = p_goal - pos_cbf
        e_int_cbf = np.clip(e_int_cbf + err * dt, -0.5, 0.5)
        d_err = (err - e_prev_cbf) / dt
        e_prev_cbf = err.copy()

        v_nom = kp * err + ki * e_int_cbf + kd * d_err
        if np.linalg.norm(v_nom) > v_max:
            v_nom = v_nom * (v_max / np.linalg.norm(v_nom))

        # CBF QP formulation
        G_rows = []
        h_rows = []
        for i, ob in enumerate(obs_list):
            diff = pos_cbf - ob['center']
            dist = np.linalg.norm(diff)
            n = diff / dist if dist > 1e-4 else np.array([1.0, 0.0])
            h_i = dist - ob['D']
            h_hist[i].append(h_i)
            G_rows.append(-n)
            h_rows.append(alpha * h_i)

            # Transverse circulation for symmetry breaking
            collinear = abs(np.dot(n, v_nom) / (np.linalg.norm(v_nom) + 1e-6))
            if collinear > 0.70 and h_i < 0.20:
                tangent = np.array([-n[1], n[0]])
                v_nom = v_nom + 0.35 * tangent

        G_rows.extend([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        h_rows.extend([v_max, v_max, v_max, v_max])

        solvers.options['show_progress'] = False
        P_qp = matrix(np.eye(2, dtype=np.float64))
        q_qp = matrix(-v_nom.astype(np.float64))
        G_qp = matrix(np.array(G_rows, dtype=np.float64))
        h_qp = matrix(np.array(h_rows, dtype=np.float64))
        sol = solvers.qp(P_qp, q_qp, G_qp, h_qp)
        v_star = np.array(sol['x']).flatten()

        pos_cbf = pos_cbf + v_star * dt
        traj_cbf.append(pos_cbf.copy())

        if np.linalg.norm(pos_cbf - p_goal) < 0.08:
            break

    traj_pid = np.array(traj_pid)
    traj_cbf = np.array(traj_cbf)

    # Plot Trajectories on ax1
    for i, ob in enumerate(obs_list):
        c = ob['center']
        circ_safe = Circle(c, ob['D'], facecolor='mistyrose', edgecolor='crimson', linestyle='--', lw=1.5, alpha=0.5)
        circ_core = Circle(c, ob['R'], facecolor='firebrick', edgecolor='darkred', lw=2.0)
        ax1.add_patch(circ_safe)
        ax1.add_patch(circ_core)
        ax1.text(c[0], c[1], f"Obs {i+1}", color='white', fontweight='bold', ha='center', va='center', fontsize=9.5)

    ax1.plot(traj_pid[:, 0], traj_pid[:, 1], 'r--', lw=2.5, label='Unfiltered PID (Crashes into Obs 1)')
    ax1.scatter(traj_pid[-1, 0], traj_pid[-1, 1], color='red', marker='X', s=160, zorder=6, label='Crash Location')

    ax1.plot(traj_cbf[:, 0], traj_cbf[:, 1], 'b-', lw=3.0, label='Multi-Obstacle CBF-QP (Safe Corridor)')
    ax1.scatter(p_start[0], p_start[1], color='black', s=100, marker='o', label='Start (0.2, 0.2)')
    ax1.scatter(p_goal[0], p_goal[1], color='gold', s=200, marker='*', edgecolor='black', label='Goal (2.5, 2.4)')

    ax1.set_xlim(-0.05, 2.85)
    ax1.set_ylim(-0.05, 2.85)
    ax1.set_xlabel('Workspace $X$ (m)', fontsize=12)
    ax1.set_ylabel('Workspace $Y$ (m)', fontsize=12)
    ax1.set_title('(a) Multi-Obstacle Flight Path Comparison', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='lower right', framealpha=0.92, fontsize=9.0)

    # Plot Barrier Histories on ax2
    colors = ['crimson', 'darkorange', 'purple']
    for i in range(len(obs_list)):
        ax2.plot(time_hist[:len(h_hist[i])], h_hist[i], color=colors[i], lw=2.2, label=f'Barrier $h_{i+1}(t)$ (Obs {i+1})')
    ax2.axhline(0.0, color='black', linestyle='--', lw=2.0, label='Safety Shell ($h=0$)')
    ax2.fill_between([0, time_hist[-1]], 0, -0.3, color='red', alpha=0.15, label='Unsafe Zone ($h < 0$)')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Safety Margin $h_i(t)$ (m)', fontsize=12)
    ax2.set_title('(b) Per-Obstacle Safety Margin Invariance', fontweight='bold')
    ax2.set_ylim(-0.15, 1.8)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='upper right', framealpha=0.92, fontsize=9.0)

    out_path = os.path.join(OUT_DIR, 'flight_trajectories_comparison.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


if __name__ == '__main__':
    generate_workspace_pid_vs_cbf()
    generate_velocity_space_polytope()
    generate_flight_trajectories_comparison()

