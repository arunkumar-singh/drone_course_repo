#!/usr/bin/env python3
"""
generate_qp_graphics.py
Generates high-resolution pedagogical graphics for the Quadratic Programming Tutorial:
1. qp_2d_contour_geometry.png: 2D QP level curves, unconstrained vs. constrained optimum, KKT normal balance.
2. multi_cbf_polytope_geometry.png: Multi-obstacle CBF safe velocity polytope and QP projection.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Output directory
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure matplotlib styling
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


def generate_2d_qp_contour_geometry():
    """Generates 2D QP level curves and constraint geometry showing unconstrained vs constrained optimum."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)

    # Cost: f(x) = 0.5 * x^T P x + q^T x + r
    # Let P = [[4, 1], [1, 2]], q = [-6, -4]^T
    # Unconstrained optimum: P x = -q -> [[4, 1], [1, 2]] [x1, x2]^T = [6, 4]^T
    # det(P) = 8 - 1 = 7.
    # P^-1 = (1/7) [[2, -1], [-1, 4]]
    # x* = (1/7) [[2, -1], [-1, 4]] [6, 4]^T = (1/7) [8, 10]^T = [1.143, 1.429]^T
    P = np.array([[4.0, 1.0], [1.0, 2.0]])
    q = np.array([-6.0, -4.0])
    x_unc = np.linalg.solve(P, -q)  # [8/7, 10/7]

    # Grid for contour plot
    x1 = np.linspace(-0.5, 2.5, 400)
    x2 = np.linspace(-0.5, 2.5, 400)
    X1, X2 = np.meshgrid(x1, x2)

    # Evaluate cost: f(x) = 0.5*(4*x1^2 + 2*x1*x2 + 2*x2^2) - 6*x1 - 4*x2 + 10
    Z = 0.5 * (P[0, 0] * X1**2 + 2 * P[0, 1] * X1 * X2 + P[1, 1] * X2**2) + q[0] * X1 + q[1] * X2 + 10.0

    # Constraint 1: g^T x <= h -> x1 + 1.5*x2 <= 2.2
    # Normal g = [1.0, 1.5], h = 2.2
    # At unconstrained optimum: x1 + 1.5*x2 = 1.143 + 1.5*1.429 = 3.286 > 2.2 (VIOLATED!)
    g = np.array([1.0, 1.5])
    h = 2.2

    # Constrained optimum by KKT:
    # P x + q + lambda * g = 0
    # g^T x = h
    # x = -P^-1 q - lambda P^-1 g = x_unc - lambda P^-1 g
    # g^T (x_unc - lambda P^-1 g) = h -> lambda = (g^T x_unc - h) / (g^T P^-1 g)
    P_inv = np.linalg.inv(P)
    lam = (np.dot(g, x_unc) - h) / np.dot(g, P_inv @ g)
    x_con = x_unc - lam * (P_inv @ g)

    # Plot contours
    levels = np.array([4.2, 4.5, 5.0, 5.8, 6.8, 8.2, 10.0, 12.5, 15.5])
    cs = ax.contour(X1, X2, Z, levels=levels, cmap='viridis_r', alpha=0.85, linewidths=1.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='f=%.1f')

    # Shade feasible region: g1*x1 + g2*x2 <= h -> x2 <= (h - g1*x1)/g2
    x1_line = np.linspace(-0.5, 2.5, 500)
    x2_line = (h - g[0] * x1_line) / g[1]

    # Feasible polygon (below constraint line)
    poly_pts = [(-0.5, -0.5), (2.5, -0.5), (2.5, (h - g[0] * 2.5) / g[1]), (-0.5, (h - g[0] * -0.5) / g[1])]
    poly = Polygon(poly_pts, facecolor='lightgreen', alpha=0.25, edgecolor='darkgreen', lw=2.0, linestyle='--', label='Feasible Set $\\mathcal{F}: \\mathbf{g}^T \\mathbf{x} \\leq h$')
    ax.add_patch(poly)

    # Draw constraint boundary line
    ax.plot(x1_line, x2_line, color='darkgreen', lw=2.5, label='Active Boundary: $\\mathbf{g}^T \\mathbf{x} = h$')

    # Shade forbidden region
    ax.fill_between(x1_line, x2_line, 2.5, color='salmon', alpha=0.20, label='Infeasible Half-Space')

    # Plot Unconstrained Minimizer
    ax.scatter(x_unc[0], x_unc[1], color='crimson', s=140, zorder=5, marker='X', edgecolor='black', lw=1.2, label='Unconstrained Optimum $\\mathbf{x}^*_{\\mathrm{unc}}$ (Infeasible)')
    ax.text(x_unc[0] + 0.05, x_unc[1] + 0.06, '$\\mathbf{x}^*_{\\mathrm{unc}} = [1.14, 1.43]^T$', color='crimson', fontweight='bold', fontsize=10)

    # Plot Constrained Minimizer
    ax.scatter(x_con[0], x_con[1], color='blue', s=160, zorder=5, marker='o', edgecolor='black', lw=1.2, label='Constrained Optimum $\\mathbf{x}^*_{\\mathrm{con}}$ (KKT Point)')
    ax.text(x_con[0] - 0.75, x_con[1] + 0.08, f'$\\mathbf{{x}}^*_{{\\mathrm{{con}}}} = [{x_con[0]:.2f}, {x_con[1]:.2f}]^T$', color='darkblue', fontweight='bold', fontsize=10)

    # Draw tangent line at x_con
    slope_line = -g[0] / g[1]
    t_vals = np.linspace(-0.6, 0.6, 50)
    ax.plot(x_con[0] + t_vals, x_con[1] + slope_line * t_vals, color='black', linestyle=':', lw=1.5, alpha=0.7)

    # Draw gradient of cost and normal vector of constraint at x_con
    # grad f(x_con) = P x_con + q
    grad_f = P @ x_con + q
    # We know grad f(x_con) = -lambda * g.
    # Plot -grad f (descent direction pointing INTO infeasible set)
    scale = 0.28
    ax.annotate('', xy=(x_con[0] - scale * grad_f[0], x_con[1] - scale * grad_f[1]), xytext=(x_con[0], x_con[1]),
                arrowprops=dict(facecolor='red', edgecolor='darkred', width=2, headwidth=8, shrink=0))
    ax.text(x_con[0] - scale * grad_f[0] + 0.04, x_con[1] - scale * grad_f[1] + 0.02, '$-\\nabla f(\\mathbf{x}^*)$ (Descent)', color='darkred', fontweight='bold', fontsize=10)

    # Plot constraint outward normal g
    ax.annotate('', xy=(x_con[0] + scale * g[0] * 1.5, x_con[1] + scale * g[1] * 1.5), xytext=(x_con[0], x_con[1]),
                arrowprops=dict(facecolor='darkgreen', edgecolor='black', width=2, headwidth=8, shrink=0))
    ax.text(x_con[0] + scale * g[0] * 1.5 + 0.04, x_con[1] + scale * g[1] * 1.5 - 0.05, '$\\mathbf{g}$ (Constraint Normal)', color='darkgreen', fontweight='bold', fontsize=10)

    ax.set_xlim(-0.2, 2.3)
    ax.set_ylim(-0.2, 2.3)
    ax.set_xlabel('$x_1$', fontsize=13)
    ax.set_ylabel('$x_2$', fontsize=13)
    ax.set_title('Geometric Anatomy of a 2D Constrained Quadratic Program', fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower left', framealpha=0.92, fontsize=9.5)

    out_path = os.path.join(OUT_DIR, 'qp_2d_contour_geometry.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def generate_multi_cbf_polytope_geometry():
    """Generates 2D velocity space diagram showing Multi-Obstacle CBF Safe Polytope and QP Projection."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)

    # Quadrotor position p = [0.8, 0.6]
    # Desired velocity v_des = [1.8, 1.4]
    # Obstacle 1 at p1 = [1.2, 0.4] -> normal n1 = (p - p1)/||p - p1||
    # Obstacle 2 at p2 = [0.6, 1.1] -> normal n2 = (p - p2)/||p - p2||
    p = np.array([0.8, 0.6])
    p1 = np.array([1.2, 0.4])
    p2 = np.array([0.5, 1.0])

    diff1 = p - p1
    n1 = diff1 / np.linalg.norm(diff1)  # [-0.4, 0.2]/0.447 = [-0.894, 0.447]
    h1 = np.linalg.norm(diff1) - 0.40   # 0.447 - 0.40 = 0.047 (close!)

    diff2 = p - p2
    n2 = diff2 / np.linalg.norm(diff2)  # [0.3, -0.4]/0.5 = [0.6, -0.8]
    h2 = np.linalg.norm(diff2) - 0.45   # 0.50 - 0.45 = 0.050 (close!)

    alpha = 1.0
    # Constraint 1: -n1^T v <= alpha * h1
    # Constraint 2: -n2^T v <= alpha * h2
    b1 = alpha * h1
    b2 = alpha * h2

    # Grid in velocity space
    vx = np.linspace(-0.5, 2.5, 400)
    vy = np.linspace(-0.5, 2.5, 400)
    VX, VY = np.meshgrid(vx, vy)

    # Nominal velocity
    v_des = np.array([1.8, 1.5])

    # Objective: 0.5 * ||v - v_des||^2
    Z_v = 0.5 * ((VX - v_des[0])**2 + (VY - v_des[1])**2)

    # Contours of objective (circles centered at v_des)
    levels = [0.05, 0.2, 0.5, 0.9, 1.4, 2.0, 2.8]
    cs = ax.contour(VX, VY, Z_v, levels=levels, cmap='Blues_r', alpha=0.8, linewidths=1.4)
    ax.clabel(cs, inline=True, fontsize=8, fmt='||v-v_des||^2/2=%.2f')

    # Plot constraint lines:
    # -n1[0]*vx - n1[1]*vy = b1 -> vy = (b1 + n1[0]*vx)/(-n1[1])
    vx_line = np.linspace(-0.5, 2.5, 500)
    vy_line1 = -(b1 + n1[0] * vx_line) / n1[1]
    vy_line2 = -(b2 + n2[0] * vx_line) / n2[1]

    # Speed limits: vx <= 2.0, vy <= 2.0
    v_max = 2.0
    ax.axvline(v_max, color='gray', linestyle=':', lw=1.5, label='Box Limit $v_x \\leq v_{\\max}$')
    ax.axhline(v_max, color='gray', linestyle=':', lw=1.5, label='Box Limit $v_y \\leq v_{\\max}$')

    # Plot barrier boundary lines
    ax.plot(vx_line, vy_line1, color='purple', lw=2.5, label='Barrier 1: $-\\nabla h_1^T \\mathbf{v} = \\alpha_1 h_1$')
    ax.plot(vx_line, vy_line2, color='darkorange', lw=2.5, label='Barrier 2: $-\\nabla h_2^T \\mathbf{v} = \\alpha_2 h_2$')

    # Solve QP using CVXOPT
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
    P_qp = matrix(np.eye(2, dtype=np.float64))
    q_qp = matrix(-v_des.astype(np.float64))
    G_qp = matrix(np.array([
        -n1,
        -n2,
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0]
    ], dtype=np.float64))
    h_qp = matrix(np.array([b1, b2, v_max, v_max, v_max, v_max], dtype=np.float64))

    sol = solvers.qp(P_qp, q_qp, G_qp, h_qp)
    v_star = np.array(sol['x']).flatten()

    # Shade safe polytope region (intersection)
    mask = (-n1[0] * VX - n1[1] * VY <= b1) & (-n2[0] * VX - n2[1] * VY <= b2) & (VX <= v_max) & (VY <= v_max) & (VX >= -0.5) & (VY >= -0.5)
    ax.contourf(VX, VY, mask.astype(float), levels=[0.5, 1.5], colors=['mediumseagreen'], alpha=0.25)
    ax.plot([], [], color='mediumseagreen', lw=8, alpha=0.4, label='Admissible Velocity Polytope $\\mathcal{K}(\\mathbf{p})$')

    # Plot Nominal Velocity
    ax.scatter(v_des[0], v_des[1], color='crimson', s=160, zorder=6, marker='X', edgecolor='black', lw=1.2, label='Nominal Command $\\mathbf{v}_{\\mathrm{des}}$ (Infeasible)')
    ax.text(v_des[0] + 0.06, v_des[1] + 0.06, '$\\mathbf{v}_{\\mathrm{des}} = [1.8, 1.5]^T$', color='crimson', fontweight='bold', fontsize=10.5)

    # Plot Optimal Safe Velocity
    ax.scatter(v_star[0], v_star[1], color='blue', s=180, zorder=6, marker='*', edgecolor='black', lw=1.2, label='Safe Filtered Command $\\mathbf{v}^*$ (QP Optimum)')
    ax.text(v_star[0] - 0.70, v_star[1] - 0.16, f'$\\mathbf{{v}}^* = [{v_star[0]:.2f}, {v_star[1]:.2f}]^T$', color='darkblue', fontweight='bold', fontsize=10.5)

    # Draw projection vector from v_des to v_star
    ax.annotate('', xy=(v_star[0], v_star[1]), xytext=(v_des[0], v_des[1]),
                arrowprops=dict(facecolor='blue', edgecolor='darkblue', width=2.2, headwidth=9, linestyle='--'))
    ax.text(0.5 * (v_des[0] + v_star[0]) + 0.08, 0.5 * (v_des[1] + v_star[1]) + 0.05, 'Minimal Projection\n$\\|\\mathbf{v}^* - \\mathbf{v}_{\\mathrm{des}}\\|^2$', color='darkblue', fontweight='bold', fontsize=9.5)

    # Plot normal vectors on barrier lines pointing into forbidden region
    ax.annotate('', xy=(0.8 - 0.3 * n1[0], 0.8 - 0.3 * n1[1]), xytext=(0.8, 0.8),
                arrowprops=dict(facecolor='purple', edgecolor='black', width=1.5, headwidth=6))
    ax.text(0.8 - 0.3 * n1[0] - 0.35, 0.8 - 0.3 * n1[1] + 0.05, '$-\\nabla h_1$', color='purple', fontweight='bold', fontsize=9)

    ax.annotate('', xy=(1.0 - 0.3 * n2[0], 0.3 - 0.3 * n2[1]), xytext=(1.0, 0.3),
                arrowprops=dict(facecolor='darkorange', edgecolor='black', width=1.5, headwidth=6))
    ax.text(1.0 - 0.3 * n2[0] + 0.05, 0.3 - 0.3 * n2[1] - 0.05, '$-\\nabla h_2$', color='darkorange', fontweight='bold', fontsize=9)

    ax.set_xlim(-0.2, 2.4)
    ax.set_ylim(-0.2, 2.4)
    ax.set_xlabel('Commanded Velocity $v_x$ (m/s)', fontsize=13)
    ax.set_ylabel('Commanded Velocity $v_y$ (m/s)', fontsize=13)
    ax.set_title('Multi-Obstacle CBF Velocity Space: Polyhedral Safe Set $\\mathcal{K}(\\mathbf{p})$ and QP Filter', fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower left', framealpha=0.92, fontsize=9.5)

    out_path = os.path.join(OUT_DIR, 'multi_cbf_polytope_geometry.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


if __name__ == '__main__':
    generate_2d_qp_contour_geometry()
    generate_multi_cbf_polytope_geometry()
