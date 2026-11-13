#!/usr/bin/env python3
"""
generate_trajectory_qp_graphics.py

Generates high-resolution publication-quality figures for the polynomial
trajectory planning QP tutorial:
1. single_segment_profiles.png: Minimum jerk vs minimum snap trajectories
2. time_interval_sweep.png: Effect of duration T on trajectory, velocity, acceleration, and jerk
3. time_scaling_analysis.png: Analytical vs numerical cost scaling J(T) ~ T^{-(2r-1)}
4. multi_waypoint_3d_trajectory.png: 3D multi-waypoint flight path with waypoints and velocity arrows
5. multi_segment_profiles.png: Time history of position, velocity, acceleration across multi-waypoint mission
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cvxopt
from cvxopt import matrix, solvers

# Mute CVXOPT progress output
solvers.options['show_progress'] = False

# Set high-quality styling
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2.2,
    'grid.alpha': 0.35,
    'grid.linestyle': '--'
})

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Helper Functions: Polynomial Differentiation & Hessian Construction
# =============================================================================

def poly_derivative_coeff(i, r):
    """Computes d^r/dt^r (t^i) evaluated coefficient: i! / (i-r)!"""
    if i < r:
        return 0.0
    val = 1.0
    for k in range(r):
        val *= (i - k)
    return val


def compute_Q_matrix(N, r, T):
    r"""
    Computes the (N+1) x (N+1) Hessian matrix Q(T) for minimum derivative of order r:
    J = \int_0^T (p^(r)(t))^2 dt = c^T Q c
    """
    Q = np.zeros((N + 1, N + 1), dtype=np.float64)
    for i in range(r, N + 1):
        for j in range(r, N + 1):
            ci = poly_derivative_coeff(i, r)
            cj = poly_derivative_coeff(j, r)
            exp_val = i + j - 2 * r + 1
            Q[i, j] = (ci * cj / exp_val) * (T ** exp_val)
    return Q


def eval_poly_derivative_basis(N, d, t):
    """
    Returns basis vector [d^d/dt^d(t^0), ..., d^d/dt^d(t^N)]^T at time t.
    """
    vec = np.zeros(N + 1, dtype=np.float64)
    for i in range(d, N + 1):
        ci = poly_derivative_coeff(i, d)
        vec[i] = ci * (t ** (i - d))
    return vec


# =============================================================================
# 1. Single Segment Polynomial QP Solver (CVXOPT)
# =============================================================================

def solve_single_segment_qp(N, r, T, p0, pf, v0=0.0, vf=0.0, a0=0.0, af=0.0, j0=0.0, jf=0.0,
                            v_max=None, a_max=None, n_colloc=20):
def solve_single_segment_qp(N, r, T, p0, pf, v0=0.0, vf=0.0, a0=0.0, af=0.0, j0=0.0, jf=0.0):
    """
    Solves for polynomial coefficients c of degree N minimizing r-th derivative over [0, T]
    subject to boundary conditions at t=0 and t=T, and optional inequality bounds.
    subject to boundary equality conditions at t=0 and t=T.
    Formulation: min (1/2) c^T P c + q^T c  subject to  A c = b
    where P = 2 Q(T), q = 0.
    """
    Q = compute_Q_matrix(N, r, T)
    P = matrix(2.0 * Q)
    q = matrix(np.zeros(N + 1, dtype=np.float64))

    # Boundary equality constraints
    A_rows = []
    b_rows = []

    # t = 0
    A_rows.append(eval_poly_derivative_basis(N, 0, 0.0))
    b_rows.append(p0)
    A_rows.append(eval_poly_derivative_basis(N, 1, 0.0))
    b_rows.append(v0)
    A_rows.append(eval_poly_derivative_basis(N, 2, 0.0))
    b_rows.append(a0)
    if r >= 4:
        A_rows.append(eval_poly_derivative_basis(N, 3, 0.0))
        b_rows.append(j0)

    # t = T
    A_rows.append(eval_poly_derivative_basis(N, 0, T))
    b_rows.append(pf)
    A_rows.append(eval_poly_derivative_basis(N, 1, T))
    b_rows.append(vf)
    A_rows.append(eval_poly_derivative_basis(N, 2, T))
    b_rows.append(af)
    if r >= 4:
        A_rows.append(eval_poly_derivative_basis(N, 3, T))
        b_rows.append(jf)

    A_mat = matrix(np.array(A_rows, dtype=np.float64))
    b_mat = matrix(np.array(b_rows, dtype=np.float64))

    # Optional inequality constraints on velocity and acceleration
    G_rows = []
    h_rows = []
    if v_max is not None or a_max is not None:
        ts = np.linspace(0.0, T, n_colloc)
        for t_val in ts:
            if v_max is not None:
                bv = eval_poly_derivative_basis(N, 1, t_val)
                G_rows.append(bv)
                h_rows.append(v_max)
                G_rows.append(-bv)
                h_rows.append(v_max)
            if a_max is not None:
                ba = eval_poly_derivative_basis(N, 2, t_val)
                G_rows.append(ba)
                h_rows.append(a_max)
                G_rows.append(-ba)
                h_rows.append(a_max)
    # Solve equality-constrained QP via CVXOPT
    sol = solvers.qp(P, q, None, None, A_mat, b_mat)

    if G_rows:
        G_mat = matrix(np.array(G_rows, dtype=np.float64))
        h_mat = matrix(np.array(h_rows, dtype=np.float64))
        sol = solvers.qp(P, q, G_mat, h_mat, A_mat, b_mat)
    else:
        sol = solvers.qp(P, q, None, None, A_mat, b_mat)

    c = np.array(sol['x']).flatten()
    cost = float(np.dot(c, np.dot(Q, c)))
    return c, cost


def evaluate_trajectory(c, N, T, n_pts=200):
    """Evaluates position, velocity, acceleration, jerk, and snap at n_pts evenly spaced times."""
    t_vals = np.linspace(0.0, T, n_pts)
    p = np.zeros(n_pts)
    v = np.zeros(n_pts)
    a = np.zeros(n_pts)
    j = np.zeros(n_pts)
    s = np.zeros(n_pts)

    for idx, t in enumerate(t_vals):
        p[idx] = np.dot(eval_poly_derivative_basis(N, 0, t), c)
        v[idx] = np.dot(eval_poly_derivative_basis(N, 1, t), c)
        a[idx] = np.dot(eval_poly_derivative_basis(N, 2, t), c)
        j[idx] = np.dot(eval_poly_derivative_basis(N, 3, t), c)
        s[idx] = np.dot(eval_poly_derivative_basis(N, 4, t), c)

    return t_vals, p, v, a, j, s


# =============================================================================
# 2. Multi-Waypoint Piecewise Polynomial QP Solver (CVXOPT)
# =============================================================================

def solve_multi_waypoint_qp(waypoints, segment_times, N=7, r=4):
    """
    Solves piecewise polynomial trajectory for a 1D sequence of waypoints.
    waypoints: list or 1D array of waypoint values [w_0, w_1, ..., w_M]
    segment_times: list of durations [T_1, T_2, ..., T_M]
    N: degree of polynomial (default 7 for minimum snap)
    r: derivative order to minimize (default 4 for minimum snap)
    """
    M = len(segment_times)
    n_vars_seg = N + 1
    total_vars = M * n_vars_seg

    # 1. Total Hessian P_tot = 2 * blkdiag(Q_1, ..., Q_M)
    Q_tot = np.zeros((total_vars, total_vars), dtype=np.float64)
    for m in range(M):
        Qm = compute_Q_matrix(N, r, segment_times[m])
        idx = m * n_vars_seg
        Q_tot[idx:idx + n_vars_seg, idx:idx + n_vars_seg] = Qm

    P_mat = matrix(2.0 * Q_tot)
    q_mat = matrix(np.zeros(total_vars, dtype=np.float64))

    # 2. Equality constraints
    A_rows = []
    b_rows = []

    # (a) Waypoint passing constraints:
    # Each segment m starts at w_{m} and ends at w_{m+1}
    for m in range(M):
        idx = m * n_vars_seg
        Tm = segment_times[m]

        # p_m(0) = w_m
        row_start = np.zeros(total_vars)
        row_start[idx:idx + n_vars_seg] = eval_poly_derivative_basis(N, 0, 0.0)
        A_rows.append(row_start)
        b_rows.append(waypoints[m])

        # p_m(T_m) = w_{m+1}
        row_end = np.zeros(total_vars)
        row_end[idx:idx + n_vars_seg] = eval_poly_derivative_basis(N, 0, Tm)
        A_rows.append(row_end)
        b_rows.append(waypoints[m + 1])

    # (b) Boundary conditions at start (m=0) and end (m=M-1):
    # Velocity, acceleration, and jerk = 0 at start and goal
    for d in range(1, r):
        # Initial boundary at t=0 of segment 0
        row_init = np.zeros(total_vars)
        row_init[0:n_vars_seg] = eval_poly_derivative_basis(N, d, 0.0)
        A_rows.append(row_init)
        b_rows.append(0.0)

        # Final boundary at t=T_M of segment M-1
        row_final = np.zeros(total_vars)
        row_final[(M - 1) * n_vars_seg:] = eval_poly_derivative_basis(N, d, segment_times[-1])
        A_rows.append(row_final)
        b_rows.append(0.0)

    # (c) Continuity constraints at internal junctions m = 0, ..., M-2
    # p_m^(d)(T_m) = p_{m+1}^(d)(0) for d = 1, ..., r
    for m in range(M - 1):
        idx_m = m * n_vars_seg
        idx_next = (m + 1) * n_vars_seg
        Tm = segment_times[m]

        for d in range(1, r + 1):
            row_cont = np.zeros(total_vars)
            row_cont[idx_m:idx_m + n_vars_seg] = eval_poly_derivative_basis(N, d, Tm)
            row_cont[idx_next:idx_next + n_vars_seg] = -eval_poly_derivative_basis(N, d, 0.0)
            A_rows.append(row_cont)
            b_rows.append(0.0)

    A_mat = matrix(np.array(A_rows, dtype=np.float64))
    b_mat = matrix(np.array(b_rows, dtype=np.float64))

    sol = solvers.qp(P_mat, q_mat, None, None, A_mat, b_mat)
    c_tot = np.array(sol['x']).flatten()
    cost = float(np.dot(c_tot, np.dot(Q_tot, c_tot)))

    # Segment-wise coefficients
    c_segs = [c_tot[m * n_vars_seg:(m + 1) * n_vars_seg] for m in range(M)]
    return c_segs, cost


def evaluate_multi_waypoint_trajectory(c_segs, segment_times, N=7, pts_per_seg=100):
    """Evaluates entire piecewise trajectory over continuous global time."""
    M = len(segment_times)
    t_global = []
    p_all = []
    v_all = []
    a_all = []
    j_all = []

    t_start = 0.0
    for m in range(M):
        Tm = segment_times[m]
        tau, p, v, a, j, _ = evaluate_trajectory(c_segs[m], N, Tm, n_pts=pts_per_seg)
        # Avoid duplicate point at junction
        if m > 0:
            tau = tau[1:]
            p = p[1:]
            v = v[1:]
            a = a[1:]
            j = j[1:]

        t_global.extend(tau + t_start)
        p_all.extend(p)
        v_all.extend(v)
        a_all.extend(a)
        j_all.extend(j)
        t_start += Tm

    return np.array(t_global), np.array(p_all), np.array(v_all), np.array(a_all), np.array(j_all)


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def generate_figure1_single_segment():
    """Generates Figure 1: Single segment minimum jerk vs minimum snap profiles."""
    print("Generating Figure 1: Single segment profiles (Min Jerk vs Min Snap)...")
    p0, pf = 0.0, 3.0
    T = 2.0

    # 1. Minimum Jerk (r=3, N=5 quintic)
    c_jerk, cost_jerk = solve_single_segment_qp(N=5, r=3, T=T, p0=p0, pf=pf)
    t_j, p_j, v_j, a_j, j_j, _ = evaluate_trajectory(c_jerk, 5, T, 300)

    # 2. Minimum Snap (r=4, N=7 septic)
    c_snap, cost_snap = solve_single_segment_qp(N=7, r=4, T=T, p0=p0, pf=pf)
    t_s, p_s, v_s, a_s, j_s, s_s = evaluate_trajectory(c_snap, 7, T, 300)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) Position
    ax = axes[0, 0]
    ax.plot(t_j, p_j, color="#1f77b4", lw=2.5, label="Min Jerk ($r=3, N=5$)")
    ax.plot(t_s, p_s, color="#d62728", lw=2.5, linestyle="--", label="Min Snap ($r=4, N=7$)")
    ax.scatter([0, T], [p0, pf], color="black", s=60, zorder=5, label="Waypoints")
    ax.set_ylabel("Position $p(t)$ (m)")
    ax.set_title("(a) Position Profile $p(t)$", fontweight="bold")
    ax.grid(True)
    ax.legend(loc="upper left")

    # (b) Velocity
    ax = axes[0, 1]
    ax.plot(t_j, v_j, color="#1f77b4", lw=2.5, label=f"Min Jerk ($v_{{max}}={np.max(v_j):.2f}$ m/s)")
    ax.plot(t_s, v_s, color="#d62728", lw=2.5, linestyle="--", label=f"Min Snap ($v_{{max}}={np.max(v_s):.2f}$ m/s)")
    ax.set_ylabel("Velocity $v(t)$ (m/s)")
    ax.set_title("(b) Velocity Profile $v(t) = \\dot{p}(t)$", fontweight="bold")
    ax.grid(True)
    ax.legend(loc="upper right")

    # (c) Acceleration
    ax = axes[1, 0]
    ax.plot(t_j, a_j, color="#1f77b4", lw=2.5, label=f"Min Jerk ($a_{{max}}={np.max(a_j):.2f}$ m/s$^2$)")
    ax.plot(t_s, a_s, color="#d62728", lw=2.5, linestyle="--", label=f"Min Snap ($a_{{max}}={np.max(a_s):.2f}$ m/s$^2$)")
    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel("Acceleration $a(t)$ (m/s$^2$)")
    ax.set_title("(c) Acceleration Profile $a(t) = \\ddot{p}(t)$", fontweight="bold")
    ax.grid(True)
    ax.legend(loc="upper right")

    # (d) Jerk
    ax = axes[1, 1]
    ax.plot(t_j, j_j, color="#1f77b4", lw=2.5, label="Min Jerk ($j(0) \\neq 0$)")
    ax.plot(t_s, j_s, color="#d62728", lw=2.5, linestyle="--", label="Min Snap ($j(0)=j(T)=0$)")
    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel("Jerk $j(t)$ (m/s$^3$)")
    ax.set_title("(d) Jerk Profile $j(t) = p^{(3)}(t)$", fontweight="bold")
    ax.grid(True)
    ax.legend(loc="upper right")

    plt.suptitle("Single Segment Trajectory Optimization: Minimum Jerk vs. Minimum Snap ($T=2.0$ s, $\\Delta p = 3.0$ m)",
                 fontweight="bold", y=1.005)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "single_segment_profiles.png")
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def generate_figure2_time_interval_sweep():
    """Generates Figure 2: Trajectory profiles as duration T varies."""
    print("Generating Figure 2: Time interval sweep profiles...")
    p0, pf = 0.0, 3.0
    T_list = [1.0, 1.5, 2.0, 3.0, 4.0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(T_list)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for T_val, col in zip(T_list, colors):
        c, cost = solve_single_segment_qp(N=5, r=3, T=T_val, p0=p0, pf=pf)
        t, p, v, a, j, _ = evaluate_trajectory(c, 5, T_val, 200)

        axes[0, 0].plot(t, p, color=col, lw=2.2, label=f"$T={T_val:.1f}$ s")
        axes[0, 1].plot(t, v, color=col, lw=2.2, label=f"$T={T_val:.1f}$ s ($v_{{max}}={np.max(v):.2f}$)")
        axes[1, 0].plot(t, a, color=col, lw=2.2, label=f"$T={T_val:.1f}$ s ($a_{{max}}={np.max(a):.2f}$)")
        axes[1, 1].plot(t, j, color=col, lw=2.2, label=f"$T={T_val:.1f}$ s ($j_{{max}}={np.max(j):.2f}$)")

    # Formatting
    axes[0, 0].set_ylabel("Position $p(t)$ (m)")
    axes[0, 0].set_title("(a) Position Profiles across Durations $T$", fontweight="bold")
    axes[0, 0].grid(True)
    axes[0, 0].legend(loc="upper left", fontsize=9)

    axes[0, 1].set_ylabel("Velocity $v(t)$ (m/s)")
    axes[0, 1].set_title("(b) Velocity Profiles: $v_{max} \\propto 1/T$", fontweight="bold")
    axes[0, 1].grid(True)
    axes[0, 1].legend(loc="upper right", fontsize=9)

    axes[1, 0].set_xlabel("Time $t$ (s)")
    axes[1, 0].set_ylabel("Acceleration $a(t)$ (m/s$^2$)")
    axes[1, 0].set_title("(c) Acceleration Profiles: $a_{max} \\propto 1/T^2$", fontweight="bold")
    axes[1, 0].grid(True)
    axes[1, 0].legend(loc="upper right", fontsize=9)

    axes[1, 1].set_xlabel("Time $t$ (s)")
    axes[1, 1].set_ylabel("Jerk $j(t)$ (m/s$^3$)")
    axes[1, 1].set_title("(d) Jerk Profiles: $j_{max} \\propto 1/T^3$", fontweight="bold")
    axes[1, 1].grid(True)
    axes[1, 1].legend(loc="upper right", fontsize=9)

    plt.suptitle("Impact of Varying the Time Interval $T$ on Minimum Jerk Single-Segment Profiles ($\\Delta p = 3$ m)",
                 fontweight="bold", y=1.005)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "time_interval_sweep.png")
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def generate_figure3_time_scaling_analysis():
    """Generates Figure 3: Cost and peak actuation scaling vs T."""
    print("Generating Figure 3: Time scaling analysis & scaling laws...")
    p0, pf = 0.0, 3.0
    dp = pf - p0
    T_sweep = np.linspace(0.6, 5.0, 50)

    costs_jerk = []
    vmax_jerk = []
    amax_jerk = []

    costs_snap = []
    vmax_snap = []
    amax_snap = []

    for T_val in T_sweep:
        # Min Jerk (N=5, r=3)
        c_j, cost_j = solve_single_segment_qp(N=5, r=3, T=T_val, p0=p0, pf=pf)
        _, _, v_j, a_j, _, _ = evaluate_trajectory(c_j, 5, T_val, 150)
        costs_jerk.append(cost_j)
        vmax_jerk.append(np.max(v_j))
        amax_jerk.append(np.max(a_j))

        # Min Snap (N=7, r=4)
        c_s, cost_s = solve_single_segment_qp(N=7, r=4, T=T_val, p0=p0, pf=pf)
        _, _, v_s, a_s, _, _ = evaluate_trajectory(c_s, 7, T_val, 150)
        costs_snap.append(cost_s)
        vmax_snap.append(np.max(v_s))
        amax_snap.append(np.max(a_s))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # (a) Log-Log Cost Scaling: J ~ T^(-(2r-1))
    ax = axes[0]
    ax.loglog(T_sweep, costs_jerk, 'b-', lw=2.5, label="Min Jerk ($J_{jerk} \\propto T^{-5}$)")
    ax.loglog(T_sweep, costs_snap, 'r--', lw=2.5, label="Min Snap ($J_{snap} \\propto T^{-7}$)")
    # Theoretical slopes
    ref_T = np.array([1.5, 4.0])
    ref_jerk = costs_jerk[15] * (ref_T / T_sweep[15]) ** (-5)
    ref_snap = costs_snap[15] * (ref_T / T_sweep[15]) ** (-7)
    ax.plot(ref_T, ref_jerk, 'k:', lw=1.8, label="Theory Slope $-5$")
    ax.plot(ref_T, ref_snap, 'm:', lw=1.8, label="Theory Slope $-7$")
    ax.set_xlabel("Duration $T$ (s)")
    ax.set_ylabel("Trajectory Cost $J = \\int_0^T (p^{(r)})^2 dt$")
    ax.set_title("(a) Objective Cost vs. Duration (Log-Log)", fontweight="bold")
    ax.grid(True, which="both")
    ax.legend(loc="upper right", fontsize=9)

    # (b) Peak Velocity vs Duration: v_max ~ 1/T
    ax = axes[1]
    ax.plot(T_sweep, vmax_jerk, 'b-', lw=2.5, label="Min Jerk: $v_{max} = 1.875 \\,\\Delta p / T$")
    ax.plot(T_sweep, vmax_snap, 'r--', lw=2.5, label="Min Snap: $v_{max} = 2.188 \\,\\Delta p / T$")
    ax.axhline(2.5, color="gray", linestyle=":", lw=1.5, label="Feasible Limit $v_{lim}=2.5$ m/s")
    ax.set_xlabel("Duration $T$ (s)")
    ax.set_ylabel("Peak Velocity $v_{max}$ (m/s)")
    ax.set_title("(b) Peak Velocity $v_{max} \\propto T^{-1}$", fontweight="bold")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9)

    # (c) Peak Acceleration vs Duration: a_max ~ 1/T^2
    ax = axes[2]
    ax.plot(T_sweep, amax_jerk, 'b-', lw=2.5, label="Min Jerk: $a_{max} = 5.77 \\,\\Delta p / T^2$")
    ax.plot(T_sweep, amax_snap, 'r--', lw=2.5, label="Min Snap: $a_{max} = 7.51 \\,\\Delta p / T^2$")
    ax.axhline(6.0, color="gray", linestyle=":", lw=1.5, label="Motor Limit $a_{lim}=6.0$ m/s$^2$")
    ax.set_xlabel("Duration $T$ (s)")
    ax.set_ylabel("Peak Acceleration $a_{max}$ (m/s$^2$)")
    ax.set_title("(c) Peak Acceleration $a_{max} \\propto T^{-2}$", fontweight="bold")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9)

    plt.suptitle("Analytical and Empirical Scaling of Minimum Jerk & Snap Optimization vs. Segment Duration $T$",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "time_scaling_analysis.png")
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def generate_figure4_multi_waypoint_3d():
    """Generates Figure 4: 3D Multi-waypoint flight path with velocity vectors."""
    print("Generating Figure 4: 3D Multi-waypoint quadrotor trajectory...")
    # 5 waypoints in 3D: Takeoff -> WP1 -> WP2 -> WP3 -> Landing/Hover
    wps = np.array([
        [0.0, 0.0, 0.0],    # Start / ground
        [1.5, 0.5, 1.2],    # WP 1: Takeoff climb
        [3.0, 2.5, 1.8],    # WP 2: High turn
        [1.0, 3.5, 1.4],    # WP 3: Return corridor
        [0.0, 2.0, 1.0],    # WP 4: Final hover
    ])

    M = len(wps) - 1
    # Segment times allocated based on Euclidean distance
    dists = [np.linalg.norm(wps[m + 1] - wps[m]) for m in range(M)]
    avg_speed = 1.2  # m/s
    seg_times = [max(1.5, d / avg_speed) for d in dists]

    N = 7  # septic polynomials
    r = 4  # minimum snap

    # Solve independent QPs for x, y, and z
    c_x, cost_x = solve_multi_waypoint_qp(wps[:, 0], seg_times, N=N, r=r)
    c_y, cost_y = solve_multi_waypoint_qp(wps[:, 1], seg_times, N=N, r=r)
    c_z, cost_z = solve_multi_waypoint_qp(wps[:, 2], seg_times, N=N, r=r)

    # Evaluate continuous trajectories
    t_tot, px, vx, ax, jx = evaluate_multi_waypoint_trajectory(c_x, seg_times, N=N, pts_per_seg=100)
    _, py, vy, ay, jy = evaluate_multi_waypoint_trajectory(c_y, seg_times, N=N, pts_per_seg=100)
    _, pz, vz, az, jz = evaluate_multi_waypoint_trajectory(c_z, seg_times, N=N, pts_per_seg=100)

    # Plot 3D trajectory
    fig = plt.figure(figsize=(11, 8.5))
    ax3d = fig.add_subplot(111, projection='3d')

    # Color segments differently to highlight junctions
    cum_times = np.cumsum([0.0] + seg_times)
    seg_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for m in range(M):
        mask = (t_tot >= cum_times[m] - 1e-4) & (t_tot <= cum_times[m + 1] + 1e-4)
        ax3d.plot(px[mask], py[mask], pz[mask], color=seg_colors[m % len(seg_colors)],
                  lw=3.2, label=f"Segment {m+1} ($T_{m+1}={seg_times[m]:.1f}$ s)")

    # Plot waypoints
    ax3d.scatter(wps[:, 0], wps[:, 1], wps[:, 2], color="darkred", s=110, marker="o",
                 edgecolor="black", zorder=10, label="Waypoints $\\mathbf{w}_m$")
    for m, wp in enumerate(wps):
        label_text = f"$\\mathbf{{w}}_{m}$" if m > 0 and m < M else ("Start $\\mathbf{w}_0$" if m == 0 else "Goal $\\mathbf{w}_4$")
        ax3d.text(wp[0] + 0.08, wp[1] + 0.08, wp[2] + 0.12, label_text,
                  color="black", fontweight="bold", fontsize=11)

    # Plot sampled velocity vectors along the path
    sample_indices = np.linspace(0, len(t_tot) - 1, 16, dtype=int)
    arrow_scale = 0.25
    ax3d.quiver(px[sample_indices], py[sample_indices], pz[sample_indices],
                vx[sample_indices] * arrow_scale, vy[sample_indices] * arrow_scale, vz[sample_indices] * arrow_scale,
                color="teal", lw=1.5, arrow_length_ratio=0.35, label="Velocity $\\mathbf{v}(t)$")

    ax3d.set_xlabel("X (m)", labelpad=10)
    ax3d.set_ylabel("Y (m)", labelpad=10)
    ax3d.set_zlabel("Z (m)", labelpad=10)
    ax3d.set_title("3D Multi-Waypoint Minimum Snap Trajectory via Decoupled QP ($N=7, r=4$)",
                   fontweight="bold", pad=15)
    ax3d.view_init(elev=26, azim=-62)
    ax3d.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "multi_waypoint_3d_trajectory.png")
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Generate Figure 5: Multi-segment profile showing continuity
    generate_figure5_multi_segment_profiles(t_tot, px, py, pz, vx, vy, vz, ax, ay, az, cum_times, wps)


def generate_figure5_multi_segment_profiles(t, px, py, pz, vx, vy, vz, ax, ay, az, cum_times, wps):
    """Generates Figure 5: Multi-segment state time histories showing C^k continuity."""
    print("Generating Figure 5: Multi-segment continuity time histories...")
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # (a) Position x(t), y(t), z(t)
    axes[0].plot(t, px, 'r-', lw=2.2, label="$x(t)$")
    axes[0].plot(t, py, 'g-', lw=2.2, label="$y(t)$")
    axes[0].plot(t, pz, 'b-', lw=2.2, label="$z(t)$")
    for t_junc in cum_times[1:-1]:
        axes[0].axvline(t_junc, color="gray", linestyle=":", lw=1.5)
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("(a) 3D Position Coordinates vs. Time", fontweight="bold")
    axes[0].grid(True)
    axes[0].legend(loc="upper left")

    # (b) Velocity components & speed
    axes[1].plot(t, vx, 'r--', lw=1.8, label="$v_x(t)$")
    axes[1].plot(t, vy, 'g--', lw=1.8, label="$v_y(t)$")
    axes[1].plot(t, vz, 'b--', lw=1.8, label="$v_z(t)$")
    axes[1].plot(t, speed, 'k-', lw=2.5, label="Total Speed $\\|\\mathbf{v}(t)\\|$")
    for t_junc in cum_times[1:-1]:
        axes[1].axvline(t_junc, color="gray", linestyle=":", lw=1.5)
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].set_title("(b) Velocity Profiles ($C^1$ Smooth across Junctions)", fontweight="bold")
    axes[1].grid(True)
    axes[1].legend(loc="upper left")

    # (c) Acceleration components & magnitude
    axes[2].plot(t, ax, 'r--', lw=1.8, label="$a_x(t)$")
    axes[2].plot(t, ay, 'g--', lw=1.8, label="$a_y(t)$")
    axes[2].plot(t, az, 'b--', lw=1.8, label="$a_z(t)$")
    axes[2].plot(t, acc_mag, 'm-', lw=2.5, label="Total Acceleration $\\|\\mathbf{a}(t)\\|$")
    for t_junc in cum_times[1:-1]:
        axes[2].axvline(t_junc, color="gray", linestyle=":", lw=1.5, label="Junction" if t_junc == cum_times[1] else "")
    axes[2].set_xlabel("Total Mission Time $t$ (s)")
    axes[2].set_ylabel("Acceleration (m/s$^2$)")
    axes[2].set_title("(c) Acceleration Profiles ($C^2$ Smooth across Junctions)", fontweight="bold")
    axes[2].grid(True)
    axes[2].legend(loc="upper left")

    plt.suptitle("Time Histories of 3D State Derivatives for 4-Segment Multi-Waypoint Trajectory ($N=7, r=4$)",
                 fontweight="bold", y=1.008)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "multi_segment_profiles.png")
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    print("=== Generating Polynomial Trajectory QP Figures ===")
    generate_figure1_single_segment()
    generate_figure2_time_interval_sweep()
    generate_figure3_time_scaling_analysis()
    generate_figure4_multi_waypoint_3d()
    print("=== All Figures Successfully Generated ===")
