"""
High-Precision Quadrotor Multi-Scenario Navigation Demos using native MuJoCo Visualizer.

Scenarios:
  1. 3D Multi-Waypoint Square Patrol with Waypoint Convergence (quadrotor_waypoint_nav.mp4)
  2. Continuous 3D Figure-8 Tracking (quadrotor_figure8_nav.mp4)
  3. Full Arena Overview Point-to-Point Flight (quadrotor_arena_overview.mp4)
"""

import os
import mediapy as media
import numpy as np
from quadrotor_env import QuadrotorEnv


class HighPrecisionQuadrotorController:
    """
    Cascaded Position & Attitude PID Controller with exact inverse allocation mixer,
    yaw heading compensation, and anti-windup integration.
    """
    def __init__(self, mass=0.500, g=9.81, max_thrust=3.0, d=0.12, c_tau=0.015):
        self.mass = mass
        self.g = g
        self.max_thrust = max_thrust
        self.d = d
        self.c_tau = c_tau
        self.int_x = 0.0
        self.int_y = 0.0
        self.int_z = 0.0

    def reset(self):
        self.int_x = 0.0
        self.int_y = 0.0
        self.int_z = 0.0

    def compute(self, state, target_pos, dt=0.02):
        pos = state[0:3]
        vel = state[3:6]
        roll, pitch, yaw = state[6:9]
        p, q, r = state[9:12]

        # 1. Altitude (Z) PID Control with Tilt Compensation
        z_err = target_pos[2] - pos[2]
        self.int_z = np.clip(self.int_z + z_err * dt, -0.4, 0.4)
        az_cmd = 12.0 * z_err - 6.0 * vel[2] + 4.0 * self.int_z

        tilt_comp = np.cos(roll) * np.cos(pitch)
        tilt_comp = np.clip(tilt_comp, 0.65, 1.0)
        total_thrust = self.mass * (self.g + az_cmd) / tilt_comp
        total_thrust = np.clip(total_thrust, 0.1 * self.mass * self.g, 1.9 * self.mass * self.g)

        # 2. Horizontal (XY) Position PID Control in World Frame
        x_err = target_pos[0] - pos[0]
        y_err = target_pos[1] - pos[1]

        self.int_x = np.clip(self.int_x + x_err * dt, -0.3, 0.3)
        self.int_y = np.clip(self.int_y + y_err * dt, -0.3, 0.3)

        ax_world = 1.8 * x_err - 2.0 * vel[0] + 0.5 * self.int_x
        ay_world = 1.8 * y_err - 2.0 * vel[1] + 0.5 * self.int_y

        # Rotate world accelerations into Quadrotor Heading (Yaw) Frame
        cy, sy = np.cos(yaw), np.sin(yaw)
        ax_body =  cy * ax_world + sy * ay_world
        ay_body = -sy * ax_world + cy * ay_world

        pitch_des = np.clip(ax_body / self.g, -0.22, 0.22)
        roll_des = np.clip(-ay_body / self.g, -0.22, 0.22)

        # 3. Attitude PD Control -> Body Torques
        tau_x = 0.075 * (roll_des - roll) - 0.015 * p
        tau_y = 0.075 * (pitch_des - pitch) - 0.015 * q
        tau_z = 0.020 * (0.0 - yaw) - 0.005 * r

        # 4. Exact Analytical Inverse Mixer Allocation
        inv_4d = 1.0 / (4.0 * self.d)
        inv_4c = 1.0 / (4.0 * self.c_tau)
        base = total_thrust / 4.0

        t0 = base + inv_4d * tau_x - inv_4d * tau_y + inv_4c * tau_z
        t1 = base + inv_4d * tau_x + inv_4d * tau_y - inv_4c * tau_z
        t2 = base - inv_4d * tau_x + inv_4d * tau_y + inv_4c * tau_z
        t3 = base - inv_4d * tau_x - inv_4d * tau_y - inv_4c * tau_z

        thrusts = np.clip([t0, t1, t2, t3], 0.0, self.max_thrust)
        return (thrusts / (0.5 * self.max_thrust) - 1.0).astype(np.float32)


def run_multi_waypoint_demo(output_dir):
    """
    Demo 1: 3D Multi-Waypoint Patrol with close waypoint convergence.
    """
    print("\n" + "=" * 65)
    print("DEMO 1: 3D Multi-Waypoint Navigation Tour (Tracking Camera)")
    print("=" * 65)

    waypoints = [
        np.array([ 0.0,  0.0, 1.0]),  # Takeoff / Center
        np.array([ 1.0,  1.0, 1.4]),  # Waypoint 1: Front-Right
        np.array([-1.0,  1.0, 1.6]),  # Waypoint 2: Rear-Right
        np.array([-1.0, -1.0, 1.2]),  # Waypoint 3: Rear-Left
        np.array([ 1.0, -1.0, 1.5]),  # Waypoint 4: Front-Left
        np.array([ 0.0,  0.0, 1.0]),  # Return Home Hover
    ]

    env = QuadrotorEnv(
        target_pos=waypoints[0],
        dt=0.01,
        frame_skip=2,
        max_steps=400,
        render_mode="rgb_array",
        camera_name="track_cam",
        render_width=720,
        render_height=540,
    )

    obs, info = env.reset(seed=42)
    ctrl = HighPrecisionQuadrotorController(mass=env.mass, max_thrust=env.max_motor_thrust)
    frames = [env.render()]

    wp_idx = 0
    current_target = waypoints[wp_idx]
    env.set_target_pos(current_target)
    stay_counter = 0

    for step in range(1, 380):
        pos = obs[0:3]
        dist = np.linalg.norm(pos - current_target)

        # Transition to next waypoint when close (< 15cm) and held for 10 steps
        if dist < 0.18:
            stay_counter += 1
            if stay_counter > 10 and wp_idx < len(waypoints) - 1:
                wp_idx += 1
                current_target = waypoints[wp_idx]
                env.set_target_pos(current_target)
                stay_counter = 0
                print(f"--> [Step {step:03d}] Converged to WP{wp_idx-1}! Next Waypoint {wp_idx}: {current_target}")
        else:
            stay_counter = 0

        action = ctrl.compute(obs, current_target, dt=0.02)
        obs, reward, term, trunc, info = env.step(action)
        frames.append(env.render())

        if step % 40 == 0 or step == 379:
            print(f"Step {step:03d} | Pos: [{pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}] | Target WP{wp_idx}: [{current_target[0]:5.2f}, {current_target[1]:5.2f}, {current_target[2]:5.2f}] | Error: {dist*100:5.1f} cm")

        if term or trunc:
            break

    env.close()

    video_path = os.path.join(output_dir, "quadrotor_waypoint_nav.mp4")
    media.write_video(video_path, frames, fps=30)
    print(f"Saved: {video_path} ({len(frames)} frames, {os.path.getsize(video_path)/1024:.1f} KB)")
    return video_path


def run_figure8_trajectory_demo(output_dir):
    """
    Demo 2: Dynamic 3D Figure-8 Continuous Tracking.
    """
    print("\n" + "=" * 65)
    print("DEMO 2: Dynamic 3D Figure-8 Continuous Tracking (Tracking Camera)")
    print("=" * 65)

    env = QuadrotorEnv(
        target_pos=(0.0, 0.0, 1.0),
        dt=0.01,
        frame_skip=2,
        max_steps=350,
        render_mode="rgb_array",
        camera_name="track_cam",
        render_width=720,
        render_height=540,
    )

    obs, info = env.reset(seed=42)
    ctrl = HighPrecisionQuadrotorController(mass=env.mass, max_thrust=env.max_motor_thrust)
    frames = [env.render()]

    omega = 0.035
    scale_x = 1.0
    scale_y = 0.8
    base_z = 1.2
    amp_z = 0.25

    for step in range(1, 300):
        t = step * omega
        tgt_x = scale_x * np.sin(t)
        tgt_y = scale_y * np.sin(2.0 * t) / 2.0
        tgt_z = base_z + amp_z * np.cos(t)
        dynamic_target = np.array([tgt_x, tgt_y, tgt_z], dtype=np.float32)

        env.set_target_pos(dynamic_target)
        action = ctrl.compute(obs, dynamic_target, dt=0.02)

        obs, reward, term, trunc, info = env.step(action)
        frames.append(env.render())

        if step % 50 == 0:
            pos = obs[0:3]
            dist = np.linalg.norm(pos - dynamic_target)
            print(f"Step {step:03d} | Drone: [{pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}] | Target: [{tgt_x:5.2f}, {tgt_y:5.2f}, {tgt_z:5.2f}] | Error: {dist*100:5.1f} cm")

        if term or trunc:
            break

    env.close()

    video_path = os.path.join(output_dir, "quadrotor_figure8_nav.mp4")
    media.write_video(video_path, frames, fps=30)
    print(f"Saved: {video_path} ({len(frames)} frames, {os.path.getsize(video_path)/1024:.1f} KB)")
    return video_path


def run_arena_overview_demo(output_dir):
    """
    Demo 3: Point-to-Point Navigation captured with Wide Arena Camera (Always in frame).
    """
    print("\n" + "=" * 65)
    print("DEMO 3: Wide Arena Camera View (Guaranteed 100% In-Frame)")
    print("=" * 65)

    targets = [
        np.array([ 0.0,  0.0, 1.0]),
        np.array([ 1.2,  1.0, 1.6]),
        np.array([-1.2, -1.0, 0.9]),
        np.array([ 0.0,  0.0, 1.0]),
    ]

    env = QuadrotorEnv(
        target_pos=targets[0],
        dt=0.01,
        frame_skip=2,
        max_steps=350,
        render_mode="rgb_array",
        camera_name="arena_cam",
        render_width=720,
        render_height=540,
    )

    obs, info = env.reset(seed=42)
    ctrl = HighPrecisionQuadrotorController(mass=env.mass, max_thrust=env.max_motor_thrust)
    frames = [env.render()]

    target_idx = 0
    current_target = targets[target_idx]
    env.set_target_pos(current_target)
    stay_count = 0

    for step in range(1, 320):
        pos = obs[0:3]
        dist = np.linalg.norm(pos - current_target)

        if dist < 0.18:
            stay_count += 1
            if stay_count > 15 and target_idx < len(targets) - 1:
                target_idx += 1
                current_target = targets[target_idx]
                env.set_target_pos(current_target)
                stay_count = 0
                print(f"--> Target Switch: Navigating to Target {target_idx} -> {current_target}")
        else:
            stay_count = 0

        action = ctrl.compute(obs, current_target, dt=0.02)
        obs, reward, term, trunc, info = env.step(action)
        frames.append(env.render())

        if step % 50 == 0:
            print(f"Step {step:03d} | Pos: [{pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}] | Target {target_idx}: [{current_target[0]:5.2f}, {current_target[1]:5.2f}, {current_target[2]:5.2f}] | Error: {dist*100:5.1f} cm")

        if term or trunc:
            break

    env.close()

    video_path = os.path.join(output_dir, "quadrotor_arena_overview.mp4")
    media.write_video(video_path, frames, fps=30)
    print(f"Saved: {video_path} ({len(frames)} frames, {os.path.getsize(video_path)/1024:.1f} KB)")
    return video_path


def main():
    print("=" * 65)
    print("MuJoCo Quadrotor: Generating Point-to-Point Navigation Videos")
    print("=" * 65)

    output_dir = os.path.dirname(__file__)

    # Run all 3 point-to-point navigation demos
    v1 = run_multi_waypoint_demo(output_dir)
    v2 = run_figure8_trajectory_demo(output_dir)
    v3 = run_arena_overview_demo(output_dir)

    print("\n" + "=" * 65)
    print("ALL DEMO VIDEOS SUCCESSFULLY GENERATED!")
    print(f"1. Multi-Waypoint Patrol:   {v1}")
    print(f"2. Continuous 3D Figure-8:  {v2}")
    print(f"3. Wide Arena Overview:     {v3}")
    print("=" * 65)


if __name__ == "__main__":
    main()
