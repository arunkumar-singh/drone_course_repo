"""
Quadrotor 6-DOF Gym / Gymnasium Environment.

This file demonstrates how to build a custom Gymnasium environment
following the standard Gymnasium / Gym template.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class QuadrotorEnv(gym.Env):
    """
    A 6-DOF Quadrotor environment following the Gymnasium API.

    State Representation (12-dim):
        [0:3]   Position (x, y, z) in meters [world frame]
        [3:6]   Linear velocity (vx, vy, vz) in m/s [world frame]
        [6:9]   Euler angles (roll, pitch, yaw) in radians
        [9:12]  Angular velocity (p, q, r) in rad/s [body frame]

    Action Space (4-dim):
        Normalized rotor thrust commands [-1.0, 1.0] for the 4 motors.
        Scaled internally to individual rotor thrusts [0, max_thrust_per_motor].
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, target_pos=(0.0, 0.0, 1.0), dt=0.01, max_steps=500):
        super().__init__()

        # --- Simulation & Physical Parameters ---
        self.dt = dt
        self.max_steps = max_steps
        self.mass = 0.500           # Drone mass in kg (e.g. 500g)
        self.arm_length = 0.17      # Arm length in meters
        self.g = 9.81               # Gravity (m/s^2)

        # Moments of inertia (diag matrix kg*m^2)
        self.Ixx = 2.5e-3
        self.Iyy = 2.5e-3
        self.Izz = 4.5e-3
        self.I = np.array([self.Ixx, self.Iyy, self.Izz])

        # Motor thrust limits
        self.hover_thrust = (self.mass * self.g) / 4.0  # Thrust per motor at hover
        self.max_motor_thrust = 2.0 * self.hover_thrust  # Max thrust (thrust-to-weight ratio = 2.0)
        self.torque_coeff = 0.01                        # Yaw torque to thrust ratio (c_tau)

        # Target goal position
        self.target_pos = np.array(target_pos, dtype=np.float32)

        # --- Define Action and Observation Spaces (Gym Template Requirement) ---
        # Action: 4 motor commands in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Observation: 12 state variables
        # [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        high_obs = np.array([
            10.0, 10.0, 10.0,       # pos limits (m)
            20.0, 20.0, 20.0,       # vel limits (m/s)
            np.pi, np.pi, np.pi,    # attitude limits (rad)
            20.0, 20.0, 20.0,       # angular rate limits (rad/s)
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-high_obs,
            high=high_obs,
            dtype=np.float32,
        )

        # State container & step counter
        self.state = np.zeros(12, dtype=np.float32)
        self.current_step = 0

    def _euler_to_rotation_matrix(self, roll, pitch, yaw):
        """Convert Euler angles (roll, pitch, yaw) -> 3x3 World-from-Body Rotation Matrix."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - cr * cy],
            [-sp,     sr * cp,                cr * cp],
        ], dtype=np.float32)
        return R

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.

        Returns:
            obs (np.ndarray): The initial 12-dim observation.
            info (dict): Auxiliary diagnostic info.
        """
        super().reset(seed=seed)

        # Randomize initial state slightly around origin
        pos = self.np_random.uniform(low=[-0.1, -0.1, 0.0], high=[0.1, 0.1, 0.2])
        vel = np.zeros(3, dtype=np.float32)
        angles = self.np_random.uniform(low=-0.05, high=0.05, size=(3,)).astype(np.float32)
        ang_vel = np.zeros(3, dtype=np.float32)

        self.state = np.concatenate([pos, vel, angles, ang_vel]).astype(np.float32)
        self.current_step = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _get_obs(self):
        return self.state.copy()

    def _get_info(self):
        pos = self.state[0:3]
        dist_to_target = np.linalg.norm(pos - self.target_pos)
        return {
            "step": self.current_step,
            "dist_to_target": float(dist_to_target),
            "altitude": float(pos[2]),
        }

    def step(self, action):
        """
        Apply action, step the physics forward by dt, and compute reward.

        Args:
            action (np.ndarray): 4 motor inputs in [-1.0, 1.0].

        Returns:
            obs (np.ndarray): Next observation (12-dim).
            reward (float): Step reward.
            terminated (bool): Whether episode terminated due to failure/success.
            truncated (bool): Whether episode ended due to time limit.
            info (dict): Extra diagnostic info.
        """
        self.current_step += 1

        # 1. Map action [-1, 1] to motor thrusts [0, max_motor_thrust]
        action = np.clip(action, -1.0, 1.0)
        thrusts = (action + 1.0) * 0.5 * self.max_motor_thrust  # [T1, T2, T3, T4]

        # 2. Extract current state
        pos = self.state[0:3]
        vel = self.state[3:6]
        roll, pitch, yaw = self.state[6:9]
        omega = self.state[9:12]  # [p, q, r]

        # 3. Compute forces and moments
        # Total collective thrust along body z-axis
        total_thrust = np.sum(thrusts)
        R = self._euler_to_rotation_matrix(roll, pitch, yaw)

        # Force in world frame: Gravity + Rotated Thrust
        thrust_world = R @ np.array([0.0, 0.0, total_thrust], dtype=np.float32)
        gravity = np.array([0.0, 0.0, -self.mass * self.g], dtype=np.float32)
        accel = (thrust_world + gravity) / self.mass

        # Torques in body frame (Standard X-configuration):
        l = self.arm_length / np.sqrt(2.0)
        tau_x = l * (thrusts[3] + thrusts[0] - thrusts[1] - thrusts[2])
        tau_y = l * (thrusts[0] + thrusts[1] - thrusts[2] - thrusts[3])
        tau_z = self.torque_coeff * (thrusts[0] - thrusts[1] + thrusts[2] - thrusts[3])
        torques = np.array([tau_x, tau_y, tau_z], dtype=np.float32)

        # Angular acceleration (Euler's equations: I*dw/dt = tau - w x (I*w))
        gyro = np.cross(omega, self.I * omega)
        alpha = (torques - gyro) / self.I

        # 4. Integrate dynamics (Semi-Implicit Euler integration)
        new_vel = vel + accel * self.dt
        new_pos = pos + new_vel * self.dt

        new_omega = omega + alpha * self.dt
        new_angles = np.array([roll, pitch, yaw]) + new_omega * self.dt

        # Update state
        self.state = np.concatenate([new_pos, new_vel, new_angles, new_omega]).astype(np.float32)

        # 5. Compute Reward
        pos_error = np.linalg.norm(new_pos - self.target_pos)
        vel_penalty = 0.1 * np.linalg.norm(new_vel)
        angle_penalty = 0.2 * np.linalg.norm(new_angles[0:2])  # Penalize tilt (roll/pitch)
        action_penalty = 0.01 * np.sum(np.square(action))

        # Dense tracking reward
        reward = float(1.0 / (1.0 + pos_error) - vel_penalty - angle_penalty - action_penalty)

        # 6. Check Termination & Truncation conditions
        crashed = new_pos[2] < 0.0 or abs(new_angles[0]) > np.deg2rad(60) or abs(new_angles[1]) > np.deg2rad(60)
        out_of_bounds = np.any(np.abs(new_pos[0:2]) > 5.0) or new_pos[2] > 5.0
        terminated = bool(crashed or out_of_bounds)

        # Truncated if maximum steps reached
        truncated = bool(self.current_step >= self.max_steps)

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info


# =====================================================================
# Demonstration & Usage of the Gym Template
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("1. Creating Quadrotor Gym Environment Instance")
    print("=" * 65)
    env = QuadrotorEnv(target_pos=(0.0, 0.0, 1.0), dt=0.02, max_steps=100)

    print(f"Action Space:         {env.action_space}")
    print(f"Observation Space:    {env.observation_space}")
    print(f"Hover Thrust / motor: {env.hover_thrust:.4f} N")

    print("\n" + "=" * 65)
    print("2. Environment Reset (env.reset)")
    print("=" * 65)
    obs, info = env.reset(seed=42)
    print("Initial Observation (12-dim state):")
    print(f"  Position (x, y, z):       {obs[0:3]}")
    print(f"  Linear Velocity:          {obs[3:6]}")
    print(f"  Euler Angles (r, p, y):   {obs[6:9]}")
    print(f"  Angular Velocity (p,q,r): {obs[9:12]}")
    print(f"Initial Info Dictionary:    {info}")

    print("\n" + "=" * 65)
    print("3. Stepping Through Actions (env.step)")
    print("=" * 65)

    total_reward = 0.0
    num_steps = 10

    for step in range(1, num_steps + 1):
        # Action Options:
        # 1. Random action: action = env.action_space.sample()
        # 2. Hover action:  action = np.zeros(4, dtype=np.float32) (maps 0.0 to hover thrust)
        action = np.zeros(4, dtype=np.float32)

        # Step the environment: returns (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        pos = obs[0:3]
        vel = obs[3:6]
        rpy_deg = np.rad2deg(obs[6:9])

        print(
            f"Step {step:02d} | "
            f"Pos: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}] | "
            f"Vel Z: {vel[2]:6.3f} m/s | "
            f"RPY (deg): [{rpy_deg[0]:5.1f}, {rpy_deg[1]:5.1f}, {rpy_deg[2]:5.1f}] | "
            f"Reward: {reward:6.3f} | "
            f"Term/Trunc: {terminated}/{truncated}"
        )

        if terminated or truncated:
            print("Episode reached terminal state. Resetting environment...")
            obs, info = env.reset()

    print("\n" + "=" * 65)
    print(f"Completed {num_steps} steps. Total Reward Accumulated: {total_reward:.4f}")
    print("=" * 65)

