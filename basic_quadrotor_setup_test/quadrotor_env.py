"""
6-DOF Quadrotor Gymnasium Environment with Native MuJoCo Physics & Visualizer.
"""

import os
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class QuadrotorEnv(gym.Env):
    """
    Quadrotor Gymnasium Environment powered directly by MuJoCo physics engine
    and MuJoCo native offscreen visualizer.

    State Representation (12-dim):
        [0:3]   Position (x, y, z) in meters [world frame]
        [3:6]   Linear velocity (vx, vy, vz) in m/s [world frame]
        [6:9]   Euler angles (roll, pitch, yaw) in radians
        [9:12]  Angular velocity (p, q, r) in rad/s [body frame]

    Action Space (4-dim):
        Normalized rotor thrust commands [-1.0, 1.0] for the 4 motors.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        xml_path=None,
        target_pos=(0.0, 0.0, 1.0),
        dt=0.01,
        frame_skip=2,
        max_steps=500,
        render_mode="rgb_array",
        camera_name="track_cam",
        render_width=720,
        render_height=540,
    ):
        super().__init__()

        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "quadrotor.xml")
        self.xml_path = xml_path

        # Load MuJoCo Model and Data
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.model.opt.timestep = dt
        self.data = mujoco.MjData(self.model)

        self.dt = dt
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.target_pos = np.array(target_pos, dtype=np.float32)

        # Precise Quadrotor Body Mass (0.500 kg)
        self.quadrotor_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
        self.mass = float(self.model.body_mass[self.quadrotor_body_id])
        self.g = 9.81
        self.hover_thrust = (self.mass * self.g) / 4.0
        self.max_motor_thrust = float(self.model.actuator_ctrlrange[0, 1])

        # Action Space: 4 normalized inputs in [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32,
        )

        # Observation Space: 12-dim state
        high_obs = np.array([
            10.0, 10.0, 10.0,       # pos (m)
            20.0, 20.0, 20.0,       # vel (m/s)
            np.pi, np.pi, np.pi,    # roll, pitch, yaw (rad)
            20.0, 20.0, 20.0,       # p, q, r (rad/s)
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-high_obs,
            high=high_obs,
            dtype=np.float32,
        )

        self.current_step = 0

        # MuJoCo Offscreen Renderer
        self.renderer = None
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)

    def set_target_pos(self, target_pos):
        """Dynamically update target waypoint position in physics and visualizer."""
        self.target_pos = np.array(target_pos, dtype=np.float32)
        self.data.mocap_pos[0] = self.target_pos

    def _quat_to_euler(self, q):
        """Convert MuJoCo quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw]."""
        w, x, y, z = q
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        pitch = np.arcsin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw], dtype=np.float32)

    def _get_obs(self):
        """Extract 12-dim state from MuJoCo Data."""
        pos = self.data.qpos[0:3].copy()
        quat = self.data.qpos[3:7].copy()
        rpy = self._quat_to_euler(quat)
        vel = self.data.qvel[0:3].copy()
        omega = self.data.qvel[3:6].copy()

        return np.concatenate([pos, vel, rpy, omega]).astype(np.float32)

    def _get_info(self):
        pos = self.data.qpos[0:3]
        dist = np.linalg.norm(pos - self.target_pos)
        return {
            "step": self.current_step,
            "dist_to_target": float(dist),
            "altitude": float(pos[2]),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset MuJoCo physics
        mujoco.mj_resetData(self.model, self.data)

        # Set target position in mocap body
        self.data.mocap_pos[0] = self.target_pos

        # Initialize quadrotor at origin near ground
        init_pos = self.np_random.uniform(low=[-0.02, -0.02, 0.2], high=[0.02, 0.02, 0.22])
        self.data.qpos[0:3] = init_pos
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0

        # Step forward once to compute kinematics
        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        self.current_step += 1

        # 1. Map action [-1.0, 1.0] -> Motor thrust commands [0, max_motor_thrust]
        action = np.clip(action, -1.0, 1.0)
        thrusts = (action + 1.0) * 0.5 * self.max_motor_thrust
        self.data.ctrl[:] = thrusts

        # 2. Step MuJoCo physics forward
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # 3. Get state observation
        obs = self._get_obs()
        pos = obs[0:3]
        vel = obs[3:6]
        rpy = obs[6:9]

        # 4. Compute Reward
        pos_error = np.linalg.norm(pos - self.target_pos)
        vel_penalty = 0.1 * np.linalg.norm(vel)
        tilt_penalty = 0.2 * np.linalg.norm(rpy[0:2])
        action_penalty = 0.01 * np.sum(np.square(action))

        reward = float(1.0 / (1.0 + pos_error) - vel_penalty - tilt_penalty - action_penalty)

        # 5. Check Termination & Truncation
        crashed = pos[2] < 0.05 or abs(rpy[0]) > np.deg2rad(70) or abs(rpy[1]) > np.deg2rad(70)
        out_of_bounds = np.any(np.abs(pos[0:2]) > 6.0) or pos[2] > 6.0
        terminated = bool(crashed or out_of_bounds)
        truncated = bool(self.current_step >= self.max_steps)

        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def render(self, camera_name=None):
        """Render RGB frame directly from MuJoCo Visualizer."""
        if self.renderer is None:
            return None

        cam = camera_name if camera_name is not None else self.camera_name
        self.renderer.update_scene(self.data, camera=cam)
        rgb_frame = self.renderer.render()
        return rgb_frame

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
