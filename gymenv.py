import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import time

class QuadrupedEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path("assembly_v5/mjcf/assembly_v5.xml")
        self.data = mujoco.MjData(self.model)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.model.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.model.nq + self.model.nv,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self.render()
        observation = self._get_observation()
        reward = 0.0  # Define your reward function here
        terminated = False  # Define your termination condition here
        info = {}
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def _get_observation(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)