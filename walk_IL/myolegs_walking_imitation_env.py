#!/usr/bin/env python3
"""
MyoLegs Walking Imitation Learning Environment
A Gymnasium environment for training the MyoLegs humanoid model to walk using expert motion capture data.
Uses imitation learning with rewards based on expert trajectory matching, height, and velocity.
"""

import numpy as np
import pandas as pd
import pickle
import mujoco
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import os
from typing import Dict, Any, Tuple, Optional
import torch
from stable_baselines3 import PPO


class MyoLegsWalkingImitationEnv(gym.Env):
    """
    Imitation learning environment for MyoLegs walking using expert data.
    
    Action Space: Continuous muscle activation levels [0, 1]
    Observation Space: Joint positions, velocities, torso orientation, foot contacts, muscle states, and expert reference
    Reward: Expert trajectory matching + height maintenance + forward velocity + alive bonus
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, expert_data_path: str = "walk_IL/data/expert_data.pkl", 
                 render_mode: Optional[str] = None):
        super().__init__()
        
        # Load MyoLegs model with proper path resolution
        model_path = self._find_model_path()
        
        print(f"Loading MyoLegs model from: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Load expert data
        self.expert_data = self._load_expert_data(expert_data_path)
        self.expert_timestep = 0
        self.expert_cycle_length = len(self.expert_data)
        
        # Environment parameters
        self.dt = self.model.opt.timestep
        self.max_episode_steps = min(1000, self.expert_cycle_length * 2)  # Allow 2 full cycles
        self.current_step = 0
        
        # Get muscle actuator names
        self.muscle_names = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self.muscle_names.append(name)
        
        print(f"Found {len(self.muscle_names)} muscle actuators")
        
        # Action space: muscle activation levels [0, 1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(len(self.muscle_names),), dtype=np.float32
        )
        
        # Joint names matching expert data
        self.joint_names = [
            'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 
            'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
            'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 
            'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l'
        ]
        
        # Create joint mapping
        self.joint_qpos_map = {}
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                qpos_idx = self.model.jnt_qposadr[joint_id]
                self.joint_qpos_map[joint_name] = qpos_idx
        
        # Observation space
        obs_dim = self._get_observation_dimension()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Target walking parameters
        self.target_height = 0.92  # Natural walking height
        self.target_velocity = 1.0  # Target forward velocity (m/s)
        self.height_threshold = 0.8  # Minimum height to maintain
        
        # Rendering
        self.render_mode = render_mode
        self.viewer = None
        
        print(f"✅ MyoLegs Walking Imitation Environment initialized")
        print(f"   - Action space: {self.action_space.shape} (muscle activations)")
        print(f"   - Observation space: {self.observation_space.shape}")
        print(f"   - Expert data: {self.expert_cycle_length} frames")
        print(f"   - Max episode steps: {self.max_episode_steps}")
    
    def _find_model_path(self) -> str:
        """Find the MyoLegs model file with robust path resolution."""
        # List of possible paths to try
        possible_paths = [
            # Relative to current script directory
            os.path.join(os.path.dirname(__file__), "model", "myolegs", "myolegs.xml"),
            os.path.join(os.path.dirname(__file__), "..", "model", "myolegs", "myolegs.xml"),
            # Relative to current working directory
            "model/myolegs/myolegs.xml",
            "../model/myolegs/myolegs.xml",
            # Absolute path based on current working directory
            os.path.join(os.getcwd(), "model", "myolegs", "myolegs.xml"),
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        
        # If none found, raise informative error
        raise FileNotFoundError(
            f"MyoLegs model not found. Tried paths:\n" + 
            "\n".join(f"  - {os.path.abspath(p)}" for p in possible_paths) +
            f"\nCurrent working directory: {os.getcwd()}"
        )
        
    def _load_expert_data(self, data_path: str) -> pd.DataFrame:
        """Load expert motion capture data."""
        # Convert to absolute path for subprocess safety
        abs_data_path = os.path.abspath(data_path)
        
        if not os.path.exists(abs_data_path):
            raise FileNotFoundError(f"Expert data not found: {abs_data_path}")
        
        print(f"Loading expert data from: {abs_data_path}")
        with open(abs_data_path, 'rb') as f:
            expert_data = pickle.load(f)
        
        if 'qpos' not in expert_data:
            raise ValueError("Expert data must contain 'qpos' key")
        
        qpos_data = expert_data['qpos']
        print(f"✅ Loaded expert data with {len(qpos_data)} frames")
        
        return qpos_data
    
    def _get_observation_dimension(self) -> int:
        """Calculate observation dimension."""
        # Joint positions (14) + joint velocities (14) + pelvis pose (7) + pelvis velocity (6) 
        # + foot contacts (2) + muscle activations (80) + expert joint positions (14) + expert phase (1) = 138
        return 14 + 14 + 7 + 6 + 2 + len(self.muscle_names) + 14 + 1
    
    def _get_expert_reference(self) -> np.ndarray:
        """Get current expert reference joint positions."""
        expert_frame = self.expert_data.iloc[self.expert_timestep % self.expert_cycle_length]
        expert_joints = []
        
        for joint_name in self.joint_names:
            if joint_name in expert_frame:
                expert_joints.append(float(expert_frame[joint_name]))
            else:
                expert_joints.append(0.0)
        
        return np.array(expert_joints)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation including expert reference."""
        # Current joint positions and velocities
        joint_positions = []
        joint_velocities = []
        
        for joint_name in self.joint_names:
            if joint_name in self.joint_qpos_map:
                qpos_idx = self.joint_qpos_map[joint_name]
                joint_positions.append(self.data.qpos[qpos_idx])
                # Get corresponding velocity
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0 and joint_id < len(self.data.qvel):
                    joint_velocities.append(self.data.qvel[joint_id])
                else:
                    joint_velocities.append(0.0)
            else:
                joint_positions.append(0.0)
                joint_velocities.append(0.0)
        
        # Pelvis (root body) pose and velocity
        pelvis_pos = self.data.qpos[:3]  # x, y, z position
        pelvis_quat = self.data.qpos[3:7]  # quaternion orientation
        pelvis_vel = self.data.qvel[:6]  # linear and angular velocity
        
        # Foot contact detection
        foot_contacts = self._get_foot_contacts()
        
        # Current muscle activations
        muscle_activations = self.data.ctrl.copy()
        
        # Expert reference
        expert_joints = self._get_expert_reference()
        
        # Phase information (normalized cycle position)
        phase = (self.expert_timestep % self.expert_cycle_length) / self.expert_cycle_length
        
        # Combine all observations
        obs = np.concatenate([
            joint_positions,      # 14 elements
            joint_velocities,     # 14 elements  
            pelvis_pos,          # 3 elements
            pelvis_quat,         # 4 elements
            pelvis_vel,          # 6 elements
            foot_contacts,       # 2 elements
            muscle_activations,  # 80 elements
            expert_joints,       # 14 elements
            [phase]              # 1 element
        ]).astype(np.float32)
        
        return obs
    
    def _get_foot_contacts(self) -> np.ndarray:
        """Detect foot contact with ground."""
        contacts = np.zeros(2)  # [right_foot, left_foot]
        
        foot_geom_names = ['r_foot_col4', 'l_foot_col4']
        
        for i, foot_geom_name in enumerate(foot_geom_names):
            foot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_geom_name)
            
            if foot_geom_id >= 0:
                for j in range(self.data.ncon):
                    contact = self.data.contact[j]
                    if (contact.geom1 == foot_geom_id or contact.geom2 == foot_geom_id):
                        other_geom = contact.geom2 if contact.geom1 == foot_geom_id else contact.geom1
                        floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
                        if other_geom == floor_geom_id:
                            contacts[i] = 1.0
                            break
        
        return contacts
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on expert matching, height, and velocity."""
        reward = 0.0
        
        # Get current and expert joint positions
        current_joints = []
        for joint_name in self.joint_names:
            if joint_name in self.joint_qpos_map:
                qpos_idx = self.joint_qpos_map[joint_name]
                current_joints.append(self.data.qpos[qpos_idx])
            else:
                current_joints.append(0.0)
        
        current_joints = np.array(current_joints)
        expert_joints = self._get_expert_reference()
        
        # 1. Expert trajectory matching reward (MUCH MORE FORGIVING)
        joint_error = np.linalg.norm(current_joints - expert_joints)
        tracking_reward = np.exp(-2 * joint_error)  # Much more forgiving (was -10)
        reward += 5.0 * tracking_reward  # Lower weight (was 15.0)
        
        # 2. Height maintenance reward (POSITIVE REWARDS)
        pelvis_height = self.data.qpos[2]
        if pelvis_height >= self.height_threshold:
            height_reward = 2.0  # Fixed positive reward for being upright
            reward += height_reward
        else:
            # Much smaller penalty (was -10.0)
            height_penalty = -5.0 * (self.height_threshold - pelvis_height)
            reward += height_penalty
        
        # 3. Forward velocity reward (POSITIVE FOCUS)
        forward_velocity = self.data.qvel[0]
        if forward_velocity > 0.0:  # Any forward movement is good
            velocity_reward = min(2.0, forward_velocity)  # Cap at 2.0, reward any forward motion
            reward += velocity_reward
        # Remove penalty for not moving forward
        
        # 4. Upright orientation reward (ALWAYS POSITIVE)
        pelvis_quat = self.data.qpos[3:7]
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, pelvis_quat)
        rot_mat = rot_mat.reshape(3, 3)
        up_vec = rot_mat[:, 2]
        uprightness = max(0, up_vec[2])
        reward += 2.0 * uprightness  # Always positive
        
        # 5. Energy efficiency (MUCH SMALLER PENALTY)
        muscle_activations = self.data.ctrl
        energy_penalty = np.mean(muscle_activations**2)
        reward -= 0.1 * energy_penalty  # Much smaller (was 0.5)
        
        # 6. Stability (SMALLER PENALTY)
        lateral_velocity = abs(self.data.qvel[1])
        angular_velocity = np.linalg.norm(self.data.qvel[3:6])
        stability_penalty = 0.2 * lateral_velocity + 0.1 * angular_velocity  # Much smaller
        reward -= stability_penalty
        
        # 7. BASE SURVIVAL REWARD
        reward += 3.0  # Large base reward for staying alive
        
        # 8. REMOVE HARSH TRACKING PENALTY
        # if joint_error > 1.0:  # REMOVED
        #     reward -= 2.0
        
        # 9. Foot contact reward (POSITIVE)
        foot_contacts = self._get_foot_contacts()
        contact_reward = np.sum(foot_contacts) * 0.5  # Small positive reward
        reward += contact_reward
        
        return reward
    
    def _is_done(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if fallen
        pelvis_height = self.data.qpos[2]
        if pelvis_height < 0.6:  # Significant fall
            return True
        
        # Terminate if too tilted
        pelvis_quat = self.data.qpos[3:7]
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, pelvis_quat)
        rot_mat = rot_mat.reshape(3, 3)
        up_vec = rot_mat[:, 2]
        if up_vec[2] < 0.2:  # Tilted more than ~78 degrees
            return True
        
        # Time limit
        if self.current_step >= self.max_episode_steps:
            return True
        
        return False
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset expert timestep (start from a random point in the cycle)
        if seed is not None:
            np.random.seed(seed)
        self.expert_timestep = np.random.randint(0, self.expert_cycle_length)
        
        # Set initial pose from expert data (with small perturbations)
        expert_frame = self.expert_data.iloc[self.expert_timestep]
        
        # Set root position
        self.data.qpos[0] = np.random.normal(0, 0.02)  # x position with noise
        self.data.qpos[1] = np.random.normal(0, 0.02)  # y position with noise
        self.data.qpos[2] = self.target_height + np.random.normal(0, 0.01)  # z position
        
        # Set upright orientation with small perturbations
        quat_noise = np.random.normal(0, 0.02, 3)
        self.data.qpos[3] = 1.0  # w component
        self.data.qpos[4:7] = quat_noise  # x, y, z components
        quat = self.data.qpos[3:7]
        quat = quat / np.linalg.norm(quat)
        self.data.qpos[3:7] = quat
        
        # Set joint angles from expert data with small noise
        for joint_name in self.joint_names:
            if joint_name in self.joint_qpos_map and joint_name in expert_frame:
                qpos_idx = self.joint_qpos_map[joint_name]
                expert_angle = float(expert_frame[joint_name])
                noise = np.random.normal(0, 0.05)  # Small joint angle noise
                self.data.qpos[qpos_idx] = expert_angle + noise
        
        # Set small random velocities
        self.data.qvel[:] = np.random.normal(0, 0.1, len(self.data.qvel))
        
        # Set initial muscle activation
        baseline_activation = 0.2 + np.random.normal(0, 0.02, len(self.muscle_names))
        self.data.ctrl[:] = np.clip(baseline_activation, 0.1, 0.4)
        
        # Reset previous action for smoothing
        self.prev_action = self.data.ctrl.copy()
        
        # Forward dynamics
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0
        
        obs = self._get_observation()
        info = {
            'expert_timestep': self.expert_timestep,
            'expert_phase': self.expert_timestep / self.expert_cycle_length
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        # Ensure action is within bounds
        action = np.clip(action, 0.0, 1.0)
        
        # Smooth action transitions (muscle activation can't change instantly)
        if hasattr(self, 'prev_action'):
            max_change = 0.1  # Max 10% change per timestep
            action = np.clip(action, 
                            self.prev_action - max_change, 
                            self.prev_action + max_change)
        
        self.prev_action = action.copy()
        
        # Apply muscle activations
        self.data.ctrl[:] = action
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Advance expert timestep
        self.expert_timestep += 1
        
        # Get observation and reward
        obs = self._get_observation()
        reward = self._calculate_reward()
        
        # Check termination
        terminated = self._is_done()
        truncated = False
        
        self.current_step += 1
        
        # Calculate tracking error for info
        current_joints = []
        for joint_name in self.joint_names:
            if joint_name in self.joint_qpos_map:
                qpos_idx = self.joint_qpos_map[joint_name]
                current_joints.append(self.data.qpos[qpos_idx])
            else:
                current_joints.append(0.0)
        current_joints = np.array(current_joints)
        expert_joints = self._get_expert_reference()
        tracking_error = np.linalg.norm(current_joints - expert_joints)
        
        info = {
            'pelvis_height': self.data.qpos[2],
            'forward_velocity': self.data.qvel[0],
            'tracking_error': tracking_error,
            'expert_timestep': self.expert_timestep % self.expert_cycle_length,
            'expert_phase': (self.expert_timestep % self.expert_cycle_length) / self.expert_cycle_length,
            'foot_contacts': self._get_foot_contacts(),
            'muscle_activation_mean': np.mean(self.data.ctrl),
            'episode_step': self.current_step
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.distance = 4.0
                self.viewer.cam.elevation = -15
                self.viewer.cam.azimuth = 135
            
            self.viewer.sync()
        
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            return renderer.render()
    
    def close(self):
        """Close the environment."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# Register the environment
register(
    id='MyoLegsWalkingImitation-v0',
    entry_point='myolegs_walking_imitation_env:MyoLegsWalkingImitationEnv',
    max_episode_steps=1000,
    reward_threshold=500.0,
)


if __name__ == "__main__":
    # Test the environment
    env = MyoLegsWalkingImitationEnv(render_mode="human")
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Expert cycle length: {env.expert_cycle_length}")
    
    # Allow larger policy updates but with structured constraints
    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512, 256, 256],  # Keep larger networks for capacity
            vf=[512, 512, 256, 256]
        ),
        activation_fn=torch.nn.Tanh,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2e-4,          # Moderate learning rate
        n_steps=2048,                # Keep larger steps for exploration
        batch_size=64,               # Smaller batches for more frequent updates
        n_epochs=8,                  # Moderate epochs
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,             # Moderate clipping - allows exploration but not chaos
        ent_coef=0.005,              # Moderate entropy - encourage exploration initially
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
    )
    
    for i in range(2000):
        # Use a simple policy: baseline activation + small variation
        action = np.clip(0.25 + 0.1 * np.random.randn(len(env.muscle_names)), 0, 1)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            print(f"Final info: {info}")
            obs, reset_info = env.reset()
        
        if i % 100 == 0:
            print(f"Step {i}, Reward: {reward:.3f}, Height: {info.get('pelvis_height', 0):.3f}, "
                  f"Velocity: {info.get('forward_velocity', 0):.3f}, "
                  f"Tracking Error: {info.get('tracking_error', 0):.3f}")
    
    env.close() 