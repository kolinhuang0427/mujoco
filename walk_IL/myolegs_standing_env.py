#!/usr/bin/env python3
"""
MyoLegs Standing Environment
A Gymnasium environment for training the MyoLegs humanoid model to stand upright.
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import os
from typing import Dict, Any, Tuple, Optional


class MyoLegsStandingEnv(gym.Env):
    """
    Custom environment for training MyoLegs to stand using muscle actuators.
    
    Action Space: Continuous muscle activation levels [0, 1]
    Observation Space: Joint positions, velocities, torso orientation, foot contacts, and muscle states
    Reward: Encourages upright posture and stability
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Load model
        model_path = os.path.join(os.path.dirname(__file__), "model", "myolegs", "myolegs.xml")
        if not os.path.exists(model_path):
            # Try alternative path
            model_path = "../model/myolegs/myolegs.xml"
        
        print(f"Loading MyoLegs model from: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Environment parameters
        self.dt = self.model.opt.timestep
        self.max_episode_steps = 1000
        self.current_step = 0
        
        # Get all muscle actuator names
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
        
        # Get key joint names for observation
        self.joint_names = [
            'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 
            'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
            'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 
            'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l'
        ]
        
        # Observation space
        obs_dim = self._get_observation_dimension()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Standing height reference (pelvis target height)
        self.target_height = 0.92  # Natural standing height for MyoLegs model
        
        # Rendering
        self.render_mode = render_mode
        self.viewer = None
        
        print(f"✅ MyoLegs Standing Environment initialized")
        print(f"   - Action space: {self.action_space.shape} (muscle activations)")
        print(f"   - Observation space: {self.observation_space.shape}")
        print(f"   - Max episode steps: {self.max_episode_steps}")
        
    def _get_observation_dimension(self) -> int:
        """Calculate observation dimension."""
        # Joint positions (14) + joint velocities (14) + pelvis pose (7) + pelvis velocity (6) 
        # + foot contacts (2) + muscle activations (80) = 123
        return 14 + 14 + 7 + 6 + 2 + len(self.muscle_names)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Joint positions and velocities for key joints
        joint_positions = []
        joint_velocities = []
        
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            
            if joint_id >= 0 and joint_id < len(self.data.qpos):
                joint_positions.append(self.data.qpos[joint_id])
                if joint_id < len(self.data.qvel):
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
        
        # Combine all observations
        obs = np.concatenate([
            joint_positions,      # 14 elements
            joint_velocities,     # 14 elements  
            pelvis_pos,          # 3 elements
            pelvis_quat,         # 4 elements
            pelvis_vel,          # 6 elements
            foot_contacts,       # 2 elements
            muscle_activations   # 80 elements
        ]).astype(np.float32)
        
        return obs
    
    def _get_foot_contacts(self) -> np.ndarray:
        """Detect foot contact with ground."""
        contacts = np.zeros(2)  # [right_foot, left_foot]
        
        # Check contacts for right and left feet
        foot_geom_names = ['r_foot_col4', 'l_foot_col4']  # Main foot collision geoms
        
        for i, foot_geom_name in enumerate(foot_geom_names):
            foot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_geom_name)
            
            if foot_geom_id >= 0:
                for j in range(self.data.ncon):
                    contact = self.data.contact[j]
                    if (contact.geom1 == foot_geom_id or contact.geom2 == foot_geom_id):
                        # Check if contact is with floor
                        other_geom = contact.geom2 if contact.geom1 == foot_geom_id else contact.geom1
                        floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
                        if other_geom == floor_geom_id:
                            contacts[i] = 1.0
                            break
        
        return contacts
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current state."""
        reward = 0.0
        
        # 1. Upright posture reward
        pelvis_pos = self.data.qpos[:3]
        pelvis_quat = self.data.qpos[3:7]
        
        # Height reward (encourage standing tall)
        height_error = abs(pelvis_pos[2] - self.target_height)
        height_reward = np.exp(-10 * height_error)  # Exponential reward for being at target height
        reward += 3.0 * height_reward
        
        # Upright orientation reward  
        # Convert quaternion to rotation matrix and check z-axis alignment
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, pelvis_quat)
        rot_mat = rot_mat.reshape(3, 3)
        up_vec = rot_mat[:, 2]  # z-axis of pelvis
        uprightness = up_vec[2]  # How much the z-axis points up
        upright_reward = max(0, uprightness)  # Reward for being upright
        reward += 4.0 * upright_reward
        
        # 2. Stability rewards
        pelvis_vel = self.data.qvel[:6]
        
        # Minimal velocity reward (encourage staying still)
        lin_vel_penalty = np.linalg.norm(pelvis_vel[:3])
        ang_vel_penalty = np.linalg.norm(pelvis_vel[3:])
        velocity_reward = np.exp(-3 * (lin_vel_penalty + ang_vel_penalty))
        reward += 1.5 * velocity_reward
        
        # 3. Joint posture reward (encourage natural standing pose)
        joint_positions = []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0 and joint_id < len(self.data.qpos):
                joint_positions.append(self.data.qpos[joint_id])
            else:
                joint_positions.append(0.0)
        
        joint_positions = np.array(joint_positions)
        # Encourage slight knee bend and neutral hip position
        target_pose = np.array([
            -0.05, 0.0, 0.0, 0.1, -0.05, 0.0, 0.0,  # Right leg
            -0.05, 0.0, 0.0, 0.1, -0.05, 0.0, 0.0   # Left leg
        ])
        joint_deviation = np.linalg.norm(joint_positions - target_pose)
        posture_reward = np.exp(-1 * joint_deviation)
        reward += 1.0 * posture_reward
        
        # 4. Foot contact reward (encourage both feet on ground)
        foot_contacts = self._get_foot_contacts()
        contact_reward = np.sum(foot_contacts) / 2.0  # Normalize to [0, 1]
        reward += 2.0 * contact_reward
        
        # 5. Muscle coordination reward (encourage balanced muscle activation)
        muscle_activations = self.data.ctrl
        
        # Penalize excessive muscle activation (energy efficiency)
        energy_penalty = np.mean(muscle_activations**2)
        reward -= 0.1 * energy_penalty
        
        # Encourage symmetric muscle activation between legs
        n_muscles_per_leg = len(self.muscle_names) // 2
        right_muscles = muscle_activations[:n_muscles_per_leg]
        left_muscles = muscle_activations[n_muscles_per_leg:]
        symmetry_reward = np.exp(-2 * np.linalg.norm(right_muscles - left_muscles))
        reward += 0.5 * symmetry_reward
        
        # 6. Encourage baseline muscle tone for postural stability
        baseline_activation = 0.15
        muscle_tone_reward = np.exp(-5 * np.mean((muscle_activations - baseline_activation)**2))
        reward += 0.5 * muscle_tone_reward
        
        return reward
    
    def _is_done(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if fallen (pelvis too low or too tilted)
        pelvis_pos = self.data.qpos[:3]
        pelvis_quat = self.data.qpos[3:7]
        
        # Height check - natural standing height is ~0.98m, so terminate if much lower
        if pelvis_pos[2] < 0.7:  # Pelvis below 70cm (significant fall)
            return True
        
        # Tilt check
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, pelvis_quat)
        rot_mat = rot_mat.reshape(3, 3)
        up_vec = rot_mat[:, 2]
        if up_vec[2] < 0.3:  # Tilted more than ~70 degrees
            return True
        
        # Time limit
        if self.current_step >= self.max_episode_steps:
            return True
        
        return False
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset simulation to natural pose
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial pose (standing position with small random perturbations)
        if seed is not None:
            np.random.seed(seed)
        
        # Set root body (freejoint) pose: position + quaternion
        # Position: slight variation around natural standing position
        self.data.qpos[0] = np.random.normal(0, 0.01)    # x position
        self.data.qpos[1] = np.random.normal(0, 0.01)    # y position  
        self.data.qpos[2] = 1.0 + np.random.normal(0, 0.01)  # z position (height)
        
        # Quaternion: upright orientation with small perturbations
        # Start with identity quaternion [1, 0, 0, 0] (no rotation)
        quat_noise = np.random.normal(0, 0.02, 3)  # Small rotation noise
        self.data.qpos[3] = 1.0  # w component
        self.data.qpos[4:7] = quat_noise  # x, y, z components
        
        # Normalize quaternion
        quat = self.data.qpos[3:7]
        quat = quat / np.linalg.norm(quat)
        self.data.qpos[3:7] = quat
        
        # Add small random perturbations to other joint angles
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0 and joint_id < len(self.data.qpos):
                # Find the qpos index for this joint
                qpos_idx = self.model.jnt_qposadr[joint_id]
                if qpos_idx >= 7:  # Skip the freejoint (first 7 elements)
                    noise = np.random.normal(0, 0.02)  # Small joint angle noise
                    self.data.qpos[qpos_idx] = noise
        
        # Zero velocities
        self.data.qvel[:] = 0.0
        
        # Set initial muscle activation (baseline muscle tone for postural stability)
        baseline_activation = 0.15 + np.random.normal(0, 0.01, len(self.muscle_names))
        self.data.ctrl[:] = np.clip(baseline_activation, 0.05, 0.3)  # More constrained range
        
        # Forward dynamics to settle into stable pose
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        # Ensure action is within bounds [0, 1]
        action = np.clip(action, 0.0, 1.0)
        
        # Apply muscle activations
        self.data.ctrl[:] = action
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get observation and reward
        obs = self._get_observation()
        reward = self._calculate_reward()
        
        # Check termination
        terminated = self._is_done()
        truncated = False
        
        self.current_step += 1
        
        # Info dict
        info = {
            'pelvis_height': self.data.qpos[2],
            'uprightness': self._get_uprightness(),
            'foot_contacts': self._get_foot_contacts(),
            'muscle_activation_mean': np.mean(self.data.ctrl),
            'episode_step': self.current_step
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_uprightness(self) -> float:
        """Get uprightness measure (0-1, 1 being perfectly upright)."""
        pelvis_quat = self.data.qpos[3:7]
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, pelvis_quat)
        rot_mat = rot_mat.reshape(3, 3)
        up_vec = rot_mat[:, 2]
        return max(0, up_vec[2])
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.distance = 4.0
                self.viewer.cam.elevation = -20
                self.viewer.cam.azimuth = 135
            
            self.viewer.sync()
        
        elif self.render_mode == "rgb_array":
            import mujoco
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
    id='MyoLegsStanding-v0',
    entry_point='myolegs_standing_env:MyoLegsStandingEnv',
    max_episode_steps=1000,
    reward_threshold=900.0,
)


if __name__ == "__main__":
    # Test the environment
    env = MyoLegsStandingEnv(render_mode="human")
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Number of muscles: {len(env.muscle_names)}")
    
    for i in range(1000):
        # Start with baseline muscle activation + small random variation
        action = np.clip(0.15 + 0.05 * np.random.randn(len(env.muscle_names)), 0, 1)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            print(f"Final info: {info}")
            obs, reset_info = env.reset()
        
        if i % 100 == 0:
            print(f"Step {i}, Reward: {reward:.3f}, Height: {info.get('pelvis_height', 0):.3f}, "
                  f"Uprightness: {info.get('uprightness', 0):.3f}")
    
    env.close() 