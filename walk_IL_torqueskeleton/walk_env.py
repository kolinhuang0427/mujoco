#!/usr/bin/env python3
"""
Torque Skeleton Walking Imitation Learning Environment
A Gymnasium environment for training the humanoid torque skeleton model to walk using expert data.
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


class TorqueSkeletonWalkingEnv(gym.Env):
    """
    Imitation learning environment for torque skeleton walking using expert data.
    
    Action Space: Continuous motor torques [-1, 1] (normalized to motor control ranges)
    Observation Space: Joint positions, velocities, torso orientation, foot contacts, and expert reference
    Reward: Expert trajectory matching + height maintenance + forward velocity + alive bonus
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, expert_data_path: str = "data/expert_data.pkl", 
                 render_mode: Optional[str] = None):
        super().__init__()
        
        # Load torque skeleton model with proper path resolution
        model_path = self._find_model_path()
        
        print(f"Loading torque skeleton model from: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Store expert data path but don't load until needed
        self.expert_data_path = expert_data_path
        self.expert_data = None
        self.expert_timestep = 0
        self.expert_cycle_length = 3000  # Default based on our data
        
        # Environment parameters
        self.dt = self.model.opt.timestep
        self.base_episode_duration = 5.0  # Base episode length in seconds
        self.stage_duration_increment = 2.0  # Add 2 seconds per stage
        self.current_step = 0
        
        # Initialize stage before calculating max steps
        self.stage: int = 0  # Start at stage 0 (standing), then 1 (walking), then 2 (imitation)
        self.max_episode_steps = self._calculate_max_steps()
        
        # Joint names matching expert data (leg joints only)
        self.joint_names = [
            'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 
            'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
            'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 
            'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l'
        ]
        
        # Get motor actuator names and control ranges - ONLY for joints in expert data
        self.motor_names = []
        self.motor_ctrl_ranges = []
        self.active_motor_indices = []  # Track which motors are active
        self.all_motor_names = []  # Keep track of all motors for disabling unused ones
        
        # Create mapping from joint names to expected motor names
        joint_to_motor_map = {
            'hip_flexion_r': 'mot_hip_flexion_r',
            'hip_adduction_r': 'mot_hip_adduction_r', 
            'hip_rotation_r': 'mot_hip_rotation_r',
            'knee_angle_r': 'mot_knee_angle_r',
            'ankle_angle_r': 'mot_ankle_angle_r',
            'subtalar_angle_r': 'mot_subtalar_angle_r',
            'mtp_angle_r': 'mot_mtp_angle_r',
            'hip_flexion_l': 'mot_hip_flexion_l',
            'hip_adduction_l': 'mot_hip_adduction_l',
            'hip_rotation_l': 'mot_hip_rotation_l', 
            'knee_angle_l': 'mot_knee_angle_l',
            'ankle_angle_l': 'mot_ankle_angle_l',
            'subtalar_angle_l': 'mot_subtalar_angle_l',
            'mtp_angle_l': 'mot_mtp_angle_l'
        }
        
        # First pass: collect all motor names
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self.all_motor_names.append((i, name))
        
        # Second pass: only include motors for joints in expert data
        for joint_name in self.joint_names:
            expected_motor_name = joint_to_motor_map.get(joint_name)
            if expected_motor_name:
                # Find the motor index
                for motor_idx, motor_name in self.all_motor_names:
                    if motor_name == expected_motor_name:
                        self.motor_names.append(motor_name)
                        ctrl_range = self.model.actuator_ctrlrange[motor_idx]
                        self.motor_ctrl_ranges.append(ctrl_range)
                        self.active_motor_indices.append(motor_idx)
                        break
                else:
                    print(f"Warning: Motor for joint '{joint_name}' (expected '{expected_motor_name}') not found")
        
        self.motor_ctrl_ranges = np.array(self.motor_ctrl_ranges)
        print(f"Found {len(self.motor_names)} active motor actuators (out of {len(self.all_motor_names)} total)")
        print(f"Active motors: {self.motor_names}")
        
        # Create set of inactive motor indices for efficient lookup
        active_set = set(self.active_motor_indices)
        self.inactive_motor_indices = [idx for idx, _ in self.all_motor_names if idx not in active_set]
        print(f"Disabled {len(self.inactive_motor_indices)} motor actuators")
        
        # Action space: normalized motor torques [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.motor_names),), dtype=np.float32
        )
        
        # Create joint mapping
        self.joint_qpos_map = {}
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                qpos_idx = self.model.jnt_qposadr[joint_id]
                self.joint_qpos_map[joint_name] = qpos_idx
            else:
                print(f"Warning: Joint {joint_name} not found in model")
        
        # Observation space
        obs_dim = self._get_observation_dimension()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Target walking parameters
        self.target_height = 0.5  # Natural walking height for this model
        self.target_velocity = 1.0  # Target forward velocity (m/s)
        self.height_threshold = 0.3  # Minimum height to maintain
        
        # Rendering
        self.render_mode = render_mode
        self.viewer = None
        
        # Curriculum learning parameters
        self.global_step: int = 0
        self.stage_change_step: int = 0
        self.standing_steps: int = 0
        self.walking_steps: int = 0
        self.prev_stage: int = 0
        
        # Tracking reward schedule
        self.tracking_warmup_steps: int = 100_000
        self.tracking_weight_max: float = 8.0
        self.tracking_weight_min: float = 1.0
        
        # Energy penalty schedule
        self.energy_warmup_steps: int = 500_000
        self.energy_penalty_max: float = 0.05
        
        # Previous values for smoothing rewards
        self.prev_forward_velocity: float = 0.0
        self.prev_action = None
        
        # Episode success tracking
        self.consecutive_success_episodes: int = 0
        self.required_success_episodes: int = 3
        self.current_episode_standing: bool = False
        self.current_episode_walking: bool = False
        self.current_episode_tracking: bool = False
        
        # Joint error threshold for imitation stage
        self.joint_error_threshold: float = 0.5  # radians
        
        print(f"✅ Torque Skeleton Walking Environment initialized")
        print(f"   - Action space: {self.action_space.shape} (motor torques)")
        print(f"   - Observation space: {self.observation_space.shape}")
        print(f"   - Simulation timestep: {self.dt}s ({1/self.dt:.0f} Hz)")
        print(f"   - Episode length: Stage {self.stage} = {self.base_episode_duration + (self.stage * self.stage_duration_increment)}s ({self.max_episode_steps} steps)")
        print(f"   - Expert data: {self.expert_cycle_length} frames")
    
    def _calculate_max_steps(self) -> int:
        """Calculate max episode steps based on current stage."""
        episode_duration = self.base_episode_duration + (self.stage * self.stage_duration_increment)
        steps = int(episode_duration / self.dt)
        return steps
    
    def _find_model_path(self) -> str:
        """Find the torque skeleton model file with robust path resolution."""
        possible_paths = [
            # Relative to current script directory
            os.path.join(os.path.dirname(__file__), "..", "model", "torque_skeleton", "humanoid_torque.xml"),
            # Relative to current working directory
            "model/torque_skeleton/humanoid_torque.xml",
            "../model/torque_skeleton/humanoid_torque.xml",
            # Absolute path based on current working directory
            os.path.join(os.getcwd(), "model", "torque_skeleton", "humanoid_torque.xml"),
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        
        raise FileNotFoundError(
            f"Torque skeleton model not found. Tried paths:\n" + 
            "\n".join(f"  - {os.path.abspath(p)}" for p in possible_paths) +
            f"\nCurrent working directory: {os.getcwd()}"
        )
        
    def _load_expert_data(self):
        """Load expert motion capture data when needed (stage 2+)."""
        if self.expert_data is not None:
            return  # Already loaded
            
        abs_data_path = os.path.abspath(self.expert_data_path)
        
        if not os.path.exists(abs_data_path):
            raise FileNotFoundError(f"Expert data not found: {abs_data_path}")
        
        print(f"🎯 Loading expert data for stage 2: {abs_data_path}")
        with open(abs_data_path, 'rb') as f:
            expert_data = pickle.load(f)
        
        if 'qpos' not in expert_data:
            raise ValueError("Expert data must contain 'qpos' key")
        
        qpos_data = expert_data['qpos']
        self.expert_data = qpos_data
        self.expert_cycle_length = len(qpos_data)
        
        print(f"Expert data loaded: {self.expert_cycle_length} frames")
        print(f"Joint columns: {list(qpos_data.columns)}")
    
    def _get_observation_dimension(self) -> int:
        """Calculate observation space dimension."""
        # Root position (3) + root orientation quaternion (4) + root velocity (6)
        root_dim = 13
        
        # Joint positions and velocities (for all actuated joints)
        joint_dim = 2 * len(self.joint_names)  # pos + vel
        
        # Foot contacts (4 contact points)
        contact_dim = 4
        
        # Expert reference (only in stage 2+, but we include in obs space)
        expert_dim = len(self.joint_names)
        
        # Additional body state (center of mass, orientation features)
        body_dim = 6
        
        total_dim = root_dim + joint_dim + contact_dim + expert_dim + body_dim
        return total_dim
    
    def _get_expert_reference(self) -> np.ndarray:
        """Get expert joint angles for current timestep."""
        if self.expert_data is None or self.stage < 2:
            return np.zeros(len(self.joint_names))
        
        expert_frame = self.expert_data.iloc[self.expert_timestep % self.expert_cycle_length]
        expert_joints = []
        
        for joint_name in self.joint_names:
            if joint_name in expert_frame:
                expert_joints.append(float(expert_frame[joint_name]))
            else:
                expert_joints.append(0.0)
        
        return np.array(expert_joints)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs_parts = []
        
        # Root state (position, orientation, velocity)
        obs_parts.append(self.data.qpos[:3])  # root position
        obs_parts.append(self.data.qpos[3:7])  # root quaternion  
        obs_parts.append(self.data.qvel[:6])   # root linear and angular velocity
        
        # Joint positions and velocities
        joint_pos = []
        joint_vel = []
        for joint_name in self.joint_names:
            if joint_name in self.joint_qpos_map:
                qpos_idx = self.joint_qpos_map[joint_name]
                joint_pos.append(self.data.qpos[qpos_idx])
                # Find corresponding qvel index
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qvel_idx = self.model.jnt_dofadr[joint_id]
                joint_vel.append(self.data.qvel[qvel_idx])
            else:
                joint_pos.append(0.0)
                joint_vel.append(0.0)
        
        obs_parts.append(np.array(joint_pos))
        obs_parts.append(np.array(joint_vel))
        
        # Foot contacts
        foot_contacts = self._get_foot_contacts()
        obs_parts.append(foot_contacts)
        
        # Expert reference
        expert_ref = self._get_expert_reference()
        obs_parts.append(expert_ref)
        
        # Additional body state features
        pelvis_quat = self.data.qpos[3:7]
        rot_mat_tmp = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat_tmp, pelvis_quat)
        rot_mat_tmp = rot_mat_tmp.reshape(3, 3)
        up_vec = rot_mat_tmp[:, 2]
        forward_vec = rot_mat_tmp[:, 1]
        
        body_features = np.array([
            up_vec[2],  # uprightness
            forward_vec[0],  # forward orientation
            self.data.qpos[2],  # height
            self.data.qvel[0],  # forward velocity
            self.data.qvel[1],  # lateral velocity
            self.data.qvel[2],  # vertical velocity
        ])
        obs_parts.append(body_features)
        
        return np.concatenate(obs_parts)
    
    def _get_foot_contacts(self) -> np.ndarray:
        """Get foot contact information."""
        contacts = np.zeros(4)
        
        # Check for contacts with floor
        for i, contact in enumerate(self.data.contact[:self.data.ncon]):
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            
            if geom1_name == "floor" or geom2_name == "floor":
                other_geom = geom1_name if geom2_name == "floor" else geom2_name
                
                # Map foot geoms to contact indices
                if "r_foot" in other_geom:
                    contacts[0] = 1.0  # right heel
                    contacts[1] = 1.0  # right toe
                elif "l_foot" in other_geom:
                    contacts[2] = 1.0  # left heel  
                    contacts[3] = 1.0  # left toe
                elif "r_bofoot" in other_geom:
                    contacts[1] = 1.0  # right toe
                elif "l_bofoot" in other_geom:
                    contacts[3] = 1.0  # left toe
        
        return contacts
    
    def _ramp_weight(self, full_weight: float) -> float:
        """Linearly ramp a weight from 0 to full_weight after a stage change."""
        stage_duration = self.global_step - self.stage_change_step
        ramp_steps = max(1, int(0.05 * max(1, self.global_step)))
        progress = np.clip(stage_duration / ramp_steps, 0.0, 1.0)
        return full_weight * progress
    
    def _update_curriculum(self, forward_velocity: float, uprightness: float):
        """Update stage depending on performance."""
        self.prev_stage = self.stage
        
        # Update consecutive standing counter
        if uprightness > 0.8 and self.data.qpos[2] >= self.height_threshold:
            self.standing_steps += 1
        else:
            self.standing_steps = 0
        
        # Update consecutive walking counter
        if forward_velocity > 0.5:
            self.walking_steps += 1
        else:
            self.walking_steps = 0
        
        # Check for stage advancement based on episode success
        standing_threshold = int(0.7 * self.max_episode_steps)
        walking_threshold = int(0.6 * self.max_episode_steps)
        
        if self.stage == 0:  # Standing stage
            if self.standing_steps >= standing_threshold:
                self.current_episode_standing = True
        elif self.stage == 1:  # Walking stage
            if (self.standing_steps >= standing_threshold and 
                self.walking_steps >= walking_threshold):
                self.current_episode_walking = True
        
        # Stage change detection
        if self.stage != self.prev_stage:
            self.stage_change_step = self.global_step
    
    def _check_episode_completion_and_advance_stage(self):
        """Check if episode meets criteria and advance stage if needed."""
        if self.stage == 0 and self.current_episode_standing:
            self.consecutive_success_episodes += 1
        elif self.stage == 1 and self.current_episode_walking:
            self.consecutive_success_episodes += 1
        elif self.stage >= 2 and self.current_episode_tracking:
            self.consecutive_success_episodes += 1
        else:
            self.consecutive_success_episodes = 0
        
        # Advance stage if enough successful episodes
        if (self.consecutive_success_episodes >= self.required_success_episodes and 
            self.stage < 2):
            self.stage += 1
            self.consecutive_success_episodes = 0
            self.max_episode_steps = self._calculate_max_steps()
            
            # Load expert data when entering stage 2
            if self.stage == 2:
                self._load_expert_data()
            
            print(f"🎉 Advanced to stage {self.stage}!")
            print(f"   New episode length: {self.max_episode_steps} steps")
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on expert matching, height, and velocity."""
        # Update curriculum
        pelvis_quat = self.data.qpos[3:7]
        rot_mat_tmp = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat_tmp, pelvis_quat)
        rot_mat_tmp = rot_mat_tmp.reshape(3, 3)
        up_vec_tmp = rot_mat_tmp[:, 2]
        uprightness = max(0.0, up_vec_tmp[2])
        forward_velocity = self.data.qvel[0]
        self._update_curriculum(forward_velocity, uprightness)
        
        reward = 0.0
        current_stage = self.stage
        
        # 1. Height and uprightness (always active)
        pelvis_height = self.data.qpos[2]
        if self.target_height - 0.05 <= pelvis_height <= self.target_height + 0.05:
            reward += 3.0
        if pelvis_height >= self.height_threshold:
            reward += 5.0
        reward += 2.0 * uprightness
        reward += 2.0  # Survival bonus
        
        # 2. Foot contact rewards (for stability)
        foot_contacts = self._get_foot_contacts()
        reward += 0.3 * np.sum(foot_contacts)
        
        if current_stage == 0:  # Standing stage
            # Penalize forward motion, reward stability
            reward -= 0.8 * abs(forward_velocity)
            reward += 2.0 * (1.0 - abs(forward_velocity))
            
        elif current_stage >= 1:  # Walking stages
            # 3. Forward velocity reward
            vel_weight = 4.0 + self._ramp_weight(2.0)
            reward += vel_weight * np.clip(forward_velocity, 0, 2.0)
            
            # Acceleration bonus
            acceleration = forward_velocity - self.prev_forward_velocity
            self.prev_forward_velocity = forward_velocity
            reward += 2.0 * np.clip(acceleration, 0, 1.0)
            
            if current_stage >= 2:  # Imitation stage
                # 4. Joint tracking reward
                current_joints = []
                for joint_name in self.joint_names:
                    if joint_name in self.joint_qpos_map:
                        qpos_idx = self.joint_qpos_map[joint_name]
                        current_joints.append(self.data.qpos[qpos_idx])
                    else:
                        current_joints.append(0.0)
                
                current_joints = np.array(current_joints)
                expert_joints = self._get_expert_reference()
                joint_errors = np.abs(current_joints - expert_joints)
                
                # Exponential reward for joint tracking
                tracking_weight = self.tracking_weight_min + (
                    self.tracking_weight_max - self.tracking_weight_min
                ) * np.clip(self.global_step / self.tracking_warmup_steps, 0, 1)
                
                joint_reward = tracking_weight * np.exp(-2.0 * np.mean(joint_errors))
                reward += joint_reward
                
                # Check if tracking is good enough for episode success
                if np.mean(joint_errors) < self.joint_error_threshold:
                    self.current_episode_tracking = True
        
        # 5. Energy penalty (smoothness)
        if self.global_step > self.energy_warmup_steps and self.prev_action is not None:
            # Only consider active motor actions for energy penalty
            current_active_actions = np.array([self.data.ctrl[idx] for idx in self.active_motor_indices])
            action_change = np.linalg.norm(current_active_actions - self.prev_action)
            energy_penalty = self.energy_penalty_max * action_change
            reward -= energy_penalty
        
        return reward
    
    def _is_done(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if fallen
        if self.data.qpos[2] < 0.2:  # Pelvis too low
            return True
        
        # Terminate if tipped over
        pelvis_quat = self.data.qpos[3:7]
        rot_mat_tmp = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat_tmp, pelvis_quat)
        rot_mat_tmp = rot_mat_tmp.reshape(3, 3)
        up_vec = rot_mat_tmp[:, 2]
        if up_vec[2] < 0.3:  # Severely tipped
            return True
        
        # Terminate if moved too far sideways
        if abs(self.data.qpos[1]) > 2.0:
            return True
        
        # Normal episode length termination
        if self.current_step >= self.max_episode_steps:
            return True
        
        return False
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        # Check episode completion and advance stage
        self._check_episode_completion_and_advance_stage()
        
        # Reset episode flags
        self.current_episode_standing = False
        self.current_episode_walking = False
        self.current_episode_tracking = False
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset to default standing position for stages 0 and 1
        if self.stage < 2:
            # Default standing position
            self.expert_timestep = 0
            
            # Set root position (standing upright)
            self.data.qpos[0] = 0.0  # x position
            self.data.qpos[1] = 0.0  # y position  
            self.data.qpos[2] = self.target_height  # z position
            
            # Set upright orientation
            self.data.qpos[3] = 1.0  # w component
            self.data.qpos[4:7] = 0.0  # x, y, z components
            
            # Set joints to neutral standing angles
            standing_angles = {
                'hip_flexion_r': -0.02,
                'hip_adduction_r': 0.0,
                'hip_rotation_r': 0.0,
                'knee_angle_r': 0.05,
                'ankle_angle_r': -0.02,
                'subtalar_angle_r': 0.0,
                'mtp_angle_r': 0.0,
                'hip_flexion_l': -0.02,
                'hip_adduction_l': 0.0,
                'hip_rotation_l': 0.0,
                'knee_angle_l': 0.05,
                'ankle_angle_l': -0.02,
                'subtalar_angle_l': 0.0,
                'mtp_angle_l': 0.0
            }
            
            for joint_name in self.joint_names:
                if joint_name in self.joint_qpos_map:
                    qpos_idx = self.joint_qpos_map[joint_name]
                    if joint_name in standing_angles:
                        self.data.qpos[qpos_idx] = standing_angles[joint_name]
                    else:
                        self.data.qpos[qpos_idx] = 0.0
        else:
            # Use expert data for initialization (stage 2+)
            if self.expert_data is not None:
                self.expert_timestep = np.random.randint(0, self.expert_cycle_length)
                expert_frame = self.expert_data.iloc[self.expert_timestep]
                
                # Set root position with small noise
                self.data.qpos[0] = np.random.normal(0, 0.02)  # x position
                self.data.qpos[1] = np.random.normal(0, 0.02)  # y position
                self.data.qpos[2] = self.target_height + np.random.normal(0, 0.01)  # z position
                
                # Set upright orientation with small perturbations
                quat_noise = np.random.normal(0, 0.02, 3)
                self.data.qpos[3] = 1.0
                self.data.qpos[4:7] = quat_noise
                quat = self.data.qpos[3:7]
                quat = quat / np.linalg.norm(quat)
                self.data.qpos[3:7] = quat
                
                # Set joint angles from expert data with noise
                for joint_name in self.joint_names:
                    if joint_name in self.joint_qpos_map and joint_name in expert_frame:
                        qpos_idx = self.joint_qpos_map[joint_name]
                        expert_angle = float(expert_frame[joint_name])
                        noise = np.random.normal(0, 0.03)
                        self.data.qpos[qpos_idx] = expert_angle + noise
        
        # Set small random velocities for walking stages
        if self.stage < 2:
            self.data.qvel[:] = 0.0  # Start stationary
        else:
            self.data.qvel[:] = np.random.normal(0, 0.1, len(self.data.qvel))
        
        # Initialize motor controls to zero
        self.data.ctrl[:] = 0.0
        self.prev_action = np.zeros(len(self.motor_names))  # Only track active motors
        
        # Ensure inactive motors are explicitly set to zero
        for motor_idx in self.inactive_motor_indices:
            self.data.ctrl[motor_idx] = 0.0
        
        # Forward dynamics
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0
        self.prev_forward_velocity = 0.0
        
        obs = self._get_observation()
        
        info = {
            'stage': self.stage,
            'expert_timestep': self.expert_timestep,
            'episode_steps': self.max_episode_steps,
            'global_step': self.global_step,
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        # Ensure action is within bounds
        action = np.clip(action, -1.0, 1.0)
        
        # Initialize all motor controls to zero
        self.data.ctrl[:] = 0.0
        
        # Convert normalized action to motor control values for ACTIVE motors only
        for i, motor_idx in enumerate(self.active_motor_indices):
            low, high = self.motor_ctrl_ranges[i]
            motor_command = low + (action[i] + 1.0) * 0.5 * (high - low)
            self.data.ctrl[motor_idx] = motor_command
        
        # Ensure inactive motors remain at zero (redundant but explicit)
        for motor_idx in self.inactive_motor_indices:
            self.data.ctrl[motor_idx] = 0.0
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Advance expert timestep
        if self.stage >= 2 and self.expert_data is not None:
            self.expert_timestep = (self.expert_timestep + 1) % self.expert_cycle_length
        
        # Increment counters
        self.global_step += 1
        self.current_step += 1
        
        # Get observation and reward
        obs = self._get_observation()
        reward = self._calculate_reward()
        
        # Check termination
        terminated = self._is_done()
        truncated = False
        
        # Calculate tracking error for info
        if self.stage >= 2:
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
        else:
            tracking_error = 0.0
        
        # Store current active motor actions for next step
        self.prev_action = np.array([self.data.ctrl[idx] for idx in self.active_motor_indices])
        
        info = {
            'stage': self.stage,
            'expert_timestep': self.expert_timestep,
            'tracking_error': tracking_error,
            'pelvis_height': self.data.qpos[2],
            'forward_velocity': self.data.qvel[0],
            'global_step': self.global_step,
            'episode_step': self.current_step,
            'standing_steps': self.standing_steps,
            'walking_steps': self.walking_steps,
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
            # Create renderer and camera
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            camera = mujoco.MjvCamera()
            
            # Set camera to follow the humanoid
            camera.lookat[:] = self.data.qpos[:3]
            camera.distance = 3.0
            camera.elevation = -20
            camera.azimuth = 90
            
            renderer.update_scene(self.data, camera)
            return renderer.render()
    
    def close(self):
        """Close the environment."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# Register the environment
register(
    id='TorqueSkeletonWalking-v0',
    entry_point='walk_env:TorqueSkeletonWalkingEnv',
    max_episode_steps=5000,
    reward_threshold=1000.0,
)


if __name__ == "__main__":
    # Test the environment
    env = TorqueSkeletonWalkingEnv(render_mode="human")
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Motor names: {env.motor_names}")
    print(f"Control ranges: {env.motor_ctrl_ranges}")
    
    for i in range(1000):
        # Random policy for testing
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            print(f"Final info: {info}")
            obs, reset_info = env.reset()
        
        if i % 100 == 0:
            print(f"Step {i}, Reward: {reward:.3f}, Height: {info['pelvis_height']:.3f}, "
                  f"Velocity: {info['forward_velocity']:.3f}, Stage: {info['stage']}")
    
    env.close()
