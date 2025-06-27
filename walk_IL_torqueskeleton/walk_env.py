#!/usr/bin/env python3
"""
Torque Skeleton Walking Imitation Learning Environment
A Gymnasium environment for training the humanoid torque skeleton model to walk using expert data.
Uses imitation learning with rewards based on expert trajectory matching, height, and velocity.

=== CRITICAL COORDINATE SYSTEM INFORMATION ===

The XML model uses: pelvis pos="0 0 0.975" quat="0.5 0.5 0.5 0.5"

This quaternion [0.5, 0.5, 0.5, 0.5] represents a 120° rotation around the [1,1,1] diagonal axis.
It creates a cyclic permutation of the standard coordinate system: X→Y→Z→X

Mathematical breakdown:
- Rotation angle: θ = 120° (2π/3 radians)  
- Rotation axis: [1,1,1] normalized = [1/√3, 1/√3, 1/√3]
- Quaternion formula: [cos(θ/2), x·sin(θ/2), y·sin(θ/2), z·sin(θ/2)]
- Result: [cos(60°), (1/√3)·sin(60°), (1/√3)·sin(60°), (1/√3)·sin(60°)]
- Simplifies to: [0.5, 0.5, 0.5, 0.5]

Coordinate system transformation:
    BEFORE ROTATION          AFTER 120° ROTATION
         Z                         X
         |                         |
         |                         |  
         +-----Y                   +-----Z
        /                         /
       /                         /
      X                         Y

Standard basis → Rotated basis:
- X-axis [1,0,0] → [0,1,0] (X becomes Y direction)
- Y-axis [0,1,0] → [0,0,1] (Y becomes Z direction)  
- Z-axis [0,0,1] → [1,0,0] (Z becomes X direction)

This means for the humanoid:
- UP direction = X-axis component (up_vec[0])
- FORWARD direction = Y-axis  
- LATERAL direction = Z-axis
- HEIGHT coordinate = qpos[1] (pelvis_tz, not pelvis_ty)

=== CRITICAL DISTINCTION: JOINT COORDINATES vs WORLD COORDINATES ===

**IMPORTANT**: Due to the complex pelvis rotation quaternion [0.5, 0.5, 0.5, 0.5], 
joint coordinates (qpos/qvel) DO NOT directly correspond to world coordinates!

Joint ordering: pelvis_tx (qpos[0]), pelvis_tz (qpos[1]), pelvis_ty (qpos[2])
Due to coordinate rotation by quaternion [0.5, 0.5, 0.5, 0.5]:
- qpos[0] = pelvis_tx → Does NOT equal world Y position directly
- qpos[1] = pelvis_tz → Does NOT equal world X position directly  
- qpos[2] = pelvis_ty → Does NOT equal world Z position directly

**CORRECT METHOD: Always use world coordinates for physical measurements**
- Actual HEIGHT = world Z coordinate (data.xpos[pelvis_body_id][2])
- Actual FORWARD velocity = qvel[0] (maps to world Y direction after transformation)
- Actual LATERAL velocity = qvel[1] (maps to world X direction after transformation)
- Actual VERTICAL velocity = qvel[2] (maps to world Z direction after transformation)

**Robot's Orientation in World Frame:**
Due to the initial pelvis quaternion rotation:
- Robot's "UP" direction = local Y-axis (rot_mat[:, 1]) 
- Robot's "FORWARD" direction = local Z-axis (rot_mat[:, 2])
- Robot's "LATERAL" direction = local X-axis (rot_mat[:, 0])

**Key Discovery**: Joint coordinates are transformed by the pelvis quaternion and 
cannot be used directly as world coordinates. Always use:
1. data.xpos[body_id] for actual world positions
2. Proper dot products with world axes for orientation measurements
3. qvel components map to world directions but require understanding the transformation

**Previous Bug**: Environment was incorrectly using qpos[1] for height instead of 
the actual world Z coordinate data.xpos[pelvis_body_id][2], leading to completely 
wrong height readings when the robot fell.

=== END COORDINATE SYSTEM INFO ===
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
    Reward: Expert trajectory matching + height maintenance + forward velocity
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, expert_data_path: str = "data/expert_data.pkl", 
                 render_mode: Optional[str] = None,
                 episode_length: float = 5.0):
        super().__init__()
        
        # Load torque skeleton model with proper path resolution
        model_path = self._find_model_path()
        
        print(f"Loading torque skeleton model from: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Load expert data immediately
        self.expert_data_path = expert_data_path
        self._load_expert_data()
        
        # Environment parameters
        self.dt = self.model.opt.timestep
        
        # Validate episode length against expert data duration
        expert_duration = self.expert_cycle_length * (1.0 / 500.0)  # Expert data duration
        if episode_length > expert_duration - 1.0:  # Leave 1s buffer
            print(f"⚠️  WARNING: Episode length ({episode_length}s) exceeds expert data duration ({expert_duration:.1f}s)")
            print(f"   Recommended maximum: {expert_duration - 1.0:.1f}s to avoid discontinuous looping")
        
        self.episode_length = episode_length
        self.max_episode_steps = int(episode_length / self.dt)
        self.current_step = 0
        
        # Expert data timing parameters
        self.expert_dt = 1.0 / 500.0  # Expert data recorded at 500 Hz
        self.expert_step_ratio = self.expert_dt / self.dt  # How many sim steps per expert frame
        self.expert_timestep_float = 0.0  # Float tracking for sub-step precision
        
        # Reset parameters to avoid discontinuous wrap-around
        self.max_episode_expert_frames = int(self.episode_length / self.expert_dt)  # Frames needed for full episode
        self.reset_buffer_frames = 500  # Safe starting range to avoid wrap-around
        
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
        
        # Target parameters
        self.target_height = 0.975  # Natural walking height for this model
        self.target_velocity = 1.0  # Target forward velocity (m/s)
        self.height_threshold = 0.6  # Minimum height to maintain
        
        # Rendering
        self.render_mode = render_mode
        self.viewer = None
        
        # Expert tracking
        self.expert_timestep = 0
        
        print(f"✅ Torque Skeleton Walking Environment initialized")
        print(f"   - Action space: {self.action_space.shape} (motor torques)")
        print(f"   - Observation space: {self.observation_space.shape}")
        print(f"   - Simulation timestep: {self.dt}s ({1/self.dt:.0f} Hz)")
        print(f"   - Expert data timestep: {self.expert_dt}s ({1/self.expert_dt:.0f} Hz)")
        print(f"   - Expert step ratio: {self.expert_step_ratio:.1f} sim steps per expert frame")
        print(f"   - Episode length: {self.episode_length}s ({self.max_episode_steps} steps)")
        print(f"   - Expert data: {self.expert_cycle_length} frames ({self.expert_cycle_length * self.expert_dt:.1f}s duration)")
        print(f"   - Episode uses: {self.max_episode_expert_frames} expert frames max")
        print(f"   - Reset range: frames 0-{min(self.reset_buffer_frames, self.expert_cycle_length - self.max_episode_expert_frames - 100)}")
    
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
        """Load expert motion capture data."""
        abs_data_path = os.path.abspath(self.expert_data_path)
        
        if not os.path.exists(abs_data_path):
            raise FileNotFoundError(f"Expert data not found: {abs_data_path}")
        
        print(f"🎯 Loading expert data: {abs_data_path}")
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
        
        # Expert reference
        expert_dim = len(self.joint_names)
        
        # Additional body state (center of mass, orientation features)
        body_dim = 6
        
        total_dim = root_dim + joint_dim + contact_dim + expert_dim + body_dim
        return total_dim
    
    def _get_expert_reference(self) -> np.ndarray:
        """Get expert joint angles for current timestep."""
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
        pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        body_quat = self.data.xquat[pelvis_body_id]
        rot_mat_tmp = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat_tmp, body_quat)
        rot_mat_tmp = rot_mat_tmp.reshape(3, 3)
        
        # Get actual world height from pelvis body position
        pelvis_world_pos = self.data.xpos[pelvis_body_id]
        
        # Robot's "up" direction is the local Y-axis (rot_mat[:, 1])
        robot_up_direction = rot_mat_tmp[:, 1]  # Robot's local Y-axis in world coordinates
        world_up_direction = np.array([0, 0, 1])
        uprightness = np.dot(robot_up_direction, world_up_direction)  # Uprightness measure
        
        forward_vec = rot_mat_tmp[:, 1]
        
        body_features = np.array([
            uprightness,  # uprightness using dot product with world up
            forward_vec[0],  # forward orientation
            pelvis_world_pos[2],  # height (actual world Z coordinate)
            self.data.qvel[0],  # forward velocity (world Y direction)
            self.data.qvel[2],  # vertical velocity (world Z direction)
            self.data.qvel[1],  # lateral velocity (world X direction)
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
                
                # Map foot geoms to contact indices with more precision
                if other_geom == "r_foot":
                    contacts[0] = 1.0  # right heel/main foot
                elif other_geom == "r_bofoot":
                    contacts[1] = 1.0  # right toe
                elif other_geom == "l_foot":
                    contacts[2] = 1.0  # left heel/main foot
                elif other_geom == "l_bofoot":
                    contacts[3] = 1.0  # left toe
        
        return contacts
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on joint tracking, height, and velocity."""
        reward = 0.0
        
        # Get current joint positions
        current_joints = []
        for joint_name in self.joint_names:
            if joint_name in self.joint_qpos_map:
                qpos_idx = self.joint_qpos_map[joint_name]
                current_joints.append(self.data.qpos[qpos_idx])
            else:
                current_joints.append(0.0)
        current_joints = np.array(current_joints)
        
        # Get expert reference
        expert_joints = self._get_expert_reference()
        
        # 1. Joint tracking reward (DOMINANT component for imitation learning)
        self.joint_errors = np.abs(current_joints - expert_joints)
        joint_reward = np.exp(-1.0 * np.log (np.mean(self.joint_errors)))  # Less aggressive decay
        joint_reward_weighted = joint_reward  # Much higher weight
        reward += joint_reward_weighted
        
        # 2. Height maintenance
        pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        pelvis_world_pos = self.data.xpos[pelvis_body_id]
        height = pelvis_world_pos[2]
        height_reward = np.exp(-5.0 * abs(height - self.target_height))
        height_reward_weighted = 2.0 * height_reward
        reward += height_reward_weighted
        
        # 3. Forward velocity reward
        forward_velocity = self.data.qvel[0]  # qvel[0] is forward velocity
        velocity_reward = np.exp(-2.0 * abs(forward_velocity - self.target_velocity))
        velocity_reward += forward_velocity if forward_velocity < 0.0 else 0.0
        velocity_reward_weighted = 5.0 * velocity_reward 
        reward += velocity_reward_weighted
        
        # 4. Uprightness
        body_quat = self.data.xquat[pelvis_body_id]
        rot_mat_tmp = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat_tmp, body_quat)
        rot_mat_tmp = rot_mat_tmp.reshape(3, 3)
        robot_up_direction = rot_mat_tmp[:, 1]  # Robot's local Y-axis
        world_up_direction = np.array([0, 0, 1])
        uprightness = np.dot(robot_up_direction, world_up_direction)
        uprightness_reward_weighted = 1.0 * max(0, uprightness)
        reward += uprightness_reward_weighted
        
        # 5. Alive bonus
        alive_bonus = self.current_step / 200.0
        reward += alive_bonus
        
        # Store individual reward components for logging
        self.last_reward_components = {
            'joint_tracking': float(joint_reward_weighted),
            'height': float(height_reward_weighted), 
            'velocity': float(velocity_reward_weighted),
            'uprightness': float(uprightness_reward_weighted),
            'alive_bonus': float(alive_bonus),
            'total': float(reward),
            # Raw values for analysis
            'joint_tracking_raw': float(joint_reward),
            'height_raw': float(height_reward),
            'velocity_raw': float(velocity_reward),
            'uprightness_raw': float(max(0, uprightness)),
            'mean_joint_error': float(np.mean(self.joint_errors))
        }
        
        return reward
    
    def _is_done(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if fallen - check actual world height
        pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        pelvis_world_pos = self.data.xpos[pelvis_body_id]
        if pelvis_world_pos[2] < self.height_threshold:
            return True
        
        # Terminate if tipped over - use proper uprightness calculation
        body_quat = self.data.xquat[pelvis_body_id]
        rot_mat_tmp = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat_tmp, body_quat)
        rot_mat_tmp = rot_mat_tmp.reshape(3, 3)
        
        # Robot's up direction is local Y-axis
        robot_up_direction = rot_mat_tmp[:, 1]
        world_up_direction = np.array([0, 0, 1])
        uprightness = np.dot(robot_up_direction, world_up_direction)
        
        # Terminate if severely tipped over
        if uprightness < 0.2:
            return True
        
        # Terminate if moved too far sideways
        if abs(self.data.qpos[2]) > 3.0:
            return True
        
        # Normal episode length termination
        if self.current_step >= self.max_episode_steps:
            return True
        
        return False
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set root position and orientation to match XML defaults
        self.data.qpos[0] = 0.975  # pelvis_tx (HEIGHT due to rotation)
        self.data.qpos[1] = 0.0    # pelvis_tz (forward/back position)
        self.data.qpos[2] = 0.0    # pelvis_ty (lateral position)
        self.data.qpos[3] = 0.0    # pelvis_tilt (Euler Z rotation)
        self.data.qpos[4] = 0.0    # pelvis_list (Euler X rotation)
        self.data.qpos[5] = 0.0    # pelvis_rotation (Euler Y rotation)
        
        # Start from a random point in the first portion of expert trajectory
        # Avoid starting near the end to prevent discontinuous wrap-around during episode
        max_start_frame = min(self.reset_buffer_frames, 
                             self.expert_cycle_length - self.max_episode_expert_frames - 10)  # 10 frame safety buffer
        max_start_frame = max(1, max_start_frame)  # Ensure at least 1 frame available
        self.expert_timestep = np.random.randint(0, max_start_frame)
        self.expert_timestep_float = float(self.expert_timestep)
        expert_frame = self.expert_data.iloc[self.expert_timestep]
        
        # Add small noise to root position
        self.data.qpos[0] += np.random.normal(0, 0.02)  # HEIGHT noise
        self.data.qpos[1] += np.random.normal(0, 0.02)  # forward/back position noise
        self.data.qpos[2] += np.random.normal(0, 0.01)  # lateral position noise
        
        # Set joint angles from expert data with noise
        for joint_name in self.joint_names:
            if joint_name in self.joint_qpos_map and joint_name in expert_frame:
                qpos_idx = self.joint_qpos_map[joint_name]
                expert_angle = float(expert_frame[joint_name])
                noise = np.random.normal(0, 0.05)  # Small noise for robustness
                self.data.qpos[qpos_idx] = expert_angle + noise
        
        # Set small random velocities
        self.data.qvel[:] = np.random.normal(0, 0.1, len(self.data.qvel))
        
        # Initialize motor controls to zero
        self.data.ctrl[:] = 0.0
        
        # Ensure inactive motors are explicitly set to zero
        for motor_idx in self.inactive_motor_indices:
            self.data.ctrl[motor_idx] = 0.0
        
        # Forward dynamics
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0
        
        obs = self._get_observation()
        
        info = {
            'expert_timestep': self.expert_timestep,
            'episode_steps': self.max_episode_steps,
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
        
        # Ensure inactive motors remain at zero
        for motor_idx in self.inactive_motor_indices:
            self.data.ctrl[motor_idx] = 0.0
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Advance expert timestep with proper timing
        # Only advance expert frame when enough simulation time has passed
        self.expert_timestep_float += 1.0 / self.expert_step_ratio
        self.expert_timestep = int(self.expert_timestep_float) % self.expert_cycle_length
        
        # Increment counters
        self.current_step += 1
        
        # Get observation and reward
        obs = self._get_observation()
        reward = self._calculate_reward()
        
        # Check termination
        terminated = self._is_done()
        truncated = False
        
        tracking_error = np.linalg.norm(self.joint_errors)
        
        # Get pelvis world position for info
        pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        pelvis_world_pos = self.data.xpos[pelvis_body_id]
        
        info = {
            'expert_timestep': self.expert_timestep,
            'tracking_error': tracking_error,
            'pelvis_height': pelvis_world_pos[2],
            'forward_velocity': self.data.qvel[0],
            'episode_step': self.current_step,
        }
        
        # Add reward components if available
        if hasattr(self, 'last_reward_components'):
            info.update(self.last_reward_components)
        
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
                  f"Velocity: {info['forward_velocity']:.3f}, Tracking: {info['tracking_error']:.3f}")
    
    env.close()
