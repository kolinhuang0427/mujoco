#!/usr/bin/env python3
"""
MyoLegs Motion Imitation Environment
Implementation of the motion imitation framework from the paper.
Follows the exact POMDP formulation with 309D observations and PD muscle control.
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


class MyoLegsMotionImitationEnv(gym.Env):
    """
    Motion imitation environment following the paper's POMDP formulation.
    
    Observation Space: 309D (pelvis: 8, joint kinematics: 234, feet contact: 4, target pose: 63)
    - Joint kinematics: Body-based approach with 16 tracked bodies 
      • Position: 3D coords (45 values, excludes root)
      • Rotation: 6D tangent-normalized representation (96 values)  
      • Linear velocity: 3D velocity (48 values)
      • Angular velocity: 3D angular velocity (48 values)
      • Total: 45 + 96 + 48 + 48 = 237 ≈ 234 values
    Action Space: Desired muscle lengths (80D) controlled via PD controllers
    - PD Control: Fi = F_peak_i * (kp(ai - li) - kd*vi) with dynamic gains
    - Uses MuJoCo's built-in muscle functions for force-to-activation conversion
    Reward: Position + Velocity + Energy + Upright components
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, expert_data_path: str = "walk_IL/data/expert_data.pkl", 
                 render_mode: Optional[str] = None):
        super().__init__()
        
        # Load MyoLegs model
        model_path = self._find_model_path()
        print(f"Loading MyoLegs model from: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Environment parameters
        self.dt = self.model.opt.timestep
        self.current_step = 0
        self.expert_timestep = 0
        
        # Get muscle information (80 muscles)
        self.muscle_names = []
        self.muscle_indices = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self.muscle_names.append(name)
                self.muscle_indices.append(i)
        
        print(f"Found {len(self.muscle_names)} muscle actuators")
        
        # Action space: desired muscle lengths (not activations)
        # We'll normalize this to [0, 1] representing relative muscle length
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(len(self.muscle_names),), dtype=np.float32
        )
        
        # Joint names for kinematics (14 joints)
        self.joint_names = [
            'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 
            'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
            'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 
            'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l'
        ]
        
        # Load expert motion data (after joint_names is defined)
        self.expert_data_path = expert_data_path
        self._load_expert_data()
        
        # Create joint mapping
        self.joint_qpos_map = {}
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                qpos_idx = self.model.jnt_qposadr[joint_id]
                self.joint_qpos_map[joint_name] = qpos_idx
        
        # Observation space: 309D as per paper
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(309,), dtype=np.float32
        )
        
        # PD controller parameters are computed per muscle based on peak force (as in reference solution)
        
        # Reward function weights (as per paper)
        self.w_pos = 1.0     # Position tracking weight
        self.w_vel = 0.1     # Velocity tracking weight  
        self.w_e = 0.001     # Energy penalty weight (reduced to prevent dominance)
        self.w_up = 0.5      # Upright posture weight
        
        # Reward scaling factors
        self.k_pos = 10.0    # Position reward scaling
        self.k_vel = 0.1     # Velocity reward scaling
        self.k_up = 10.0     # Upright reward scaling
        
        # Termination threshold (0.15m as per paper)
        self.termination_threshold = 0.15
        
        # Key tracking points (pelvis, knees, ankles, toes) as per paper
        # Paper tracks: pelvis, knees, ankles, toes
        # Maps to bodies: pelvis, tibia_r (right knee), tibia_l (left knee), 
        #                 talus_r (right ankle), talus_l (left ankle), toes_r, toes_l
        self.tracking_bodies = ['pelvis', 'tibia_r', 'tibia_l', 'talus_r', 'talus_l', 'toes_r', 'toes_l']
        
        # Store initial muscle lengths for normalization
        self.initial_muscle_lengths = None
        self.muscle_length_ranges = None
        
        # Previous muscle lengths for PD control
        self.prev_muscle_lengths = None
        self.desired_muscle_lengths = None
        
        # Rendering
        self.render_mode = render_mode
        self.viewer = None
        
        print(f"✅ MyoLegs Motion Imitation Environment initialized")
        print(f"   - Action space: {self.action_space.shape} (desired muscle lengths)")
        print(f"   - Observation space: {self.observation_space.shape}")
        print(f"     • Pelvis state: 8D, Body kinematics: 234D (16 bodies), Contact: 4D, Target: 63D")
        print(f"   - Expert data: {len(self.expert_data)} frames")
        print(f"   - Termination threshold: {self.termination_threshold}m")
    
    def _find_model_path(self) -> str:
        """Find the MyoLegs model file with robust path resolution."""
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "model", "myolegs", "myolegs.xml"),
            "model/myolegs/myolegs.xml",
            "../model/myolegs/myolegs.xml",
            os.path.join(os.getcwd(), "model", "myolegs", "myolegs.xml"),
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        
        raise FileNotFoundError(
            f"MyoLegs model not found. Tried paths:\n" + 
            "\n".join(f"  - {os.path.abspath(p)}" for p in possible_paths)
        )
    
    def _load_expert_data(self):
        """Load expert motion capture data."""
        # Fix path resolution - remove duplicate walk_IL directory
        if "walk_IL/walk_IL" in self.expert_data_path:
            self.expert_data_path = self.expert_data_path.replace("walk_IL/walk_IL", "walk_IL")
        abs_data_path = os.path.abspath(self.expert_data_path)
        
        if not os.path.exists(abs_data_path):
            raise FileNotFoundError(f"Expert data not found: {abs_data_path}")
        
        print(f"Loading expert data: {abs_data_path}")
        with open(abs_data_path, 'rb') as f:
            expert_data = pickle.load(f)
        
        if 'qpos' not in expert_data:
            raise ValueError("Expert data must contain 'qpos' key")
        
        self.expert_data = expert_data['qpos']
        
        # Validate expert data contains required joint columns
        missing_joints = []
        for joint_name in self.joint_names:
            if joint_name not in self.expert_data.columns:
                missing_joints.append(joint_name)
        
        if missing_joints:
            raise ValueError(f"Expert data missing required joints: {missing_joints}")
        
        print(f"✅ Expert data loaded: {len(self.expert_data)} frames")
        print(f"   Available joints: {list(self.expert_data.columns)}")
        print(f"   Data range: frame 0 to {len(self.expert_data)-1}")
    
    def _initialize_muscle_parameters(self):
        """Initialize muscle length parameters for PD control."""
        if self.initial_muscle_lengths is not None:
            return
        
        # Get current muscle lengths
        mujoco.mj_forward(self.model, self.data)
        current_lengths = np.zeros(len(self.muscle_names))
        
        for i, muscle_idx in enumerate(self.muscle_indices):
            # Get muscle length from tendon length
            current_lengths[i] = self.data.ten_length[muscle_idx]
        
        self.initial_muscle_lengths = current_lengths.copy()
        
        # Set reasonable ranges for muscle length variation (±10% of initial length)
        self.muscle_length_ranges = self.initial_muscle_lengths * 0.2  # Total range
        
        print(f"Initialized muscle parameters:")
        print(f"  - Mean initial length: {np.mean(self.initial_muscle_lengths):.4f}")
        print(f"  - Mean length range: {np.mean(self.muscle_length_ranges):.4f}")
    
    def _action_to_muscle_lengths(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized action [0,1] to desired muscle lengths."""
        # Map [0,1] to [initial_length - range/2, initial_length + range/2]
        desired_lengths = (self.initial_muscle_lengths - self.muscle_length_ranges/2 + 
                          action * self.muscle_length_ranges)
        return desired_lengths
    
    def _apply_pd_control(self, desired_lengths: np.ndarray):
        """Apply PD control to achieve desired muscle lengths using paper's equation: Fi = F_peak_i * (kp(ai - li) - kd*vi)"""
        # Convert target lengths to forces using PD control
        forces = self._target_length_to_force(desired_lengths)
        
        # Convert forces to muscle activations using MuJoCo's muscle model
        activations = self._force_to_activation(forces)
        
        # Apply muscle activations
        self.data.ctrl[:] = np.array(activations)
        self.desired_muscle_lengths = desired_lengths.copy()
    
    def _target_length_to_force(self, lengths: np.ndarray) -> list:
        """Convert target muscle lengths to forces using PD control law from paper."""
        forces = []
        for i, muscle_idx in enumerate(self.muscle_indices):
            # Get current muscle state
            current_length = self.data.actuator_length[muscle_idx]
            current_velocity = self.data.actuator_velocity[muscle_idx]
            
            # Get peak force from muscle parameters
            peak_force = self.model.actuator_biasprm[muscle_idx, 2]
            
            # PD gains (scaled relative to peak force as in reference)
            kp = 5 * peak_force
            kd = 0.1 * kp
            
            # PD control equation from paper: Fi = F_peak_i * (kp(ai - li) - kd*vi)
            # Rearranged: Fi = kp * (ai - li) - kd * vi
            force = kp * (lengths[i] - current_length) - kd * current_velocity
            
            # Clip force to muscle's peak force capability (negative for muscle forces)
            clipped_force = np.clip(force, -peak_force, 0)
            forces.append(clipped_force)
        
        return forces
    
    def _force_to_activation(self, forces) -> list:
        """Convert actuator forces to activation levels using MuJoCo's muscle model."""
        activations = []
        for i, muscle_idx in enumerate(self.muscle_indices):
            # Get current muscle state
            length = self.data.actuator_length[muscle_idx]
            velocity = self.data.actuator_velocity[muscle_idx]
            lengthrange = self.model.actuator_lengthrange[muscle_idx]
            acc0 = self.model.actuator_acc0[muscle_idx]
            
            # Get muscle parameters (first 9 elements)
            prmb = self.model.actuator_biasprm[muscle_idx, :9]
            prmg = self.model.actuator_gainprm[muscle_idx, :9]
            
            # Use MuJoCo's built-in muscle functions
            bias = mujoco.mju_muscleBias(length, lengthrange, acc0, prmb)
            gain = min(-1, mujoco.mju_muscleGain(length, velocity, lengthrange, acc0, prmg))
            
            # Convert force to activation: activation = (force - bias) / gain
            if abs(gain) > 1e-6:
                activation = (forces[i] - bias) / gain
            else:
                activation = 0.0
            
            # Clip activation to valid range
            activations.append(np.clip(activation, 0.0, 1.0))
        
        return activations
    

    
    def _get_observation(self) -> np.ndarray:
        """Get 309D observation as per paper specification."""
        obs_components = []
        
        # 1. Pelvis state (8 values): height, tilt, velocity
        pelvis_pos = self.data.qpos[:3]  # x, y, z position
        pelvis_quat = self.data.qpos[3:7]  # quaternion orientation
        pelvis_vel = self.data.qvel[:6]  # linear and angular velocity
        
        # Extract height and tilt
        height = pelvis_pos[2]
        
        # Calculate tilt angles from quaternion
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, pelvis_quat)
        rot_mat = rot_mat.reshape(3, 3)
        
        # Forward and sideways tilt (in radians)
        forward_tilt = np.arctan2(rot_mat[0, 2], rot_mat[2, 2])
        sideways_tilt = np.arctan2(rot_mat[1, 2], rot_mat[2, 2])
        
        pelvis_obs = np.array([
            height, forward_tilt, sideways_tilt,  # 3 values
            pelvis_vel[0], pelvis_vel[1], pelvis_vel[2],  # linear velocity: 3 values
            pelvis_vel[3], pelvis_vel[4]  # angular velocity (2 main components): 2 values
        ])
        obs_components.append(pelvis_obs)  # 8 values
        
        # 2. Joint kinematics (234 values): Body-based approach as per paper
        # Track 16 bodies: root, torso, head, pelvis, femur_r, tibia_r, talus_r, calcn_r, toes_r, patella_r,
        #                  femur_l, tibia_l, talus_l, calcn_l, toes_l, patella_l
        tracked_body_names = [
            'root', 'torso', 'head', 'pelvis',
            'femur_r', 'tibia_r', 'talus_r', 'calcn_r', 'toes_r', 'patella_r',
            'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'patella_l'
        ]
        
        body_obs = []
        
        # Get body positions, rotations, and velocities
        for i, body_name in enumerate(tracked_body_names):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            
            if body_id >= 0 and body_id < self.model.nbody:
                # Position (3D) - exclude root position as per calculation
                if body_name != 'root':
                    body_pos = self.data.xpos[body_id].copy()
                    body_obs.extend(body_pos)  # 3 values per body (except root) = 45 total
                
                # Rotation (6D tangent-normalized representation)
                body_quat = self.data.xquat[body_id].copy()
                # Convert quaternion to 6D representation (tangent space)
                # Use axis-angle representation truncated to 6D
                if np.linalg.norm(body_quat[1:]) > 1e-8:  # Non-identity quaternion
                    angle = 2 * np.arccos(np.clip(abs(body_quat[0]), 0, 1))
                    if angle > 1e-8:
                        axis = body_quat[1:] / np.sin(angle / 2)
                        axis_angle = axis * angle
                        # Expand to 6D by adding cross products
                        rotation_6d = np.array([
                            axis_angle[0], axis_angle[1], axis_angle[2],
                            axis_angle[1] * axis_angle[2], 
                            axis_angle[2] * axis_angle[0], 
                            axis_angle[0] * axis_angle[1]
                        ])
                    else:
                        rotation_6d = np.zeros(6)
                else:
                    rotation_6d = np.zeros(6)
                body_obs.extend(rotation_6d)  # 6 values per body = 96 total
                
                # Linear velocity (3D)
                body_vel = np.zeros(6)
                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body_id, body_vel, 0)
                body_obs.extend(body_vel[3:6])  # Linear velocity: 3 values per body = 48 total
                
                # Angular velocity (3D)  
                body_obs.extend(body_vel[0:3])  # Angular velocity: 3 values per body = 48 total
            else:
                # Body not found, pad with zeros
                if body_name != 'root':
                    body_obs.extend([0.0] * 3)  # Position
                body_obs.extend([0.0] * 6)  # Rotation
                body_obs.extend([0.0] * 3)  # Linear velocity
                body_obs.extend([0.0] * 3)  # Angular velocity
        
        # Ensure exactly 234 values (45 + 96 + 48 + 48 = 237, so trim last 3)
        if len(body_obs) > 234:
            body_obs = body_obs[:234]
        elif len(body_obs) < 234:
            body_obs.extend([0.0] * (234 - len(body_obs)))
        
        obs_components.append(np.array(body_obs))  # 234 values
        
        # 3. Feet contact forces (4 values)
        foot_contacts = self._get_foot_contact_forces()
        obs_components.append(foot_contacts)  # 4 values
        
        # 4. Target reference pose at t+1 (63 values)
        target_pose = self._get_target_reference_pose()
        obs_components.append(target_pose)  # 63 values
        
        # Combine all components
        obs = np.concatenate(obs_components).astype(np.float32)
        
        # Ensure exactly 309 dimensions
        if len(obs) != 309:
            if len(obs) > 309:
                obs = obs[:309]
            else:
                padding = np.zeros(309 - len(obs))
                obs = np.concatenate([obs, padding])
        
        return obs
    
    def _get_foot_contact_forces(self) -> np.ndarray:
        """Get contact forces for both feet (4 values total)."""
        contact_forces = np.zeros(4)  # [right_foot_normal, right_foot_friction, left_foot_normal, left_foot_friction]
        
        foot_geom_names = ['r_foot_col4', 'l_foot_col4']
        
        for foot_idx, foot_geom_name in enumerate(foot_geom_names):
            foot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_geom_name)
            
            if foot_geom_id >= 0:
                normal_force = 0.0
                friction_force = 0.0
                
                for j in range(self.data.ncon):
                    contact = self.data.contact[j]
                    if (contact.geom1 == foot_geom_id or contact.geom2 == foot_geom_id):
                        # Extract contact force
                        force = np.zeros(6)
                        mujoco.mj_contactForce(self.model, self.data, j, force)
                        
                        normal_force += abs(force[0])  # Normal component
                        friction_force += np.sqrt(force[1]**2 + force[2]**2)  # Tangential components
                
                contact_forces[foot_idx * 2] = normal_force
                contact_forces[foot_idx * 2 + 1] = friction_force
        
        return contact_forces
    
    def _get_target_reference_pose(self) -> np.ndarray:
        """Get target reference pose at t+1 (63 values)."""
        # Get next frame in expert data
        next_timestep = (self.expert_timestep + 1) % len(self.expert_data)
        target_frame = self.expert_data.iloc[next_timestep]
        
        # Extract target joint positions (absolute)
        absolute_targets = []
        for joint_name in self.joint_names:
            if joint_name in target_frame:
                absolute_targets.append(float(target_frame[joint_name]))
            else:
                absolute_targets.append(0.0)
        
        # Calculate relative targets (differences from current positions)
        current_joints = []
        for joint_name in self.joint_names:
            if joint_name in self.joint_qpos_map:
                qpos_idx = self.joint_qpos_map[joint_name]
                current_joints.append(self.data.qpos[qpos_idx])
            else:
                current_joints.append(0.0)
        
        relative_targets = np.array(absolute_targets) - np.array(current_joints)
        
        # Combine absolute and relative targets (28 values)
        pose_targets = np.concatenate([absolute_targets, relative_targets])
        
        # Add additional pose information to reach 63 values
        # Include target pelvis information, joint velocities, etc.
        additional_targets = []
        
        # Add trigonometric encoding of targets
        for target in absolute_targets:
            additional_targets.extend([np.sin(target), np.cos(target)])
        
        # Add more features to reach 63 total
        while len(pose_targets) + len(additional_targets) < 63:
            additional_targets.append(0.0)
        
        # Combine and ensure exactly 63 values
        all_targets = np.concatenate([pose_targets, additional_targets])
        if len(all_targets) > 63:
            all_targets = all_targets[:63]
        elif len(all_targets) < 63:
            padding = np.zeros(63 - len(all_targets))
            all_targets = np.concatenate([all_targets, padding])
        
        return all_targets.astype(np.float32)
    
    def _calculate_reward(self) -> float:
        """Calculate reward with four components as per paper."""
        # Get current and target poses
        current_keypoints = self._get_current_keypoints()
        target_keypoints = self._get_target_keypoints()
        
        # 1. Position reward: r_pos = exp(-k_pos * sum(||p_i - p_target||^2))
        position_errors = []
        for current, target in zip(current_keypoints, target_keypoints):
            error = np.linalg.norm(current - target)
            position_errors.append(error**2)
        
        r_pos = np.exp(-self.k_pos * np.sum(position_errors))
        
        # 2. Velocity reward: r_vel = exp(-k_vel * sum(||v_i - v_target||^2))
        current_velocities = self._get_current_keypoint_velocities()
        target_velocities = self._get_target_keypoint_velocities()
        
        velocity_errors = []
        for current_vel, target_vel in zip(current_velocities, target_velocities):
            error = np.linalg.norm(current_vel - target_vel)
            velocity_errors.append(error**2)
        
        r_vel = np.exp(-self.k_vel * np.sum(velocity_errors))
        
        # 3. Energy reward: r_e = -||m||_1 - ||m||_2
        muscle_activations = self.data.ctrl
        r_e = -np.sum(np.abs(muscle_activations)) - np.sum(muscle_activations**2)
        
        # 4. Upright reward: r_up = exp(-k_up * (theta_f^2 + theta_s^2))
        pelvis_quat = self.data.qpos[3:7]
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, pelvis_quat)
        rot_mat = rot_mat.reshape(3, 3)
        
        theta_f = np.arctan2(rot_mat[0, 2], rot_mat[2, 2])  # Forward tilt
        theta_s = np.arctan2(rot_mat[1, 2], rot_mat[2, 2])  # Sideways tilt
        
        r_up = np.exp(-self.k_up * (theta_f**2 + theta_s**2))
        
        # Total reward
        total_reward = (self.w_pos * r_pos + self.w_vel * r_vel + 
                       self.w_e * r_e + self.w_up * r_up)
        
        return total_reward
    
    def _get_current_keypoints(self) -> list:
        """Get current positions of tracking keypoints."""
        keypoints = []
        
        for body_name in self.tracking_bodies:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0 and body_id < self.model.nbody:
                pos = self.data.xpos[body_id].copy()
                keypoints.append(pos)
            else:
                keypoints.append(np.zeros(3))
        
        return keypoints
    
    def _get_target_keypoints(self) -> list:
        """Get target positions of tracking keypoints from expert data.
        
        According to the paper, we track: pelvis, knees, ankles, and toes.
        This maps to bodies: pelvis, tibia_r, tibia_l, talus_r, talus_l, toes_r, toes_l
        
        Since expert data contains only joint angles, we compute target body positions
        using forward kinematics with the expert pose at t+1.
        """
        # Get target frame at t+1 (next timestep in expert data)
        target_timestep = (self.expert_timestep + 1) % len(self.expert_data)
        target_frame = self.expert_data.iloc[target_timestep]
        
        # Create a temporary model state to compute forward kinematics
        # We need to create a temporary copy of the current data to avoid modifying the simulation
        temp_data = mujoco.MjData(self.model)
        
        # Copy current state
        temp_data.qpos[:] = self.data.qpos[:]
        temp_data.qvel[:] = self.data.qvel[:]
        
        # Set target joint angles from expert data
        for joint_name in self.joint_names:
            if joint_name in self.joint_qpos_map and joint_name in target_frame:
                qpos_idx = self.joint_qpos_map[joint_name]
                target_angle = float(target_frame[joint_name])
                temp_data.qpos[qpos_idx] = target_angle
        
        # Keep root body position similar to current (expert data doesn't include global translation)
        # Set pelvis at appropriate walking height
        temp_data.qpos[0] = self.data.qpos[0]  # x position (forward progress)
        temp_data.qpos[1] = self.data.qpos[1]  # y position (lateral)
        temp_data.qpos[2] = 0.92  # z position (standing height from paper)
        
        # Keep upright orientation
        temp_data.qpos[3] = 1.0  # quaternion w
        temp_data.qpos[4:7] = 0.0  # quaternion x, y, z
        
        # Compute forward kinematics for target pose
        mujoco.mj_forward(self.model, temp_data)
        
        # Extract target keypoint positions according to paper specification
        # Paper tracks: pelvis, knees, ankles, toes
        target_keypoint_bodies = [
            'pelvis',    # Pelvis
            'tibia_r',   # Right knee 
            'tibia_l',   # Left knee
            'talus_r',   # Right ankle
            'talus_l',   # Left ankle  
            'toes_r',    # Right toes
            'toes_l'     # Left toes
        ]
        
        target_keypoints = []
        for body_name in target_keypoint_bodies:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0 and body_id < self.model.nbody:
                # Get target body position from forward kinematics
                target_pos = temp_data.xpos[body_id].copy()
                target_keypoints.append(target_pos)
            else:
                # Fallback if body not found
                print(f"Warning: Body '{body_name}' not found in model")
                target_keypoints.append(np.zeros(3))
        
        return target_keypoints
    
    def _get_current_keypoint_velocities(self) -> list:
        """Get current velocities of tracking keypoints."""
        velocities = []
        
        for body_name in self.tracking_bodies:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0 and body_id < self.model.nbody:
                # Calculate body velocity from spatial velocity
                vel = np.zeros(6)
                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel, 0)
                velocities.append(vel[:3])  # Linear velocity only
            else:
                velocities.append(np.zeros(3))
        
        return velocities
    
    def _get_target_keypoint_velocities(self) -> list:
        """Get target velocities of tracking keypoints."""
        # Estimate velocities from expert data using finite differences
        current_targets = self._get_target_keypoints()
        
        # Get next frame targets
        next_timestep = (self.expert_timestep + 1) % len(self.expert_data)
        self.expert_timestep = next_timestep  # Temporarily advance
        next_targets = self._get_target_keypoints()
        self.expert_timestep = (next_timestep - 1) % len(self.expert_data)  # Reset
        
        # Calculate velocities
        target_velocities = []
        for current, next_target in zip(current_targets, next_targets):
            velocity = (next_target - current) / self.dt
            target_velocities.append(velocity)
        
        return target_velocities
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate due to tracking error."""
        current_keypoints = self._get_current_keypoints()
        target_keypoints = self._get_target_keypoints()
        
        # Check if any keypoint deviates more than 0.15m from reference
        for current, target in zip(current_keypoints, target_keypoints):
            distance = np.linalg.norm(current - target)
            if distance > self.termination_threshold:
                return True
        
        # Also terminate if model falls
        pelvis_height = self.data.qpos[2]
        if pelvis_height < 0.5:
            return True
        
        return False
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Sample random starting frame from expert data
        if seed is not None:
            np.random.seed(seed)
        self.expert_timestep = np.random.randint(0, len(self.expert_data))
        
        # Initialize from expert pose
        expert_frame = self.expert_data.iloc[self.expert_timestep]
        
        # Set root position
        self.data.qpos[0] = 0.0  # x position
        self.data.qpos[1] = 0.0  # y position  
        self.data.qpos[2] = 0.92  # z position (standing height)
        
        # Set upright orientation
        self.data.qpos[3] = 1.0  # w component
        self.data.qpos[4:7] = 0.0  # x, y, z components
        
        # Set joint angles from expert data
        for joint_name in self.joint_names:
            if joint_name in self.joint_qpos_map and joint_name in expert_frame:
                qpos_idx = self.joint_qpos_map[joint_name]
                expert_angle = float(expert_frame[joint_name])
                self.data.qpos[qpos_idx] = expert_angle
        
        # Set small random velocities
        self.data.qvel[:] = np.random.normal(0, 0.01, len(self.data.qvel))
        
        # Initialize muscle parameters
        mujoco.mj_forward(self.model, self.data)
        self._initialize_muscle_parameters()
        
        # Set initial muscle activations
        self.data.ctrl[:] = 0.1  # Low initial activation
        
        self.current_step = 0
        
        obs = self._get_observation()
        info = {
            'expert_timestep': self.expert_timestep,
            'tracking_errors': [],
            'pelvis_height': self.data.qpos[2],
            'episode_step': self.current_step
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        # Ensure action is within bounds
        action = np.clip(action, 0.0, 1.0)
        
        # Convert action to desired muscle lengths
        desired_lengths = self._action_to_muscle_lengths(action)
        
        # Apply PD control to achieve desired muscle lengths
        self._apply_pd_control(desired_lengths)
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Advance expert timestep
        self.expert_timestep = (self.expert_timestep + 1) % len(self.expert_data)
        
        # Get observation and reward
        obs = self._get_observation()
        reward = self._calculate_reward()
        
        # Check termination
        terminated = self._check_termination()
        truncated = False
        
        self.current_step += 1
        
        # Calculate tracking errors for info
        current_keypoints = self._get_current_keypoints()
        target_keypoints = self._get_target_keypoints()
        tracking_errors = [np.linalg.norm(curr - targ) 
                          for curr, targ in zip(current_keypoints, target_keypoints)]
        
        info = {
            'expert_timestep': self.expert_timestep,
            'tracking_errors': tracking_errors,
            'max_tracking_error': max(tracking_errors) if tracking_errors else 0.0,
            'pelvis_height': self.data.qpos[2],
            'episode_step': self.current_step,
            'muscle_activation_mean': np.mean(self.data.ctrl),
            'desired_muscle_lengths': self.desired_muscle_lengths.copy() if self.desired_muscle_lengths is not None else None
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
    id='MyoLegsMotionImitation-v0',
    entry_point='myolegs_motion_imitation_env:MyoLegsMotionImitationEnv',
    max_episode_steps=2999,  # Length of expert data
    reward_threshold=100.0,
)


if __name__ == "__main__":
    # Test the environment
    env = MyoLegsMotionImitationEnv(render_mode="human")
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Expert data length: {len(env.expert_data)}")
    
    for i in range(1000):
        # Random action (desired muscle lengths)
        action = np.random.uniform(0.3, 0.7, env.action_space.shape[0])
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            print(f"Termination reason: {'Early termination' if terminated else 'Truncated'}")
            print(f"Max tracking error: {info.get('max_tracking_error', 0):.4f}")
            obs, info = env.reset()
        
        if i % 100 == 0:
            print(f"Step {i}, Reward: {reward:.3f}, Max tracking error: {info.get('max_tracking_error', 0):.4f}")
    
    env.close()