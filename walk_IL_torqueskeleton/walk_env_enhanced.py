#!/usr/bin/env python3
"""
Enhanced Torque Skeleton Walking Environment with LocoMujoco-style Modifications
Implements the same XML modification system as LocoMujoco for proper physics behavior.

Key Features:
- Box feet implementation (default: enabled)
- Arm disabling (default: enabled) 
- Dynamic XML modification using mjcf
- Proper collision group management
- Consistent timestep control
- Stable physics simulation

This addresses the "bouncing" behavior by applying the same settings
that LocoMujoco uses by default.
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
from dm_control import mjcf


class EnhancedTorqueSkeletonWalkingEnv(gym.Env):
    """
    Enhanced torque skeleton walking environment with LocoMujoco-style XML modifications.
    
    Features:
    - Box feet (default: enabled) - replaces complex foot geometry with simple boxes
    - Disabled arms (default: enabled) - removes arm joints and motors
    - Proper collision groups - manages contact detection correctly
    - Consistent timestep control - prevents physics instabilities
    - Dynamic XML modification - modifies model at runtime
    
    Action Space: Continuous motor torques [-1, 1] (normalized)
    Observation Space: Joint positions, velocities, torso orientation, foot contacts, expert reference
    Reward: Expert trajectory matching + height maintenance + forward velocity
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, 
                 expert_data_path: str = "data/expert_data.pkl", 
                 render_mode: Optional[str] = None,
                 episode_length: float = 5.0,
                 use_box_feet: bool = True,
                 disable_arms: bool = True,
                 alpha_box_feet: float = 0.5,
                 n_substeps: int = 10,
                 timestep: float = 0.001):
        """
        Initialize the enhanced environment.
        
        Args:
            expert_data_path: Path to expert trajectory data
            render_mode: Rendering mode ("human", "rgb_array", None)
            episode_length: Episode duration in seconds
            use_box_feet: Whether to use box feet instead of complex foot geometry
            disable_arms: Whether to disable arm joints and motors
            alpha_box_feet: Transparency of box feet (0.0-1.0)
            n_substeps: Number of physics substeps per control step
            timestep: Physics timestep in seconds
        """
        super().__init__()
        
        # Store configuration
        self.use_box_feet = use_box_feet
        self.disable_arms = disable_arms
        self.alpha_box_feet = alpha_box_feet
        self.n_substeps = n_substeps
        self.physics_timestep = timestep
        
        # Load and modify XML
        model_path = self._find_model_path()
        print(f"Loading torque skeleton model from: {model_path}")
        
        # Load XML and apply modifications (LocoMujoco style)
        xml_handle = mjcf.from_path(model_path)
        xml_handle, active_joints, active_motors = self._apply_xml_modifications(xml_handle)
        
        # Create MuJoCo model from modified XML with proper asset handling
        self.model, temp_dir = self._create_model_from_xml_handle(xml_handle, model_path)
        self.data = mujoco.MjData(self.model)
        self._temp_dir = temp_dir  # Store for cleanup
        
        # Set physics parameters to match LocoMujoco
        self._configure_physics()
        
        # Load expert data
        self.expert_data_path = expert_data_path
        self._load_expert_data()
        
        # Environment parameters
        self.dt = self.n_substeps * self.model.opt.timestep
        self.episode_length = episode_length
        self.max_episode_steps = int(episode_length / self.dt)
        self.current_step = 0
        
        # Expert data timing
        self.expert_dt = 1.0 / 500.0  # Expert data at 500 Hz
        self.expert_step_ratio = self.expert_dt / self.dt
        self.expert_timestep_float = 0.0
        
        # Reset parameters
        self.max_episode_expert_frames = int(self.episode_length / self.expert_dt)
        self.reset_buffer_frames = 500
        
        # Use only active joints (after XML modifications)
        self.joint_names = active_joints
        print(f"Active joints after modifications: {len(self.joint_names)}")
        print(f"Joints: {self.joint_names}")
        
        # Setup motors for active joints only
        self.motor_names = active_motors
        self.motor_ctrl_ranges = []
        self.active_motor_indices = []
        
        # Get control ranges for active motors
        for motor_name in self.motor_names:
            for i in range(self.model.nu):
                actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if actuator_name == motor_name:
                    ctrl_range = self.model.actuator_ctrlrange[i]
                    self.motor_ctrl_ranges.append(ctrl_range)
                    self.active_motor_indices.append(i)
                    break
        
        self.motor_ctrl_ranges = np.array(self.motor_ctrl_ranges)
        print(f"Found {len(self.motor_names)} active motors")
        print(f"Motors: {self.motor_names}")
        
        # Action space: normalized motor torques [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.motor_names),), dtype=np.float32
        )
        
        # Create joint mapping for active joints only
        self.joint_qpos_map = {}
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                qpos_idx = self.model.jnt_qposadr[joint_id]
                self.joint_qpos_map[joint_name] = qpos_idx
            else:
                print(f"Warning: Joint {joint_name} not found in modified model")
        
        # Setup collision groups (LocoMujoco style)
        self._setup_collision_groups()
        
        # Observation space
        obs_dim = self._get_observation_dimension()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Target parameters (LocoMujoco defaults)
        self.target_height = 0.975
        self.target_velocity = 1.0
        self.height_threshold = 0.6
        
        # Rendering
        self.render_mode = render_mode
        self.viewer = None
        
        # Expert tracking
        self.expert_timestep = 0
        
        print(f"✅ Enhanced Torque Skeleton Environment initialized")
        print(f"   - Box feet: {'enabled' if self.use_box_feet else 'disabled'}")
        print(f"   - Arms: {'disabled' if self.disable_arms else 'enabled'}")
        print(f"   - Physics timestep: {self.model.opt.timestep}s")
        print(f"   - Control timestep: {self.dt}s ({self.n_substeps} substeps)")
        print(f"   - Action space: {self.action_space.shape}")
        print(f"   - Observation space: {self.observation_space.shape}")
    
    def _apply_xml_modifications(self, xml_handle):
        """
        Apply LocoMujoco-style XML modifications.
        
        Returns:
            Modified XML handle, list of active joints, list of active motors
        """
        print("Applying XML modifications...")
        
        # Get modification specifications (LocoMujoco style)
        joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_xml_modifications()
        
        # Store original joint/motor lists before removal
        original_joints = [
            'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 
            'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
            'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 
            'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l'
        ]
        
        original_motors = [
            'mot_hip_flexion_r', 'mot_hip_adduction_r', 'mot_hip_rotation_r',
            'mot_knee_angle_r', 'mot_ankle_angle_r', 'mot_subtalar_angle_r', 'mot_mtp_angle_r',
            'mot_hip_flexion_l', 'mot_hip_adduction_l', 'mot_hip_rotation_l',
            'mot_knee_angle_l', 'mot_ankle_angle_l', 'mot_subtalar_angle_l', 'mot_mtp_angle_l'
        ]
        
        # Add arm joints/motors to original lists if not disabled
        if not self.disable_arms:
            arm_joints = [
                "arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r", "wrist_flex_r",
                "wrist_dev_r", "arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l", "pro_sup_l",
                "wrist_flex_l", "wrist_dev_l"
            ]
            arm_motors = [
                "mot_shoulder_flex_r", "mot_shoulder_add_r", "mot_shoulder_rot_r", "mot_elbow_flex_r",
                "mot_pro_sup_r", "mot_wrist_flex_r", "mot_wrist_dev_r", "mot_shoulder_flex_l",
                "mot_shoulder_add_l", "mot_shoulder_rot_l", "mot_elbow_flex_l", "mot_pro_sup_l",
                "mot_wrist_flex_l", "mot_wrist_dev_l"
            ]
            original_joints.extend(arm_joints)
            original_motors.extend(arm_motors)
        
        # Remove specified joints, motors, and constraints
        xml_handle = self._delete_from_xml_handle(xml_handle, joints_to_remove, motors_to_remove, equ_constr_to_remove)
        
        # Add box feet if enabled
        if self.use_box_feet:
            xml_handle = self._add_box_feet_to_xml_handle(xml_handle, self.alpha_box_feet)
            print("✓ Box feet added")
        
        # Reorient arms if disabled (LocoMujoco style)
        if self.disable_arms:
            xml_handle = self._reorient_arms(xml_handle)
            print("✓ Arms disabled and reoriented")
        
        # Calculate active joints and motors (those not removed)
        active_joints = [j for j in original_joints if j not in joints_to_remove]
        active_motors = [m for m in original_motors if m not in motors_to_remove]
        
        print(f"✓ Removed {len(joints_to_remove)} joints, {len(motors_to_remove)} motors")
        print(f"✓ Active: {len(active_joints)} joints, {len(active_motors)} motors")
        
        return xml_handle, active_joints, active_motors
    
    def _get_xml_modifications(self):
        """
        Get lists of joints/motors/constraints to remove (LocoMujoco style).
        """
        joints_to_remove = []
        motors_to_remove = []
        equ_constr_to_remove = []
        
        if self.use_box_feet:
            # Remove complex foot joints when using box feet
            joints_to_remove += ["subtalar_angle_l", "mtp_angle_l", "subtalar_angle_r", "mtp_angle_r"]
            motors_to_remove += ["mot_subtalar_angle_l", "mot_mtp_angle_l", "mot_subtalar_angle_r", "mot_mtp_angle_r"]
            equ_constr_to_remove += [j + "_constraint" for j in joints_to_remove]
        
        if self.disable_arms:
            # Remove all arm joints and motors
            joints_to_remove += [
                "arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r", "wrist_flex_r",
                "wrist_dev_r", "arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l", "pro_sup_l",
                "wrist_flex_l", "wrist_dev_l"
            ]
            motors_to_remove += [
                "mot_shoulder_flex_r", "mot_shoulder_add_r", "mot_shoulder_rot_r", "mot_elbow_flex_r",
                "mot_pro_sup_r", "mot_wrist_flex_r", "mot_wrist_dev_r", "mot_shoulder_flex_l",
                "mot_shoulder_add_l", "mot_shoulder_rot_l", "mot_elbow_flex_l", "mot_pro_sup_l",
                "mot_wrist_flex_l", "mot_wrist_dev_l"
            ]
            equ_constr_to_remove += [
                "wrist_flex_r_constraint", "wrist_dev_r_constraint",
                "wrist_flex_l_constraint", "wrist_dev_l_constraint"
            ]
        
        return joints_to_remove, motors_to_remove, equ_constr_to_remove
    
    @staticmethod
    def _delete_from_xml_handle(xml_handle, joints_to_remove, motors_to_remove, equ_constraints):
        """
        Delete joints, motors and constraints from XML handle (LocoMujoco implementation).
        """
        for j in joints_to_remove:
            try:
                j_handle = xml_handle.find("joint", j)
                if j_handle is not None:
                    j_handle.remove()
            except:
                pass  # Joint might not exist
        
        for m in motors_to_remove:
            try:
                m_handle = xml_handle.find("actuator", m)
                if m_handle is not None:
                    m_handle.remove()
            except:
                pass  # Motor might not exist
        
        for e in equ_constraints:
            try:
                e_handle = xml_handle.find("equality", e)
                if e_handle is not None:
                    e_handle.remove()
            except:
                pass  # Constraint might not exist
        
        return xml_handle
    
    @staticmethod
    def _add_box_feet_to_xml_handle(xml_handle, alpha_box_feet, scaling=1.0):
        """
        Add box feet and make original feet non-collidable (LocoMujoco implementation).
        """
        try:
            # Add box feet to toes
            toe_l = xml_handle.find("body", "toes_l")
            if toe_l is not None:
                size = np.array([0.112, 0.03, 0.05]) * scaling
                pos = np.array([-0.09, 0.019, 0.0]) * scaling
                toe_l.add("geom", name="foot_box_l", type="box", size=size.tolist(), pos=pos.tolist(),
                          rgba=[0.5, 0.5, 0.5, alpha_box_feet], euler=[0.0, 0.15, 0.0])
            
            toe_r = xml_handle.find("body", "toes_r")
            if toe_r is not None:
                size = np.array([0.112, 0.03, 0.05]) * scaling
                pos = np.array([-0.09, 0.019, 0.0]) * scaling
                toe_r.add("geom", name="foot_box_r", type="box", size=size.tolist(), pos=pos.tolist(),
                          rgba=[0.5, 0.5, 0.5, alpha_box_feet], euler=[0.0, -0.15, 0.0])
            
            # Make original feet non-collidable
            geoms_to_disable = ["r_foot", "r_bofoot", "l_foot", "l_bofoot"]
            for geom_name in geoms_to_disable:
                geom = xml_handle.find("geom", geom_name)
                if geom is not None:
                    geom.contype = 0
                    geom.conaffinity = 0
                    
        except Exception as e:
            print(f"Warning: Could not add box feet: {e}")
        
        return xml_handle
    
    @staticmethod
    def _reorient_arms(xml_handle):
        """
        Reorient arms when disabled (LocoMujoco implementation).
        """
        try:
            h = xml_handle.find("body", "humerus_l")
            if h is not None:
                h.quat = [1.0, -0.1, -1.0, -0.1]
            
            h = xml_handle.find("body", "ulna_l")
            if h is not None:
                h.quat = [1.0, 0.6, 0.0, 0.0]
            
            h = xml_handle.find("body", "humerus_r")
            if h is not None:
                h.quat = [1.0, 0.1, 1.0, -0.1]
            
            h = xml_handle.find("body", "ulna_r")
            if h is not None:
                h.quat = [1.0, -0.6, 0.0, 0.0]
        except Exception as e:
            print(f"Warning: Could not reorient arms: {e}")
        
        return xml_handle
    
    def _create_model_from_xml_handle(self, xml_handle, original_model_path):
        """
        Create MuJoCo model from XML handle with proper asset management.
        This ensures mesh files are properly accessible.
        """
        import tempfile
        import shutil
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Get the directory containing the original model and meshes
            model_dir = os.path.dirname(original_model_path)
            
            # Export the modified XML with all assets to the temp directory
            mjcf.export_with_assets(xml_handle, temp_dir, "modified_model.xml")
            
            # Copy the mesh directory to the temp directory if it doesn't exist
            temp_mesh_dir = os.path.join(temp_dir, "meshes")
            original_mesh_dir = os.path.join(model_dir, "meshes")
            
            if not os.path.exists(temp_mesh_dir) and os.path.exists(original_mesh_dir):
                shutil.copytree(original_mesh_dir, temp_mesh_dir)
                print("✓ Copied mesh files to temporary directory")
            
            # Load the model from the temporary file
            temp_model_path = os.path.join(temp_dir, "modified_model.xml")
            model = mujoco.MjModel.from_xml_path(temp_model_path)
            
            return model, temp_dir
            
        except Exception as e:
            # Cleanup on failure
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise e
    
    def _configure_physics(self):
        """
        Configure physics parameters to match LocoMujoco defaults.
        """
        # Set timestep
        self.model.opt.timestep = self.physics_timestep
        
        # Set integration method and physics parameters (LocoMujoco defaults)
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4  # RK4 integration
        
        # Set solver parameters for stability
        self.model.opt.iterations = 50  # Solver iterations
        self.model.opt.tolerance = 1e-10  # Solver tolerance
        
        # Set contact parameters
        self.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL  # Pyramidal friction cone
        
        print(f"✓ Physics configured: timestep={self.model.opt.timestep}, substeps={self.n_substeps}")
    
    def _setup_collision_groups(self):
        """
        Setup collision groups based on modifications (LocoMujoco style).
        """
        if self.use_box_feet:
            self.collision_groups = [
                ("floor", ["floor"]),
                ("foot_r", ["foot_box_r"]),
                ("foot_l", ["foot_box_l"])
            ]
        else:
            self.collision_groups = [
                ("floor", ["floor"]),
                ("foot_r", ["r_foot"]),
                ("front_foot_r", ["r_bofoot"]),
                ("foot_l", ["l_foot"]),
                ("front_foot_l", ["l_bofoot"])
            ]
        
        print(f"✓ Collision groups: {[group[0] for group in self.collision_groups]}")
    
    def _find_model_path(self) -> str:
        """Find the torque skeleton model file."""
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "model", "torque_skeleton", "humanoid_torque.xml"),
            "model/torque_skeleton/humanoid_torque.xml",
            "../model/torque_skeleton/humanoid_torque.xml",
            os.path.join(os.getcwd(), "model", "torque_skeleton", "humanoid_torque.xml"),
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        
        raise FileNotFoundError(
            f"Torque skeleton model not found. Tried paths:\n" + 
            "\n".join(f"  - {os.path.abspath(p)}" for p in possible_paths)
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
    
    def _get_observation_dimension(self) -> int:
        """Calculate observation space dimension."""
        # Root position (3) + root orientation quaternion (4) + root velocity (6)
        root_dim = 13
        
        # Joint positions and velocities (for active joints only)
        joint_dim = 2 * len(self.joint_names)
        
        # Foot contacts (depends on box feet setting)
        contact_dim = 2 if self.use_box_feet else 4
        
        # Expert reference
        expert_dim = len(self.joint_names)
        
        # Additional body state
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
        
        # Root state
        obs_parts.append(self.data.qpos[:3])   # root position
        obs_parts.append(self.data.qpos[3:7])  # root quaternion  
        obs_parts.append(self.data.qvel[:6])   # root linear and angular velocity
        
        # Joint positions and velocities (active joints only)
        joint_pos = []
        joint_vel = []
        for joint_name in self.joint_names:
            if joint_name in self.joint_qpos_map:
                qpos_idx = self.joint_qpos_map[joint_name]
                joint_pos.append(self.data.qpos[qpos_idx])
                # Find corresponding qvel index
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0:
                    qvel_idx = self.model.jnt_dofadr[joint_id]
                    joint_vel.append(self.data.qvel[qvel_idx])
                else:
                    joint_vel.append(0.0)
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
        if pelvis_body_id >= 0:
            body_quat = self.data.xquat[pelvis_body_id]
            rot_mat_tmp = np.zeros(9)
            mujoco.mju_quat2Mat(rot_mat_tmp, body_quat)
            rot_mat_tmp = rot_mat_tmp.reshape(3, 3)
            
            pelvis_world_pos = self.data.xpos[pelvis_body_id]
            robot_up_direction = rot_mat_tmp[:, 1]
            world_up_direction = np.array([0, 0, 1])
            uprightness = np.dot(robot_up_direction, world_up_direction)
            
            body_features = np.array([
                uprightness,
                rot_mat_tmp[1, 1],  # forward orientation
                pelvis_world_pos[2],  # height
                self.data.qvel[0],  # forward velocity
                self.data.qvel[2],  # vertical velocity
                self.data.qvel[1],  # lateral velocity
            ])
        else:
            body_features = np.zeros(6)
        
        obs_parts.append(body_features)
        
        return np.concatenate(obs_parts)
    
    def _get_foot_contacts(self) -> np.ndarray:
        """Get foot contact information based on collision groups."""
        if self.use_box_feet:
            contacts = np.zeros(2)  # Only left and right box feet
            
            for i, contact in enumerate(self.data.contact[:self.data.ncon]):
                geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                
                if geom1_name == "floor" or geom2_name == "floor":
                    other_geom = geom1_name if geom2_name == "floor" else geom2_name
                    
                    if other_geom == "foot_box_r":
                        contacts[0] = 1.0  # right foot
                    elif other_geom == "foot_box_l":
                        contacts[1] = 1.0  # left foot
        else:
            contacts = np.zeros(4)  # Original foot contact setup
            
            for i, contact in enumerate(self.data.contact[:self.data.ncon]):
                geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                
                if geom1_name == "floor" or geom2_name == "floor":
                    other_geom = geom1_name if geom2_name == "floor" else geom2_name
                    
                    if other_geom == "r_foot":
                        contacts[0] = 1.0
                    elif other_geom == "r_bofoot":
                        contacts[1] = 1.0
                    elif other_geom == "l_foot":
                        contacts[2] = 1.0
                    elif other_geom == "l_bofoot":
                        contacts[3] = 1.0
        
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
        
        # Joint tracking reward
        self.joint_errors = np.abs(current_joints - expert_joints)
        joint_reward = np.exp(-1.0 * np.mean(self.joint_errors))
        reward += joint_reward
        
        # Height maintenance
        pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        if pelvis_body_id >= 0:
            pelvis_world_pos = self.data.xpos[pelvis_body_id]
            height = pelvis_world_pos[2]
            height_reward = np.exp(-5.0 * abs(height - self.target_height))
            reward += 2.0 * height_reward
            
            # Forward velocity reward
            forward_velocity = self.data.qvel[0]
            velocity_reward = np.exp(-2.0 * abs(forward_velocity - self.target_velocity))
            reward += 5.0 * velocity_reward
            
            # Uprightness
            body_quat = self.data.xquat[pelvis_body_id]
            rot_mat_tmp = np.zeros(9)
            mujoco.mju_quat2Mat(rot_mat_tmp, body_quat)
            rot_mat_tmp = rot_mat_tmp.reshape(3, 3)
            robot_up_direction = rot_mat_tmp[:, 1]
            world_up_direction = np.array([0, 0, 1])
            uprightness = np.dot(robot_up_direction, world_up_direction)
            reward += 1.0 * max(0, uprightness)
        
        # Alive bonus
        reward += self.current_step / 200.0
        
        return reward
    
    def _is_done(self) -> bool:
        """Check if episode should terminate."""
        pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        if pelvis_body_id >= 0:
            pelvis_world_pos = self.data.xpos[pelvis_body_id]
            if pelvis_world_pos[2] < self.height_threshold:
                return True
            
            body_quat = self.data.xquat[pelvis_body_id]
            rot_mat_tmp = np.zeros(9)
            mujoco.mju_quat2Mat(rot_mat_tmp, body_quat)
            rot_mat_tmp = rot_mat_tmp.reshape(3, 3)
            robot_up_direction = rot_mat_tmp[:, 1]
            world_up_direction = np.array([0, 0, 1])
            uprightness = np.dot(robot_up_direction, world_up_direction)
            
            if uprightness < 0.2:
                return True
        
        if abs(self.data.qpos[2]) > 3.0:
            return True
        
        if self.current_step >= self.max_episode_steps:
            return True
        
        return False
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set root position and orientation
        self.data.qpos[0] = 0.975  # HEIGHT
        self.data.qpos[1] = 0.0    # forward/back
        self.data.qpos[2] = 0.0    # lateral
        self.data.qpos[3] = 0.0    # pelvis_tilt
        self.data.qpos[4] = 0.0    # pelvis_list
        self.data.qpos[5] = 0.0    # pelvis_rotation
        
        # Random start from expert trajectory
        max_start_frame = min(self.reset_buffer_frames, 
                             self.expert_cycle_length - self.max_episode_expert_frames - 10)
        max_start_frame = max(1, max_start_frame)
        self.expert_timestep = np.random.randint(0, max_start_frame)
        self.expert_timestep_float = float(self.expert_timestep)
        expert_frame = self.expert_data.iloc[self.expert_timestep]
        
        # Add noise to root position
        self.data.qpos[0] += np.random.normal(0, 0.02)
        self.data.qpos[1] += np.random.normal(0, 0.02)
        self.data.qpos[2] += np.random.normal(0, 0.01)
        
        # Set joint angles from expert data with noise
        for joint_name in self.joint_names:
            if joint_name in self.joint_qpos_map and joint_name in expert_frame:
                qpos_idx = self.joint_qpos_map[joint_name]
                expert_angle = float(expert_frame[joint_name])
                noise = np.random.normal(0, 0.05)
                self.data.qpos[qpos_idx] = expert_angle + noise
        
        # Set small random velocities
        self.data.qvel[:] = np.random.normal(0, 0.1, len(self.data.qvel))
        
        # Initialize motor controls to zero
        self.data.ctrl[:] = 0.0
        
        # Forward dynamics
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0
        
        obs = self._get_observation()
        
        info = {
            'expert_timestep': self.expert_timestep,
            'episode_steps': self.max_episode_steps,
            'use_box_feet': self.use_box_feet,
            'disable_arms': self.disable_arms,
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step with proper substep control."""
        # Ensure action is within bounds
        action = np.clip(action, -1.0, 1.0)
        
        # Initialize all motor controls to zero
        self.data.ctrl[:] = 0.0
        
        # Convert normalized action to motor control values for active motors only
        for i, motor_idx in enumerate(self.active_motor_indices):
            if i < len(action):
                low, high = self.motor_ctrl_ranges[i]
                motor_command = low + (action[i] + 1.0) * 0.5 * (high - low)
                self.data.ctrl[motor_idx] = motor_command
        
        # Step simulation with proper substeps (LocoMujoco style)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        
        # Advance expert timestep
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
        
        # Info
        pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        if pelvis_body_id >= 0:
            pelvis_world_pos = self.data.xpos[pelvis_body_id]
            height = pelvis_world_pos[2]
        else:
            height = 0.0
        
        info = {
            'expert_timestep': self.expert_timestep,
            'pelvis_height': height,
            'forward_velocity': self.data.qvel[0],
            'episode_step': self.current_step,
            'use_box_feet': self.use_box_feet,
            'disable_arms': self.disable_arms,
        }
        
        if hasattr(self, 'joint_errors'):
            info['tracking_error'] = np.linalg.norm(self.joint_errors)
        
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
            camera = mujoco.MjvCamera()
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
        
        # Clean up temporary directory
        if hasattr(self, '_temp_dir') and self._temp_dir:
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None


# Register the environment
register(
    id='EnhancedTorqueSkeletonWalking-v0',
    entry_point='walk_env_enhanced:EnhancedTorqueSkeletonWalkingEnv',
    max_episode_steps=5000,
    reward_threshold=1000.0,
)


if __name__ == "__main__":
    print("Testing Enhanced Torque Skeleton Walking Environment")
    print("="*60)
    
    # Test with different configurations
    configs = [
        {"use_box_feet": True, "disable_arms": True, "name": "Box Feet + No Arms (LocoMujoco default)"},
        {"use_box_feet": False, "disable_arms": True, "name": "Original Feet + No Arms"},
        {"use_box_feet": True, "disable_arms": False, "name": "Box Feet + Arms"},
        {"use_box_feet": False, "disable_arms": False, "name": "Original Setup"},
    ]
    
    for config in configs:
        name = config.pop("name")
        print(f"\n🧪 Testing: {name}")
        print("-" * 40)
        
        try:
            env = EnhancedTorqueSkeletonWalkingEnv(render_mode=None, **config)
            
            obs, info = env.reset()
            print(f"✓ Environment initialized successfully")
            print(f"  - Action space: {env.action_space.shape}")
            print(f"  - Observation space: {env.observation_space.shape}")
            print(f"  - Active joints: {len(env.joint_names)}")
            print(f"  - Active motors: {len(env.motor_names)}")
            print(f"  - Physics timestep: {env.model.opt.timestep}")
            print(f"  - Control timestep: {env.dt}")
            
            # Test a few steps
            for i in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if i == 0:
                    print(f"  - First step reward: {reward:.3f}")
                    print(f"  - Height: {info['pelvis_height']:.3f}")
                    print(f"  - Forward velocity: {info['forward_velocity']:.3f}")
            
            env.close()
            print(f"✓ Test completed successfully")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print(f"\n{'='*60}")
    print("All tests completed!")