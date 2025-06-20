#!/usr/bin/env python3
"""
Interactive MyoLegs Model Viewer
This script loads the MyoLegs model and renders it with MuJoCo's viewer for real-time visualization.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os

class MyoLegsEnv:
    """Environment wrapper for the MyoLegs model with rendering."""
    
    def __init__(self, render_mode="human"):
        """Initialize the MyoLegs environment."""
        self.render_mode = render_mode
        
        # Load the model
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, "myolegs.xml")
        
        print(f"Loading MyoLegs model from: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize viewer
        self.viewer = None
        
        # Get muscle and joint names for control
        self.muscle_names = self._get_muscle_names()
        self.joint_names = self._get_joint_names()
        
        # Initialize control parameters
        self.muscle_activation = np.zeros(self.model.nu)
        self.simulation_time = 0.0
        
        print(f"✅ MyoLegs environment initialized!")
        print(f"   - {self.model.nu} muscle actuators")
        print(f"   - {self.model.njnt} joints")
        print(f"   - {self.model.nsensor} sensors")
        
        # Properly initialize the model state
        self._initialize_stable_state()
        
    def _initialize_stable_state(self):
        """Initialize the model in a stable state to prevent initial instability."""
        print("Initializing stable state...")
        
        # Set standing pose
        self._set_standing_pose()
        
        # Apply baseline muscle activation for stability
        baseline_activation = 0.15  # Stronger initial muscle tone
        self.data.ctrl[:] = baseline_activation
        
        # Let the model settle for a few steps with muscle activation
        for i in range(100):
            mujoco.mj_step(self.model, self.data)
        
        # Reset simulation time after settling
        self.simulation_time = 0.0
        self.data.time = 0.0
        
        print("✅ Model settled in stable state")
        
    def _get_muscle_names(self):
        """Get names of all muscle actuators."""
        names = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                names.append(name)
        return names
    
    def _get_joint_names(self):
        """Get names of all joints."""
        names = []
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                names.append(name)
        return names
    
    def reset(self):
        """Reset the environment to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        
        # Set a natural standing pose
        self._set_standing_pose()
        
        # Apply stable muscle activation immediately
        self.data.ctrl[:] = 0.15
        
        # Let model settle briefly
        for i in range(50):
            mujoco.mj_step(self.model, self.data)
        
        # Reset simulation time
        self.simulation_time = 0.0
        self.data.time = 0.0
        
        return self._get_observation()
    
    def _set_standing_pose(self):
        """Set the model to a natural standing pose."""
        # Define joint angles for standing position
        standing_pose = {
            'hip_flexion_r': -0.05,    # Slight hip extension for stability
            'hip_flexion_l': -0.05,
            'hip_adduction_r': 0.0,
            'hip_adduction_l': 0.0,
            'hip_rotation_r': 0.0,
            'hip_rotation_l': 0.0,
            'knee_angle_r': 0.1,       # Slight knee bend
            'knee_angle_l': 0.1,
            'ankle_angle_r': -0.05,    # Slight plantarflexion for balance
            'ankle_angle_l': -0.05,
            'subtalar_angle_r': 0.0,
            'subtalar_angle_l': 0.0,
            'mtp_angle_r': 0.0,
            'mtp_angle_l': 0.0
        }
        
        # Apply joint positions
        for joint_name, angle in standing_pose.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0 and joint_id < len(self.data.qpos):
                self.data.qpos[joint_id] = angle
        
        # Set root position to proper height (pelvis height for standing)
        # Find the root joint (usually the first 7 DOF for free body)
        if len(self.data.qpos) > 2:
            self.data.qpos[2] = 0.92  # Set pelvis height to ~92cm
        
        # Zero out velocities for stability
        self.data.qvel[:] = 0.0
        
        # Forward kinematics to update positions
        mujoco.mj_forward(self.model, self.data)
    
    def step(self, action=None):
        """Step the simulation forward."""
        if action is not None:
            # Apply muscle activations
            self.data.ctrl[:] = np.clip(action, 0, 1)
        else:
            # Default: stable muscle tone to maintain posture
            base_activation = 0.12
            variation = 0.03 * np.sin(0.1 * self.simulation_time * 50)  # Gentle variation
            self.data.ctrl[:] = base_activation + variation * np.random.random(self.model.nu) * 0.5
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        self.simulation_time += self.model.opt.timestep
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current state observation."""
        return {
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
            'sensor_data': self.data.sensordata.copy(),
            'time': self.simulation_time
        }
    
    def render(self):
        """Render the environment."""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 3.0  # Set camera distance
            self.viewer.cam.elevation = -20  # Set camera elevation
            self.viewer.cam.azimuth = 135    # Set camera azimuth
        
        # Sync viewer with current state
        self.viewer.sync()
    
    def close(self):
        """Close the environment."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

def demo_basic_standing():
    """Demo 1: Basic standing with muscle tone."""
    print("Demo 1: Basic Standing with Muscle Tone")
    print("=" * 50)
    
    env = MyoLegsEnv()
    # Note: Model is already initialized in stable state, no need to reset
    
    print("Controls:")
    print("- Close the viewer window to exit")
    print("- The model will maintain standing posture with slight muscle activations")
    
    try:
        steps = 0
        while True:
            # Gentle muscle activation variation for natural standing
            base_activation = 0.12
            variation = 0.02 * np.sin(0.05 * steps) 
            muscle_activation = base_activation + variation * np.random.random(env.model.nu) * 0.3
            
            obs = env.step(muscle_activation)
            env.render()
            
            # Print some info every 500 steps
            if steps % 500 == 0:
                root_height = obs['qpos'][2] if len(obs['qpos']) > 2 else 0
                print(f"Step {steps}: Root height = {root_height:.3f}m, Time = {obs['time']:.2f}s")
            
            steps += 1
            time.sleep(0.001)  # Small delay for real-time visualization
            
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    finally:
        env.close()

def demo_muscle_activation_patterns():
    """Demo 2: Systematic muscle activation patterns."""
    print("Demo 2: Muscle Activation Patterns")
    print("=" * 50)
    
    env = MyoLegsEnv()
    
    print("Controls:")
    print("- Close the viewer window to exit")
    print("- Watch different muscle groups activate in sequence")
    
    # Define muscle groups
    muscle_groups = {
        'hip_flexors': ['psoas_r', 'psoas_l', 'iliacus_r', 'iliacus_l', 'recfem_r', 'recfem_l'],
        'quadriceps': ['recfem_r', 'recfem_l', 'vasmed_r', 'vasmed_l', 'vaslat_r', 'vaslat_l', 'vasint_r', 'vasint_l'],
        'hamstrings': ['bflh_r', 'bflh_l', 'bfsh_r', 'bfsh_l', 'semimem_r', 'semimem_l', 'semiten_r', 'semiten_l'],
        'calves': ['gasmed_r', 'gasmed_l', 'gaslat_r', 'gaslat_l', 'soleus_r', 'soleus_l']
    }
    
    try:
        steps = 0
        cycle_duration = 1000  # Steps per muscle group
        
        while True:
            # Determine which muscle group to activate
            cycle_position = (steps // cycle_duration) % len(muscle_groups)
            group_names = list(muscle_groups.keys())
            current_group = group_names[cycle_position]
            
            # Base activation for stability
            muscle_activation = np.full(env.model.nu, 0.12)
            
            # Enhanced activation for current muscle group
            activation_strength = 0.25 + 0.15 * np.sin(0.01 * (steps % cycle_duration))
            
            for muscle_name in muscle_groups[current_group]:
                muscle_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, muscle_name)
                if muscle_id >= 0:
                    muscle_activation[muscle_id] = activation_strength
            
            obs = env.step(muscle_activation)
            env.render()
            
            # Print info about current activation
            if steps % cycle_duration == 0:
                print(f"Now activating: {current_group}")
            
            steps += 1
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    finally:
        env.close()

def demo_walking_simulation():
    """Demo 3: Simple walking-like motion."""
    print("Demo 3: Walking-like Motion")
    print("=" * 50)
    
    env = MyoLegsEnv()
    
    print("Controls:")
    print("- Close the viewer window to exit")
    print("- Simulating alternating leg muscle activation for walking")
    
    try:
        steps = 0
        gait_frequency = 0.003  # Slower walking frequency for stability
        
        while True:
            # Base muscle activation for stability
            muscle_activation = np.full(env.model.nu, 0.12)
            
            # Alternating leg activation (simplified gait pattern)
            right_phase = np.sin(2 * np.pi * gait_frequency * steps)
            left_phase = np.sin(2 * np.pi * gait_frequency * steps + np.pi)
            
            # Right leg muscles
            right_muscles = ['recfem_r', 'gasmed_r', 'tibant_r', 'glmed1_r']
            for muscle_name in right_muscles:
                muscle_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, muscle_name)
                if muscle_id >= 0:
                    activation = 0.15 + 0.2 * max(0, right_phase)
                    muscle_activation[muscle_id] = activation
            
            # Left leg muscles
            left_muscles = ['recfem_l', 'gasmed_l', 'tibant_l', 'glmed1_l']
            for muscle_name in left_muscles:
                muscle_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, muscle_name)
                if muscle_id >= 0:
                    activation = 0.15 + 0.2 * max(0, left_phase)
                    muscle_activation[muscle_id] = activation
            
            obs = env.step(muscle_activation)
            env.render()
            
            if steps % 500 == 0:
                print(f"Step {steps}: Gait cycle at {(steps * gait_frequency) % 1:.2f}")
            
            steps += 1
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    finally:
        env.close()

def main():
    """Main function to run demos."""
    print("MyoLegs Interactive Viewer")
    print("=" * 50)
    print("Choose a demo to run:")
    print("1. Basic Standing (default)")
    print("2. Muscle Activation Patterns")
    print("3. Walking Simulation")
    print()
    
    try:
        choice = input("Enter choice (1-3, or press Enter for default): ").strip()
        
        if choice == "2":
            demo_muscle_activation_patterns()
        elif choice == "3":
            demo_walking_simulation()
        else:
            demo_basic_standing()
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the required dependencies installed:")
        print("- mujoco")
        print("- numpy")

if __name__ == "__main__":
    main()