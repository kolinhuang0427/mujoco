#!/usr/bin/env python3
"""
Render trained Torque Skeleton Walking Agent
Visualize the trained agent's walking behavior with MuJoCo viewer.
Based on the working render_walking_imitation.py pattern.
"""

import argparse
import os
import glob
import time
import numpy as np
import mujoco
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from walk_env import TorqueSkeletonWalkingEnv


def render_walking_model(model_path: str, episodes: int = 3, expert_data_path: str = "data/expert_data.pkl"):
    """
    Load and render the trained walking model.
    
    Args:
        model_path: Path to the trained model
        episodes: Number of episodes to render
        expert_data_path: Path to expert data
    """
    print(f"🎬 Loading and rendering walking model: {model_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Create base environment
    env = TorqueSkeletonWalkingEnv(expert_data_path=expert_data_path, render_mode="human")
    
    # Create vectorized environment exactly as in training
    env = Monitor(env)
    env_vec = DummyVecEnv([lambda: env])
    
    # Load normalization statistics if available
    vec_normalize_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    if os.path.exists(vec_normalize_path):
        print("📊 Loading normalization statistics...")
        env_norm = VecNormalize.load(vec_normalize_path, env_vec)
        env_norm.training = False
        env_norm.norm_reward = False
        print(f"   ✅ Loaded normalization statistics")
    else:
        print("⚠️ No normalization found - using raw environment")
        env_norm = env_vec
    
    # Load the trained model
    print("🤖 Loading trained model...")
    model = PPO.load(model_path, env=env_norm)
    print("✅ Model loaded successfully!")
    
    print(f"\n🚀 Starting {episodes} episodes...")
    print("💡 MuJoCo viewer should open in a separate window")
    print("   Use mouse to rotate view, scroll to zoom, Space to pause")
    print("=" * 60)
    
    try:
        for episode in range(episodes):
            print(f"\n📺 Episode {episode + 1}/{episodes}")
            print("-" * 40)
            
            # Reset environment
            obs = env_norm.reset()
            
            episode_reward = 0
            episode_steps = 0
            start_time = time.time()
            
            # Track metrics
            heights = []
            velocities = []
            tracking_errors = []
            
            done = False
            while not done:
                # Get action from trained model
                action, _ = model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, done, info = env_norm.step(action)
                
                # Handle vectorized environment returns
                if isinstance(done, np.ndarray):
                    done = done[0]
                if isinstance(reward, np.ndarray):
                    reward = reward[0]
                if isinstance(info, list) and len(info) > 0:
                    info = info[0]
                
                episode_reward += reward
                episode_steps += 1
                
                # Get the actual environment for metrics and rendering
                if hasattr(env_norm, 'venv'):  # VecNormalize wrapper
                    actual_env = env_norm.venv.envs[0].env  # Unwrap Monitor and DummyVecEnv
                else:  # Direct DummyVecEnv
                    actual_env = env_norm.envs[0].env  # Unwrap Monitor and DummyVecEnv
                
                # Collect metrics directly from MuJoCo data (use actual world coordinates)
                pelvis_body_id = mujoco.mj_name2id(actual_env.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
                pelvis_world_pos = actual_env.data.xpos[pelvis_body_id]
                
                # Calculate uprightness using same method as environment
                body_quat = actual_env.data.xquat[pelvis_body_id]
                rot_mat = np.zeros(9)
                mujoco.mju_quat2Mat(rot_mat, body_quat)
                rot_mat = rot_mat.reshape(3, 3)
                robot_up_direction = rot_mat[:, 1]  # Robot's local Y-axis
                world_up_direction = np.array([0, 0, 1])
                uprightness = np.dot(robot_up_direction, world_up_direction)
                
                heights.append(pelvis_world_pos[2])  # Actual world height (Z coordinate)
                velocities.append(actual_env.data.qvel[0])  # Forward velocity (qvel[0] maps to world Y direction)  
                tracking_errors.append(info.get('tracking_error', 0))
                
                # Track uprightness
                if not hasattr(render_walking_model, 'uprightness_values'):
                    render_walking_model.uprightness_values = []
                render_walking_model.uprightness_values.append(uprightness)
                
                # Render the actual environment
                actual_env.render()
                
                # Print progress
                if episode_steps % 100 == 0:
                    print(f"  Step {episode_steps:3d}: "
                          f"Height={pelvis_world_pos[2]:.3f}m, "
                          f"Vel={actual_env.data.qvel[0]:.3f}m/s, "
                          f"Uprightness={uprightness:.3f}")
                
                # Control frame rate
                time.sleep(0.02)  # ~50 FPS
            
            # Episode summary
            episode_time = time.time() - start_time
            avg_height = np.mean(heights) if heights else 0
            avg_velocity = np.mean(velocities) if velocities else 0
            avg_tracking_error = np.mean(tracking_errors) if tracking_errors else 0
            avg_uprightness = np.mean(render_walking_model.uprightness_values) if hasattr(render_walking_model, 'uprightness_values') and render_walking_model.uprightness_values else 0
            
            print(f"\n📊 Episode {episode + 1} Summary:")
            print(f"  ⏱️  Duration: {episode_time:.1f}s ({episode_steps} steps)")
            print(f"  🏆 Total Reward: {episode_reward:.1f}")
            print(f"  📏 Average Height: {avg_height:.3f}m (target: 0.975m)")
            print(f"  🏃 Average Velocity: {avg_velocity:.3f}m/s (target: 1.0m/s)")
            print(f"  📐 Average Uprightness: {avg_uprightness:.3f} (-1=upside down, +1=upright)")
            if avg_tracking_error > 0:
                print(f"  🎯 Average Tracking Error: {avg_tracking_error:.3f}")
            
            # Performance assessment
            if avg_height > 0.9 and abs(avg_velocity - 1.0) < 0.3 and avg_uprightness > 0.8:
                print("  🌟 EXCELLENT performance!")
            elif avg_height > 0.8 and abs(avg_velocity - 1.0) < 0.5 and avg_uprightness > 0.6:
                print("  ✅ GOOD performance")
            elif avg_height > 0.7 and avg_uprightness > 0.3:
                print("  👍 Making progress")
            else:
                print("  ⚠️  Needs improvement")
            
            print("=" * 60)
            
            # Wait between episodes
            if episode < episodes - 1:
                print("Press Enter for next episode...")
                input()
    
    except KeyboardInterrupt:
        print("\n⏹️ Rendering interrupted by user")
    
    finally:
        print(f"\n🎬 Rendering complete!")
        env_norm.close()


def find_latest_model(log_dir: str = "logs") -> tuple:
    """Find the latest trained model and normalization stats."""
    
    # Find all log directories
    log_dirs = glob.glob(os.path.join(log_dir, "torque_skeleton_walking_*"))
    if not log_dirs:
        raise FileNotFoundError(f"No training logs found in {log_dir}")
    
    # Get the most recent log directory
    latest_log_dir = max(log_dirs, key=os.path.getctime)
    
    # Look for models in order of preference
    model_files = [
        "best_model.zip",
        "final_model.zip",
    ]
    
    # Also check for checkpoint models
    checkpoint_files = glob.glob(os.path.join(latest_log_dir, "torque_skeleton_walking_*.zip"))
    if checkpoint_files:
        # Sort by modification time, get latest
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        model_files.insert(0, os.path.basename(latest_checkpoint))
    
    model_path = None
    for model_file in model_files:
        potential_path = os.path.join(latest_log_dir, model_file)
        if os.path.exists(potential_path):
            model_path = potential_path
            break
    
    if model_path is None:
        raise FileNotFoundError(f"No model files found in {latest_log_dir}")
    
    return model_path, latest_log_dir


def main():
    parser = argparse.ArgumentParser(description="Render trained torque skeleton walking agent")
    
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (default: auto-find latest)")
    parser.add_argument("--expert_data", type=str, default="data/expert_data.pkl",
                        help="Path to expert data")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to render")
    
    args = parser.parse_args()
    
    try:
        # Find model if not specified
        if args.model is None:
            print("🔍 Auto-finding latest trained model...")
            model_path, log_dir = find_latest_model()
            print(f"   Found model: {model_path}")
            print(f"   From training: {os.path.basename(log_dir)}")
        else:
            model_path = args.model
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
        
        if not os.path.exists(args.expert_data):
            print(f"❌ Expert data not found: {args.expert_data}")
            return
        
        # Render the model
        render_walking_model(model_path, args.episodes, args.expert_data)
        
    except KeyboardInterrupt:
        print("\n⏹️ Rendering interrupted by user")
    except Exception as e:
        print(f"❌ Error during rendering: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 