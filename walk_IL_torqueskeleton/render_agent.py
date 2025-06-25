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
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from walk_env import TorqueSkeletonWalkingEnv


def render_walking_model(model_path: str, episodes: int = 3, expert_data_path: str = "data/expert_data.pkl", stage: int = None):
    """
    Load and render the trained walking model.
    
    Args:
        model_path: Path to the trained model
        episodes: Number of episodes to render
        expert_data_path: Path to expert data
        stage: Stage to initialize environment with (0: standing, 1: walking, 2: imitation)
               If None, will use the stage from the trained model
    """
    print(f"🎬 Loading and rendering walking model: {model_path}")
    
    # Create environment directly (not wrapped)
    env = TorqueSkeletonWalkingEnv(expert_data_path=expert_data_path, render_mode="human")
    
    # Override stage if specified
    if stage is not None:
        print(f"🎯 Setting environment to stage {stage}")
        env.stage = stage
        env.max_episode_steps = env._calculate_max_steps()
        
        # Load expert data if needed for stages 2+
        if stage >= 2:
            env._load_expert_data()
            
        # Update episode duration info
        episode_duration = env.base_episode_duration + (env.stage * env.stage_duration_increment)
        print(f"   📏 Episode length: {episode_duration}s ({env.max_episode_steps} steps)")
    
    # Load normalization if available
    vec_normalize_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    if os.path.exists(vec_normalize_path):
        print("📊 Loading normalization statistics...")
        env_dummy = DummyVecEnv([lambda: env])
        env_norm = VecNormalize.load(vec_normalize_path, env_dummy)
        env_norm.training = False
        env_norm.norm_reward = False
    else:
        print("⚠️ No normalization found, using environment directly")
        env_norm = None
    
    # Load the trained model
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    if env_norm is not None:
        model = PPO.load(model_path, env=env_norm)
    else:
        model = PPO.load(model_path)
    print("✅ Model loaded successfully!")
    
    print(f"\n🚀 Starting {episodes} episodes...")
    print("💡 MuJoCo viewer should open in a separate window")
    print("   Use mouse to rotate view, Space to pause, Esc to close")
    print("=" * 60)
    
    try:
        for episode in range(episodes):
            if env_norm is not None:
                obs = env_norm.reset()
            else:
                obs, info = env.reset()
                obs = np.array([obs])
            
            episode_reward = 0
            episode_steps = 0
            start_time = time.time()
            
            print(f"\n📺 Episode {episode + 1}/{episodes}")
            print("-" * 40)
            
            # Track metrics
            heights = []
            velocities = []
            tracking_errors = []
            stages = []
            
            done = False
            while not done:
                # Get action from trained model
                action, _ = model.predict(obs, deterministic=True)
                
                # Step environment
                if env_norm is not None:
                    obs, reward, done, info = env_norm.step(action)
                    # Get info from the underlying environment
                    info_dict = env.step(action[0])[-1] if hasattr(env, 'step') else {}
                else:
                    obs, reward, terminated, truncated, info_dict = env.step(action[0])
                    done = terminated or truncated
                    obs = np.array([obs])
                    reward = np.array([reward])
                
                episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                episode_steps += 1
                
                # Collect metrics
                if isinstance(info_dict, dict):
                    heights.append(info_dict.get('pelvis_height', 0))
                    velocities.append(info_dict.get('forward_velocity', 0))
                    tracking_errors.append(info_dict.get('tracking_error', 0))
                    stages.append(info_dict.get('stage', 0))
                
                # Render - this uses the environment directly, not the wrapped version
                env.render()
                
                # Print info every 100 steps
                if episode_steps % 100 == 0 and isinstance(info_dict, dict):
                    print(f"  Step {episode_steps:3d}: Stage={info_dict.get('stage', 0)}, "
                          f"Height={info_dict.get('pelvis_height', 0):.3f}m, "
                          f"Vel={info_dict.get('forward_velocity', 0):.3f}m/s, "
                          f"Track_Err={info_dict.get('tracking_error', 0):.3f}")
                
                # Small delay for visualization
                time.sleep(0.02)  # 50 FPS max
            
            # Episode summary
            episode_time = time.time() - start_time
            avg_height = np.mean(heights) if heights else 0
            avg_velocity = np.mean(velocities) if velocities else 0
            avg_tracking_error = np.mean(tracking_errors) if tracking_errors else 0
            final_stage = stages[-1] if stages else 0
            
            print(f"\n📊 Episode {episode + 1} Summary:")
            print(f"  ⏱️  Duration: {episode_time:.1f}s ({episode_steps} steps)")
            print(f"  🏆 Total Reward: {episode_reward:.1f}")
            print(f"  🎯 Final Stage: {final_stage}")
            print(f"  📏 Average Height: {avg_height:.3f}m (target: 0.975m)")
            print(f"  🏃 Average Velocity: {avg_velocity:.3f}m/s (target: 1.0m/s)")
            if avg_tracking_error > 0:
                print(f"  🎯 Average Tracking Error: {avg_tracking_error:.3f}")
            
            # Performance assessment
            if final_stage >= 2 and avg_height > 0.9 and abs(avg_velocity - 1.0) < 0.3 and avg_tracking_error < 0.5:
                print("  🌟 EXCELLENT walking performance!")
            elif final_stage >= 1 and avg_height > 0.8 and abs(avg_velocity - 1.0) < 0.5:
                print("  ✅ GOOD walking performance")
            elif avg_height > 0.8:
                print("  👍 Standing well, working on walking")
            else:
                print("  ⚠️  Needs more training")
            
            print("=" * 60)
            
            # Wait between episodes
            if episode < episodes - 1:
                print("Press Enter for next episode...")
                input()
    
    except KeyboardInterrupt:
        print("\n⏹️ Rendering interrupted by user")
    
    finally:
        print(f"\n🎬 Rendering complete!")
        env.close()


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
    parser.add_argument("--stage", type=int, default=None, choices=[0, 1, 2],
                        help="Stage to initialize environment with (0: standing, 1: walking, 2: imitation)")
    
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
        
        # Display stage information
        if args.stage is not None:
            stage_descriptions = {
                0: "Standing - Focus on maintaining upright posture",
                1: "Walking - Learning forward motion and basic locomotion", 
                2: "Imitation - Matching expert motion capture data"
            }
            print(f"🎯 Selected Stage {args.stage}: {stage_descriptions[args.stage]}")
        else:
            print("🤖 Using model's learned stage progression")
        
        # Render the model
        render_walking_model(model_path, args.episodes, args.expert_data, args.stage)
        
    except KeyboardInterrupt:
        print("\n⏹️ Rendering interrupted by user")
    except Exception as e:
        print(f"❌ Error during rendering: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 