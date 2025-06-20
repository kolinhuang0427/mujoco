#!/usr/bin/env python3
"""
Render Trained MyoLegs Walking Imitation Model
Loads the trained walking imitation model and visualizes the performance.
"""

import numpy as np
import time
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from myolegs_walking_imitation_env import MyoLegsWalkingImitationEnv


def render_walking_model(model_path: str, episodes: int = 3, expert_data_path: str = "walk_IL/data/expert_data.pkl"):
    """
    Load and render the trained walking imitation model.
    
    Args:
        model_path: Path to the trained model (without .zip extension)
        episodes: Number of episodes to render
        expert_data_path: Path to expert data
    """
    print(f"🎬 Loading and rendering walking model: {model_path}")
    
    # Create environment
    env = MyoLegsWalkingImitationEnv(expert_data_path=expert_data_path, render_mode="human")
    
    # Load normalization if available
    vec_normalize_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    if os.path.exists(vec_normalize_path):
        print("Loading normalization statistics...")
        env_dummy = DummyVecEnv([lambda: env])
        env_norm = VecNormalize.load(vec_normalize_path, env_dummy)
        env_norm.training = False
    else:
        print("No normalization found, using environment directly")
        env_norm = None
    
    # Load the trained model
    if not os.path.exists(model_path + ".zip"):
        print(f"❌ Model file not found: {model_path}.zip")
        return
    
    if env_norm is not None:
        model = PPO.load(model_path, env=env_norm)
    else:
        model = PPO.load(model_path, env=env)
    print("✅ Model loaded successfully!")
    
    print(f"\n🚀 Starting {episodes} episodes...")
    print("=" * 60)
    
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
        
        done = False
        while not done:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            if env_norm is not None:
                obs, reward, done, info = env_norm.step(action)
                # Get info from the underlying environment
                info_dict = env.unwrapped.step(action)[-1] if hasattr(env, 'unwrapped') else {}
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
            
            # Render
            env.render()
            
            # Print info every 50 steps
            if episode_steps % 50 == 0 and isinstance(info_dict, dict):
                print(f"  Step {episode_steps:3d}: Height={info_dict.get('pelvis_height', 0):.3f}m, "
                      f"Vel={info_dict.get('forward_velocity', 0):.3f}m/s, "
                      f"Track_Err={info_dict.get('tracking_error', 0):.3f}, "
                      f"Phase={info_dict.get('expert_phase', 0):.2f}")
            
            # Small delay for visualization
            time.sleep(0.02)  # 50 FPS max
        
        # Episode summary
        episode_time = time.time() - start_time
        avg_height = np.mean(heights) if heights else 0
        avg_velocity = np.mean(velocities) if velocities else 0
        avg_tracking_error = np.mean(tracking_errors) if tracking_errors else 0
        
        print(f"\n📊 Episode {episode + 1} Summary:")
        print(f"  ⏱️  Duration: {episode_time:.1f}s ({episode_steps} steps)")
        print(f"  🏆 Total Reward: {episode_reward:.1f}")
        print(f"  📏 Average Height: {avg_height:.3f}m (target: 0.92m)")
        print(f"  🏃 Average Velocity: {avg_velocity:.3f}m/s (target: 1.0m/s)")
        print(f"  🎯 Average Tracking Error: {avg_tracking_error:.3f}")
        
        # Performance assessment
        if avg_height > 0.85 and abs(avg_velocity - 1.0) < 0.3 and avg_tracking_error < 0.5:
            print("  ✅ EXCELLENT walking performance!")
        elif avg_height > 0.8 and abs(avg_velocity - 1.0) < 0.5 and avg_tracking_error < 1.0:
            print("  👍 GOOD walking performance")
        else:
            print("  ⚠️  Needs improvement")
        
        print("=" * 60)
        
        # Wait between episodes
        if episode < episodes - 1:
            input("Press Enter for next episode...")
    
    print(f"\n🎬 Rendering complete!")
    env.close()


def find_latest_walking_model():
    """Find the latest trained walking model."""
    logs_dir = "walk_IL/logs"
    if not os.path.exists(logs_dir):
        return None
    
    model_dirs = []
    for item in os.listdir(logs_dir):
        item_path = os.path.join(logs_dir, item)
        if os.path.isdir(item_path) and "walking_imitation" in item:
            final_model = os.path.join(item_path, "best_model.zip")
            if os.path.exists(final_model):
                model_dirs.append((item_path, final_model))
    
    if not model_dirs:
        return None
    
    # Use the most recent model
    latest_dir = max(model_dirs, key=lambda x: os.path.getctime(x[0]))
    return latest_dir[1][:-4]  # Remove .zip extension


def main():
    parser = argparse.ArgumentParser(description="Render trained MyoLegs walking imitation model")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to trained model (without .zip extension)")
    parser.add_argument("--expert_data", type=str, default="walk_IL/data/expert_data.pkl",
                       help="Path to expert data")
    parser.add_argument("--episodes", type=int, default=3,
                       help="Number of episodes to render")
    
    args = parser.parse_args()
    
    # Find model if not specified
    if args.model is None:
        print("🔍 Finding latest walking model...")
        args.model = find_latest_walking_model()
    
    if args.model is None:
        print("❌ No trained walking models found.")
        print("   Please train a model first: python train_walking_imitation.py")
        return
    
    if not os.path.exists(args.model + ".zip"):
        print(f"❌ Model not found: {args.model}.zip")
        return
    
    if not os.path.exists(args.expert_data):
        print(f"❌ Expert data not found: {args.expert_data}")
        return
    
    render_walking_model(args.model, args.episodes, args.expert_data)


if __name__ == "__main__":
    main() 