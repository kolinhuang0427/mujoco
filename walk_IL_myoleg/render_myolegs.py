#!/usr/bin/env python3
"""
Render Trained MyoLegs Model
Loads the trained standing model and visualizes the performance with real-time metrics.
"""

import numpy as np
import time
from stable_baselines3 import PPO
from myolegs_standing_env import MyoLegsStandingEnv
import os
import argparse


def render_trained_model(model_path: str, episodes: int = 5, render_mode: str = "human"):
    """
    Load and render the trained MyoLegs model.
    
    Args:
        model_path: Path to the trained model
        episodes: Number of episodes to render
        render_mode: "human" for interactive viewer, "rgb_array" for headless
    """
    print(f"🎬 Loading and rendering model: {model_path}")
    
    # Create environment with rendering
    env = MyoLegsStandingEnv(render_mode=render_mode)
    
    # Load the trained model
    if not os.path.exists(model_path + ".zip"):
        print(f"❌ Model file not found: {model_path}.zip")
        return
    
    model = PPO.load(model_path, env=env)
    print("✅ Model loaded successfully!")
    
    print(f"\n🚀 Starting {episodes} episodes with rendering...")
    print("=" * 60)
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        start_time = time.time()
        
        print(f"\n📺 Episode {episode + 1}/{episodes}")
        print("-" * 40)
        
        # Track metrics throughout episode
        heights = []
        uprightness_scores = []
        muscle_activations = []
        
        done = False
        while not done:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_steps += 1
            
            # Collect metrics
            heights.append(info.get('pelvis_height', 0))
            uprightness_scores.append(info.get('uprightness', 0))
            muscle_activations.append(info.get('muscle_activation_mean', 0))
            
            # Render the environment
            env.render()
            
            # Print real-time info every 100 steps
            if episode_steps % 100 == 0:
                print(f"  Step {episode_steps:4d}: Height={info.get('pelvis_height', 0):.3f}m, "
                      f"Upright={info.get('uprightness', 0):.3f}, "
                      f"Reward={reward:.1f}, "
                      f"Muscle={info.get('muscle_activation_mean', 0):.3f}")
            
            # Small delay for human viewing
            if render_mode == "human":
                time.sleep(0.01)  # 100 FPS max
        
        # Episode summary
        episode_time = time.time() - start_time
        avg_height = np.mean(heights) if heights else 0
        avg_uprightness = np.mean(uprightness_scores) if uprightness_scores else 0
        avg_muscle = np.mean(muscle_activations) if muscle_activations else 0
        
        print(f"\n📊 Episode {episode + 1} Summary:")
        print(f"  ⏱️  Duration: {episode_time:.1f}s ({episode_steps} steps)")
        print(f"  🏆 Total Reward: {episode_reward:.1f}")
        print(f"  📏 Average Height: {avg_height:.3f}m (target: 0.98m)")
        print(f"  ⬆️  Average Uprightness: {avg_uprightness:.3f}")
        print(f"  💪 Average Muscle Activation: {avg_muscle:.3f}")
        
        # Performance assessment
        if avg_height > 0.9 and avg_uprightness > 0.8:
            print("  ✅ EXCELLENT standing performance!")
        elif avg_height > 0.8 and avg_uprightness > 0.6:
            print("  👍 GOOD standing performance")
        else:
            print("  ⚠️  Needs improvement")
        
        print("=" * 60)
        
        # Wait between episodes for human viewing
        if render_mode == "human" and episode < episodes - 1:
            input("Press Enter for next episode...")
    
    print(f"\n🎬 Rendering complete! Showed {episodes} episodes.")
    env.close()


def find_latest_model():
    """
    Find the latest trained model in the logs directory.
    
    Returns:
        str: Path to the latest model (without .zip extension) or None if not found
    """
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        print("❌ No logs directory found. Please train a model first.")
        return None
    
    print("🔍 Looking for the latest trained model in logs directory...")
    model_dirs = []
    for item in os.listdir(logs_dir):
        item_path = os.path.join(logs_dir, item)
        if os.path.isdir(item_path) and "myolegs_standing" in item:
            final_model = os.path.join(item_path, "final_model.zip")
            if os.path.exists(final_model):
                model_dirs.append((item_path, final_model))
    
    if not model_dirs:
        print("❌ No trained models found. Please train a model first.")
        return None
    
    # Use the most recent model
    latest_dir = max(model_dirs, key=lambda x: os.path.getctime(x[0]))
    model_path = latest_dir[1][:-4]  # Remove .zip extension
    print(f"✅ Found latest model: {model_path}")
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Render trained MyoLegs standing model")
    parser.add_argument("--model", type=str, 
                       default=None,
                       help="Path to trained model (without .zip extension). If not specified, uses the latest model.")
    parser.add_argument("--episodes", type=int, default=3,
                       help="Number of episodes to render")
    parser.add_argument("--render_mode", type=str, default="human",
                       choices=["human", "rgb_array"],
                       help="Rendering mode")
    
    args = parser.parse_args()
    
    # If no model specified, or specified model doesn't exist, find the latest one
    if args.model is None:
        print("🔄 No model specified, finding the latest model...")
        args.model = find_latest_model()
    elif not os.path.exists(args.model + ".zip"):
        print(f"❌ Specified model not found: {args.model}.zip")
        print("🔄 Finding the latest model instead...")
        args.model = find_latest_model()
    
    if args.model is None:
        return
    
    render_trained_model(args.model, args.episodes, args.render_mode)


if __name__ == "__main__":
    main() 