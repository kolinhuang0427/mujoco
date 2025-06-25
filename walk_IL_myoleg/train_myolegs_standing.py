#!/usr/bin/env python3
"""
Training Script for MyoLegs Standing Task
Uses Stable Baselines3 PPO to train the muscle-driven humanoid model to stand upright.
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Add current directory to path for importing environment
sys.path.append(os.getcwd())

from myolegs_standing_env import MyoLegsStandingEnv
import gymnasium as gym

# Stable Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback


class ProgressCallback(BaseCallback):
    """Custom callback to log training progress and visualize performance."""
    
    def __init__(self, log_interval: int = 1000):
        super().__init__()
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Track episode progress
        self.current_episode_reward += self.locals.get('rewards', [0])[0]
        self.current_episode_length += 1
        
        # Log when episode ends
        if self.locals.get('dones', [False])[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    'episode_reward': self.current_episode_reward,
                    'episode_length': self.current_episode_length,
                    'global_step': self.num_timesteps
                })
            
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # Regular logging
        if self.num_timesteps % self.log_interval == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])  # Last 100 episodes
                mean_length = np.mean(self.episode_lengths[-100:])
                
                print(f"Step {self.num_timesteps}: "
                      f"Mean Reward (100 ep): {mean_reward:.2f}, "
                      f"Mean Length: {mean_length:.1f}")
        
        return True


def create_env(render_mode=None):
    """Create and wrap the environment."""
    env = MyoLegsStandingEnv(render_mode=render_mode)
    env = Monitor(env)
    return env


def train_myolegs(args):
    """Main training function."""
    
    # Set up experiment directory
    exp_name = f"myolegs_standing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join("logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"🚀 Starting MyoLegs Standing Training")
    print(f"   - Total timesteps: {args.total_timesteps:,}")
    print(f"   - Number of environments: {args.n_envs}")
    print(f"   - Log directory: {log_dir}")
    
    # Initialize Weights & Biases if requested
    if args.use_wandb:
        wandb.init(
            project="myolegs-standing",
            name=exp_name,
            config={
                "algorithm": "PPO",
                "total_timesteps": args.total_timesteps,
                "n_envs": args.n_envs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "n_epochs": args.n_epochs,
            }
        )
    
    # Create vectorized environment
    print("Creating environments...")
    if args.n_envs > 1:
        # Use subprocess environments for parallel training
        env = make_vec_env(
            create_env, 
            n_envs=args.n_envs, 
            vec_env_cls=SubprocVecEnv
        )
    else:
        env = make_vec_env(create_env, n_envs=1)
    
    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create evaluation environment
    eval_env = VecNormalize(
        make_vec_env(create_env, n_envs=1),
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0
    )
    
    # Define PPO model
    print("Initializing PPO model...")
    
    # Network architecture optimized for muscle control
    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512, 256, 256],  # Policy network
            vf=[512, 512, 256, 256]   # Value function network
        ),
        activation_fn=torch.nn.Tanh,
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        device=args.device
    )
    
    # Set up callbacks
    callbacks = []
    
    # Progress callback
    progress_callback = ProgressCallback(log_interval=args.log_interval)
    callbacks.append(progress_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=log_dir,
        name_prefix="myolegs_standing"
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    callbacks.append(eval_callback)
    
    # Wandb callback
    if args.use_wandb:
        wandb_callback = WandbCallback(
            gradient_save_freq=args.save_freq,
            model_save_path=log_dir,
            verbose=2,
        )
        callbacks.append(wandb_callback)
    
    callback_list = CallbackList(callbacks)
    
    # Start training
    print("🏃 Starting training...")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Observation space shape: {env.observation_space.shape}")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback_list,
            progress_bar=True
        )
        
        print("✅ Training completed successfully!")
        
        # Save final model
        final_model_path = os.path.join(log_dir, "final_model")
        model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        # Save environment normalization stats
        env.save(os.path.join(log_dir, "vec_normalize.pkl"))
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        # Save current model
        model.save(os.path.join(log_dir, "interrupted_model"))
        print("Model saved before interruption")
    
    finally:
        env.close()
        eval_env.close()
        
        if args.use_wandb:
            wandb.finish()
    
    return log_dir, model


def evaluate_model(model_path, env_normalize_path=None, n_episodes=10, render=True):
    """Evaluate a trained model."""
    print(f"🔍 Evaluating model: {model_path}")
    
    # Create environment
    env = create_env(render_mode="human" if render else None)
    
    # Wrap with normalization if available
    if env_normalize_path and os.path.exists(env_normalize_path):
        env = VecNormalize.load(env_normalize_path, make_vec_env(create_env, n_envs=1))
        env.training = False
        env.norm_reward = False
    else:
        env = make_vec_env(lambda: env, n_envs=1)
    
    # Load model
    model = PPO.load(model_path)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"Episode {episode + 1}/{n_episodes}")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            episode_length += 1
            
            if render and episode < 3:  # Only render first few episodes
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"  Reward: {episode_reward:.2f}, Length: {episode_length}")
    
    env.close()
    
    print(f"\n📊 Evaluation Results:")
    print(f"  Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    return episode_rewards, episode_lengths


def main():
    parser = argparse.ArgumentParser(description="Train MyoLegs to stand")
    
    # Training parameters
    parser.add_argument("--total_timesteps", type=int, default=2_000_000,
                        help="Total training timesteps")
    parser.add_argument("--n_envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Number of steps per environment per update")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Minibatch size")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip_range", type=float, default=0.2,
                        help="PPO clipping range")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="Value function coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="Max gradient norm")
    
    # Logging and evaluation
    parser.add_argument("--log_interval", type=int, default=1000,
                        help="Log interval")
    parser.add_argument("--save_freq", type=int, default=50000,
                        help="Save frequency")
    parser.add_argument("--eval_freq", type=int, default=25000,
                        help="Evaluation frequency")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases logging")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu, cuda, auto)")
    
    # Mode
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"],
                        help="Mode: train or evaluate")
    parser.add_argument("--model_path", type=str,
                        help="Path to model for evaluation")
    parser.add_argument("--env_normalize_path", type=str,
                        help="Path to environment normalization file")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        log_dir, model = train_myolegs(args)
        print(f"\n🎯 Training completed! Check logs at: {log_dir}")
        
        # Quick evaluation
        print("\n🔍 Running quick evaluation...")
        model_path = os.path.join(log_dir, "final_model")
        env_normalize_path = os.path.join(log_dir, "vec_normalize.pkl")
        evaluate_model(model_path, env_normalize_path, n_episodes=3, render=True)
        
    elif args.mode == "evaluate":
        if not args.model_path:
            print("❌ Model path required for evaluation mode")
            return
        
        evaluate_model(
            args.model_path, 
            args.env_normalize_path, 
            n_episodes=10, 
            render=True
        )


if __name__ == "__main__":
    main() 