#!/usr/bin/env python3
"""
Enhanced Training script for Torque Skeleton Walking Imitation Learning
Uses the enhanced environment with LocoMujoco-style XML modifications for stable physics.

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

import argparse
import os
import time
from datetime import datetime
from collections import deque
from typing import Dict, Any

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback, EvalCallback
)

# Set matplotlib backend to non-interactive before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import our enhanced environment
from walk_env_enhanced import EnhancedTorqueSkeletonWalkingEnv

# Try to import wandb for logging
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available for logging")


class ProgressCallback(BaseCallback):
    """Log training progress to console and wandb."""
    
    def __init__(self, log_interval=10000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.last_log_step = 0
        
        # Track metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.tracking_errors = deque(maxlen=100)
        self.heights = deque(maxlen=100)
        self.velocities = deque(maxlen=100)
        
        # Track individual reward components
        self.joint_tracking_rewards = deque(maxlen=100)
        self.height_rewards = deque(maxlen=100)
        self.velocity_rewards = deque(maxlen=100)
        self.uprightness_rewards = deque(maxlen=100)
        self.alive_bonus_rewards = deque(maxlen=100)
        
        # Track raw reward values for analysis
        self.joint_tracking_raw = deque(maxlen=100)
        self.height_raw = deque(maxlen=100)
        self.velocity_raw = deque(maxlen=100)
        self.uprightness_raw = deque(maxlen=100)
        self.mean_joint_errors = deque(maxlen=100)
    
    def _on_step(self) -> bool:
        # Collect episode info
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                ep_info = info['episode']
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
            
            # Collect environment-specific metrics
            if 'tracking_error' in info:
                self.tracking_errors.append(info['tracking_error'])
            if 'pelvis_height' in info:
                self.heights.append(info['pelvis_height'])
            if 'forward_velocity' in info:
                self.velocities.append(info['forward_velocity'])
            
            # Collect individual reward components
            if 'joint_tracking' in info:
                self.joint_tracking_rewards.append(info['joint_tracking'])
            if 'height' in info:
                self.height_rewards.append(info['height'])
            if 'velocity' in info:
                self.velocity_rewards.append(info['velocity'])
            if 'uprightness' in info:
                self.uprightness_rewards.append(info['uprightness'])
            if 'alive_bonus' in info:
                self.alive_bonus_rewards.append(info['alive_bonus'])
            
            # Collect raw reward values
            if 'joint_tracking_raw' in info:
                self.joint_tracking_raw.append(info['joint_tracking_raw'])
            if 'height_raw' in info:
                self.height_raw.append(info['height_raw'])
            if 'velocity_raw' in info:
                self.velocity_raw.append(info['velocity_raw'])
            if 'uprightness_raw' in info:
                self.uprightness_raw.append(info['uprightness_raw'])
            if 'mean_joint_error' in info:
                self.mean_joint_errors.append(info['mean_joint_error'])
        
        # Log progress periodically
        if self.num_timesteps - self.last_log_step >= self.log_interval:
            self.last_log_step = self.num_timesteps
            
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
            avg_tracking_error = np.mean(self.tracking_errors) if self.tracking_errors else 0
            avg_height = np.mean(self.heights) if self.heights else 0
            avg_velocity = np.mean(self.velocities) if self.velocities else 0
            
            # Calculate average reward components
            avg_joint_tracking = np.mean(self.joint_tracking_rewards) if self.joint_tracking_rewards else 0
            avg_height_reward = np.mean(self.height_rewards) if self.height_rewards else 0
            avg_velocity_reward = np.mean(self.velocity_rewards) if self.velocity_rewards else 0
            avg_uprightness_reward = np.mean(self.uprightness_rewards) if self.uprightness_rewards else 0
            avg_alive_bonus = np.mean(self.alive_bonus_rewards) if self.alive_bonus_rewards else 0
            
            # Calculate average raw values
            avg_joint_tracking_raw = np.mean(self.joint_tracking_raw) if self.joint_tracking_raw else 0
            avg_height_raw = np.mean(self.height_raw) if self.height_raw else 0
            avg_velocity_raw = np.mean(self.velocity_raw) if self.velocity_raw else 0
            avg_uprightness_raw = np.mean(self.uprightness_raw) if self.uprightness_raw else 0
            avg_mean_joint_error = np.mean(self.mean_joint_errors) if self.mean_joint_errors else 0
            
            print(f"\n🏃 Enhanced Training Progress - Step {self.num_timesteps:,}")
            print(f"   Avg Reward: {avg_reward:.2f}")
            print(f"   Avg Episode Length: {avg_length:.0f}")
            print(f"   Avg Height: {avg_height:.3f}m")
            print(f"   Avg Velocity: {avg_velocity:.3f}m/s")
            if avg_tracking_error > 0:
                print(f"   Avg Tracking Error: {avg_tracking_error:.3f}")
            
            # Print reward component breakdown
            if avg_joint_tracking > 0:
                print(f"   Reward Components:")
                print(f"     Joint Tracking: {avg_joint_tracking:.2f} (raw: {avg_joint_tracking_raw:.3f})")
                print(f"     Height: {avg_height_reward:.2f} (raw: {avg_height_raw:.3f})")
                print(f"     Velocity: {avg_velocity_reward:.2f} (raw: {avg_velocity_raw:.3f})")
                print(f"     Uprightness: {avg_uprightness_reward:.2f} (raw: {avg_uprightness_raw:.3f})")
                print(f"     Alive Bonus: {avg_alive_bonus:.2f}")
                print(f"     Mean Joint Error: {avg_mean_joint_error:.3f}")
            
            # Log to wandb if available
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb_log = {
                    "train/episode_reward_mean": avg_reward,
                    "train/episode_length_mean": avg_length,
                    "env/pelvis_height_mean": avg_height,
                    "env/forward_velocity_mean": avg_velocity,
                    "env/tracking_error_mean": avg_tracking_error,
                    "global_step": self.num_timesteps,
                }
                
                # Add reward components section
                if avg_joint_tracking > 0:
                    wandb_log.update({
                        # Weighted reward components
                        "rewards/joint_tracking": avg_joint_tracking,
                        "rewards/height": avg_height_reward,
                        "rewards/velocity": avg_velocity_reward,
                        "rewards/uprightness": avg_uprightness_reward,
                        "rewards/alive_bonus": avg_alive_bonus,
                        
                        # Raw reward values (before weighting)
                        "rewards_raw/joint_tracking": avg_joint_tracking_raw,
                        "rewards_raw/height": avg_height_raw,
                        "rewards_raw/velocity": avg_velocity_raw,
                        "rewards_raw/uprightness": avg_uprightness_raw,
                        
                        # Analysis metrics
                        "analysis/mean_joint_error": avg_mean_joint_error,
                        "analysis/reward_ratio_joint_to_alive": avg_joint_tracking / max(avg_alive_bonus, 0.01),
                        "analysis/reward_total_secondary": avg_height_reward + avg_velocity_reward + avg_uprightness_reward + avg_alive_bonus,
                    })
                
                wandb.log(wandb_log, step=self.num_timesteps)
        
        return True


class PlottingCallback(BaseCallback):
    """Create training plots."""
    
    def __init__(self, log_dir, plot_interval=50000, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.plot_interval = plot_interval
        self.last_plot_step = 0
        
        # Data storage
        self.timesteps = []
        self.rewards = []
        self.episode_lengths = []
        self.tracking_errors = []
        self.heights = []
        self.velocities = []
        
        # Reward component tracking
        self.joint_tracking_rewards = []
        self.height_rewards = []
        self.velocity_rewards = []
        self.uprightness_rewards = []
        
        os.makedirs(os.path.join(log_dir, "plots"), exist_ok=True)
    
    def _on_step(self) -> bool:
        # Collect data
        infos = self.locals.get('infos', [])
        current_rewards = []
        current_lengths = []
        current_tracking_errors = []
        current_heights = []
        current_velocities = []
        
        current_joint_tracking = []
        current_height_rewards = []
        current_velocity_rewards = []
        current_uprightness_rewards = []
        
        for info in infos:
            if 'episode' in info:
                current_rewards.append(info['episode']['r'])
                current_lengths.append(info['episode']['l'])
            
            if 'tracking_error' in info:
                current_tracking_errors.append(info['tracking_error'])
            if 'pelvis_height' in info:
                current_heights.append(info['pelvis_height'])
            if 'forward_velocity' in info:
                current_velocities.append(info['forward_velocity'])
            
            # Reward components
            if 'joint_tracking' in info:
                current_joint_tracking.append(info['joint_tracking'])
            if 'height' in info:
                current_height_rewards.append(info['height'])
            if 'velocity' in info:
                current_velocity_rewards.append(info['velocity'])
            if 'uprightness' in info:
                current_uprightness_rewards.append(info['uprightness'])
        
        # Store data if available
        if current_rewards:
            self.timesteps.append(self.num_timesteps)
            self.rewards.append(np.mean(current_rewards))
            self.episode_lengths.append(np.mean(current_lengths))
            
            if current_tracking_errors:
                self.tracking_errors.append(np.mean(current_tracking_errors))
            if current_heights:
                self.heights.append(np.mean(current_heights))
            if current_velocities:
                self.velocities.append(np.mean(current_velocities))
                
            # Reward components
            if current_joint_tracking:
                self.joint_tracking_rewards.append(np.mean(current_joint_tracking))
            if current_height_rewards:
                self.height_rewards.append(np.mean(current_height_rewards))
            if current_velocity_rewards:
                self.velocity_rewards.append(np.mean(current_velocity_rewards))
            if current_uprightness_rewards:
                self.uprightness_rewards.append(np.mean(current_uprightness_rewards))
        
        # Create plots periodically
        if self.num_timesteps - self.last_plot_step >= self.plot_interval:
            self.last_plot_step = self.num_timesteps
            self._create_plots()
        
        return True
    
    def _create_plots(self):
        """Create training progress plots."""
        if not self.timesteps:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Enhanced Torque Skeleton Training Progress (Step {self.num_timesteps:,})', fontsize=16)
        
        # Plot 1: Rewards
        if self.rewards:
            axes[0,0].plot(self.timesteps, self.rewards, 'b-', linewidth=2)
            axes[0,0].set_title('Episode Rewards')
            axes[0,0].set_xlabel('Timesteps')
            axes[0,0].set_ylabel('Average Reward')
            axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Episode Length
        if self.episode_lengths:
            axes[0,1].plot(self.timesteps, self.episode_lengths, 'g-', linewidth=2)
            axes[0,1].set_title('Episode Length')
            axes[0,1].set_xlabel('Timesteps')
            axes[0,1].set_ylabel('Average Steps')
            axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Height and Velocity
        if self.heights and self.velocities:
            ax1 = axes[0,2]
            ax2 = ax1.twinx()
            
            line1 = ax1.plot(self.timesteps, self.heights, 'r-', linewidth=2, label='Height (m)')
            line2 = ax2.plot(self.timesteps, self.velocities, 'b-', linewidth=2, label='Velocity (m/s)')
            
            ax1.set_xlabel('Timesteps')
            ax1.set_ylabel('Height (m)', color='r')
            ax2.set_ylabel('Velocity (m/s)', color='b')
            ax1.tick_params(axis='y', labelcolor='r')
            ax2.tick_params(axis='y', labelcolor='b')
            ax1.grid(True, alpha=0.3)
            
            # Add legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
            axes[0,2].set_title('Height & Velocity')
        
        # Plot 4: Tracking Error
        if self.tracking_errors:
            axes[1,0].plot(self.timesteps, self.tracking_errors, 'm-', linewidth=2)
            axes[1,0].set_title('Joint Tracking Error')
            axes[1,0].set_xlabel('Timesteps')
            axes[1,0].set_ylabel('RMS Error (rad)')
            axes[1,0].grid(True, alpha=0.3)
        
        # Plot 5: Reward Components
        if (self.joint_tracking_rewards and self.height_rewards and 
            self.velocity_rewards and self.uprightness_rewards):
            axes[1,1].plot(self.timesteps, self.joint_tracking_rewards, 'r-', linewidth=2, label='Joint Tracking')
            axes[1,1].plot(self.timesteps, self.height_rewards, 'g-', linewidth=2, label='Height')
            axes[1,1].plot(self.timesteps, self.velocity_rewards, 'b-', linewidth=2, label='Velocity')
            axes[1,1].plot(self.timesteps, self.uprightness_rewards, 'm-', linewidth=2, label='Uprightness')
            axes[1,1].set_title('Reward Components')
            axes[1,1].set_xlabel('Timesteps')
            axes[1,1].set_ylabel('Reward Value')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        # Plot 6: Environment Status (success rate, stability metrics)
        if self.heights and self.episode_lengths:
            # Calculate stability metrics
            stable_episodes = [1 if h > 0.6 else 0 for h in self.heights]  # Height threshold
            long_episodes = [1 if l > 100 else 0 for l in self.episode_lengths]  # Episode length threshold
            
            # Rolling averages for smoother display
            window_size = min(10, len(stable_episodes))
            if window_size > 1:
                stable_rate = [np.mean(stable_episodes[max(0, i-window_size):i+1]) for i in range(len(stable_episodes))]
                long_rate = [np.mean(long_episodes[max(0, i-window_size):i+1]) for i in range(len(long_episodes))]
                
                axes[1,2].plot(self.timesteps, stable_rate, 'g-', linewidth=2, label='Stability Rate')
                axes[1,2].plot(self.timesteps, long_rate, 'b-', linewidth=2, label='Long Episode Rate')
                axes[1,2].set_title('Success Metrics')
                axes[1,2].set_xlabel('Timesteps')
                axes[1,2].set_ylabel('Success Rate')
                axes[1,2].set_ylim(0, 1.1)
                axes[1,2].legend()
                axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.log_dir, "plots", f"training_progress_{self.num_timesteps:08d}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save the latest plot with a consistent name
        latest_plot_path = os.path.join(self.log_dir, "plots", "latest_progress.png")
        plt.figure(figsize=(18, 12))
        plt.suptitle(f'Enhanced Torque Skeleton Training Progress (Step {self.num_timesteps:,})', fontsize=16)
        
        # Recreate the same plots in the "latest" figure
        self._create_plots()  # This will be saved as latest
        
        print(f"📊 Plots saved: {plot_path}")


def create_enhanced_env(expert_data_path="data/expert_data.pkl", render_mode=None, 
                       use_box_feet=True, disable_arms=True, **kwargs):
    """Create and wrap the enhanced environment."""
    env = EnhancedTorqueSkeletonWalkingEnv(
        expert_data_path=expert_data_path, 
        render_mode=render_mode,
        use_box_feet=use_box_feet,
        disable_arms=disable_arms,
        **kwargs
    )
    env = Monitor(env)
    return env


def train_enhanced_torque_skeleton_walking(args):
    """Train the enhanced torque skeleton walking agent."""
    
    # Initialize wandb
    if WANDB_AVAILABLE:
        # Set wandb API key
        os.environ["WANDB_API_KEY"] = "e21f03ade561068a6e94fb2339b320871562c8d1"
        
        # Initialize wandb run
        wandb_run = wandb.init(
            project="enhanced-torque-skeleton-walking",
            name=f"enhanced_walk_IL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "algorithm": "PPO",
                "env": "EnhancedTorqueSkeletonWalking-v0",
                "environment_type": "enhanced_with_loco_mujoco_modifications",
                "use_box_feet": args.use_box_feet,
                "disable_arms": args.disable_arms,
                "total_timesteps": args.total_timesteps,
                "n_envs": args.n_envs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "n_epochs": args.n_epochs,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "clip_range": args.clip_range,
                "ent_coef": args.ent_coef,
                "vf_coef": args.vf_coef,
                "max_grad_norm": args.max_grad_norm,
                "n_steps": args.n_steps,
                "expert_data_path": args.expert_data,
                "seed": args.seed,
            },
            tags=["imitation_learning", "humanoid", "walking", "mujoco", "enhanced", "loco_mujoco_style"],
            notes="Enhanced torque skeleton humanoid walking with LocoMujoco-style XML modifications for stable physics",
            sync_tensorboard=True,
        )
        print(f"✅ Wandb initialized: {wandb_run.url}")
    else:
        wandb_run = None
        print("⚠️ Wandb not available, skipping online logging")
    
    # Set random seeds for reproducibility
    if hasattr(args, 'seed'):
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        print(f"🎲 Random seed set to: {args.seed}")
    
    # Check if expert data exists
    if not os.path.exists(args.expert_data):
        print(f"❌ Expert data not found: {args.expert_data}")
        print("Please ensure expert data is available at the specified path")
        return
    
    # Set up experiment directory
    exp_name = f"enhanced_torque_skeleton_walking_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join("logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"🚀 Starting Enhanced Torque Skeleton Walking Training")
    print(f"   - Environment: Enhanced with LocoMujoco-style modifications")
    print(f"   - Box feet: {'✅ Enabled' if args.use_box_feet else '❌ Disabled'}")
    print(f"   - Arms disabled: {'✅ Enabled' if args.disable_arms else '❌ Disabled'}")
    print(f"   - Expert data: {args.expert_data}")
    print(f"   - Total timesteps: {args.total_timesteps:,}")
    print(f"   - Number of environments: {args.n_envs}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Device: {args.device}")
    print(f"   - Matplotlib backend: {matplotlib.get_backend()}")
    print(f"   - Log directory: {log_dir}")
    
    # Create vectorized environment
    print("Creating enhanced environments...")
    if args.n_envs > 1:
        env = SubprocVecEnv([
            lambda: create_enhanced_env(
                args.expert_data, 
                use_box_feet=args.use_box_feet,
                disable_arms=args.disable_arms
            ) 
            for _ in range(args.n_envs)
        ])
    else:
        env = DummyVecEnv([
            lambda: create_enhanced_env(
                args.expert_data,
                use_box_feet=args.use_box_feet,
                disable_arms=args.disable_arms
            )
        ])
    
    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create evaluation environment
    eval_env = VecNormalize(
        DummyVecEnv([
            lambda: create_enhanced_env(
                args.expert_data,
                use_box_feet=args.use_box_feet,
                disable_arms=args.disable_arms
            )
        ]),
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0
    )
    
    # Define PPO model for enhanced torque control
    print("Initializing PPO model for enhanced environment...")
    
    policy_kwargs = dict(
        net_arch=dict(
            pi=[1024, 768, 512, 256],  # 4-layer network for torque control
            vf=[1024, 768, 512, 256]   # Matching capacity for value function
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
    
    print(f"Model parameters: ~{sum(p.numel() for p in model.policy.parameters()):,}")
    
    # Set up callbacks
    callbacks = []
    
    # Progress callback
    progress_callback = ProgressCallback(log_interval=args.log_interval, verbose=1)
    callbacks.append(progress_callback)
    
    # Plotting callback
    if not args.no_plots:
        plotting_callback = PlottingCallback(log_dir, plot_interval=args.plot_interval, verbose=1)
        callbacks.append(plotting_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=log_dir,
        name_prefix="enhanced_torque_skeleton_walking",
        verbose=1
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
        n_eval_episodes=5,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # Wandb callback
    if wandb_run is not None:
        wandb_callback = WandbCallback(
            gradient_save_freq=args.save_freq,
            model_save_path=log_dir,
            verbose=2,
        )
        callbacks.append(wandb_callback)
    
    callback_list = CallbackList(callbacks)
    
    # Start training
    print("🏃 Starting enhanced training...")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Observation space shape: {env.observation_space.shape}")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback_list,
            progress_bar=True
        )
        
        # Save final model
        final_model_path = os.path.join(log_dir, "final_model")
        model.save(final_model_path)
        
        # Save normalization statistics
        env.save(os.path.join(log_dir, "vec_normalize.pkl"))
        
        print(f"✅ Enhanced training completed!")
        print(f"   - Final model saved: {final_model_path}")
        print(f"   - Logs saved: {log_dir}")
        if not args.no_plots:
            print(f"   - Plots saved: {os.path.join(log_dir, 'plots')}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        interrupted_model_path = os.path.join(log_dir, "interrupted_model")
        model.save(interrupted_model_path)
        print(f"Model saved: {interrupted_model_path}")
    
    finally:
        env.close()
        if eval_env:
            eval_env.close()
        
        if wandb_run is not None:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train enhanced torque skeleton walking with LocoMujoco-style modifications")
    
    # Data parameters
    parser.add_argument("--expert_data", type=str, default="data/expert_data.pkl",
                        help="Path to expert motion capture data")
    
    # Environment parameters (LocoMujoco-style)
    parser.add_argument("--use_box_feet", action="store_true", default=True,
                        help="Use box feet (LocoMujoco style) for stable contact")
    parser.add_argument("--disable_arms", action="store_true", default=True,
                        help="Disable arm joints (LocoMujoco style) to focus on locomotion")
    parser.add_argument("--no_box_feet", dest="use_box_feet", action="store_false",
                        help="Disable box feet (use original foot geometry)")
    parser.add_argument("--enable_arms", dest="disable_arms", action="store_false",
                        help="Enable arm joints")
    
    # Training parameters
    parser.add_argument("--total_timesteps", type=int, default=10_000_000,  # Reduced for initial testing
                        help="Total training timesteps")
    parser.add_argument("--n_envs", type=int, default=32,  # Reduced for stability
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
    parser.add_argument("--log_interval", type=int, default=10000,
                        help="Log interval")
    parser.add_argument("--save_freq", type=int, default=100000,
                        help="Model save frequency")
    parser.add_argument("--eval_freq", type=int, default=50000,
                        help="Evaluation frequency")
    parser.add_argument("--plot_interval", type=int, default=50000,
                        help="Plot creation interval")
    
    # System parameters
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu, cuda, auto)")
    parser.add_argument("--no_plots", action="store_true", default=True,
                        help="Disable plot generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🖥️  Using device: {args.device}")
    print(f"🎯 Enhanced Training Configuration:")
    print(f"   - Environment: Enhanced Torque Skeleton with LocoMujoco modifications")
    print(f"   - Box feet: {'✅ Enabled' if args.use_box_feet else '❌ Disabled'}")
    print(f"   - Arms disabled: {'✅ Enabled' if args.disable_arms else '❌ Disabled'}")
    print(f"   - Expert data: {args.expert_data}")
    print(f"   - Total training steps: {args.total_timesteps:,}")
    print(f"   - Parallel environments: {args.n_envs}")
    
    train_enhanced_torque_skeleton_walking(args)


if __name__ == "__main__":
    main()