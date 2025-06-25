#!/usr/bin/env python3
"""
Train MyoLegs Motion Imitation
Implementation of the training framework from the paper using PPO and optional Lattice exploration.
Uses the exact 6-hidden layer MLP architecture: [2048, 1536, 1024, 1024, 512, 512]
"""

import os
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Weights & Biases not available. Install with: pip install wandb")

from myolegs_motion_imitation_env import MyoLegsMotionImitationEnv


class LatticeNoiseLayer(nn.Module):
    """
    Lattice exploration layer that injects correlated noise into latent space.
    Based on the Lattice method for improved exploration in high-dimensional control.
    """
    def __init__(self, latent_dim: int, noise_std: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.noise_std = noise_std
        self.register_buffer('noise_scale', torch.ones(latent_dim) * noise_std)
        
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        if training and self.training:
            # Generate correlated noise in latent space
            batch_size = x.shape[0]
            
            # Create structured noise that respects muscle correlations
            # This is a simplified version - full Lattice uses learned correlations
            noise = torch.randn_like(x) * self.noise_scale
            
            # Add temporal correlation (simple moving average)
            if hasattr(self, '_prev_noise'):
                correlation_factor = 0.7  # Temporal correlation strength
                noise = correlation_factor * self._prev_noise + (1 - correlation_factor) * noise
            
            self._prev_noise = noise.detach()
            return x + noise
        else:
            return x


class CustomMLP(BaseFeaturesExtractor):
    """
    Custom MLP feature extractor implementing the exact architecture from the paper:
    6-hidden layer [2048, 1536, 1024, 1024, 512, 512] MLP with optional Lattice exploration.
    """
    
    def __init__(self, observation_space, features_dim: int = 512, use_lattice: bool = False):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]  # 309 for motion imitation
        
        # 6-hidden layer architecture with SiLU activation
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.SiLU(),
            nn.Linear(2048, 1536),
            nn.SiLU(),
            nn.Linear(1536, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
        )
        
        # Optional Lattice exploration layer
        self.use_lattice = use_lattice
        if use_lattice:
            self.lattice_layer = LatticeNoiseLayer(512, noise_std=0.1)
        
        self._features_dim = 512
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.mlp(observations)
        
        if self.use_lattice:
            x = self.lattice_layer(x, training=self.training)
        
        return x


class MotionImitationCallback(BaseCallback):
    """Custom callback for motion imitation training metrics."""
    
    def __init__(self, log_interval=1000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.last_log_step = 0
        
        # Track motion imitation specific metrics
        self.tracking_errors = deque(maxlen=1000)
        self.position_rewards = deque(maxlen=1000)
        self.velocity_rewards = deque(maxlen=1000)
        self.energy_rewards = deque(maxlen=1000)
        self.upright_rewards = deque(maxlen=1000)
        self.early_terminations = 0
        self.total_episodes = 0
    
    def _on_step(self) -> bool:
        # Collect episode info
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.total_episodes += 1
                
            # Collect tracking metrics
            if 'max_tracking_error' in info:
                self.tracking_errors.append(info['max_tracking_error'])
            
            # Check for early termination
            if 'episode_step' in info and info['episode_step'] < 2999:  # Max episode length
                if self.locals.get('dones', [False])[0]:  # If episode ended early
                    self.early_terminations += 1
        
        # Log progress periodically
        if self.num_timesteps - self.last_log_step >= self.log_interval:
            self.last_log_step = self.num_timesteps
            
            avg_tracking_error = np.mean(self.tracking_errors) if self.tracking_errors else 0
            early_term_rate = self.early_terminations / max(self.total_episodes, 1)
            
            print(f"Training step: {self.num_timesteps:,}")
            print(f"  Avg tracking error: {avg_tracking_error:.4f}m")
            print(f"  Early termination rate: {early_term_rate:.2%}")
            print(f"  Total episodes: {self.total_episodes}")
            
            # Reset counters
            self.early_terminations = 0
            self.total_episodes = 0
        
        return True


class PlottingCallback(BaseCallback):
    """Enhanced plotting callback for motion imitation metrics."""
    
    def __init__(self, log_dir, plot_interval=50000, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.plot_interval = plot_interval
        self.last_plot_step = 0
        
        # Storage for metrics
        self.timesteps = []
        self.episode_rewards = deque(maxlen=100)
        self.mean_rewards = []
        self.tracking_errors = deque(maxlen=1000)
        self.mean_tracking_errors = []
        self.early_termination_rates = []
        self.muscle_activations = deque(maxlen=1000)
        
        # Create plots directory
        self.plots_dir = os.path.join(log_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Episode counters
        self.episodes_count = 0
        self.early_terminations = 0
        
        plt.ioff()
    
    def _on_step(self) -> bool:
        # Collect metrics
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episodes_count += 1
                
                # Check if episode ended early
                if info.get('episode_step', 0) < 2999:
                    self.early_terminations += 1
            
            if 'max_tracking_error' in info:
                self.tracking_errors.append(info['max_tracking_error'])
            
            if 'muscle_activation_mean' in info:
                self.muscle_activations.append(info['muscle_activation_mean'])
        
        # Store aggregated metrics
        if len(self.episode_rewards) > 0:
            self.timesteps.append(self.num_timesteps)
            self.mean_rewards.append(np.mean(self.episode_rewards))
            
            if len(self.tracking_errors) > 0:
                self.mean_tracking_errors.append(np.mean(list(self.tracking_errors)[-100:]))
            else:
                self.mean_tracking_errors.append(0)
            
            # Calculate early termination rate
            if self.episodes_count > 0:
                term_rate = self.early_terminations / self.episodes_count
                self.early_termination_rates.append(term_rate)
                # Reset counters
                self.episodes_count = 0
                self.early_terminations = 0
            else:
                self.early_termination_rates.append(0)
        
        # Create plots periodically
        if (self.num_timesteps - self.last_plot_step >= self.plot_interval and 
            len(self.timesteps) > 10):
            self._create_plots()
            self.last_plot_step = self.num_timesteps
        
        return True
    
    def _create_plots(self):
        """Create motion imitation specific plots."""
        try:
            fig = plt.figure(figsize=(16, 12))
            fig.suptitle(f'Motion Imitation Training Progress - Step {self.num_timesteps:,}', fontsize=16)
            
            # Plot 1: Episode Rewards
            ax1 = plt.subplot(2, 3, 1)
            if len(self.timesteps) > 0 and len(self.mean_rewards) > 0:
                ax1.plot(self.timesteps, self.mean_rewards, 'b-', linewidth=2)
            ax1.set_title('Mean Episode Reward')
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Mean Reward')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Tracking Errors
            ax2 = plt.subplot(2, 3, 2)
            if len(self.timesteps) > 0 and len(self.mean_tracking_errors) > 0:
                ax2.plot(self.timesteps[-len(self.mean_tracking_errors):], self.mean_tracking_errors, 'r-', linewidth=2)
                ax2.axhline(y=0.15, color='orange', linestyle='--', alpha=0.7, label='Termination threshold')
                ax2.legend()
            ax2.set_title('Mean Tracking Error')
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('Tracking Error (m)')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Early Termination Rate
            ax3 = plt.subplot(2, 3, 3)
            if len(self.timesteps) > 0 and len(self.early_termination_rates) > 0:
                ax3.plot(self.timesteps[-len(self.early_termination_rates):], self.early_termination_rates, 'orange', linewidth=2)
            ax3.set_title('Early Termination Rate')
            ax3.set_xlabel('Training Steps')
            ax3.set_ylabel('Termination Rate')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Recent Tracking Errors Distribution
            ax4 = plt.subplot(2, 3, 4)
            if len(self.tracking_errors) > 10:
                recent_errors = list(self.tracking_errors)[-500:]
                ax4.hist(recent_errors, bins=30, alpha=0.7, color='green')
                ax4.axvline(x=0.15, color='red', linestyle='--', alpha=0.7, label='Termination threshold')
                ax4.legend()
            ax4.set_title('Recent Tracking Error Distribution')
            ax4.set_xlabel('Tracking Error (m)')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Muscle Activations
            ax5 = plt.subplot(2, 3, 5)
            if len(self.muscle_activations) > 10:
                recent_muscles = list(self.muscle_activations)[-500:]
                ax5.plot(recent_muscles, 'purple', alpha=0.7)
                ax5.axhline(y=0.1, color='blue', linestyle='--', alpha=0.7, label='Low activation')
                ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='High activation')
                ax5.legend()
            ax5.set_title('Recent Muscle Activations')
            ax5.set_xlabel('Recent Steps')
            ax5.set_ylabel('Mean Activation')
            ax5.grid(True, alpha=0.3)
            
            # Plot 6: Training Progress Summary
            ax6 = plt.subplot(2, 3, 6)
            if len(self.timesteps) > 0:
                # Create a summary text
                summary_text = f"""Training Summary
Total Steps: {self.num_timesteps:,}
Latest Reward: {self.mean_rewards[-1]:.2f}
Latest Tracking Error: {self.mean_tracking_errors[-1]:.4f}m
Latest Termination Rate: {self.early_termination_rates[-1]:.2%}

Target Performance:
- Tracking Error < 0.15m
- Low Termination Rate
- Stable Muscle Control
"""
                ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax6.set_title('Training Summary')
            ax6.axis('off')
            
            plt.tight_layout()
            
            # Save plots
            plot_path = os.path.join(self.plots_dir, f'motion_imitation_progress_{self.num_timesteps:07d}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            
            latest_path = os.path.join(self.plots_dir, 'latest_progress.png')
            plt.savefig(latest_path, dpi=150, bbox_inches='tight')
            
            plt.close(fig)
            
            print(f"📊 Motion imitation plots saved: {plot_path}")
            
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")
            plt.close('all')


def create_env(expert_data_path="walk_IL/data/expert_data.pkl", render_mode=None):
    """Create and wrap the motion imitation environment."""
    env = MyoLegsMotionImitationEnv(expert_data_path=expert_data_path, render_mode=render_mode)
    env = Monitor(env)
    return env


def train_motion_imitation(args):
    """Main training function for motion imitation."""
    
    # Check if expert data exists
    if not os.path.exists(args.expert_data):
        print(f"❌ Expert data not found: {args.expert_data}")
        print("Please ensure expert data is available at the specified path")
        return
    
    # Set up experiment directory
    exp_name = f"motion_imitation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # if args.use_lattice:
    #     exp_name += "_lattice"
    
    log_dir = os.path.join("walk_IL/logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"🚀 Starting MyoLegs Motion Imitation Training")
    print(f"   - Expert data: {args.expert_data} (2999 frames at 500 FPS)")
    print(f"   - Total timesteps: {args.total_timesteps:,}")
    print(f"   - Number of environments: {args.n_envs}")
    print(f"   - Steps per env per update: {args.n_steps}")
    print(f"   - Total samples per update: {args.n_envs * args.n_steps:,}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Network architecture: [2048, 1536, 1024, 1024, 512, 512] + SiLU")
    print(f"   - Max gradient norm: {args.max_grad_norm}")
    print(f"   - Log directory: {log_dir}")
    
    # Initialize Weights & Biases if requested
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="myolegs-motion-imitation",
            name=exp_name,
            config={
                "algorithm": "PPO",
                "environment": "MyoLegsMotionImitation-v0",
                "total_timesteps": args.total_timesteps,
                "n_envs": args.n_envs,
                "learning_rate": args.learning_rate,
                "architecture": "[2048, 1536, 1024, 1024, 512, 512]",
                "use_lattice": args.use_lattice,
                "expert_data": args.expert_data,
            }
        )
    
    # Create vectorized environment
    print("Creating environments...")
    if args.n_envs > 1:
        env = make_vec_env(
            lambda: create_env(args.expert_data), 
            n_envs=args.n_envs, 
            vec_env_cls=SubprocVecEnv
        )
    else:
        env = make_vec_env(lambda: create_env(args.expert_data), n_envs=1)
    
    # Normalize observations (but not rewards for motion imitation)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # Create evaluation environment
    eval_env = VecNormalize(
        make_vec_env(lambda: create_env(args.expert_data), n_envs=1),
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0
    )
    
    # Define PPO model with paper's architecture
    print("Initializing PPO model with paper's architecture...")
    
    # Policy kwargs implementing the paper's 6-layer MLP: [2048, 1536, 1024, 1024, 512, 512]
    policy_kwargs = dict(
        net_arch=dict(
            pi=[2048, 1536, 1024, 1024, 512, 512],  # Policy network
            vf=[2048, 1536, 1024, 1024, 512, 512]   # Value network
        ),
        activation_fn=torch.nn.SiLU,
    )
    
    model = PPO(
        "MlpPolicy",  # Use standard MLP policy with custom feature extractor
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
    
    # Print model architecture info
    print(f"✅ Model initialized:")
    print(f"   - Policy network: 6-layer MLP [2048, 1536, 1024, 1024, 512, 512] + SiLU")
    print(f"   - Value network: 6-layer MLP [2048, 1536, 1024, 1024, 512, 512] + SiLU")
    print(f"   - Activation function: SiLU")
    print(f"   - Total parameters: ~{sum(p.numel() for p in model.policy.parameters()):,}")
    
    # Set up callbacks
    callbacks = []
    
    # Motion imitation callback
    motion_callback = MotionImitationCallback(log_interval=args.log_interval, verbose=1)
    callbacks.append(motion_callback)
    
    # Plotting callback
    if not args.no_plots:
        plotting_callback = PlottingCallback(log_dir, plot_interval=args.plot_interval, verbose=1)
        callbacks.append(plotting_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=log_dir,
        name_prefix="motion_imitation",
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
        n_eval_episodes=3,  # Fewer episodes since they can be long
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # Wandb callback
    if args.use_wandb and WANDB_AVAILABLE:
        wandb_callback = WandbCallback(
            gradient_save_freq=args.save_freq,
            model_save_path=log_dir,
            verbose=2,
        )
        callbacks.append(wandb_callback)
    
    callback_list = CallbackList(callbacks)
    
    # Start training
    print("🏃 Starting motion imitation training...")
    print(f"📊 Observation space: {env.observation_space.shape} (309D)")
    print(f"   - Pelvis state: 8D, Body kinematics: 234D (16 bodies), Contact: 4D, Target: 63D")
    print(f"   - Body kinematics breakdown: Position 45D + Rotation 96D + Lin.Vel 48D + Ang.Vel 48D")
    print(f"🎮 Action space: {env.action_space.shape} (80D muscle lengths)")
    print(f"🎯 Target: Track expert motion with <0.15m keypoint error")
    print(f"⏱️  Expert data: 2999 frames @ 500 FPS = ~6 seconds walking cycle")
    
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
        
        print(f"✅ Motion imitation training completed!")
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
        
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train MyoLegs motion imitation with paper's framework")
    
    # Data parameters
    parser.add_argument("--expert_data", type=str, default="walk_IL/data/expert_data.pkl",
                        help="Path to expert motion capture data (2999 frames @ 500 FPS)")
    
    # Training parameters from config file
    parser.add_argument("--total_timesteps", type=int, default=100_000_000,
                        help="Total training timesteps (max_epoch)")
    parser.add_argument("--n_envs", type=int, default=64,
                        help="Number of parallel environments")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate (policy_lr)")
    parser.add_argument("--n_steps", type=int, default=800,
                        help="Number of steps per environment per update (min_batch_size/n_envs)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Minibatch size")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of epochs per update (opt_num_epochs)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE lambda (tau)")
    parser.add_argument("--clip_range", type=float, default=0.2,
                        help="PPO clipping range (clip_epsilon)")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="Value function coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=25.0,
                        help="Max gradient norm (policy_grad_clip)")
    
    # Architecture and exploration
    parser.add_argument("--use_lattice", action="store_true",
                        help="Use Lattice exploration method for correlated muscle control")
    
    # Logging and evaluation
    parser.add_argument("--log_interval", type=int, default=5000,
                        help="Log interval")
    parser.add_argument("--plot_interval", type=int, default=100000,
                        help="Plot creation interval")
    parser.add_argument("--save_freq", type=int, default=250000,
                        help="Save frequency")
    parser.add_argument("--eval_freq", type=int, default=100000,
                        help="Evaluation frequency")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--no_plots", action="store_true",
                        help="Disable plotting")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cpu, cuda, auto)")
    
    args = parser.parse_args()
    
    # Validate expert data exists
    if not os.path.exists(args.expert_data):
        print(f"❌ Expert data file not found: {args.expert_data}")
        print(f"   Please ensure the expert data is available")
        return
    
    train_motion_imitation(args)


if __name__ == "__main__":
    main()