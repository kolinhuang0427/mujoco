#!/usr/bin/env python3
"""
Train MyoLegs Walking Imitation Learning
Uses PPO to train the MyoLegs model to walk by imitating expert motion capture data.
"""

import os
import argparse
from datetime import datetime
import numpy as np
import torch

# Set matplotlib backend before importing pyplot (CRITICAL FIX)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Weights & Biases not available. Install with: pip install wandb")

from myolegs_walking_imitation_env import MyoLegsWalkingImitationEnv


class ProgressCallback(BaseCallback):
    """Custom callback to print training progress."""
    
    def __init__(self, log_interval=1000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.last_log_step = 0
    
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_log_step >= self.log_interval:
            self.last_log_step = self.num_timesteps
            print(f"Training step: {self.num_timesteps:,}")
        return True


class PlottingCallback(BaseCallback):
    """Callback to create reward and clipping fraction plots during training."""
    
    def __init__(self, log_dir, plot_interval=25000, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.plot_interval = plot_interval
        self.last_plot_step = 0
        
        # Storage for metrics
        self.timesteps = []
        self.episode_rewards = deque(maxlen=100)  # Keep last 100 episodes
        self.mean_rewards = []
        self.clipping_fractions = []
        self.tracking_errors = []
        self.forward_velocities = []
        self.muscle_activations = []
        self.stage_changes = []  # Store timesteps when stages changed
        
        # Create plots directory
        self.plots_dir = os.path.join(log_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set matplotlib to non-interactive mode
        plt.ioff()  # Turn off interactive mode
    
    def _on_step(self) -> bool:
        # Collect episode rewards from info
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
            # Collect other metrics if available
            if 'tracking_error' in info:
                self.tracking_errors.append(info['tracking_error'])
            if 'forward_velocity' in info:
                self.forward_velocities.append(info['forward_velocity'])
            if 'muscle_activation_mean' in info:
                self.muscle_activations.append(info['muscle_activation_mean'])
            # Track stage changes
            if 'stage_changed' in info and info['stage_changed']:
                self.stage_changes.append(self.num_timesteps)
                print(f"📊 Stage change detected at step {self.num_timesteps}")
        
        # Store current metrics
        if len(self.episode_rewards) > 0:
            self.timesteps.append(self.num_timesteps)
            self.mean_rewards.append(np.mean(self.episode_rewards))
            
            # Get clipping fraction from logger
            if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
                clip_frac = self.model.logger.name_to_value.get('train/clip_fraction', 0)
                self.clipping_fractions.append(clip_frac)
            else:
                self.clipping_fractions.append(0)
        
        # Create plots periodically
        if (self.num_timesteps - self.last_plot_step >= self.plot_interval and 
            len(self.timesteps) > 10):
            self._create_plots()
            self.last_plot_step = self.num_timesteps
        
        return True
    
    def _create_plots(self):
        """Create and save training plots."""
        try:
            # Create figure with 2x3 layout for 5 subplots
            fig = plt.figure(figsize=(18, 12))
            fig.suptitle(f'Training Progress - Step {self.num_timesteps:,}', fontsize=16)
            
            # Plot 1: Mean Episode Reward
            ax1 = plt.subplot(2, 3, 1)
            if len(self.timesteps) > 0 and len(self.mean_rewards) > 0:
                ax1.plot(self.timesteps, self.mean_rewards, 'b-', linewidth=2)
                # Add stage change lines
                for stage_step in self.stage_changes:
                    if stage_step <= max(self.timesteps):
                        ax1.axvline(x=stage_step, color='red', linestyle='--', alpha=0.7)
            ax1.set_title('Mean Episode Reward (Last 100 Episodes)')
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Mean Reward')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Clipping Fraction
            ax2 = plt.subplot(2, 3, 2)
            if len(self.clipping_fractions) > 0 and len(self.timesteps) > 0:
                steps_for_clip = self.timesteps[-len(self.clipping_fractions):]
                ax2.plot(steps_for_clip, self.clipping_fractions, 'r-', linewidth=2)
                ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Target: ~0.1')
                # Add stage change lines
                for stage_step in self.stage_changes:
                    if stage_step <= max(steps_for_clip):
                        ax2.axvline(x=stage_step, color='red', linestyle='--', alpha=0.7)
                ax2.legend()
            ax2.set_title('PPO Clipping Fraction')
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('Clipping Fraction')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Tracking Error (if available)
            ax3 = plt.subplot(2, 3, 3)
            if len(self.tracking_errors) > 10:
                recent_errors = self.tracking_errors[-1000:]  # Last 1000 steps
                ax3.plot(recent_errors, 'g-', alpha=0.7)
                ax3.set_title('Recent Tracking Error')
                ax3.set_xlabel('Recent Steps')
                ax3.set_ylabel('Joint Tracking Error')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Tracking Error\nData Collecting...', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Tracking Error')
            
            # Plot 4: Forward Velocity (if available)
            ax4 = plt.subplot(2, 3, 4)
            if len(self.forward_velocities) > 10:
                recent_vels = self.forward_velocities[-1000:]  # Last 1000 steps
                ax4.plot(recent_vels, 'm-', alpha=0.7)
                ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Target: 1.0 m/s')
                ax4.legend()
                ax4.set_title('Recent Forward Velocity')
                ax4.set_xlabel('Recent Steps')
                ax4.set_ylabel('Forward Velocity (m/s)')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Forward Velocity\nData Collecting...', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Forward Velocity')
            
            # Plot 5: Muscle Activations (NEW)
            ax5 = plt.subplot(2, 3, 5)
            if len(self.muscle_activations) > 10:
                recent_muscles = self.muscle_activations[-1000:]  # Last 1000 steps
                ax5.plot(recent_muscles, 'c-', alpha=0.7)
                ax5.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Target: ~0.3')
                ax5.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Low: 0.1')
                ax5.legend()
                ax5.set_title('Recent Muscle Activation')
                ax5.set_xlabel('Recent Steps')
                ax5.set_ylabel('Mean Muscle Activation')
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, 'Muscle Activation\nData Collecting...', 
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Muscle Activation')
            
            # Plot 6: Stage Progress (NEW)
            ax6 = plt.subplot(2, 3, 6)
            if len(self.stage_changes) > 0:
                # Create a simple timeline of stage changes
                stages = ['Standing', 'Walking', 'Motion Tracking', 'Refinement']
                y_pos = [0, 1, 2, 3]
                
                # Plot stage transitions
                prev_step = 0
                for i, stage_step in enumerate(self.stage_changes):
                    if i < len(stages) - 1:
                        ax6.barh(y_pos[i], stage_step - prev_step, left=prev_step, 
                                alpha=0.6, label=f'Stage {i}: {stages[i]}')
                        prev_step = stage_step
                
                # Add current stage
                current_stage = min(len(self.stage_changes), 3)
                if current_stage < len(stages):
                    ax6.barh(y_pos[current_stage], self.num_timesteps - prev_step, 
                            left=prev_step, alpha=0.6, 
                            label=f'Stage {current_stage}: {stages[current_stage]}')
                
                ax6.set_yticks(y_pos)
                ax6.set_yticklabels(stages)
                ax6.set_xlabel('Training Steps')
                ax6.set_title('Stage Progress')
                ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax6.text(0.5, 0.5, 'Stage 0: Standing\nWaiting for transitions...', 
                        ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Stage Progress')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.plots_dir, f'training_progress_{self.num_timesteps:07d}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            
            # Also save latest plot with fixed name
            latest_path = os.path.join(self.plots_dir, 'latest_progress.png')
            plt.savefig(latest_path, dpi=150, bbox_inches='tight')
            
            plt.close(fig)  # Important: close the figure to free memory
            
            print(f"📊 Plots saved: {plot_path}")
            
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")
            # Close any open figures to prevent memory leaks
            plt.close('all')
    
    def _on_training_end(self):
        """Create final plots when training ends."""
        if len(self.timesteps) > 0:
            self._create_plots()
            print(f"📊 Final plots saved in: {self.plots_dir}")


class StageModelSavingCallback(BaseCallback):
    """Callback to save models when stages change."""
    
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.saved_stages = set()
    
    def _on_step(self) -> bool:
        # Check for stage changes in any environment
        infos = self.locals.get('infos', [])
        for i, info in enumerate(infos):
            # More detailed debug output
            current_stage = info.get('stage', 'unknown')
            stage_changed = info.get('stage_changed', False)
            
            if 'stage' in info and current_stage not in [0, 1, 2, 3]:
                print(f"⚠️ Unexpected stage value: {current_stage}")
            
            if stage_changed:
                print(f"🔍 Callback detected stage_changed=True in env {i}, new stage={current_stage}")
                print(f"🔍 Currently saved stages: {self.saved_stages}")
                
                # Save model for the COMPLETED stage (previous stage)
                completed_stage = current_stage - 1
                
                if completed_stage >= 0 and completed_stage not in self.saved_stages:
                    # Save model for the completed stage
                    stage_model_path = os.path.join(self.save_path, f"stage_{completed_stage}_model")
                    try:
                        self.model.save(stage_model_path)
                        self.saved_stages.add(completed_stage)
                        print(f"💾 Stage {completed_stage} COMPLETED - model saved: {stage_model_path}")
                    except Exception as e:
                        print(f"❌ Failed to save completed stage {completed_stage} model: {e}")
                elif completed_stage >= 0:
                    print(f"⚠️ Completed stage {completed_stage} already saved, skipping")
        
        return True

    def _on_training_end(self) -> bool:
        """Save the final stage model when training ends."""
        # Get the current stage from any environment
        if hasattr(self, 'training_env') and hasattr(self.training_env, 'get_attr'):
            try:
                current_stages = self.training_env.get_attr('stage')
                if current_stages:
                    final_stage = current_stages[0]  # Get stage from first environment
                    if final_stage not in self.saved_stages:
                        stage_model_path = os.path.join(self.save_path, f"stage_{final_stage}_model")
                        try:
                            self.model.save(stage_model_path)
                            self.saved_stages.add(final_stage)
                            print(f"💾 Final stage {final_stage} model saved at training end: {stage_model_path}")
                        except Exception as e:
                            print(f"❌ Failed to save final stage {final_stage} model: {e}")
            except Exception as e:
                print(f"⚠️ Could not retrieve final stage for saving: {e}")
        return True


def create_env(expert_data_path="walk_IL/data/expert_data.pkl", render_mode=None):
    """Create and wrap the environment."""
    env = MyoLegsWalkingImitationEnv(expert_data_path=expert_data_path, render_mode=render_mode)
    env = Monitor(env)
    return env


def train_walking_imitation(args):
    """Main training function."""
    
    # Check if expert data exists
    if not os.path.exists(args.expert_data):
        print(f"❌ Expert data not found: {args.expert_data}")
        print("Please run convert_theia_data_labels.py first to generate expert data")
        return
    
    # Set up experiment directory
    exp_name = f"myolegs_walking_imitation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join("walk_IL/logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"🚀 Starting MyoLegs Walking Imitation Learning")
    print(f"   - Expert data: {args.expert_data}")
    print(f"   - Total timesteps: {args.total_timesteps:,}")
    print(f"   - Number of environments: {args.n_envs}")
    print(f"   - Log directory: {log_dir}")
    
    # Initialize Weights & Biases if requested
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="myolegs-walking-imitation",
            name=exp_name,
            config={
                "algorithm": "PPO",
                "environment": "MyoLegsWalkingImitation-v0",
                "total_timesteps": args.total_timesteps,
                "n_envs": args.n_envs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "n_epochs": args.n_epochs,
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
    
    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create evaluation environment
    eval_env = VecNormalize(
        make_vec_env(lambda: create_env(args.expert_data), n_envs=1),
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0
    )
    
    # Define PPO model with MODERATE parameters for 80D muscle control
    print("Initializing PPO model...")
    
    policy_kwargs = dict(
        net_arch=dict(
            pi=[2048, 1536, 1024, 1024, 512, 512],  # Large 6-layer network for complex muscle patterns
            vf=[2048, 1536, 1024, 1024, 512, 512]   # Matching capacity for value function
        ),
        activation_fn=torch.nn.Tanh,
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2e-4,          # Moderate learning rate
        n_steps=2048,                # Keep larger steps for exploration
        batch_size=64,               # Smaller batches for more frequent updates
        n_epochs=8,                  # Moderate epochs
        gamma=0.99,                  # Standard discount
        gae_lambda=0.95,             # Standard GAE
        clip_range=0.15,             # Moderate clipping - allows exploration but not chaos
        ent_coef=0.005,              # Moderate entropy - encourage exploration initially
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        device=args.device
    )
    
    # Set up callbacks
    callbacks = []
    
    # Progress callback
    progress_callback = ProgressCallback(log_interval=args.log_interval, verbose=1)
    callbacks.append(progress_callback)
    
    # Stage model saving callback
    stage_callback = StageModelSavingCallback(save_path=log_dir, verbose=1)
    callbacks.append(stage_callback)
    
    # Plotting callback (only if not disabled)
    if not args.no_plots:
        plotting_callback = PlottingCallback(log_dir, plot_interval=args.plot_interval, verbose=1)
        callbacks.append(plotting_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=log_dir,
        name_prefix="walking_imitation",
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
        n_eval_episodes=5,  # Fewer episodes for faster evaluation
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
    print("🏃 Starting training...")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Observation space shape: {env.observation_space.shape}")
    print("💪 Using action smoothing for realistic muscle control")
    print("🎯 Moderate policy updates for 80D muscle exploration")
    if not args.no_plots:
        print(f"📊 Plots will be saved to: {os.path.join(log_dir, 'plots')}")
    
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
        
        print(f"✅ Training completed!")
        print(f"   - Final model saved: {final_model_path}")
        print(f"   - Logs saved: {log_dir}")
        if not args.no_plots:
            print(f"   - Plots saved: {os.path.join(log_dir, 'plots')}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        # Save interrupted model
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
    parser = argparse.ArgumentParser(description="Train MyoLegs walking imitation learning")
    
    # Data parameters
    parser.add_argument("--expert_data", type=str, default="walk_IL/data/expert_data.pkl",
                        help="Path to expert motion capture data")
    
    # Training parameters with moderate defaults for 80D muscle control
    parser.add_argument("--total_timesteps", type=int, default=100_000_000,
                        help="Total training timesteps")
    parser.add_argument("--n_envs", type=int, default=32,
                        help="Number of parallel environments")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Number of steps per environment per update")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Minibatch size")
    parser.add_argument("--n_epochs", type=int, default=8,
                        help="Number of epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip_range", type=float, default=0.15,
                        help="PPO clipping range")
    parser.add_argument("--ent_coef", type=float, default=0.005,
                        help="Entropy coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="Value function coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="Max gradient norm")
    
    # Logging and evaluation
    parser.add_argument("--log_interval", type=int, default=10000,
                        help="Log interval")
    parser.add_argument("--plot_interval", type=int, default=250000,
                        help="Plot creation interval")
    parser.add_argument("--save_freq", type=int, default=100000,
                        help="Save frequency")
    parser.add_argument("--eval_freq", type=int, default=50000,
                        help="Evaluation frequency")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--no_plots", action="store_true",
                        help="Disable plotting (useful for headless systems)")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cpu, cuda, auto)")
    
    args = parser.parse_args()
    
    # Validate expert data exists
    if not os.path.exists(args.expert_data):
        print(f"❌ Expert data file not found: {args.expert_data}")
        print(f"   Please run: python walk_IL/data/convert_theia_data_labels.py")
        return
    
    train_walking_imitation(args)


if __name__ == "__main__":
    main() 