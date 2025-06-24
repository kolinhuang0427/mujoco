#!/usr/bin/env python3
"""
Parameter Sweep for MyoLegs Walking Imitation Learning
Systematically test different hyperparameter configurations and compare results.
"""

import os
import yaml
import json
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class ExperimentRunner:
    """Run and manage multiple training experiments with different parameters."""
    
    def __init__(self, config_file="walk_IL/experiment_configs.yaml"):
        self.config_file = config_file
        self.config = self._load_config()
        self.results_dir = Path("walk_IL/experiment_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Create sweep-specific directory
        self.sweep_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.sweep_dir = self.results_dir / f"sweep_{self.sweep_id}"
        self.sweep_dir.mkdir(exist_ok=True)
        
        print(f"🧪 Experiment sweep ID: {self.sweep_id}")
        print(f"📁 Results will be saved to: {self.sweep_dir}")
    
    def _load_config(self):
        """Load experiment configurations from YAML file."""
        with open(self.config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def run_experiment(self, exp_name, exp_config, dry_run=False):
        """Run a single experiment with given configuration."""
        print(f"\n🚀 Starting experiment: {exp_name}")
        print(f"📝 Description: {exp_config['description']}")
        
        # Merge with defaults
        config = {**self.config['defaults'], **exp_config}
        
        # Create experiment directory
        exp_dir = self.sweep_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        # Save configuration
        config_path = exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        if dry_run:
            print(f"🔍 DRY RUN - Would run with config: {config}")
            return {"status": "dry_run", "config": config}
        
        # Build command
        cmd = [
            "python", "walk_IL/train_walking_imitation.py",
            "--total_timesteps", str(config['total_timesteps']),
            "--n_envs", str(config['n_envs']),
            "--learning_rate", str(config['learning_rate']),
            "--batch_size", str(config['batch_size']),
            "--n_epochs", str(config['n_epochs']),
            "--gamma", str(config['gamma']),
            "--gae_lambda", str(config['gae_lambda']),
            "--clip_range", str(config['clip_range']),
            "--ent_coef", str(config['ent_coef']),
            "--vf_coef", str(config['vf_coef']),
            "--max_grad_norm", str(config['max_grad_norm']),
            "--plot_interval", str(config['plot_interval']),
            "--eval_freq", str(config['eval_freq']),
            "--save_freq", str(config['save_freq']),
            "--device", config['device']
        ]
        
        # Set environment variables for custom config
        env = os.environ.copy()
        env['EXPERIMENT_NAME'] = exp_name
        env['EXPERIMENT_CONFIG'] = str(config_path)
        
        # Run training
        start_time = time.time()
        try:
            print(f"⏰ Starting training at {datetime.now().strftime('%H:%M:%S')}")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)  # No timeout - let training run as long as needed
            end_time = time.time()
            
            # Save results
            result_data = {
                "experiment_name": exp_name,
                "config": config,
                "start_time": start_time,
                "end_time": end_time,
                "duration_minutes": (end_time - start_time) / 60,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "status": "completed" if result.returncode == 0 else "failed"
            }
            
            # Save result
            result_path = exp_dir / "result.json"
            with open(result_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            if result.returncode == 0:
                print(f"✅ Experiment {exp_name} completed successfully")
                print(f"⏱️  Duration: {result_data['duration_minutes']:.1f} minutes")
            else:
                print(f"❌ Experiment {exp_name} failed")
                print(f"Error: {result.stderr}")
            
            return result_data
            
        except Exception as e:
            print(f"💥 Experiment {exp_name} crashed: {e}")
            return {"status": "crashed", "config": config, "error": str(e)}
    
    def run_sweep(self, experiments=None, dry_run=False):
        """Run multiple experiments in sequence."""
        if experiments is None:
            experiments = list(self.config['experiments'].keys())
        
        print(f"🧪 Running parameter sweep with {len(experiments)} experiments")
        print(f"Experiments: {', '.join(experiments)}")
        
        results = []
        
        for i, exp_name in enumerate(experiments, 1):
            if exp_name not in self.config['experiments']:
                print(f"⚠️  Skipping unknown experiment: {exp_name}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Experiment {i}/{len(experiments)}: {exp_name}")
            print(f"{'='*60}")
            
            exp_config = self.config['experiments'][exp_name]
            result = self.run_experiment(exp_name, exp_config, dry_run)
            results.append(result)
            
            if not dry_run and result.get('status') == 'completed':
                # Give system a moment to rest between experiments
                time.sleep(10)
        
        # Save sweep summary
        sweep_summary = {
            "sweep_id": self.sweep_id,
            "experiments": experiments,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        summary_path = self.sweep_dir / "sweep_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(sweep_summary, f, indent=2)
        
        print(f"\n🎯 Sweep completed! Summary saved to: {summary_path}")
        
        if not dry_run:
            self._create_comparison_plots()
        
        return results
    
    def _create_comparison_plots(self):
        """Create plots comparing all experiments in the sweep."""
        print(f"\n📊 Creating comparison plots...")
        
        # Collect data from all experiments
        experiment_data = []
        
        for exp_dir in self.sweep_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            result_file = exp_dir / "result.json"
            if not result_file.exists():
                continue
            
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            if result.get('status') != 'completed':
                continue
            
            # Look for logs directory (training creates logs/experiment_name_timestamp/)
            logs_pattern = f"logs/*{exp_dir.name}*"
            import glob
            log_dirs = glob.glob(logs_pattern)
            
            if log_dirs:
                # Get the most recent log directory for this experiment
                latest_log_dir = max(log_dirs, key=lambda x: os.path.getctime(x))
                
                experiment_data.append({
                    'name': exp_dir.name,
                    'config': result['config'],
                    'duration': result['duration_minutes'],
                    'log_dir': latest_log_dir
                })
        
        if len(experiment_data) < 2:
            print("⚠️  Not enough completed experiments for comparison plots")
            return
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Experiment Comparison - Sweep {self.sweep_id}', fontsize=16)
        
        # Plot 1: Training duration
        names = [exp['name'] for exp in experiment_data]
        durations = [exp['duration'] for exp in experiment_data]
        
        axes[0, 0].bar(names, durations)
        axes[0, 0].set_title('Training Duration')
        axes[0, 0].set_ylabel('Minutes')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Learning rate vs Clip range
        lr_values = [exp['config']['learning_rate'] for exp in experiment_data]
        clip_values = [exp['config']['clip_range'] for exp in experiment_data]
        
        axes[0, 1].scatter(lr_values, clip_values, s=100)
        for i, name in enumerate(names):
            axes[0, 1].annotate(name, (lr_values[i], clip_values[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0, 1].set_xlabel('Learning Rate')
        axes[0, 1].set_ylabel('Clip Range')
        axes[0, 1].set_title('Learning Rate vs Clip Range')
        
        # Plot 3: Entropy coefficient comparison
        ent_values = [exp['config']['ent_coef'] for exp in experiment_data]
        axes[1, 0].bar(names, ent_values)
        axes[1, 0].set_title('Entropy Coefficient')
        axes[1, 0].set_ylabel('Entropy Coef')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Batch size vs Epochs
        batch_values = [exp['config']['batch_size'] for exp in experiment_data]
        epoch_values = [exp['config']['n_epochs'] for exp in experiment_data]
        
        axes[1, 1].scatter(batch_values, epoch_values, s=100)
        for i, name in enumerate(names):
            axes[1, 1].annotate(name, (batch_values[i], epoch_values[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 1].set_xlabel('Batch Size')
        axes[1, 1].set_ylabel('N Epochs')
        axes[1, 1].set_title('Batch Size vs N Epochs')
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_path = self.sweep_dir / "experiment_comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comparison plots saved to: {comparison_path}")
        
        # Create summary table
        summary_df = pd.DataFrame([
            {
                'Experiment': exp['name'],
                'Learning Rate': exp['config']['learning_rate'],
                'Clip Range': exp['config']['clip_range'],
                'Entropy Coef': exp['config']['ent_coef'],
                'Batch Size': exp['config']['batch_size'],
                'N Epochs': exp['config']['n_epochs'],
                'Duration (min)': f"{exp['duration']:.1f}",
                'Description': exp['config']['description']
            }
            for exp in experiment_data
        ])
        
        summary_csv_path = self.sweep_dir / "experiment_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        
        print(f"📋 Summary table saved to: {summary_csv_path}")
        print("\nExperiment Summary:")
        print(summary_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Run parameter sweep for walking imitation learning")
    parser.add_argument("--experiments", nargs="+", 
                       help="Specific experiments to run (default: all)")
    parser.add_argument("--config", type=str, default="walk_IL/experiment_configs.yaml",
                       help="Path to experiment configuration file")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be run without actually training")
    parser.add_argument("--list", action="store_true",
                       help="List available experiments")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.config)
    
    if args.list:
        print("Available experiments:")
        for name, config in runner.config['experiments'].items():
            print(f"  {name}: {config['description']}")
        return
    
    experiments = args.experiments or list(runner.config['experiments'].keys())
    runner.run_sweep(experiments, args.dry_run)


if __name__ == "__main__":
    main() 