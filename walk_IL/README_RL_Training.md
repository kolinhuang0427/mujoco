# MyoLegs Standing RL Training

Train the MyoLegs humanoid model to stand upright using reinforcement learning with muscle actuators.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test the Environment

```bash
python myolegs_standing_env.py
```

This will run a basic test with random muscle activations to verify the environment works.

### 3. Start Training

```bash
python train_myolegs_standing.py
```

For faster training with multiple CPU cores:

```bash
python train_myolegs_standing.py --n_envs 16 --total_timesteps 5000000
```

### 4. Monitor Training (Optional)

To track training with Weights & Biases:

```bash
python train_myolegs_standing.py --use_wandb
```

## 📊 Environment Details

### Action Space
- **Type**: Continuous
- **Shape**: (80,) - One activation level per muscle
- **Range**: [0, 1] - Muscle activation from 0% to 100%

### Observation Space  
- **Shape**: (123,)
- **Components**:
  - Joint positions (14): Hip, knee, ankle angles for both legs
  - Joint velocities (14): Angular velocities
  - Pelvis pose (7): 3D position + quaternion orientation  
  - Pelvis velocity (6): Linear + angular velocity
  - Foot contacts (2): Binary contact detection
  - Muscle activations (80): Current muscle states

### Reward Function
The reward encourages:
1. **Upright posture** (height and orientation)
2. **Stability** (minimal movement)
3. **Natural pose** (slight knee bend, neutral hips)
4. **Foot contact** (both feet on ground)
5. **Muscle coordination** (efficient, symmetric activation)

## 🎯 Training Options

### Basic Training
```bash
python train_myolegs_standing.py \
    --total_timesteps 2000000 \
    --n_envs 8
```

### Advanced Training
```bash
python train_myolegs_standing.py \
    --total_timesteps 5000000 \
    --n_envs 16 \
    --learning_rate 1e-4 \
    --batch_size 128 \
    --use_wandb
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--total_timesteps` | 2,000,000 | Total training steps |
| `--n_envs` | 8 | Parallel environments |
| `--learning_rate` | 3e-4 | PPO learning rate |
| `--n_steps` | 2048 | Steps per environment per update |
| `--batch_size` | 64 | Minibatch size |
| `--n_epochs` | 10 | Training epochs per update |

## 📈 Evaluation

Evaluate a trained model:

```bash
python train_myolegs_standing.py \
    --mode evaluate \
    --model_path logs/myolegs_standing_YYYYMMDD_HHMMSS/final_model \
    --env_normalize_path logs/myolegs_standing_YYYYMMDD_HHMMSS/vec_normalize.pkl
```

## 🔧 Muscle Groups

The MyoLegs model has 80 muscles (40 per leg):

### Per Leg (40 muscles):
- **Hip**: Gluteus (max/med/min), Psoas, Iliacus, Piriformis
- **Thigh**: Quadriceps (rectus femoris, vastus), Hamstrings (biceps femoris, semimembranosus, semitendinosus), Adductors, Sartorius, TFL
- **Calf**: Gastrocnemius, Soleus, Tibialis anterior/posterior
- **Foot**: Extensors (EDL, EHL), Flexors (FDL, FHL), Peroneals

## 📋 Training Tips

### Performance Optimization
- Use `--n_envs 16` or more for faster training on multi-core systems
- Enable GPU acceleration: `--device cuda` (if available)
- Monitor with `--use_wandb` for detailed metrics

### Hyperparameter Tuning
- **Lower learning rate** (1e-4) for more stable learning
- **Larger batch size** (128-256) for smoother gradients  
- **Higher entropy coefficient** (0.02) for more exploration

### Common Issues
- **Model falls immediately**: Lower learning rate, increase training time
- **Unstable training**: Use observation normalization (enabled by default)
- **Poor muscle coordination**: Adjust muscle coordination rewards in environment

## 📁 File Structure

```
├── myolegs_standing_env.py     # Gymnasium environment
├── train_myolegs_standing.py   # Training script
├── requirements.txt            # Dependencies
├── model/myolegs/             # MyoLegs model files
│   ├── myolegs.xml            # Main model file
│   └── myolegs_assets.xml     # Muscle definitions
└── logs/                      # Training logs and models
    └── myolegs_standing_*/    # Experiment folders
```

## 🎮 Expected Results

After successful training:
- **Episode length**: 800-1000 steps (stable standing)
- **Reward**: 6-8 (good posture and stability)  
- **Behavior**: Model maintains upright stance with natural muscle coordination

Training typically requires 2-5 million timesteps depending on hardware and hyperparameters.

## 🔬 Research Applications

This environment can be extended for:
- **Bipedal locomotion**: Modify rewards for walking/running
- **Muscle pathology**: Simulate muscle weakness or injury
- **Prosthetics control**: Test artificial muscle control strategies
- **Human biomechanics**: Study muscle activation patterns

## 📚 References

- [MyoSuite](https://sites.google.com/view/myosuite): Musculoskeletal simulation
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/): RL algorithms
- [MuJoCo](https://mujoco.org/): Physics simulation 