# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESCP (Environment-Specific Contextual Policy) - A meta-reinforcement learning framework for adapting to sudden environment changes using context-sensitive policies with RMDM (Recurrent Model-based Dynamics Model).

Paper: [Adapt to Environment Sudden Changes by Learning a Context Sensitive Policy](https://www.aaai.org/AAAI22Papers/AAAI-6573.LuoF.pdf)

## Core Architecture

### Entry Point
- `main.py` - Initializes Ray, creates SAC instance, and starts training
- `algorithms/sac.py` - Main SAC algorithm implementation with RMDM integration

### Key Components

**Algorithms** (`algorithms/`)
- `sac.py` - Soft Actor-Critic with meta-learning support
- `RMDM.py` - Recurrent Model-based Dynamics Model for environment representation
- `contrastive.py` - Contrastive learning loss for representation learning

**Models** (`models/`)
- `policy.py` - Actor network with environment parameter conditioning
- `value.py` - Critic networks (Q-functions)
- `rnn_base.py` - RNN-based models for temporal dependencies
- `mlp_base.py` - MLP baseline models
- `transition.py` - Dynamics model

**Environments** (`envs/`)
- `nonstationary_env.py` - Wrapper for environments with changing physical parameters (gravity, mass, damping)
- `delayed_env.py` - Wrapper for observation delay simulation
- `nonstationary_delayed_env.py` - Combined non-stationary + delayed environment
- `delayed_distribution.py` - Delay distributions (Gamma, Uniform, DoubleGaussian)
- `env_wrapper.py` - Environment decoration factory
- `grid_world.py`, `grid_world_general.py` - Custom grid world environments

**Agent** (`agent/`)
- `Agent.py` - Contains `EnvRemoteArray` for parallel environment execution using Ray

**Utils** (`utils/`)
- `replay_memory.py` - Experience replay buffer
- `history_construct.py` - Constructs history sequences for RNN input
- `torch_utils.py` - PyTorch utilities
- `zfilter.py` - Running mean/std normalization

**Logging** (`log_util/`)
- `logger.py` - TensorBoard logging and experiment tracking
- `logger_base.py` - Base logger class

**Parameters** (`parameter/`)
- `Parameter.py` - All hyperparameters and configuration
- `private_config.py` - Private configuration (credentials, paths)

## Common Commands

### Training

**Basic training (non-stationary environment):**
```bash
python main.py --env_name HalfCheetah-v2 --rnn_fix_length 16 --seed 5 \
  --task_num 40 --max_iter_num 2000 \
  --varying_params dof_damping_1_dim \
  --test_task_num 40 --ep_dim 2 --name_suffix RMDM \
  --rbf_radius 3000 --use_rmdm --stop_pg_for_ep --bottle_neck
```

**Training with delayed environment (recommended script):**
```bash
bash train.sh
```

The `train.sh` script includes:
- Automatic GPU selection (avoids GPU 0)
- Delayed environment configuration
- Optimized update intervals
- Action history and state history configuration

**Key training parameters:**
- `--env_name`: Environment (HalfCheetah-v4, Walker2d-v4, Ant-v4, Hopper-v4, Humanoid-v4)
- `--rnn_fix_length`: RNN sequence length (TBPTT length)
- `--ep_state_history_length`: Number of recent states for EP input (H)
- `--ep_action_window_length`: Action window length per state (T)
- `--action_history_length`: Legacy parameter (same as ep_action_window_length)
- `--min_batch_size`: Steps per iteration
- `--update_interval`: Policy update frequency
- `--delay_changing_period`: Delay distribution change period (steps)
- `--delay_changing_interval`: Delay check interval (steps)
- `--delay_task_num`: Number of delay tasks

### Docker Training

```bash
docker pull sanluosizhou/selfdl:ml

docker run --rm -it --shm-size 50gb --gpus all \
  -v $PWD:/root/policy_adaptation sanluosizhou/selfdl:ml \
  -c "cd /root/policy_adaptation && python main.py [args]"
```

### Analysis

**Analyze training logs:**
```bash
python analyze_training_log.py --log log_file/[experiment_name]/log.txt
```

Generates 4 plots:
- `performance_metrics.png` - Rewards, Q-values
- `loss_metrics.png` - Loss curves
- `timing_metrics.png` - Training speed
- `statistics.png` - Training statistics

**Monitor training in real-time:**
```bash
# Use tmux (recommended)
tmux new -s training
python main.py [args]
# Detach: Ctrl+B then D
# Reattach: tmux attach -t training

# Or redirect to file
python main.py [args] 2>&1 | tee training.log
tail -f training.log
```

### Environment Setup

**Install dependencies:**
```bash
pip install -r requirement.txt
```

**Virtual environment (if .venv exists):**
```bash
source .venv/bin/activate  # Linux/Mac
.venv/Scripts/activate     # Windows
```

**MuJoCo setup:**
Follow instructions at https://github.com/openai/mujoco-py

## Architecture Details

### Meta-Learning Flow

1. **Environment Sampling**: Sample multiple tasks with different physical/delay parameters
2. **Context Encoding**: RMDM encodes environment dynamics into context vector
3. **Policy Conditioning**: Policy network takes (state, context) as input
4. **Training**: SAC updates policy/value networks across multiple tasks
5. **Adaptation**: At test time, policy adapts to new environments using learned context

### RNN Input Construction

The system uses a sliding window approach for RNN inputs:

- **State History (H)**: Recent H states
- **Action Window (T)**: For each state, the most recent T actions
- **Total RNN Input**: H × (state_dim + T × action_dim)

This is configured via:
- `--ep_state_history_length` (H)
- `--ep_action_window_length` (T)
- `--rnn_fix_length` (should match H for TBPTT)

### Delayed Environment

The delayed environment simulates observation delays:

1. **Delay Buffer**: Maintains history of observations
2. **Delay Sampling**: Samples delay from distribution (Gamma/Uniform/DoubleGaussian)
3. **Non-Stationary**: Delay distribution changes over time
4. **Meta-Learning**: Delay parameters encoded as context vector

### RMDM (Recurrent Model-based Dynamics Model)

- Learns environment representation from state-action-next_state transitions
- Uses contrastive learning to distinguish different environments
- Provides context vector for policy conditioning
- Key parameters:
  - `--use_rmdm`: Enable RMDM
  - `--bottle_neck`: Use bottleneck architecture
  - `--stop_pg_for_ep`: Stop policy gradient for environment parameters
  - `--rbf_radius`: RBF kernel radius for similarity

## Important Implementation Notes

### Ray Parallelization

- Training uses Ray for parallel environment execution
- `EnvRemoteArray` manages worker processes
- Each worker runs independent environment instances
- Synchronization happens during batch collection

### Memory Management

- `Memory` class stores transitions
- `MemoryArray` for parallel collection
- Replay buffer size controlled by `--memory_size`

### Logging

- TensorBoard logs saved to `log_file/[experiment_name]/`
- Checkpoints saved every N iterations
- Code snapshot saved at experiment start

### GPU Selection

The `train.sh` script automatically selects the least-used GPU (excluding GPU 0). To manually set GPU:
```bash
export CUDA_VISIBLE_DEVICES=1,2,3
python main.py [args]
```

## Troubleshooting

**Training crashes:**
- Check GPU memory: `nvidia-smi`
- Reduce `--min_batch_size` or `--num_threads`
- Check Ray logs for worker errors

**Poor performance:**
- Verify environment parameters are reasonable
- Check if RMDM loss is decreasing
- Ensure sufficient exploration (check Alpha values)
- Try different `--rnn_fix_length` values

**Slow training:**
- Increase `--update_interval` (trade-off: less frequent updates)
- Reduce `--num_threads` if CPU-bound
- Use `--min_batch_size` to control iteration length

## Key Metrics to Monitor

- `EpRetTest`: Test episode return (main performance metric)
- `NSDeltaVSTestRet`: Non-stationary performance gap
- `OODDeltaVSTestRet`: Out-of-distribution performance gap
- `ActorLoss`: Policy loss (should increase in absolute value as Q improves)
- `QMean`: Average Q-value (should increase)
- `Alpha`: Entropy coefficient (should decrease as policy improves)
- `rmdmLoss`: RMDM representation loss (should decrease)

Good training indicators:
- EpRetTest steadily increasing
- |Delta| metrics approaching 0
- RMDM loss decreasing
- Q values increasing
- Stable training time per iteration

## Related Documentation

- `README.md` - Installation and basic usage
- `HOW_TO_ANALYZE.md` - Detailed analysis guide with metric explanations
- `DELAYED_ENV_INTEGRATION.md` - Delayed environment integration details
- `RNN_ACTION_HISTORY_MODIFICATION.md` - RNN input construction details
- `SLIDING_WINDOW_INPUT_CHANGES.md` - Sliding window implementation

## Supported Environments

- `HalfCheetah-v2/v4` - Fast quadruped locomotion
- `Walker2d-v2/v4` - Bipedal walking
- `Hopper-v2/v4` - Single-leg hopping
- `Ant-v2/v4` - Quadruped with complex dynamics
- `Humanoid-v2/v4` - Full humanoid locomotion
- `GridWorldPlat-v2` - Custom grid world (use task_num=12)

## Varying Parameters

Physical parameters that can vary (via `--varying_params`):
- `gravity` - Gravitational acceleration
- `body_mass` - Body mass multipliers
- `dof_damping` - Joint damping coefficients
- `dof_damping_1_dim` - Single dimension damping
- See `envs/nonstationary_env.py` for full list

## Experiment Naming

Log directories follow pattern:
```
log_file/{env_name}-rnn_len_{rnn_fix_length}-ep_dim_{ep_dim}-{seed}_{name_suffix}/
```

Example:
```
log_file/Walker2d-v4-rnn_len_32-ep_dim_2-1_N/
```
