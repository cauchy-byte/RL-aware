# AGENTS.md

This file gives coding agents practical guidance for working in this repository.
It is based on the existing `CLAUDE.md`, the current scripts, and the experiment
configuration recorded under `log_file/`.

## Project Snapshot

This repository is an extended ESCP implementation for meta-reinforcement
learning under sudden environment changes. The current work focuses on delayed
and non-stationary delayed MuJoCo environments, using RMDM/contrastive learning
and a sliding-window state-action history for environment probing.

Original paper:
`Adapt to Environment Sudden Changes by Learning a Context Sensitive Policy`

Important local changes relative to the original ESCP code:
- Delayed environment wrappers are present in `envs/delayed_env.py`,
  `envs/delayed_distribution.py`, and `envs/nonstationary_delayed_env.py`.
- `Parameter.py` currently defaults to delayed training:
  `use_delay=True`, `nonstationary_delay=True`, `use_rmdm=True`,
  `use_wmcl=True`, `ep_dim=16`.
- EP/RNN input now uses aligned sliding windows:
  `H = ep_state_history_length`, `T = ep_action_window_length`.
- `action_history_length` is kept as a compatibility name for `T`.
- Existing recent RMDM delayed experiments mainly use `rnn_fix_length=16`,
  `ep_state_history_length=16`, `ep_action_window_length=10`,
  `action_history_length=10`, `update_interval=20`,
  `rmdm_update_interval=10`, `min_batch_size=1000`.

## Repository Map

- `main.py`: entry point; initializes Ray and starts SAC training.
- `algorithms/sac.py`: main Soft Actor-Critic training loop with EP/RMDM hooks.
- `algorithms/RMDM.py`: RMDM representation learning.
- `algorithms/contrastive.py`: contrastive loss for representation learning.
- `models/policy.py`: actor policy with environment parameter conditioning.
- `models/value.py`: critic/Q networks.
- `models/rnn_base.py`: RNN/GRU model components.
- `models/mlp_base.py`: MLP model components.
- `models/transition.py`: dynamics/transition model.
- `agent/Agent.py`: `EnvWorker` and `EnvRemoteArray` for Ray-based collection.
- `envs/env_wrapper.py`: environment decoration factory.
- `envs/nonstationary_env.py`: physical parameter non-stationarity.
- `envs/nonstationary_delayed_env.py`: delay distribution non-stationarity.
- `envs/delayed_distribution.py`: Gamma, Uniform, and DoubleGaussian delays.
- `utils/history_construct.py`: online construction of state/action histories.
- `utils/replay_memory.py`: replay buffers.
- `parameter/Parameter.py`: argparse configuration and experiment metadata.
- `parameter/private_config.py`: local paths/private configuration.
- `log_util/`: TensorBoard/file logging and experiment snapshots.
- `log_file/`: experiment outputs, copied code snapshots, configs, models, evals.

Related documentation:
- `CLAUDE.md`: original agent guidance.
- `README.md`: upstream installation and basic usage.
- `DELAYED_ENV_INTEGRATION.md`: delayed environment design.
- `SLIDING_WINDOW_INPUT_CHANGES.md`: current aligned state-action input design.
- `RNN_ACTION_HISTORY_MODIFICATION.md`: older action-history modification notes.
- `HOW_TO_ANALYZE.md`: training-log analysis guide.

## Environment Setup

Use the local virtual environment when available:

```bash
source .venv/bin/activate
```

Install dependencies with:

```bash
pip install -r requirement.txt
```

MuJoCo/Gym compatibility can be fragile. Prefer running commands with the
existing `.venv/bin/python` when it exists; both `train.sh` and
`evaluate_delay_shift_escp.py` already try to use it.

This directory is not currently a Git checkout in this workspace, so do not
expect `git status` or branch operations to work here.

## Common Commands

Run the current delayed-environment training configuration:

```bash
bash train.sh
```

`train.sh` currently uses:
- `ENV_NAME=HalfCheetah-v4`
- `MIN_BATCH_SIZE=1000`
- `EP_STATE_HISTORY_LENGTH=16`
- `EP_ACTION_WINDOW_LENGTH=10`
- `RNN_FIX_LENGTH=16`
- `ACTION_HISTORY_LENGTH=10`
- `RANDOM_DELAY_PER_EPISODE=1`
- `--use_wmcl --wmcl_weight 1`
- `--use_rmdm --bottle_neck --use_contrastive`
- `--rmdm_update_interval 10`
- `--rbf_radius 3000.0`
- `--delay_changing_period 10000`
- `--delay_changing_interval 500`
- `--delay_task_num 10`
- `--update_interval 20`

Run delayed shift evaluation:

```bash
bash evaluate.sh
```

`evaluate.sh` currently points to an Ant-v4 run directory. Check that the path
exists before running; the present `log_file/` entries include HalfCheetah-v4,
Hopper-v4, Humanoid-v4, Reacher-v4, and Walker2d-v4 runs, but no top-level
Ant-v4 run directory was visible.

Run evaluation directly:

```bash
.venv/bin/python evaluate_delay_shift_escp.py \
  --run-dir log_file/Walker2d-v4-use_rmdm-rnn_len_16-bottle_neck-ep_dim_16-1_N \
  --initial-delay-type gamma \
  --shifted-delay-type doublegaussian \
  --type-max-delay '{"gamma": 6, "uniform": 7, "doublegaussian": 9}' \
  --episodes 20 \
  --shift-step 50 \
  --post-shift-steps 300
```

Monitor a long run:

```bash
tmux new -s training
bash train.sh 2>&1 | tee training.log
tail -f training.log
```

## Current Experiment Reality

Recent RMDM delayed runs in `log_file/` use a consistent configuration:
- `use_delay=True`
- `initial_delay_type=gamma`
- `use_rmdm=True`
- `use_wmcl=True`
- `use_contrastive=True`
- `bottle_neck=True`
- `ep_dim=16`
- `rnn_fix_length=16`
- `ep_state_history_length=16`
- `ep_action_window_length=10`
- `action_history_length=10`
- `min_batch_size=1000`
- `update_interval=20`
- `rmdm_update_interval=10`

The active `train.sh` run is non-stationary during training:
- `nonstationary_delay=True`
- `delay_changing_period=10000`
- `delay_changing_interval=500`
- `delay_task_num=10`
- `max_delay_range_min=3`
- `max_delay_range_max=10`
- no physical `varying_params`

Many saved evaluation-style run configs set:
- `nonstationary_delay=False`
- `delay_changing_period=1000000000`
- `delay_changing_interval=1000000000`
- `delay_task_num=2`
- `max_delay_range_min=6`
- `max_delay_range_max=9`

Interpret those as fixed-delay or shift-evaluation configs, not necessarily
the current training setup.

Existing delayed shift summaries show different behavior by environment:
- `Walker2d-v4-use_rmdm-rnn_len_16-bottle_neck-ep_dim_16-1_N`: 20/20
  successful recovery episodes in the saved gamma-to-doublegaussian eval.
- `Humanoid-v4-use_rmdm-rnn_len_16-bottle_neck-ep_dim_16-1_N`: 14/20
  successful recovery episodes, with 5 not recovered.
- `Reacher-v4-use_rmdm-rnn_len_16-bottle_neck-ep_dim_16-1_N`: 0/20
  successful recovery episodes in the saved eval.
- Older Hopper/Walker baselines often used `ep_dim=2` and may include physical
  `varying_params=["gravity", "body_mass"]`.

## Sliding-Window EP Input

The current intended EP/RNN input is:
- `H = ep_state_history_length`: number of recent states/timesteps.
- `T = ep_action_window_length`: number of recent actions aligned to each state.
- For each timestep, the EP input is the state plus that timestep's flattened
  backward-looking action window.
- During training, sequence tensors should be cropped to the most recent `H`
  timesteps when sequence data exists.
- Keep `rnn_fix_length == ep_state_history_length` for fixed-length EP training.
- Keep `action_history_length == ep_action_window_length` unless deliberately
  testing old compatibility behavior.

Files that must stay consistent when touching this flow:
- `parameter/Parameter.py`
- `utils/history_construct.py`
- `agent/Agent.py`
- `algorithms/sac.py`
- `algorithms/contrastive.py`
- `models/policy.py`
- `models/value.py`
- `models/transition.py`

## Delayed Environment Notes

Supported delay distributions:
- `gamma`
- `uniform`
- `doublegaussian`

Distribution-specific max-delay caps used by evaluation:
- `gamma`: 6
- `uniform`: 9
- `doublegaussian`: 10

Common evaluation binding:

```bash
--type-max-delay '{"gamma": 6, "uniform": 7, "doublegaussian": 9}'
```

`evaluate_delay_shift_escp.py` forces evaluation to CPU, loads parameters from
the selected run directory, disables non-stationary delay during evaluation,
and manually switches delay mode at `--shift-step`.

## Coding Guidelines For Agents

- Read `CLAUDE.md` and this file before making broad changes.
- Prefer small, targeted edits. This codebase has multiple historical input
  formats and experiment branches; avoid "cleanup" changes unless requested.
- Do not delete or rewrite `log_file/` experiment outputs unless the user
  explicitly asks.
- Treat `log_file/*/codes/` as historical snapshots, not active source.
- When changing any parameter name or tensor shape, update all producers and
  consumers in the sliding-window flow.
- Be careful with boolean argparse options that use `type=bool`; shell strings
  can behave unexpectedly. Existing scripts use numeric `0`/`1` for some flags.
- Do not launch long training jobs unless the user explicitly asks.
- For short validation, prefer import/compile checks or focused script dry runs
  over full RL training.
- Generated plots, checkpoints, and logs can be large; avoid adding new outputs
  unless needed for the requested task.

## Validation Suggestions

For syntax/import sanity:

```bash
.venv/bin/python -m py_compile main.py algorithms/sac.py agent/Agent.py \
  models/policy.py models/value.py models/transition.py \
  envs/nonstationary_delayed_env.py evaluate_delay_shift_escp.py
```

For config sanity:

```bash
.venv/bin/python main.py --help >/tmp/escp_help.txt
```

For delayed shift evaluation, first verify the selected run directory has:
- `parameter.json`
- model checkpoint files expected by `Logger`/`Policy.load`
- optional `delay_shift_eval/summary.json` from previous runs

## Metrics To Watch

Training:
- `EpRetTest`: main test return.
- `NSDeltaVSTestRet`: non-stationary performance gap.
- `OODDeltaVSTestRet`: out-of-distribution performance gap.
- `ActorLoss`, `QMean`, `Alpha`: SAC health.
- `rmdmLoss`: representation learning health.

Delay shift evaluation:
- `successful_recovery_episode_count`
- `not_recovered_count`
- `short_episode_count`
- `avg_recovery_step_to_90pct`
- `avg_post_shift_steps`
- `avg_baseline_reward`
- `avg_mean_post_shift_reward`

When judging results, account for short episodes: many saved Hopper evaluations
terminate before or soon after the shift step, so recovery counts can be
misleading without checking `short_episode_count` and `avg_post_shift_steps`.
