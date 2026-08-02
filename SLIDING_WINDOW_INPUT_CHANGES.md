# Sliding Window State-Action Aligned Input – Change Log

## 1. Objective
This change replaces the previous “single flattened action history appended to every state” input with a **sliding window aligned** input:

- **H = `ep_state_history_length`**: number of recent states used as EP/RNN sequence length.
- **T = `ep_action_window_length`**: for each state timestep, use a backward-looking action window of length T aligned to that timestep.
- Training and online inference must construct **the same (state, aligned_action_window)** structure.

Key requirement implemented: during training, when sequence data exists, **crop to last H timesteps** so EP only sees the most recent H aligned pairs.


## 2. New / Updated Runtime Parameters
### Added (new)
- `--ep_state_history_length` (H)
  - Default: `8`
  - Meaning: EP/RNN sequence length (how many historical states are used).
- `--ep_action_window_length` (T)
  - Default: `10`
  - Meaning: action window length aligned to each state.

### Existing (used / aligned)
- `--rnn_fix_length`
  - Recommendation: set **equal to H** to keep TBPTT length consistent with EP input.
- `--action_history_length`
  - In this change, it is treated as a **compatibility parameter** representing **T** (action window length) in places that still use the old name.


## 3. Files Modified (What & Why)

### 3.1 `parameter/Parameter.py`
- Registered two new argparse options:
  - `ep_state_history_length` (H)
  - `ep_action_window_length` (T)
- Added a runtime **warning** when `rnn_fix_length > 0` and `rnn_fix_length != ep_state_history_length`, recommending alignment.


### 3.2 `utils/history_construct.py`
- Updated `HistoryConstructor` to support the new action window definition:
  - Maintains an action deque sized by `ep_action_window_length` (T).
  - `get_action_history_array()` now returns the current action window (flattened) that corresponds to the current timestep.
- Purpose: ensure the online rollout state augmentation produces the correct (state + aligned_action_window) structure.


### 3.3 `agent/Agent.py`
#### `EnvWorker`
- Initializes `HistoryConstructor` with:
  - `ep_state_history_length`
  - `ep_action_window_length`
- When pushing transitions into replay buffer (`mem.push`), stores:
  - `state` without the action prefix
  - `last_action` as the **action window prefix** (length `T * act_dim`)

#### `EnvRemoteArray`
- Fixed missing attributes:
  - Added `self.ep_state_history_length` and `self.ep_action_window_length` to avoid runtime error in sampling.
- Sampling (`sample1step` / `sample1step1env`) uses:
  - `action_history_size = ep_action_window_length * act_dim`
  - to split the prefix action window from the actual observation.


### 3.4 `models/policy.py`
- EP input dimension changed to use **T**:
  - from `obs_dim + act_dim * action_history_length`
  - to effectively `obs_dim + act_dim * ep_action_window_length`
- Updated `make_config_from_param()` to carry new params through model construction.
- Updated inference path (`inference_one_step`) to split input state into:
  - `lst_a` (action window prefix, size `T * act_dim`)
  - `x` (pure observation)


### 3.5 `models/value.py`
- Same EP input-dim and config propagation updates as `models/policy.py`.


### 3.6 `models/transition.py`
- Same EP input-dim update as policy/value so all EP consumers share the same aligned input definition.


### 3.7 `algorithms/sac.py`
- Updated batch construction so that, for sequence training, each timestep in the state sequence uses its **own aligned action window** (not one shared flat history).
- Enforced EP sequence length constraint:
  - When sequence tensors are present, they are cropped to the last `ep_state_history_length` timesteps.
  - `valid_num` is recomputed after truncation.
- Adjusted the ordering of computations to avoid shape mismatch with contrastive loss query tensor creation.


### 3.8 `algorithms/contrastive.py`
- Continues to use the EP query input as `torch.cat((state, last_action), dim=-1)`.
- With this change, `last_action` now represents the per-timestep aligned action window (flattened).


### 3.9 `train.sh`
- Updated training script to pass the new flags:
  - `--ep_state_history_length`
  - `--ep_action_window_length`
  - `--rnn_fix_length` (set equal to H)
- Kept `--action_history_length` as compatibility, set equal to T.


## 4. Data Flow After Change (High-Level)
- **Online rollout**:
  - `HistoryConstructor` maintains the last T actions.
  - Augmented input state is `[action_window(T) || obs]`.
  - Replay buffer stores `last_action = action_window(T)` separately.

- **Training**:
  - Replay sampling provides sequences of `state` and `last_action`.
  - `sac_update` constructs per-timestep aligned windows and crops to last H timesteps.

- **Inference**:
  - `Policy.inference_one_step` splits `[action_window(T) || obs]` and feeds EP consistently.


## 5. Recommended Config
- Set:
  - `ep_state_history_length = 8`
  - `ep_action_window_length = 10`
  - `rnn_fix_length = ep_state_history_length`
  - `action_history_length = ep_action_window_length` (compat)


## 6. Known Non-blocking Warnings
- Gym / NumPy compatibility warnings may appear at runtime; they are not caused by this change and do not block import/compile.
