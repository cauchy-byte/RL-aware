import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
if sys.executable != str(VENV_PYTHON) and VENV_PYTHON.exists():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent.Agent import EnvWorker
from envs.env_wrapper import create_env_decoration
from log_util.logger import Logger
from models.policy import Policy
from parameter.Parameter import Parameter


@dataclass
class EvalConfig:
    episodes: int
    max_steps: int
    warmup_steps: int
    shift_step: int
    post_shift_steps: int
    seed: int
    deterministic: bool
    initial_delay_type: str
    shifted_delay_type: str
    initial_max_delay: int
    shifted_max_delay: int
    initial_delay_process: str = "legacy"
    shifted_delay_process: str = "legacy"
    delay_type_max_delay: Optional[Dict[str, int]] = None


# 延迟类型允许的最大 max_delay 上限（与 envs/nonstationary_delayed_env.py 保持一致）
TYPE_MAX_DELAY_CAPS: Dict[str, int] = {
    "gamma": 6,
    "uniform": 9,
    "doublegaussian": 10,
}


def validate_type_max_delay_map(mapping: Dict[str, int]) -> None:
    """Validate per-type max_delay binding.

    For each type present in the mapping, the value must be a positive int
    that does not exceed the cap supported by the corresponding distribution
    (see ``envs.nonstationary_delayed_env.NonstationaryDelayedEnv._create_delay_distribution``).
    """
    for delay_type, max_delay in mapping.items():
        if delay_type not in TYPE_MAX_DELAY_CAPS:
            raise ValueError(
                f"Unknown delay_type '{delay_type}' in --type-max-delay; "
                f"valid types: {sorted(TYPE_MAX_DELAY_CAPS)}"
            )
        if not isinstance(max_delay, int) or max_delay <= 0:
            raise ValueError(
                f"max_delay for type '{delay_type}' must be a positive int, got {max_delay!r}"
            )
        cap = TYPE_MAX_DELAY_CAPS[delay_type]
        if max_delay > cap:
            raise ValueError(
                f"max_delay for type '{delay_type}' is {max_delay}, "
                f"which exceeds the supported cap {cap} for this distribution"
            )


def resolve_max_delay(delay_type: str, eval_cfg: EvalConfig, phase: str) -> int:
    """Resolve the max_delay for a given phase.

    When ``eval_cfg.delay_type_max_delay`` is provided, the max_delay is
    looked up from the per-type binding so that switching the delay type
    automatically changes the parameter.  Otherwise the explicit
    ``initial_max_delay`` / ``shifted_max_delay`` from the legacy
    configuration is used.
    """
    mapping = eval_cfg.delay_type_max_delay
    if mapping is not None:
        if delay_type not in mapping:
            raise ValueError(
                f"delay_type '{delay_type}' is not present in --type-max-delay binding "
                f"{mapping}; please add an entry for it"
            )
        return int(mapping[delay_type])
    if phase == "initial":
        return int(eval_cfg.initial_max_delay)
    if phase == "shifted":
        return int(eval_cfg.shifted_max_delay)
    raise ValueError(f"Unknown phase '{phase}'")


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def load_parameter_from_run(run_dir: Path) -> Parameter:
    param = Parameter(config_path=str(run_dir))
    param.set_config_path(str(run_dir))
    return param


def override_delay_eval_setup(param: Parameter, eval_cfg: EvalConfig) -> None:
    param.use_delay = True
    param.nonstationary_delay = False
    param.delay_process = eval_cfg.initial_delay_process
    param.initial_delay_type = eval_cfg.initial_delay_type
    # max_delay_range 必须覆盖到所有可能出现的 max_delay（无论是从
    # delay_type_max_delay 映射中解析出的，还是从显式的 initial/shifted
    # 字段中取得的），否则 NonstationaryDelayedEnv 的缓冲区裁剪会过窄。
    candidate_max_delays: List[int] = [
        int(eval_cfg.initial_max_delay),
        int(eval_cfg.shifted_max_delay),
    ]
    if eval_cfg.delay_type_max_delay:
        candidate_max_delays.extend(int(v) for v in eval_cfg.delay_type_max_delay.values())
    param.max_delay_range_min = min(candidate_max_delays)
    param.max_delay_range_max = max(candidate_max_delays)
    param.delay_changing_period = int(1e9)
    param.delay_changing_interval = int(1e9)
    param.delay_task_num = 2
    param.render = False
    param.model_path = None


def build_worker(param: Parameter, seed: int) -> EnvWorker:
    logger = Logger(log_to_file=False, parameter=param, force_backup=False)
    env_decoration = create_env_decoration(param)
    worker = EnvWorker(
        parameter=param,
        env_name=param.env_name,
        seed=seed,
        policy_type=Policy,
        history_len=0,
        env_decoration=env_decoration,
        env_tasks=None,
        use_true_parameter=param.use_true_parameter,
        non_stationary=False,
    )
    worker.policy.to(torch.device("cpu"))
    model_dir = Path(logger.model_output_dir)
    worker.policy.load(str(model_dir), map_location=torch.device("cpu"))
    worker.policy.eval()
    worker.policy.inference_init_hidden(1)
    return worker


def set_delay_mode(env: Any, delay_type: str, max_delay: int) -> Dict[str, Any]:
    task = {"delay_type": delay_type, "max_delay": int(max_delay)}
    env.set_delay_task(task)
    return task


def set_evaluation_delay_mode(
    env: Any,
    delay_process: str,
    delay_type: str,
    max_delay: int,
) -> Dict[str, Any]:
    """Apply one evaluation phase's process or legacy delay task."""
    current_process = getattr(env, "delay_process", "legacy")
    if current_process != delay_process and hasattr(env, "set_delay_process"):
        env.set_delay_process(delay_process)
    if delay_process != "legacy":
        return {"delay_process": delay_process}
    return set_delay_mode(env, delay_type, max_delay)


def get_delay_state(env: Any) -> Dict[str, Any]:
    return {
        "delay_process": getattr(env, "delay_process", "legacy"),
        "delay_type": getattr(env, "current_delay_type", None),
        "max_delay": int(getattr(env, "current_max_delay", -1)),
        "sampled_observation_delay": int(getattr(env, "last_sampled_delay", 0)),
        "delay_regime": getattr(env, "current_delay_regime", "legacy"),
        "delay_censored": bool(getattr(env, "delay_censored", False)),
        "delay_vector": _to_builtin(getattr(env, "delay_distribution_vector", [])),
        "env_parameter_vector": _to_builtin(getattr(env, "env_parameter_vector", [])),
    }


def compute_sliding_average(rewards: List[float], window: int) -> List[float]:
    """Compute sliding window average of rewards."""
    if len(rewards) < window:
        return rewards
    result = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        result.append(sum(rewards[start:i + 1]) / (i - start + 1))
    return result


def evaluate_episode(
    worker: EnvWorker,
    eval_cfg: EvalConfig,
    episode_index: int,
    recovery_window: int = 20,
    recovery_threshold: float = 0.9,
) -> Dict[str, Any]:
    obs = worker.reset(None)
    obs = worker.extend_state(obs)
    worker.state = obs
    worker.policy.inference_init_hidden(1)

    env = worker.env
    initial_max_delay = resolve_max_delay(eval_cfg.initial_delay_type, eval_cfg, "initial")
    shifted_max_delay = resolve_max_delay(eval_cfg.shifted_delay_type, eval_cfg, "shifted")
    initial_task = set_evaluation_delay_mode(
        env, eval_cfg.initial_delay_process, eval_cfg.initial_delay_type, initial_max_delay
    )
    shifted_task = set_evaluation_delay_mode(
        env, eval_cfg.shifted_delay_process, eval_cfg.shifted_delay_type, shifted_max_delay
    ) if eval_cfg.shift_step == 0 else (
        {"delay_process": eval_cfg.shifted_delay_process}
        if eval_cfg.shifted_delay_process != "legacy"
        else {"delay_type": eval_cfg.shifted_delay_type, "max_delay": int(shifted_max_delay)}
    )

    total_reward = 0.0
    step_rows: List[Dict[str, Any]] = []
    shift_applied = False
    done = False

    horizon = max(eval_cfg.max_steps, eval_cfg.shift_step + eval_cfg.post_shift_steps)

    for step in range(horizon):
        if (not shift_applied) and step == eval_cfg.shift_step:
            set_evaluation_delay_mode(
                env, eval_cfg.shifted_delay_process, eval_cfg.shifted_delay_type, shifted_max_delay
            )
            shift_applied = True

        state_tensor = torch.from_numpy(worker.state).to(torch.get_default_dtype()).unsqueeze(0)
        action_tensor = worker.policy.inference_one_step(state_tensor, deterministic=eval_cfg.deterministic)[0]
        action = action_tensor.detach().cpu().numpy()

        _, reward, done, next_state, _, _, _, _ = worker.step(action, render=False, need_info=True)
        total_reward += float(reward)

        row = {
            "episode": episode_index,
            "step": step,
            "reward": float(reward),
            "cumulative_reward": total_reward,
            "phase": "post_shift" if shift_applied else "pre_shift",
            "shift_applied": int(shift_applied),
            **get_delay_state(env),
        }
        step_rows.append(row)

        if done:
            break

    post_shift_rewards = [r["reward"] for r in step_rows if r["shift_applied"] == 1]
    sampled_delays = [int(r["sampled_observation_delay"]) for r in step_rows]
    censored_count = sum(1 for r in step_rows if r["delay_censored"])
    
    # Compute baseline from stable pre-shift window (skip first 20 steps)
    baseline_start = max(20, eval_cfg.shift_step - 50)
    baseline_window = [r["reward"] for r in step_rows[baseline_start:eval_cfg.shift_step]]
    baseline_reward = float(np.mean(baseline_window)) if baseline_window else None
    
    # Compute recovery step using sliding window average
    recovery_step = None
    if post_shift_rewards and baseline_reward is not None:
        post_shift_sliding = compute_sliding_average(post_shift_rewards, recovery_window)
        threshold = recovery_threshold * baseline_reward
        
        # Also compute cumulative reward from shift point
        cumulative_from_shift = []
        cumsum = 0.0
        for r in post_shift_rewards:
            cumsum += r
            cumulative_from_shift.append(cumsum)
        
        # Use sliding window: recovery when consecutive N steps all above threshold
        consecutive_above = 0
        required_consecutive = recovery_window // 2
        for idx, avg_reward in enumerate(post_shift_sliding):
            if avg_reward >= threshold:
                consecutive_above += 1
                if consecutive_above >= required_consecutive:
                    recovery_step = idx
                    break
            else:
                consecutive_above = 0

    return {
        "episode": episode_index,
        "initial_task": initial_task,
        "shifted_task": shifted_task,
        "initial_delay_process": eval_cfg.initial_delay_process,
        "shifted_delay_process": eval_cfg.shifted_delay_process,
        "shift_step": eval_cfg.shift_step,
        "sampled_delay_min": min(sampled_delays) if sampled_delays else None,
        "sampled_delay_max": max(sampled_delays) if sampled_delays else None,
        "censored_count": censored_count,
        "total_reward": total_reward,
        "steps": len(step_rows),
        "pre_shift_steps": eval_cfg.shift_step,
        "post_shift_steps": len(post_shift_rewards),
        "baseline_reward": baseline_reward,
        "baseline_window_size": len(baseline_window),
        "min_post_shift_reward": float(np.min(post_shift_rewards)) if post_shift_rewards else None,
        "max_post_shift_reward": float(np.max(post_shift_rewards)) if post_shift_rewards else None,
        "mean_post_shift_reward": float(np.mean(post_shift_rewards)) if post_shift_rewards else None,
        "recovery_step_to_90pct": recovery_step,
        "rows": step_rows,
    }


def summarize_results(results: Sequence[Dict[str, Any]], recovery_window: int = 20) -> Dict[str, Any]:
    recovery = [r["recovery_step_to_90pct"] for r in results if r["recovery_step_to_90pct"] is not None]
    min_rewards = [r["min_post_shift_reward"] for r in results if r["min_post_shift_reward"] is not None]
    max_rewards = [r["max_post_shift_reward"] for r in results if r["max_post_shift_reward"] is not None]
    mean_rewards = [r["mean_post_shift_reward"] for r in results if r["mean_post_shift_reward"] is not None]
    baseline_rewards = [r["baseline_reward"] for r in results if r["baseline_reward"] is not None]
    post_shift_step_counts = [r["post_shift_steps"] for r in results]
    
    # Count episodes by outcome
    short_episodes = [r for r in results if r["post_shift_steps"] < recovery_window]
    recovered = [r for r in results if r["recovery_step_to_90pct"] is not None]
    not_recovered = [r for r in results if r["recovery_step_to_90pct"] is None and r["post_shift_steps"] >= recovery_window]

    summary = {
        "episodes": len(results),
        "avg_recovery_step_to_90pct": float(np.mean(recovery)) if recovery else None,
        "median_recovery_step_to_90pct": float(np.median(recovery)) if recovery else None,
        "min_recovery_step": int(np.min(recovery)) if recovery else None,
        "max_recovery_step": int(np.max(recovery)) if recovery else None,
        "avg_min_post_shift_reward": float(np.mean(min_rewards)) if min_rewards else None,
        "avg_max_post_shift_reward": float(np.mean(max_rewards)) if max_rewards else None,
        "avg_mean_post_shift_reward": float(np.mean(mean_rewards)) if mean_rewards else None,
        "avg_baseline_reward": float(np.mean(baseline_rewards)) if baseline_rewards else None,
        "successful_recovery_episode_count": len(recovery),
        "short_episode_count": len(short_episodes),
        "not_recovered_count": len(not_recovered),
        "avg_post_shift_steps": float(np.mean(post_shift_step_counts)) if post_shift_step_counts else None,
        "initial_delay_process": results[0].get("initial_delay_process", "legacy") if results else "legacy",
        "shifted_delay_process": results[0].get("shifted_delay_process", "legacy") if results else "legacy",
        "shift_step": results[0].get("shift_step") if results else None,
        "sampled_delay_min": min(
            r["sampled_delay_min"] for r in results if r.get("sampled_delay_min") is not None
        ) if any(r.get("sampled_delay_min") is not None for r in results) else None,
        "sampled_delay_max": max(
            r["sampled_delay_max"] for r in results if r.get("sampled_delay_max") is not None
        ) if any(r.get("sampled_delay_max") is not None for r in results) else None,
        "censored_count": int(sum(r.get("censored_count", 0) for r in results)),
    }
    
    # Add per-episode breakdown
    if len(short_episodes) > 0:
        summary["short_episodes_detail"] = [
            {"episode": r["episode"], "post_shift_steps": r["post_shift_steps"]} 
            for r in short_episodes
        ]
    if len(not_recovered) > 0:
        summary["not_recovered_detail"] = [
            {
                "episode": r["episode"], 
                "post_shift_steps": r["post_shift_steps"],
                "mean_reward": r["mean_post_shift_reward"],
                "max_reward": r["max_post_shift_reward"],
                "baseline": r["baseline_reward"]
            } 
            for r in not_recovered
        ]
    
    return summary


def write_step_csv(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    rows = [row for result in results for row in result["rows"]]
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def moving_average(values: np.ndarray, window: int = 15) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    if len(values) < window:
        window = max(1, len(values) // 3)
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def align_post_shift_curves(results: Sequence[Dict[str, Any]]) -> Optional[np.ndarray]:
    curves: List[np.ndarray] = []
    for result in results:
        post_shift_rewards = [row["reward"] for row in result["rows"] if row["shift_applied"] == 1]
        if post_shift_rewards:
            curves.append(np.asarray(post_shift_rewards, dtype=float))
    if not curves:
        return None

    max_len = max(len(curve) for curve in curves)
    aligned = np.full((len(curves), max_len), np.nan, dtype=float)
    for i, curve in enumerate(curves):
        aligned[i, : len(curve)] = curve
    return aligned


def plot_recovery_curve(output_dir: Path, eval_cfg: EvalConfig, summary: Dict[str, Any], results: Sequence[Dict[str, Any]]) -> Optional[Path]:
    aligned = align_post_shift_curves(results)
    if aligned is None:
        return None

    mean_curve = np.nanmean(aligned, axis=0)
    std_curve = np.nanstd(aligned, axis=0)
    smoothed_curve = moving_average(mean_curve, window=15)
    x = np.arange(len(mean_curve))

    baseline = summary.get("avg_baseline_reward")
    avg_recovery = summary.get("avg_recovery_step_to_90pct")
    median_recovery = summary.get("median_recovery_step_to_90pct")
    successful_recoveries = summary.get("successful_recovery_episode_count")

    plt.figure(figsize=(10, 6), dpi=180)

    for result in results:
        post_shift_rewards = [row["reward"] for row in result["rows"] if row["shift_applied"] == 1]
        if post_shift_rewards:
            plt.plot(np.arange(len(post_shift_rewards)), post_shift_rewards, alpha=0.18, linewidth=1)

    plt.plot(x, mean_curve, color="#1f77b4", linewidth=1.5, alpha=0.55, label="Mean reward")
    plt.plot(x, smoothed_curve, color="#d62728", linewidth=2.5, label="Smoothed recovery")
    plt.fill_between(
        x,
        mean_curve - std_curve,
        mean_curve + std_curve,
        color="#1f77b4",
        alpha=0.15,
        label="±1 std",
    )

    if baseline is not None:
        plt.axhline(baseline, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Pre-shift baseline = {baseline:.2f}")
        plt.axhline(0.9 * baseline, color="#9467bd", linestyle=":", linewidth=1.5, label=f"90% baseline = {0.9 * baseline:.2f}")

    if avg_recovery is not None:
        plt.axvline(avg_recovery, color="#ff7f0e", linestyle="--", linewidth=2, label=f"Avg recovery step = {avg_recovery:.2f}")

    initial_max_delay = resolve_max_delay(eval_cfg.initial_delay_type, eval_cfg, "initial")
    shifted_max_delay = resolve_max_delay(eval_cfg.shifted_delay_type, eval_cfg, "shifted")
    plt.title(
        "Recovery Curve After Delay Distribution Shift\n"
        f"{eval_cfg.initial_delay_type}(max_delay={initial_max_delay}) → "
        f"{eval_cfg.shifted_delay_type}(max_delay={shifted_max_delay}), "
        f"shift at env step {eval_cfg.shift_step}"
    )
    plt.xlabel("Steps since distribution shift")
    plt.ylabel("Reward per step")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)

    text = (
        f"Episodes with post-shift data: {aligned.shape[0]}\n"
        f"Successful recovery episodes: {successful_recoveries}\n"
        f"Median recovery step: {median_recovery}"
    )
    plt.text(
        0.985,
        0.02,
        text,
        transform=plt.gca().transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#dddddd"},
    )

    plt.tight_layout()
    plot_path = output_dir / "recovery_curve.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    return plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ESCP model under delay distribution shift.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="/home/zhangboyuan/ESCP-master_extended/ESCP-master_extended/log_file/Walker2d-v4-rnn_len_32-ep_dim_2-1_N",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--shift-step", type=int, default=250)
    parser.add_argument("--post-shift-steps", type=int, default=500)
    parser.add_argument("--recovery-window", type=int, default=20, help="Window size for sliding average reward")
    parser.add_argument("--recovery-threshold", type=float, default=0.9, help="Recovery threshold as fraction of baseline")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--initial-delay-type", type=str, default="gamma", choices=["gamma", "uniform", "doublegaussian"])
    parser.add_argument("--shifted-delay-type", type=str, default="uniform", choices=["gamma", "uniform", "doublegaussian"])
    parser.add_argument(
        "--initial-delay-process",
        type=str,
        default="legacy",
        choices=["legacy", "ge1_23", "ge4_32", "mm1"],
    )
    parser.add_argument(
        "--shifted-delay-process",
        type=str,
        default="legacy",
        choices=["legacy", "ge1_23", "ge4_32", "mm1"],
    )
    parser.add_argument("--initial-max-delay", type=int, default=6)
    parser.add_argument("--shifted-max-delay", type=int, default=9)
    parser.add_argument(
        "--type-max-delay",
        type=str,
        default=None,
        help=(
            "Optional JSON string mapping delay_type -> max_delay, e.g. "
            "'{\"gamma\": 4, \"uniform\": 7, \"doublegaussian\": 9}'. When provided, "
            "the max_delay is derived from the active delay_type so that switching "
            "the delay type automatically changes the delay parameter. "
            "When omitted, --initial-max-delay and --shifted-max-delay are used "
            "as explicit values for the pre-/post-shift phases."
        ),
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "delay_shift_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    type_max_delay_map: Optional[Dict[str, int]] = None
    if args.type_max_delay is not None:
        try:
            raw = json.loads(args.type_max_delay)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--type-max-delay must be a valid JSON object: {exc}")
        if not isinstance(raw, dict):
            raise SystemExit("--type-max-delay must decode to a JSON object mapping type -> int")
        type_max_delay_map = {str(k): int(v) for k, v in raw.items()}
        validate_type_max_delay_map(type_max_delay_map)

    eval_cfg = EvalConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        shift_step=args.shift_step,
        post_shift_steps=args.post_shift_steps,
        seed=args.seed,
        deterministic=not args.stochastic,
        initial_delay_type=args.initial_delay_type,
        shifted_delay_type=args.shifted_delay_type,
        initial_max_delay=args.initial_max_delay,
        shifted_max_delay=args.shifted_max_delay,
        initial_delay_process=args.initial_delay_process,
        shifted_delay_process=args.shifted_delay_process,
        delay_type_max_delay=type_max_delay_map,
    )

    param = load_parameter_from_run(run_dir)
    override_delay_eval_setup(param, eval_cfg)

    initial_max_delay = resolve_max_delay(eval_cfg.initial_delay_type, eval_cfg, "initial")
    shifted_max_delay = resolve_max_delay(eval_cfg.shifted_delay_type, eval_cfg, "shifted")
    print(
        f"[delay-shift-eval] strategy: "
        f"{eval_cfg.initial_delay_process}/"
        f"{eval_cfg.initial_delay_type}(max_delay={initial_max_delay}) -> "
        f"{eval_cfg.shifted_delay_process}/"
        f"{eval_cfg.shifted_delay_type}(max_delay={shifted_max_delay}) "
        f"at env step {eval_cfg.shift_step}; "
        f"type->max_delay binding: {eval_cfg.delay_type_max_delay}"
    )

    results: List[Dict[str, Any]] = []
    for episode in range(eval_cfg.episodes):
        worker = build_worker(param, seed=eval_cfg.seed + episode)
        result = evaluate_episode(
            worker, eval_cfg, episode,
            recovery_window=args.recovery_window,
            recovery_threshold=args.recovery_threshold
        )
        results.append(result)

    summary = summarize_results(results, recovery_window=args.recovery_window)
    payload = {
        "run_dir": str(run_dir),
        "evaluation": _to_builtin(eval_cfg.__dict__),
        "summary": summary,
        "episodes": [
            {k: _to_builtin(v) for k, v in result.items() if k != "rows"}
            for result in results
        ],
    }

    with (output_dir / "summary.json").open("w") as f:
        json.dump(payload, f, indent=2)

    step_csv_path = output_dir / "step_metrics.csv"
    write_step_csv(step_csv_path, results)
    plot_path = plot_recovery_curve(output_dir, eval_cfg, summary, results)

    print(json.dumps(payload, indent=2))
    print(f"step csv saved to: {step_csv_path}")
    print(f"summary saved to: {output_dir / 'summary.json'}")
    if plot_path is not None:
        print(f"recovery curve saved to: {plot_path}")
    else:
        print("recovery curve was not generated because no post-shift samples were found")


if __name__ == "__main__":
    main()
