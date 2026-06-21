from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config_loader import load_config, resolve_path
from src.env import create_env
from src.vla import create_vla_adapter


def run_episode(env, vla, instruction: str, max_steps: int) -> tuple[bool, int, float]:
    image, _ = env.reset()
    total_reward = 0.0
    steps = 0
    success = False

    for _ in range(max_steps):
        action = vla.predict_action(image, instruction)
        image, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if info.get("success") or reward >= 1.0:
            success = True
        if done:
            break

    return success, steps, total_reward


def run_evaluation(
    prompt_modifier: str = "",
    iteration: int | None = None,
    config_path: Path | None = None,
) -> dict:
    config = load_config(config_path)
    env_backend = config["env"]["backend"]
    vla_backend = config["vla"]["backend"]
    base_instruction = config["task"]["base_instruction"]
    episodes = int(config["task"]["episodes"])
    final_instruction = f"{base_instruction} {prompt_modifier}".strip()

    env = create_env(env_backend, simpler_task=config["env"].get("simpler_task"))
    vla = create_vla_adapter(
        vla_backend,
        model_id=config["vla"].get("model_id"),
        policy_setup=config["vla"].get("policy_setup"),
        action_dim=getattr(env, "action_dim", 2),
    )

    successes = 0
    total_steps = 0
    max_steps = getattr(env, "max_episode_steps", 300)

    try:
        for _ in range(episodes):
            success, steps, _ = run_episode(env, vla, final_instruction, max_steps)
            successes += int(success)
            total_steps += steps
    finally:
        env.close()

    success_rate = successes / episodes if episodes else 0.0
    avg_steps = total_steps / episodes if episodes else 0.0

    result = {
        "iteration": iteration,
        "prompt_modifier": prompt_modifier,
        "final_instruction": final_instruction,
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "episodes": episodes,
        "backend": f"{vla.name}+{env.name}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    log_path = resolve_path(config["paths"]["latest_eval_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        for key, value in result.items():
            f.write(f"{key}: {value}\n")

    return result


def format_result(result: dict) -> str:
    lines = [f"{key}: {value}" for key, value in result.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run VLA simulation evaluation.")
    parser.add_argument("--modifier", default="", help="Prompt modifier from researcher agent.")
    parser.add_argument("--iteration", type=int, default=None, help="Current loop iteration.")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config.")
    args = parser.parse_args(argv)

    result = run_evaluation(
        prompt_modifier=args.modifier,
        iteration=args.iteration,
        config_path=args.config,
    )
    print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
