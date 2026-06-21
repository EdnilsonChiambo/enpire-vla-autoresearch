from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.config_loader import load_config, resolve_path
from src.evaluator import format_result, run_evaluation


def seed_initial_log(latest_log_path: Path) -> None:
    latest_log_path.parent.mkdir(parents=True, exist_ok=True)
    if not latest_log_path.exists():
        latest_log_path.write_text(
            "success_rate: 0.0\nNo previous experiment run yet.\n",
            encoding="utf-8",
        )


def append_history(history_path: Path, iteration: int, result_text: str) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as history:
        history.write(f"\nIteration {iteration}:\n{result_text}\n{'=' * 20}\n")


def update_best_run(best_run_path: Path, modifier: str, result: dict) -> None:
    best_run_path.parent.mkdir(parents=True, exist_ok=True)
    current_best = {"success_rate": -1.0, "prompt_modifier": "", "final_instruction": ""}
    if best_run_path.exists():
        current_best = json.loads(best_run_path.read_text(encoding="utf-8"))

    if result["success_rate"] >= current_best.get("success_rate", -1.0):
        payload = {
            "success_rate": result["success_rate"],
            "prompt_modifier": modifier,
            "final_instruction": result["final_instruction"],
            "avg_steps": result["avg_steps"],
            "backend": result["backend"],
            "timestamp": result["timestamp"],
        }
        best_run_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_researcher_modifier(config_path: Path | None, dry_run: bool) -> str:
    if dry_run:
        modifiers = [
            "carefully",
            "focusing on the handle",
            "approach slowly from above",
            "align precisely before grasping",
            "use a steady pushing motion",
        ]
        return modifiers[0]

    cmd = [sys.executable, "-m", "src.researcher_agent"]
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])
    output = subprocess.check_output(cmd, text=True).strip()
    return output.splitlines()[-1].strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ENPIRE-style VLA autoresearch loop.")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls; use fixed modifiers.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    iterations = int(config["loop"]["iterations"])
    latest_log_path = resolve_path(config["paths"]["latest_eval_log"])
    history_path = resolve_path(config["paths"]["history_tree_log"])
    best_run_path = resolve_path(config["paths"]["best_run"])

    seed_initial_log(latest_log_path)

    for i in range(iterations):
        print(f"\n--- STARTING ITERATION {i + 1} ---")

        print("Consulting AI Researcher...")
        modifier = get_researcher_modifier(args.config, dry_run=args.dry_run)
        print(f"AI Researcher proposed modifier: '{modifier}'")

        print("Running simulator evaluation...")
        if args.dry_run:
            result = run_evaluation(
                prompt_modifier=modifier,
                iteration=i + 1,
                config_path=args.config,
            )
        else:
            cmd = [
                sys.executable,
                "-m",
                "src.evaluator",
                "--modifier",
                modifier,
                "--iteration",
                str(i + 1),
            ]
            if args.config is not None:
                cmd.extend(["--config", str(args.config)])
            subprocess.run(cmd, check=True)
            result_text = latest_log_path.read_text(encoding="utf-8")
            result = {}
            for line in result_text.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    if key in {"iteration", "episodes"}:
                        result[key] = int(value)
                    elif key in {"success_rate", "avg_steps"}:
                        result[key] = float(value)
                    else:
                        result[key] = value

        result_text = format_result(result)
        append_history(history_path, i + 1, result_text)
        update_best_run(best_run_path, modifier, result)

        print(result_text)

    print("\nMVP Loop Complete! Review logs/history_tree.log to see the progression.")
    if best_run_path.exists():
        print(f"Best run saved to {best_run_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
