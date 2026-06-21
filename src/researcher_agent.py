from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from src.config_loader import load_config, resolve_path


def read_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def extract_modifier(text: str) -> str:
    match = re.search(r"\[([^\]]+)\]", text)
    if match:
        return match.group(1).strip()
    return text.strip().strip('"').strip("'")


def build_prompt(base_instruction: str, last_log: str, history: str) -> str:
    return f"""You are an AI robotics researcher optimizing a Vision-Language-Action (VLA) model in simulation.
You are inspired by ENPIRE, a harness where coding agents improve robot policies through closed-loop physical feedback.

The current base task instruction is: "{base_instruction}"

Your job in this MVP is to propose a short modifier appended to that instruction to improve VLA success rate.
Examples: "carefully", "focusing on the handle", "approach slowly from above".

Here are the results of the last experiment:
{last_log}

Recent experiment history:
{history or "No prior history yet."}

Propose the next instruction modifier to test.
Output ONLY the modifier inside square brackets, like [modifier text].
Do not write code, explanations, or full sentences outside the brackets."""


def call_openai(prompt: str, model: str) -> str:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content or ""


def call_anthropic(prompt: str, model: str) -> str:
    from anthropic import Anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = [block.text for block in response.content if hasattr(block, "text")]
    return "\n".join(parts)


def consult_researcher(config_path: Path | None = None) -> str:
    config = load_config(config_path)
    logs_dir = resolve_path(config["paths"]["logs_dir"])
    latest_log_path = resolve_path(config["paths"]["latest_eval_log"])
    history_path = resolve_path(config["paths"]["history_tree_log"])

    last_log = read_text(
        latest_log_path,
        default="Success Rate: 0%\nNo previous experiment run yet.",
    )
    history = read_text(history_path)[-4000:]

    prompt = build_prompt(config["task"]["base_instruction"], last_log, history)
    provider = config["llm"]["provider"].lower()
    model = config["llm"]["model"]

    if provider == "anthropic":
        content = call_anthropic(prompt, model)
    else:
        content = call_openai(prompt, model)

    return extract_modifier(content)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consult LLM researcher for next prompt modifier.")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config.")
    args = parser.parse_args(argv)

    modifier = consult_researcher(config_path=args.config)
    print(modifier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
