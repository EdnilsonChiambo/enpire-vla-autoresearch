# ENPIRE VLA Autoresearch MVP

A minimal **agentic autoresearch loop** for optimizing open-source Vision-Language-Action (VLA) models in simulation, inspired by [ENPIRE: Agentic Robot Policy Self-Improvement in the Real World](https://research.nvidia.com/labs/gear/enpire/).

An LLM acts as the **Researcher**: it reads evaluation logs, proposes a prompt modifier, runs a fast simulation rollout, and appends results to a local idea tree.

## Architecture

```mermaid
flowchart LR
  subgraph E [Evolution Module]
    HistoryTree[history_tree.log]
    Researcher[researcher_agent.py]
  end
  subgraph PI [Policy Improvement]
    PromptMod[prompt_modifier hypothesis]
  end
  subgraph R [Rollout Module]
    Evaluator[evaluator.py]
  end
  subgraph EN [Environment Module]
    Sim[SIMPLER or Gym-PushT]
    Verify[binary success check]
  end

  Researcher -->|reads latest_eval_log.txt| PromptMod
  PromptMod --> Evaluator
  Evaluator --> Sim
  Sim --> Verify
  Verify -->|writes latest_eval_log.txt| HistoryTree
  HistoryTree --> Researcher
  LoopRunner[loop_runner.py] --> Researcher
  LoopRunner --> Evaluator
```

## ENPIRE Module Mapping

| ENPIRE Module | MVP Component | MVP behavior | Future upgrade |
|---|---|---|---|
| **EN** (Environment) | `src/evaluator.py` env wrapper | `env.reset()`, binary success from sim state | Perception tools (SAM/YOLO) for pixel-based verify |
| **R** (Rollout) | `src/evaluator.py` episode loop | N episodes, log steps + success | Parallel rollouts, video saving |
| **PI** (Policy Improvement) | `src/researcher_agent.py` | Prompt modifier generation | LoRA training config edits, data sampler tuning |
| **E** (Evolution) | `src/loop_runner.py` + `logs/history_tree.log` | Append-only experiment log, LLM reads history | Git-branch idea tree, prune bad runs, multi-agent |

## Repository Layout

```
enpire-vla-autoresearch/
├── config/
│   ├── default.yaml          # local dev defaults
│   ├── kaggle.yaml           # T4 + openvla-4bit + simpler (spoon task)
│   └── kaggle-demo.yaml      # carrot-on-plate demo (more episodes, clearer prompt deltas)
├── notebooks/
│   └── kaggle_setup.ipynb    # one-click Kaggle setup
├── src/
│   ├── loop_runner.py        # orchestrator
│   ├── researcher_agent.py   # LLM evolution (PI + E)
│   ├── evaluator.py          # env + rollout (EN + R)
│   ├── env/                  # PushT and SIMPLER wrappers
│   └── vla/                  # mock, OpenVLA bf16/4bit, Octo adapters
├── logs/                     # runtime output (gitignored)
└── scripts/                  # setup helpers
```

## Quick Start (Mac, ~2 min)

No GPU required. Uses **Gym-PushT** + **mock VLA** to validate the full loop.

```bash
cd enpire-vla-autoresearch
bash scripts/setup_mac.sh
source .venv/bin/activate

# Validate loop without LLM API calls
python -m src.loop_runner --dry-run
```

Review output in:
- `logs/latest_eval_log.txt` — most recent experiment
- `logs/history_tree.log` — full idea tree
- `logs/best_run.json` — best modifier so far

## Full Loop with LLM Researcher

```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env

python -m src.loop_runner
```

Or run components individually:

```bash
python -m src.researcher_agent          # proposes next modifier
python -m src.evaluator --modifier "carefully focusing on the handle"
```

## GPU Path: SIMPLER + OpenVLA

For the real VLA experiments (e.g. "pick up the red mug"), use an NVIDIA GPU (T4/3090+).

```bash
bash scripts/setup_gpu.sh
pip install flash-attn==2.6.1 --no-build-isolation

git clone https://github.com/simpler-env/SimplerEnv ../SimplerEnv
cd ../SimplerEnv && pip install -e .
```

Set in `.env`:

```bash
VLA_BACKEND=openvla
ENV_BACKEND=simpler
OPENAI_API_KEY=sk-...
```

Update `config/default.yaml` if needed, then:

```bash
python -m src.loop_runner
```

**Memory note:** OpenVLA-7B (bf16) needs ~16GB VRAM. On **16GB GPUs (Kaggle T4, Colab T4)**, use `openvla-4bit` instead (see below).

## Kaggle Free GPU (T4 x2)

Kaggle gives **2× Tesla T4 (16GB each)** and **~30 GPU hours/week**. This repo uses **one T4** with **4-bit OpenVLA** (~6–7GB VRAM), leaving headroom for SIMPLER.

### Fastest path: upload the notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Settings → Accelerator → **GPU T4 x2**, enable **Internet**
3. Add-ons → Secrets → `OPENAI_API_KEY`
4. Upload or copy [`notebooks/kaggle_setup.ipynb`](notebooks/kaggle_setup.ipynb)
5. Run all cells

### Manual setup on Kaggle

```bash
git clone https://github.com/EdnilsonChiambo/enpire-vla-autoresearch.git
cd enpire-vla-autoresearch
pip install -r requirements-kaggle.txt

# Optional: SIMPLER
git clone --depth 1 https://github.com/simpler-env/SimplerEnv ../SimplerEnv
pip install -e ../SimplerEnv

export VLA_BACKEND=openvla-4bit
export ENV_BACKEND=simpler
export OPENAI_API_KEY=sk-...

python -m src.loop_runner --config config/kaggle.yaml
```

### Demo config (better prompt-modifier signal)

For a Bridge-aligned task where OpenVLA is more likely to show non-zero success and visible differences between LLM modifiers, use [`config/kaggle-demo.yaml`](config/kaggle-demo.yaml):

- Task: `PutCarrotOnPlateInScene-v1` (`widowx_carrot_on_plate`)
- 20 episodes × 5 iterations (slower, but statistically meaningful)
- Logs written to `logs/kaggle_demo/` (separate from the spoon run)

```bash
python -m src.loop_runner --config config/kaggle-demo.yaml
```

### VLA backend options

| Backend | VRAM | Best for |
|---|---|---|
| `mock` | 0 | Loop testing (Mac / Kaggle CPU) |
| `openvla-4bit` | ~6–7GB | **Kaggle T4, Colab T4** |
| `openvla` / `openvla-bf16` | ~16GB | RTX 3090 / A100 |
| `octo` | ~2–4GB | Lightweight alternative |

**Kaggle tips:**
- Do not spawn a second GPU process — keep env + VLA in one Python process
- Skip `flash-attn` on Kaggle; 4-bit adapter uses `sdpa` attention
- Pin `accelerate==0.26.0` if you hit quantized `.to()` errors (already in `requirements-kaggle.txt`)


Edit [`config/default.yaml`](config/default.yaml):

```yaml
task:
  base_instruction: "pick up the red mug"
  episodes: 5
loop:
  iterations: 5
env:
  backend: pusht          # pusht | simpler
vla:
  backend: mock           # mock | openvla-4bit | openvla | octo
llm:
  provider: openai        # openai | anthropic
  model: gpt-4o-mini
```

Environment variables in `.env` override `env.backend` and `vla.backend`.

## Two-Track Setup

| Track | Environment | VLA | Hardware |
|---|---|---|---|
| **Dev** | Gym-PushT | mock | Mac / CPU |
| **Kaggle** | SIMPLER | openvla-4bit | 1× T4 (16GB) |
| **Research** | SIMPLER | openvla (bf16) | RTX 3090+ |

## Roadmap

- [ ] LoRA fine-tuning instead of prompt-only optimization
- [ ] Octo-small adapter (scaffold included)
- [ ] Idea-tree visualization (ENPIRE-style git tree)
- [ ] RoboCasa simulation tasks
- [ ] Multi-agent parallel branches
- [ ] MRU / MTU resource utilization metrics

## References

- [ENPIRE paper & demo](https://research.nvidia.com/labs/gear/enpire/)
- [Gym-PushT](https://github.com/huggingface/gym-pusht)
- [SIMPLER](https://github.com/simpler-env/SimplerEnv)
- [OpenVLA](https://github.com/openvla/openvla)
- [Octo](https://github.com/octo-models/octo)

## License

MIT
