# ABES — Adaptive Belief Ecology System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-514%20passing-brightgreen.svg)]()

A research platform for **Belief Ecology**: treating beliefs as living, evolving entities instead of static memory entries.

Most LLM memory systems use key-value stores or vector retrieval. ABES implements something different—beliefs decay over time, contradict each other, get reinforced when similar evidence arrives, mutate under tension, and cluster around related concepts. Specialized agents manage this ecology, and reinforcement learning optimizes the control parameters.

---

## What This Is

ABES is a Python research platform implementing:

- **Belief data model** with confidence, tension, status, tags, and lineage tracking
- **15 specialized agents** for perception, mutation, resolution, decay, consistency, etc.
- **Agent scheduler** with 14-phase execution order
- **RL environment** (15-dim state, 7-dim action) with Evolution Strategy training
- **Snapshot system** capturing belief graph edges (contradiction, support, lineage)
- **Semantic clustering** with incremental assignment and automatic maintenance
- **REST API** with FastAPI for beliefs, snapshots, agents, and clusters
- **In-memory storage** for beliefs and snapshots

## What This Is Not

- No frontend exists (placeholder folder structure only)
- No benchmark scenarios defined
- No metrics collection implemented

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) — dev setup, testing, code style
- [CHANGELOG.md](CHANGELOG.md) — version history, what's built/missing
- [docs/architecture.md](docs/architecture.md) — system design, data flow, module map
- [docs/agents.md](docs/agents.md) — detailed reference for all 15 agents

---

## Current Status

| Component | Status | Tests |
|-----------|--------|-------|
| 15 Specialized Agents | ✅ Implemented | 361 tests |
| Agent Scheduler | ✅ Implemented | 20 tests |
| Safety Limits (spec 3.8) | ✅ Implemented | 18 tests |
| Belief Clustering | ✅ Implemented | 16 tests |
| REST API (FastAPI) | ✅ Implemented | 21 tests |
| RL Environment | ✅ Implemented | 20 tests |
| RL Training (ES) | ✅ Implemented | 10 tests |
| Belief/Snapshot Models | ✅ Implemented | — |
| Snapshot Edges | ✅ Implemented | — |
| Core BEL Loop | ⚠️ Code exists | **Untested** |
| Benchmark System | ✅ Implemented | 33 tests |
| Frontend | ❌ Not started | — |

**514 tests passing** — agents, clustering, API, safety, RL, benchmarks, and LLM.

---

## Agents

All 15 agents are fully implemented with tests:

| Agent | Purpose |
|-------|---------|
| PerceptionAgent | Extracts claims from raw input |
| BeliefCreatorAgent | Creates beliefs with deduplication |
| ReinforcementAgent | Strengthens beliefs on similar evidence |
| ContradictionAuditorAgent | Computes pairwise tensions |
| MutationEngineerAgent | Proposes hedged/conditional variants |
| ResolutionStrategistAgent | Integrates, splits, or deprecates |
| RelevanceCuratorAgent | Ranks by context similarity |
| DecayControllerAgent | Adjusts decay rates per belief |
| BaselineMemoryBridgeAgent | Interfaces with RAG/chat history |
| RLPolicyAgent | Outputs action from ecology state |
| RewardShaperAgent | Computes RL reward signal |
| ExperimentOrchestratorAgent | Runs scenarios and logs results |
| ConsistencyCheckerAgent | Probes for answer drift |
| NarrativeExplainerAgent | Generates human-readable explanations |
| SafetySanityAgent | Enforces limits and vetoes |

Agents run in a defined 14-phase order via `AgentScheduler`.

---

## RL Layer

**Environment** (`backend/rl/environment.py`):
- 15-dimensional state (confidence stats, tension, belief counts, etc.)
- 7-dimensional action (decay rate, thresholds, ranking weights)
- Reward shaped from task success, consistency, efficiency, stability

**Policy** (`backend/rl/policy.py`):
- Pure NumPy MLP with tanh activations
- Save/load to `.npz` files
- No PyTorch dependency

**Training** (`backend/rl/training.py`):
- Evolution Strategy (ES) optimizer
- Population-based search with gradient-free updates
- Checkpointing and early stopping

---

## Repository Structure

```
backend/
├── agents/           # 16 files: 15 agents + scheduler
├── core/
│   ├── bel/          # Loop, decay, ranking, compression, timeline
│   └── models/       # Belief, Snapshot models
├── rl/               # Environment, Policy, Training
├── storage/          # Abstract base + in-memory implementations
├── api/              # (empty)
├── benchmark/        # (empty)
├── metrics/          # (empty)
└── util/             # (empty)

frontend/             # (placeholder stubs only)
configs/              # (empty)
experiments/          # (empty)
tests/
├── agents/           # 16 test files, 361 tests
└── rl/               # 3 test files, 50 tests
```

---

## Installation

```bash
git clone https://github.com/moonrunnerkc/abes.git
cd abes

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Verify
pytest tests/ -q
# 411 passed
```

Requires Python 3.10+ and downloads `all-MiniLM-L6-v2` (~80MB) on first embedding use.

---

## Usage

```python
from backend.rl import BeliefEcologyEnv, ESTrainer, TrainingConfig

# Create environment
env = BeliefEcologyEnv()

# Train a policy
config = TrainingConfig(total_generations=50, population_size=20)
trainer = ESTrainer(env, config)
metrics = trainer.train()

print(f"Best return: {metrics.best_return:.3f}")

# Save trained policy
trainer.save_policy("policy.npz")
```

---

## Formulas

All agents implement formulas from the spec:

- **Decay**: `confidence *= decay_rate ^ hours_elapsed`
- **Contradiction**: `semantic_similarity × negation_signal`
- **Tension**: `max(contradiction_score)` across pairs
- **Ranking**: `0.4×relevance + 0.3×confidence + 0.2×recency - 0.1×tension`

---

## Known Gaps

1. `BeliefEcologyLoop` in `backend/core/bel/loop.py` has no direct tests
2. Snapshot compression, timeline, and query utilities are untested
3. No API, frontend, benchmarks, or metrics exist
4. No cluster maintenance (merge/split) implemented

---

## License

MIT © 2025-2026 Bradley R. Kinnard
