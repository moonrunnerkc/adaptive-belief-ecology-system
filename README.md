
# ABES — Adaptive Belief Ecology System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-632%20passing-brightgreen.svg)]()

A research platform for **Belief Ecology**: treating beliefs as living, evolving entities instead of static memory entries.

Most LLM memory systems use key-value stores or vector retrieval. ABES implements something different—beliefs decay over time, contradict each other, get reinforced when similar evidence arrives, mutate under tension, and cluster around related concepts. Specialized agents manage this ecology, and reinforcement learning optimizes the control parameters.

---

## Overview

ABES is a Python research platform implementing a dynamic cognitive memory architecture. Beliefs are first-class objects with confidence scores, tension values (contradiction pressure), status lifecycle, and mutation lineage. A pipeline of 15 specialized agents processes beliefs each iteration: extracting claims from input, deduplicating, reinforcing similar evidence, applying decay, detecting contradictions, proposing mutations, resolving conflicts, and ranking by relevance.

An RL layer wraps this ecology as a Gymnasium-compatible environment, enabling policy optimization via Evolution Strategy training.

---

## Key Features

| Feature | Source | Tests |
|---------|--------|-------|
| 15 Specialized Agents | [backend/agents/](backend/agents/) | 394 |
| Agent Scheduler (14 phases) | [backend/agents/scheduler.py](backend/agents/scheduler.py) | 20 |
| Belief Ecology Loop | [backend/core/bel/loop.py](backend/core/bel/loop.py) | 40 |
| RL-BEL Integration | [backend/core/bel/rl_integration.py](backend/core/bel/rl_integration.py) | 24 |
| Snapshot Timeline | [backend/core/bel/timeline.py](backend/core/bel/timeline.py) | 10 |
| RL Environment (15-dim state, 7-dim action) | [backend/rl/environment.py](backend/rl/environment.py) | 20 |
| RL Policy (NumPy MLP) | [backend/rl/policy.py](backend/rl/policy.py) | 15 |
| RL Training (Evolution Strategy) | [backend/rl/training.py](backend/rl/training.py) | 15 |
| Semantic Clustering | [backend/core/bel/clustering.py](backend/core/bel/clustering.py) | 16 |
| REST API (FastAPI) | [backend/api/](backend/api/) | 21 |
| Benchmark Scenarios | [backend/benchmark/scenarios.py](backend/benchmark/scenarios.py) | 17 |
| Baseline Memory Systems | [backend/benchmark/baselines.py](backend/benchmark/baselines.py) | 16 |
| Metrics Export (JSON/CSV/Prometheus) | [backend/metrics/](backend/metrics/) | 20 |
| Snapshot Queries | [backend/storage/snapshot_queries.py](backend/storage/snapshot_queries.py) | 24 |

**Total: 632 tests passing.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Agent Scheduler                         │
│  (14 phases: Perception → Creation → ... → Experiment)      │
└─────────────────────────────────────────────────────────────┘
							  │
							  ▼
┌─────────────────────────────────────────────────────────────┐
│                    15 Specialized Agents                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │Perception│ │ Creator  │ │ Auditor  │ │ Mutation │  ...   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘        │
└─────────────────────────────────────────────────────────────┘
							  │
							  ▼
┌─────────────────────────────────────────────────────────────┐
│                     Storage Layer                           │
│  ┌──────────────────┐    ┌───────────────────┐              │
│  │  Belief Store    │    │  Snapshot Store   │              │
│  │  (in-memory)     │    │  (compressed)     │              │
│  └──────────────────┘    └───────────────────┘              │
└─────────────────────────────────────────────────────────────┘
							  │
							  ▼
┌─────────────────────────────────────────────────────────────┐
│                      RL Layer                               │
│  ┌───────────┐    ┌──────────┐    ┌──────────┐              │
│  │Environment│ ←→ │  Policy  │ ←→ │ Trainer  │              │
│  │(15d/7d)   │    │(NumPy MLP│    │   (ES)   │              │
│  └───────────┘    └──────────┘    └──────────┘              │
└─────────────────────────────────────────────────────────────┘
```

Key modules:
- [backend/core/models/belief.py](backend/core/models/belief.py) — Belief data model
- [backend/core/models/snapshot.py](backend/core/models/snapshot.py) — Snapshot with edge relationships
- [backend/core/config.py](backend/core/config.py) — All tunable parameters via pydantic-settings

---

## Installation

```bash
git clone https://github.com/moonrunnerkc/abes.git
cd abes

python -m venv .venv
source .venv/bin/activate
pip install numpy pydantic pydantic-settings msgpack sentence-transformers
pip install pytest pytest-asyncio  # for tests
```

**Note:** Editable install (`pip install -e .`) currently fails due to missing package discovery config. Use `PYTHONPATH` as a workaround:

```bash
export PYTHONPATH=$PWD
```

Requires Python 3.10+. First embedding use downloads `all-MiniLM-L6-v2` (~80MB).

---

## Quick Start

### Run Tests

```bash
PYTHONPATH=$PWD pytest tests/ -q
# 632 passed
```

### Train an RL Policy

```python
from backend.rl import BeliefEcologyEnv, ESTrainer, TrainingConfig

env = BeliefEcologyEnv()
config = TrainingConfig(total_generations=50, population_size=20)
trainer = ESTrainer(env, config)
metrics = trainer.train()

print(f"Best return: {metrics.best_return:.3f}")
trainer.save_policy("policy.npz")
```

### Start the API

```bash
PYTHONPATH=$PWD uvicorn backend.api.app:app --reload
# Docs at http://localhost:8000/docs
```

---

## Configuration

All parameters are configurable via environment variables or `ABESSettings`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DECAY_RATE` | 0.995 | Per-hour confidence multiplier |
| `CONFIDENCE_THRESHOLD_DECAYING` | 0.3 | Status transition threshold |
| `TENSION_THRESHOLD_MUTATION` | 0.6 | Trigger for mutation proposals |
| `CLUSTER_SIMILARITY_THRESHOLD` | 0.7 | Min similarity to join cluster |
| `MAX_ACTIVE_BELIEFS` | 10,000 | Safety limit |
| `MAX_MUTATION_DEPTH` | 5 | Lineage cap |

See [backend/core/config.py](backend/core/config.py) for the full list.

---

## Testing and Verification

| Test Suite | Count | What It Verifies |
|------------|-------|------------------|
| `tests/agents/` | 394 | All 15 agents + scheduler behavior |
| `tests/core/` | 90 | BEL loop, clustering, timeline, RL integration |
| `tests/rl/` | 50 | Environment, policy, ES training |
| `tests/api/` | 21 | REST endpoint correctness |
| `tests/benchmark/` | 33 | Scenario generation, baseline memory |
| `tests/metrics/` | 20 | Metric computation, export formats |
| `tests/storage/` | 24 | Snapshot queries and retrieval |

Run all tests:
```bash
PYTHONPATH=$PWD pytest tests/ -v
```

### Test Output Summary

```
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.0.1
collected 632 items
632 passed in 4.01s
==============================
```

---

## Roadmap

- [x] ~~Add tests for `BeliefEcologyLoop`~~ ✅ (40 tests added)
- [x] ~~Add tests for snapshot timeline and queries~~ ✅ (34 tests added)
- [x] ~~Add tests for RL-BEL integration~~ ✅ (24 tests added)
- [ ] Fix editable install (`pyproject.toml` package discovery)
- [ ] Add CI workflow (GitHub Actions)
- [ ] Frontend tests
- [ ] Persistent storage backend (SQLite/PostgreSQL)
- [ ] Experiment tracking integration (MLflow/W&B)

---

## License

MIT © 2025-2026 Bradley R. Kinnard
