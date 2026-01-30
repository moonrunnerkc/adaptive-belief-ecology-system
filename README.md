# ABES — Adaptive Belief Ecology System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-636%20passing-brightgreen.svg)]()

A research platform implementing **belief ecology** — treating beliefs as living, evolving entities rather than static memory entries.

---

## Overview

ABES is an experimental cognitive memory architecture for AI systems. Unlike key-value or vector-based memory stores, ABES manages beliefs as first-class objects that:

- **Decay** over time without reinforcement
- **Accumulate tension** when contradictions are detected
- **Get reinforced** when similar evidence arrives
- **Mutate or deprecate** when tension exceeds thresholds

A pipeline of specialized agents processes beliefs each iteration, with optional reinforcement learning to tune system parameters.

**Status:** Research prototype. Not production-ready.

---

## Key Features

| Feature | Source | Tests |
|---------|--------|-------|
| Belief data model (confidence, tension, status, lineage) | [backend/core/models/belief.py](backend/core/models/belief.py) | [test_bel_loop.py](tests/core/test_bel_loop.py) |
| 14-phase agent scheduler | [backend/agents/scheduler.py](backend/agents/scheduler.py) | [test_scheduler.py](tests/agents/test_scheduler.py) |
| Perception agent (text → belief candidates) | [backend/agents/perception.py](backend/agents/perception.py) | [test_perception.py](tests/agents/test_perception.py) |
| Reinforcement agent (boost on similar evidence) | [backend/agents/reinforcement.py](backend/agents/reinforcement.py) | [test_reinforcement.py](tests/agents/test_reinforcement.py) |
| Decay controller (time-based confidence reduction) | [backend/agents/decay_controller.py](backend/agents/decay_controller.py) | [test_decay_controller.py](tests/agents/test_decay_controller.py) |
| Contradiction auditor (embedding + antonym detection) | [backend/agents/contradiction_auditor.py](backend/agents/contradiction_auditor.py) | [test_contradiction_auditor.py](tests/agents/test_contradiction_auditor.py) |
| Mutation engineer (conflict-triggered belief modification) | [backend/agents/mutation_engineer.py](backend/agents/mutation_engineer.py) | [test_mutation_engineer.py](tests/agents/test_mutation_engineer.py) |
| Semantic clustering | [backend/core/bel/clustering.py](backend/core/bel/clustering.py) | [test_clustering.py](tests/core/test_clustering.py) |
| RL environment (15D state, 7D action) | [backend/rl/environment.py](backend/rl/environment.py) | [test_environment.py](tests/rl/test_environment.py) |
| Evolution Strategy trainer | [backend/rl/training.py](backend/rl/training.py) | [test_training.py](tests/rl/test_training.py) |
| FastAPI REST + WebSocket API | [backend/api/app.py](backend/api/app.py) | [test_routes.py](tests/api/test_routes.py) |
| Chat service with Ollama LLM | [backend/chat/service.py](backend/chat/service.py) | Manual testing only |
| Next.js frontend | [frontend/](frontend/) | Manual testing only |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                       │
│                    localhost:3000/chat                      │
└─────────────────────────────────────────────────────────────┘
                              │ REST + WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend (:8000)                   │
│   /beliefs  /chat  /bel  /clusters  /snapshots  /agents    │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│   Chat Service  │ │  Agent Scheduler│ │   RL Environment    │
│ (Ollama LLM)    │ │  (14 phases)    │ │ (Gymnasium-compat)  │
└─────────────────┘ └─────────────────┘ └─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    In-Memory Belief Store                   │
│              (no persistence across restarts)               │
└─────────────────────────────────────────────────────────────┘
```

### Agent Execution Pipeline

```
Perception → Creation → Reinforcement → Decay → Contradiction →
Mutation → Resolution → Relevance → RL Policy → Consistency →
Safety → Baseline → Narrative → Experiment
```

Each agent is independently tested. See [backend/agents/](backend/agents/).

---

## Installation

**Requirements:** Python 3.10+, Node.js 18+ (frontend), Ollama (chat)

### Backend

```bash
git clone https://github.com/moonrunnerkc/adaptive-belief-ecology-system.git
cd adaptive-belief-ecology-system

python -m venv .venv
source .venv/bin/activate
pip install numpy pydantic pydantic-settings msgpack sentence-transformers httpx
pip install pytest pytest-asyncio  # dev dependencies

export PYTHONPATH=$PWD
```

### Frontend

```bash
cd frontend
npm install
```

### Ollama (required for chat)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b-instruct-q4_0
```

---

## Quick Start

**Terminal 1 — Backend:**
```bash
source .venv/bin/activate
PYTHONPATH=$PWD uvicorn backend.api.app:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend && npm run dev
```

**Terminal 3 — Ollama:**
```bash
ollama serve
```

Open http://localhost:3000/chat

---

## Configuration

All parameters are set via environment variables or [backend/core/config.py](backend/core/config.py).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DECAY_RATE` | 0.995 | Per-hour confidence multiplier |
| `CONFIDENCE_THRESHOLD_DECAYING` | 0.3 | Threshold to mark belief as decaying |
| `TENSION_THRESHOLD_MUTATION` | 0.5 | Trigger mutation proposals |
| `CLUSTER_SIMILARITY_THRESHOLD` | 0.7 | Min similarity to join cluster |
| `REINFORCEMENT_SIMILARITY_THRESHOLD` | 0.7 | Min similarity for reinforcement |
| `OLLAMA_MODEL` | llama3.1:8b-instruct-q4_0 | LLM model for chat |
| `MAX_ACTIVE_BELIEFS` | 10000 | Safety limit |

---

## Testing and Verification

### Unit Tests

```bash
source .venv/bin/activate
PYTHONPATH=$PWD pytest tests/ -q
```

**Current status:** 636 passed, 2 failed

| Suite | Count | Scope |
|-------|-------|-------|
| `tests/agents/` | 18 files | All agent modules |
| `tests/core/` | 5 files | BEL loop, clustering, timeline, RL integration |
| `tests/rl/` | 3 files | Environment, policy, training |
| `tests/api/` | 1 file | REST endpoints |
| `tests/verification/` | 3 files | Determinism, offline, conflict resolution |

### Verification Suite

Run deterministic experiments that produce machine-readable evidence:

```bash
PYTHONPATH=$PWD python experiments/run_all.py
```

**Generated artifacts:**

| File | Purpose |
|------|---------|
| [results/determinism_check.json](results/determinism_check.json) | Proves identical inputs → identical state hashes |
| [results/offline_verification.json](results/offline_verification.json) | Proves core runs without network access |
| [results/conflict_resolution_log.json](results/conflict_resolution_log.json) | Documents resolution decisions |
| [results/drift_comparison.json](results/drift_comparison.json) | Compares belief ecology vs baselines |
| [results/decay_sweep/](results/decay_sweep/) | Decay factor sensitivity analysis |

### Verified Properties

| Property | Evidence | Result |
|----------|----------|--------|
| Determinism | [determinism_check.json](results/determinism_check.json) | `deterministic: true` |
| Offline operation | [offline_verification.json](results/offline_verification.json) | `network_calls_detected: 0` |
| Conflict resolution consistency | [conflict_resolution_log.json](results/conflict_resolution_log.json) | 4/4 test cases passed |

---

## Limitations

- **In-memory only** — All state lost on server restart
- **Single-user** — No session isolation or authentication
- **No CI/CD** — Tests run locally only
- **Ollama-only LLM** — No OpenAI/Anthropic integration
- **Embedding model fixed** — Uses `all-MiniLM-L6-v2`
- **2 failing tests** — mutation_engineer, rl_environment edge cases

---

## Roadmap

Future work not yet implemented:

- [ ] Persistent storage (SQLite/PostgreSQL)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Multi-user session support
- [ ] Belief Explorer UI
- [ ] Document ingestion service
- [ ] OpenAI/Anthropic LLM providers
- [ ] Benchmark against production memory systems

---

## License

MIT © 2025-2026 Bradley R. Kinnard
