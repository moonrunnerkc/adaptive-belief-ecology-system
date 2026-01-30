
# ABES — Adaptive Belief Ecology System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-632%20passing-brightgreen.svg)]()

A research platform for **Belief Ecology**: treating beliefs as living, evolving entities instead of static memory entries.

Most LLM memory systems use key-value stores or vector retrieval. ABES implements something different—beliefs decay over time, contradict each other, get reinforced when similar evidence arrives, mutate under tension, and cluster around related concepts. Specialized agents manage this ecology, and reinforcement learning optimizes the control parameters.

---

## 🚀 What's New

### Chat Interface with Ollama LLM
ABES now includes a fully functional **chatbot** powered by a local LLM (Ollama). The chat system:
- Extracts beliefs from your messages in real-time
- Uses your stored beliefs to provide personalized, contextual responses
- Shows live belief activity (created, reinforced, evolved, tensions)
- Streams events via WebSocket for real-time UI updates

### Dashboard Hub
A central dashboard at `/` provides access to all ABES services:
- **Chat** — Conversational AI with evolving memory (active)
- **Documents** — Upload and analyze documents (coming soon)
- **Belief Explorer** — Browse and manage the belief ecology (coming soon)
- **Integrations** — External data sources via webhooks/APIs (coming soon)

### Modern UI
Clean, professional dark theme inspired by Grok:
- Black/grey color scheme with minimal semantic colors
- 3-panel chat layout: sidebar, chat area, belief activity panel
- Real-time belief event streaming
- Responsive and accessible design

---

## Overview

ABES is a Python research platform implementing a dynamic cognitive memory architecture. Beliefs are first-class objects with confidence scores, tension values (contradiction pressure), status lifecycle, and mutation lineage. A pipeline of 15 specialized agents processes beliefs each iteration: extracting claims from input, deduplicating, reinforcing similar evidence, applying decay, detecting contradictions, proposing mutations, resolving conflicts, and ranking by relevance.

An RL layer wraps this ecology as a Gymnasium-compatible environment, enabling policy optimization via Evolution Strategy training.

---

## Key Features

| Feature | Source | Description |
|---------|--------|-------------|
| **Chat Interface** | [backend/chat/](backend/chat/) | Conversational AI with belief memory |
| **LLM Integration** | [backend/llm/](backend/llm/) | Ollama provider with belief context |
| **Dashboard UI** | [frontend/](frontend/) | Next.js dashboard with service hub |
| **15 Specialized Agents** | [backend/agents/](backend/agents/) | Perception, creation, reinforcement, decay, etc. |
| **Agent Scheduler** | [backend/agents/scheduler.py](backend/agents/scheduler.py) | 14-phase execution pipeline |
| **Belief Ecology Loop** | [backend/core/bel/loop.py](backend/core/bel/loop.py) | Core iteration engine |
| **RL Environment** | [backend/rl/environment.py](backend/rl/environment.py) | Gymnasium-compatible (15d state, 7d action) |
| **RL Training** | [backend/rl/training.py](backend/rl/training.py) | Evolution Strategy optimization |
| **Semantic Clustering** | [backend/core/bel/clustering.py](backend/core/bel/clustering.py) | Belief grouping by similarity |
| **REST API** | [backend/api/](backend/api/) | FastAPI with WebSocket support |
| **Benchmark Scenarios** | [backend/benchmark/](backend/benchmark/) | Scenario generation and baselines |
| **Metrics Export** | [backend/metrics/](backend/metrics/) | JSON, CSV, Prometheus formats |

**Total: 632 tests passing.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Next.js)                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │Dashboard │ │   Chat   │ │ Explorer │ │  Docs    │  ...  │
│  │   Hub    │ │Interface │ │ (soon)   │ │ (soon)   │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │ REST + WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                    │
│  │ Chat API │ │Belief API│ │ Stats API│                    │
│  └──────────┘ └──────────┘ └──────────┘                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM Layer (Ollama)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Chat completion with belief context injection        │  │
│  │ Transforms beliefs to user-perspective for clarity   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Agent Scheduler                         │
│  (14 phases: Perception → Creation → ... → Experiment)      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    15 Specialized Agents                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │Perception│ │ Creator  │ │ Auditor  │ │ Mutation │  ...  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Storage Layer                           │
│  ┌──────────────────┐    ┌───────────────────┐             │
│  │  Belief Store    │    │  Snapshot Store   │             │
│  │  (in-memory)     │    │  (compressed)     │             │
│  └──────────────────┘    └───────────────────┘             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      RL Layer                               │
│  ┌───────────┐    ┌──────────┐    ┌──────────┐             │
│  │Environment│ ←→ │  Policy  │ ←→ │ Trainer  │             │
│  │(15d/7d)   │    │(NumPy MLP│    │   (ES)   │             │
│  └───────────┘    └──────────┘    └──────────┘             │
└─────────────────────────────────────────────────────────────┘
```

Key modules:
- [backend/core/models/belief.py](backend/core/models/belief.py) — Belief data model
- [backend/core/models/snapshot.py](backend/core/models/snapshot.py) — Snapshot with edge relationships
- [backend/core/config.py](backend/core/config.py) — All tunable parameters via pydantic-settings

---

## Installation

### Backend

```bash
git clone https://github.com/moonrunnerkc/adaptive-belief-ecology-system.git
cd adaptive-belief-ecology-system

python -m venv .venv
source .venv/bin/activate
pip install numpy pydantic pydantic-settings msgpack sentence-transformers httpx
pip install pytest pytest-asyncio  # for tests
```

### Frontend

```bash
cd frontend
npm install
```

### LLM (Ollama)

Install [Ollama](https://ollama.ai/) and pull a model:

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (llama3.1 8B recommended)
ollama pull llama3.1:8b-instruct-q4_0
```

**Note:** Set `PYTHONPATH` for running without editable install:

```bash
export PYTHONPATH=$PWD
```

Requires Python 3.10+, Node.js 18+.

---

## Quick Start

### 1. Start the Backend API

```bash
source .venv/bin/activate
PYTHONPATH=$PWD uvicorn backend.api.app:app --host 0.0.0.0 --port 8000
# API docs at http://localhost:8000/docs
```

### 2. Start the Frontend

```bash
cd frontend
npm run dev
# Dashboard at http://localhost:3000
# Chat at http://localhost:3000/chat
```

### 3. Start Ollama (if not running)

```bash
ollama serve
```

### Using the Chat

1. Open http://localhost:3000/chat
2. Send messages like "My name is Brad and I have a dog named Max"
3. Watch beliefs appear in the Activity panel
4. Ask "What do you know about me?" to see the system's memory

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

### Completed ✅
- [x] Core belief ecology with 15 specialized agents
- [x] RL environment and Evolution Strategy training
- [x] REST API with FastAPI
- [x] **Chat interface with Ollama LLM integration**
- [x] **Dashboard hub with service navigation**
- [x] **Real-time belief activity panel via WebSocket**
- [x] **Perception agent for conversational content**
- [x] **User-perspective belief context for LLM**
- [x] 632 tests passing

### In Progress 🔄
- [ ] Belief Explorer UI (browse, search, manage beliefs)
- [ ] Document ingestion service
- [ ] Belief reinforcement from repeated mentions

### Planned 📋
- [ ] Persistent storage backend (SQLite/PostgreSQL)
- [ ] External integrations (webhooks, Kafka)
- [ ] CI workflow (GitHub Actions)
- [ ] Experiment tracking (MLflow/W&B)
- [ ] Multi-user session support
- [ ] Belief export/import

---

## API Endpoints

### Chat API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat/message` | Send message, get response with beliefs |
| GET | `/chat/sessions` | List chat sessions |
| POST | `/chat/sessions` | Create new session |
| GET | `/chat/sessions/{id}` | Get session with history |
| WS | `/chat/ws` | WebSocket for real-time belief events |

### Belief API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/beliefs` | List beliefs with pagination |
| GET | `/beliefs/{id}` | Get specific belief |
| POST | `/beliefs` | Create belief manually |
| PATCH | `/beliefs/{id}` | Update belief |
| POST | `/beliefs/{id}/reinforce` | Reinforce belief |

### System API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/bel/stats` | System statistics |
| GET | `/bel/health` | Health check |
| POST | `/bel/iterate` | Run one ecology iteration |

---

## License

MIT © 2025-2026 Bradley R. Kinnard
