# ABES - Adaptive Belief Ecology System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-636%20passing-brightgreen.svg)]()

A research platform for belief ecology: treating beliefs as living, evolving entities rather than static memory entries.

---

## Overview

ABES is an experimental cognitive memory architecture for AI systems. Most memory systems use key-value stores or vector retrieval. This one is different. Beliefs here are first-class objects that decay over time, accumulate tension when they contradict each other, get reinforced when similar evidence shows up, and mutate or get deprecated when tension gets too high.

A pipeline of specialized agents processes beliefs each iteration. There's also an optional RL layer to tune system parameters automatically.

This is a research prototype. It works, but it's not production-ready.

---

## The Chatbot

ABES includes a conversational chatbot that demonstrates the belief ecology in action. The chatbot is powered by a local Ollama LLM and uses your stored beliefs to provide personalized responses.

### Why the chatbot exists

The chat interface is the primary way to interact with and test the belief ecology. When you talk to it:

1. Your messages get parsed by the perception agent to extract belief candidates
2. New beliefs are created with initial confidence scores
3. Existing similar beliefs get reinforced (confidence boost)
4. Contradicting beliefs accumulate tension
5. The LLM generates responses using your belief context

This lets you watch the ecology evolve in real time. Say "My name is Brad" a few times and watch the confidence climb. Then say "Actually my name is Sam" and watch the tension spike.

### How to use it

1. Start the backend: `PYTHONPATH=$PWD uvicorn backend.api.app:app --port 8000`
2. Start the frontend: `cd frontend && npm run dev`
3. Start Ollama: `ollama serve`
4. Open http://localhost:3000/chat

Try these interactions:
- "My name is Brad and I have two dogs named Reaper and Rocky"
- "Reaper sings like a bird"
- "What do you know about me?"
- "Actually I have three dogs" (creates tension with previous belief)

The Activity panel on the right shows belief events as they happen.

---

## Key Features

| Feature | Source | Tests |
|---------|--------|-------|
| Belief data model (confidence, tension, status, lineage) | [backend/core/models/belief.py](backend/core/models/belief.py) | [test_bel_loop.py](tests/core/test_bel_loop.py) |
| 14-phase agent scheduler | [backend/agents/scheduler.py](backend/agents/scheduler.py) | [test_scheduler.py](tests/agents/test_scheduler.py) |
| Perception agent (text to belief candidates) | [backend/agents/perception.py](backend/agents/perception.py) | [test_perception.py](tests/agents/test_perception.py) |
| Reinforcement agent (boost on similar evidence) | [backend/agents/reinforcement.py](backend/agents/reinforcement.py) | [test_reinforcement.py](tests/agents/test_reinforcement.py) |
| Decay controller (time-based confidence reduction) | [backend/agents/decay_controller.py](backend/agents/decay_controller.py) | [test_decay_controller.py](tests/agents/test_decay_controller.py) |
| Contradiction auditor (embedding + antonym detection) | [backend/agents/contradiction_auditor.py](backend/agents/contradiction_auditor.py) | [test_contradiction_auditor.py](tests/agents/test_contradiction_auditor.py) |
| Mutation engineer (conflict-triggered belief modification) | [backend/agents/mutation_engineer.py](backend/agents/mutation_engineer.py) | [test_mutation_engineer.py](tests/agents/test_mutation_engineer.py) |
| Semantic clustering | [backend/core/bel/clustering.py](backend/core/bel/clustering.py) | [test_clustering.py](tests/core/test_clustering.py) |
| RL environment (15D state, 7D action) | [backend/rl/environment.py](backend/rl/environment.py) | [test_environment.py](tests/rl/test_environment.py) |
| Evolution Strategy trainer | [backend/rl/training.py](backend/rl/training.py) | [test_training.py](tests/rl/test_training.py) |
| FastAPI REST + WebSocket API | [backend/api/app.py](backend/api/app.py) | [test_routes.py](tests/api/test_routes.py) |
| Chat service with Ollama LLM | [backend/chat/service.py](backend/chat/service.py) | Manual testing |
| Next.js frontend | [frontend/](frontend/) | Manual testing |

---

## Architecture

```
Frontend (Next.js) --> REST/WebSocket --> FastAPI Backend (:8000)
                                              |
                    +-----------+-------------+-----------+
                    |           |             |           |
               Chat Service  Agent Scheduler  RL Environment
               (Ollama LLM)  (14 phases)      (Gymnasium)
                    |           |             |
                    +-----------+-------------+
                                |
                       In-Memory Belief Store
```

### Agent Pipeline

```
Perception --> Creation --> Reinforcement --> Decay --> Contradiction -->
Mutation --> Resolution --> Relevance --> RL Policy --> Consistency -->
Safety --> Baseline --> Narrative --> Experiment
```

Each agent is independently tested. See [backend/agents/](backend/agents/).

---

## Installation

Requirements: Python 3.10+, Node.js 18+ (frontend), Ollama (chat)

### Backend

```bash
git clone https://github.com/moonrunnerkc/adaptive-belief-ecology-system.git
cd adaptive-belief-ecology-system

python -m venv .venv
source .venv/bin/activate
pip install numpy pydantic pydantic-settings msgpack sentence-transformers httpx
pip install pytest pytest-asyncio

export PYTHONPATH=$PWD
```

### Frontend

```bash
cd frontend
npm install
```

### Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b-instruct-q4_0
```

---

## Quick Start

Terminal 1 (Backend):
```bash
source .venv/bin/activate
PYTHONPATH=$PWD uvicorn backend.api.app:app --host 0.0.0.0 --port 8000
```

Terminal 2 (Frontend):
```bash
cd frontend && npm run dev
```

Terminal 3 (Ollama):
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

We ran a full verification suite to make sure everything works as claimed. Here's what we tested and what the results mean.

### Unit Tests

```bash
PYTHONPATH=$PWD pytest tests/ -q
```

Current status: **636 passed, 2 failed**

| Suite | Files | What it covers |
|-------|-------|----------------|
| tests/agents/ | 18 | All agent modules |
| tests/core/ | 5 | BEL loop, clustering, timeline, RL integration |
| tests/rl/ | 3 | Environment, policy, training |
| tests/api/ | 1 | REST endpoints |
| tests/verification/ | 3 | Determinism, offline operation, conflict resolution |

### Verification Experiments

We ran these experiments to produce hard evidence for our claims:

```bash
PYTHONPATH=$PWD python experiments/run_all.py
```

All experiments passed. Here's what each one proves:

**Determinism Check** ([results/determinism_check.json](results/determinism_check.json))
- Ran the same input sequence twice with seed 12345
- Both runs produced identical state hashes: `077ac8e32f721ef8dbb51a3613adf8e1288e9e0c02422af918327956c7dbcbe1`
- Different seeds (12345 vs 12346) produce different hashes
- This proves: given the same inputs and seed, you get byte-for-byte identical outputs

**Offline Operation** ([results/offline_verification.json](results/offline_verification.json))
- Blocked all network sockets at runtime
- Ran 5 core components (belief ingest, conflict resolution, baselines, metrics, decay simulation)
- Detected 0 network calls
- This proves: the core belief processing works without any network access

**Conflict Resolution** ([results/conflict_resolution_log.json](results/conflict_resolution_log.json))
- Tested 4 conflict scenarios with different confidence levels and ages
- Resolution actions are deterministic: WEAKEN for confidence gaps, DEFER for equal strength
- 9 total cases documented with case IDs and confidence scores
- This proves: conflict resolution follows consistent rules, not random decisions

**Drift Comparison** ([results/drift_comparison.json](results/drift_comparison.json))
- Ran 23-turn conversation with reinforcement, contradictions, and duplicates
- Compared three systems: plain LLM (no memory), append-only memory, belief ecology
- Append-only accumulated 17 beliefs and 2 contradictions
- Belief ecology maintained 0 active contradictions (tension-based resolution worked)
- This proves: the ecology manages contradictions instead of just accumulating them

**Decay Sweep** ([results/decay_sweep/](results/decay_sweep/))
- Tested decay factors: 0.999, 0.995, 0.99, 0.97, 0.95
- At 0.999: 4 beliefs retained, 9 dropped
- At 0.995 and below: 0 beliefs retained, 13 dropped
- This proves: decay factor significantly affects retention. Default of 0.995 is aggressive.

### Evidence File Hashes

For reproducibility, here are the SHA256 hashes of our evidence files:

```
ecfce79e1b80ab06a9c813e3233f634352d4064b760511c6b1b0bb5ff85a829c  results/drift_comparison.json
dcffb0e0f13ff4c28125e61d208cdc2d4c4fa8d36086aa56a3fdaa082d6db0dc  results/determinism_check.json
2be674788c8b5168348ae3e7f1157c807b18b65001ec581a45e501264aac2b95  results/offline_verification.json
391bfd85ce01eb5fa75b393ca2a69986e2114051db31403bc7d02efa522bfe26  results/conflict_resolution_log.json
d4dd4f5c4a777eac6d2954cd25030f74c9a4e7f27275f1c5e19fa21795d247c5  results/decay_sweep/decay_0.995.json
```

---

## What's Failing Right Now

Two tests are currently failing. We're actively working on them:

1. **test_mutation_engineer.py::TestProposeMutation::test_mutated_belief_has_neutral_confidence**
   - The mutation engineer is supposed to set mutated beliefs to neutral confidence (0.5)
   - Something in the confidence calculation is off
   - Low priority since mutations still work, just with different confidence values

2. **test_environment.py::TestActionDecoding::test_positive_action_increases**
   - RL environment action decoding edge case
   - Positive actions should always increase parameter values
   - The decode logic has a boundary condition bug

### Known Issues Being Worked On

- Chat requires Ollama running locally (no fallback)
- No persistence. Server restart clears all beliefs.
- Single-user only. No session isolation.
- Contradiction detection uses embeddings and antonym lists, not full semantic understanding
- The decay factor default (0.995) might be too aggressive for some use cases

---

## Limitations

- In-memory storage only. All state lost on restart.
- Single-user. No authentication or session isolation.
- No CI/CD pipeline. Tests run locally only.
- Ollama-only LLM. No OpenAI/Anthropic integration yet.
- Fixed embedding model (all-MiniLM-L6-v2)

---

## Roadmap

Not yet implemented:

- [ ] Persistent storage (SQLite/PostgreSQL)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Multi-user session support
- [ ] Belief Explorer UI
- [ ] Document ingestion service
- [ ] OpenAI/Anthropic LLM providers
- [ ] Benchmarks against production memory systems

---

## License

MIT 2025-2026 Bradley R. Kinnard
