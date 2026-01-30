# ABES — Adaptive Belief Ecology System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A research platform for **Belief Ecology**: treating beliefs as living, evolving entities instead of static memory entries.

---

## What This Is

ABES is an experimental cognitive memory architecture where beliefs:
- Decay over time without reinforcement
- Can contradict each other (tension is tracked)
- Get reinforced when similar evidence arrives
- Are processed by specialized agents each iteration

This is a **research prototype**, not a production system.

---

## Verified Claims

The following claims are backed by reproducible experiments stored in `results/`.

### Determinism
**Claim:** Given identical inputs and seed, the system produces byte-for-byte identical outputs.

**Evidence:** [results/determinism_check.json](results/determinism_check.json)
- Two runs with seed 12345 produced identical state hashes
- Different seeds (12345 vs 12346) produce different hashes

### Offline Operation
**Claim:** Core belief processing runs without network access.

**Evidence:** [results/offline_verification.json](results/offline_verification.json)
- Network sockets blocked at runtime
- 5 experiment components ran successfully
- 0 network calls detected

### Conflict Resolution Consistency
**Claim:** Conflict resolution follows deterministic rules based on confidence and age, not semantic truth.

**Evidence:** [results/conflict_resolution_log.json](results/conflict_resolution_log.json)
- 4 test cases passed
- Resolution actions: WEAKEN (for confidence gaps), DEFER (for equal strength)
- No truth inference—only consistency enforcement

### Decay Behavior
**Claim:** Decay factor significantly affects belief retention.

**Evidence:** [results/decay_sweep/](results/decay_sweep/)
| Decay Factor | Beliefs Retained | Beliefs Dropped | Churn Rate |
|--------------|------------------|-----------------|------------|
| 0.999 | 4 | 9 | 0.85 |
| 0.995 | 0 | 13 | 1.00 |
| 0.99 | 0 | 13 | 1.00 |
| 0.97 | 0 | 13 | 1.00 |
| 0.95 | 0 | 13 | 1.00 |

**Observation:** The default decay factor of 0.995 results in aggressive belief deprecation. This is intentional for forgetting transient facts, but may need tuning for specific use cases.

### Comparative Drift Benchmark
**Claim:** The belief ecology maintains fewer accumulated contradictions than append-only storage.

**Evidence:** [results/drift_comparison.json](results/drift_comparison.json)
- 23-turn conversation script with reinforcement, contradictions, and duplicates
- Metrics captured: belief count, contradiction count, entropy over time
- Systems compared: plain LLM (no memory), append-only, belief ecology

---

## Limitations and Caveats

### What Is NOT Verified

1. **LLM integration quality** — The chat interface depends on Ollama and response quality varies by model.

2. **Semantic understanding** — Contradiction detection uses embedding similarity and antonym lists, not true semantic reasoning.

3. **Scalability** — No load testing. In-memory storage only. The system is designed for research, not production workloads.

4. **RL effectiveness** — The Evolution Strategy trainer runs, but policy quality is not benchmarked against baselines.

5. **Persistence** — All state is in-memory. Server restart clears beliefs.

### Known Issues

- 2 tests currently failing (out of 638 total)
- Chat requires Ollama running locally
- No multi-user isolation

---

## Installation

### Requirements
- Python 3.10+
- Node.js 18+ (for frontend)
- Ollama (for chat)

### Backend

```bash
git clone https://github.com/moonrunnerkc/adaptive-belief-ecology-system.git
cd adaptive-belief-ecology-system

python -m venv .venv
source .venv/bin/activate
pip install numpy pydantic pydantic-settings msgpack sentence-transformers httpx
pip install pytest pytest-asyncio  # for tests

export PYTHONPATH=$PWD
```

### Frontend

```bash
cd frontend
npm install
```

### Ollama (for chat)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.1:8b-instruct-q4_0
```

---

## Quick Start

### Start Backend
```bash
source .venv/bin/activate
PYTHONPATH=$PWD uvicorn backend.api.app:app --host 0.0.0.0 --port 8000
```

### Start Frontend
```bash
cd frontend
npm run dev
# Chat at http://localhost:3000/chat
```

### Start Ollama
```bash
ollama serve
```

---

## Running Verification Suite

All experiments are deterministic and run offline.

```bash
source .venv/bin/activate
PYTHONPATH=$PWD python experiments/run_all.py
```

This generates:
```
results/
├── drift_comparison.json
├── determinism_check.json
├── offline_verification.json
├── conflict_resolution_log.json
├── verification_summary.json
└── decay_sweep/
    ├── decay_0.999.json
    ├── decay_0.995.json
    ├── decay_0.99.json
    ├── decay_0.97.json
    └── decay_0.95.json
```

---

## Architecture

```
Frontend (Next.js) → REST/WebSocket → FastAPI Backend
                                          │
                                    LLM Layer (Ollama)
                                          │
                                    Agent Scheduler
                                    (15 agents, 14 phases)
                                          │
                                    Storage (in-memory)
                                          │
                                    RL Layer (optional)
```

### Key Components

| Component | Location | Description |
|-----------|----------|-------------|
| Belief Model | [backend/core/models/belief.py](backend/core/models/belief.py) | Confidence, tension, status, lineage |
| Agent Scheduler | [backend/agents/scheduler.py](backend/agents/scheduler.py) | 14-phase execution pipeline |
| Chat Service | [backend/chat/service.py](backend/chat/service.py) | Message → beliefs → LLM response |
| Perception Agent | [backend/agents/perception.py](backend/agents/perception.py) | Extract beliefs from text |
| Reinforcement Agent | [backend/agents/reinforcement.py](backend/agents/reinforcement.py) | Boost confidence on similar evidence |
| Contradiction Auditor | [backend/agents/contradiction_auditor.py](backend/agents/contradiction_auditor.py) | Detect conflicting beliefs |

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DECAY_RATE` | 0.995 | Per-hour confidence multiplier |
| `CONFIDENCE_THRESHOLD_DECAYING` | 0.3 | Threshold for deprecation |
| `TENSION_THRESHOLD_MUTATION` | 0.6 | Trigger for mutation proposals |
| `CLUSTER_SIMILARITY_THRESHOLD` | 0.7 | Min similarity to join cluster |

See [backend/core/config.py](backend/core/config.py) for full list.

---

## Tests

```bash
PYTHONPATH=$PWD pytest tests/ -q
# 636 passed, 2 failed
```

| Suite | Description |
|-------|-------------|
| `tests/agents/` | Agent behavior |
| `tests/core/` | BEL loop, clustering |
| `tests/rl/` | Environment, training |
| `tests/verification/` | Reproducibility checks |

---

## Verification Evidence

| Claim | Evidence File | Key Metric |
|-------|---------------|------------|
| Deterministic | [results/determinism_check.json](results/determinism_check.json) | `deterministic: true` |
| Offline | [results/offline_verification.json](results/offline_verification.json) | `network_calls_detected: 0` |
| Decay sensitivity | [results/decay_sweep/](results/decay_sweep/) | Retention varies by factor |
| Conflict resolution | [results/conflict_resolution_log.json](results/conflict_resolution_log.json) | 4/4 tests passed |
| Drift comparison | [results/drift_comparison.json](results/drift_comparison.json) | 3-system comparison |

---

## Open Research Questions

1. **Optimal decay rate** — Current default (0.995) is aggressive. What's the right balance between forgetting and retention?

2. **Contradiction resolution strategy** — Current approach weakens lower-confidence beliefs. Is merge or temporal windowing better?

3. **RL reward design** — What objective function best captures "healthy" belief ecology?

4. **Semantic vs syntactic** — Current contradiction detection uses embeddings. Would LLM-based detection be more accurate? At what cost?

---

## License

MIT © 2025-2026 Bradley R. Kinnard
