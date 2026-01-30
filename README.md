# ABES – Adaptive Belief Ecology System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-411%20passing-brightgreen.svg)]()

ABES is an open-source research platform for **Belief Ecology**: treating beliefs as living, evolving entities instead of static memory entries.

Most LLM memory systems use key-value stores or vector retrieval. ABES implements something different—beliefs decay over time, contradict each other, get reinforced when similar evidence arrives, mutate under tension, and cluster around related concepts. Specialized agents manage this ecology, and reinforcement learning optimizes the control parameters.

## Current Status (January 2026)

| Component | Status | Tests |
|-----------|--------|-------|
| Belief Ecology Engine | ✅ Complete | Core loop, decay, tension, ranking |
| 15 Specialized Agents | ✅ Complete | 361 tests |
| Agent Scheduler | ✅ Complete | 20 tests |
| RL Environment | ✅ Complete | 20 tests |
| RL Training (ES) | ✅ Complete | 10 tests |
| Snapshot Edges | ✅ Complete | contradiction, support, lineage |
| Frontend | ❌ Not started | — |
| API Layer | ❌ Not started | — |

**411 tests passing** across the backend.

## What's Implemented

### Belief Ecology Loop (BEL)

The core loop in `backend/core/bel/loop.py` runs 7 steps per iteration:

1. **Load** active beliefs from storage
2. **Decay** confidence based on time since reinforcement
3. **Compute tensions** via pairwise contradiction scoring
4. **Trigger actions** (mutation/resolution candidates)
5. **Score relevance** to current context
6. **Rank** beliefs by weighted formula
7. **Snapshot** full state with edges

All formulas match the spec exactly:
- Decay: `confidence *= decay_rate ^ hours_elapsed`
- Contradiction: `semantic_similarity × negation_signal`
- Tension: `max(contradiction_score)` across pairs
- Ranking: `0.4×relevance + 0.3×confidence + 0.2×recency - 0.1×tension`

### 15 Specialized Agents

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

### Agent Scheduler

Per spec 4.2, agents run in a defined order:

```
Perception → Creation → Reinforcement → Decay → Contradiction →
Mutation → Resolution → Relevance → RL Policy → Consistency →
Safety → Baseline → Narrative → Experiment
```

Supports conditional execution, frequency control (`run_every_n`), and enable/disable per agent.

### RL Layer

**Environment** (`backend/rl/environment.py`):
- 15-dimensional state (confidence stats, tension, belief counts, etc.)
- 7-dimensional action (decay rate, thresholds, ranking weights)
- Reward shaped from task success, consistency, efficiency, stability

**Policy** (`backend/rl/policy.py`):
- Pure NumPy MLP with tanh activations
- Save/load support
- No PyTorch dependency

**Training** (`backend/rl/training.py`):
- Evolution Strategy (ES) optimizer
- Population-based search with gradient-free updates
- Checkpointing and early stopping

### Snapshot System

Each snapshot captures:
- All belief states (content, confidence, tension, status, etc.)
- **Contradiction edges**: `[(belief_a, belief_b, score)]`
- **Support edges**: high similarity without negation
- **Lineage edges**: parent → child mutations
- RL state/action for that iteration
- Agent actions taken

## Repository Structure

```
backend/
├── agents/           # 15 specialized agents + scheduler
├── core/
│   ├── bel/          # Belief Ecology Loop implementation
│   └── models/       # Belief, Snapshot, Event models
├── rl/               # Environment, Policy, Training
├── storage/          # Belief and snapshot stores
└── util/

frontend/             # (not yet implemented)
experiments/          # Scenario definitions
configs/              # Parameters and settings
tests/
├── agents/           # 361 agent tests
└── rl/               # 50 RL tests
```

## Getting Started

```bash
# Clone
git clone https://github.com/moonrunnerkc/abes.git
cd abes

# Create venv and install
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Should see: 411 passed
```

### Quick Usage

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

## What's Next

1. **FastAPI Routes** — CRUD for beliefs, snapshots, agents
2. **Frontend** — Next.js 14 with belief graph visualization
3. **Benchmark Lab** — Scenarios comparing BE vs RAG vs chat history
4. **Cluster Maintenance** — Periodic merge/split per spec 3.6.2

## Contributing

This is a research project with strict spec adherence. I'm not accepting PRs until Phase 1 is stable.

Once ready, I'll welcome:
- Bug reports
- Performance improvements
- New benchmark scenarios
- Research collaborations

Open an issue first to discuss.

## License

MIT © 2025-2026 ABES Contributors
