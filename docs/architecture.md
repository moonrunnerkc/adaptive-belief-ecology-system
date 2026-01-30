# ABES Architecture

## Overview

ABES implements a **Belief Ecology** where beliefs are living entities that decay, conflict, mutate, and cluster. The system consists of:

```
┌─────────────────────────────────────────────────────────────┐
│                     Agent Scheduler                         │
│  (14 phases: Perception → ... → Experiment)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    15 Specialized Agents                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │Perception│ │ Creator  │ │ Auditor  │ │ Mutation │  ...  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Storage Layer                            │
│  ┌──────────────────┐    ┌───────────────────┐              │
│  │  Belief Store    │    │  Snapshot Store   │              │
│  │  (in-memory)     │    │  (compressed)     │              │
│  └──────────────────┘    └───────────────────┘              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      RL Layer                                │
│  ┌───────────┐    ┌──────────┐    ┌──────────┐             │
│  │Environment│ ←→ │  Policy  │ ←→ │ Trainer  │             │
│  │(15d/7d)   │    │(NumPy MLP│    │(ES)      │             │
│  └───────────┘    └──────────┘    └──────────┘             │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Per-Iteration Flow

1. **Input arrives** → PerceptionAgent extracts candidate claims
2. **BeliefCreatorAgent** deduplicates via embedding similarity, creates Belief objects
3. **ReinforcementAgent** boosts existing beliefs matching new input
4. **DecayControllerAgent** applies time-based confidence decay
5. **ContradictionAuditorAgent** computes pairwise tensions
6. **MutationEngineerAgent** proposes hedged variants for high-tension beliefs
7. **ResolutionStrategistAgent** resolves high-confidence conflicts
8. **RelevanceCuratorAgent** ranks beliefs against current context
9. **RLPolicyAgent** outputs control parameters
10. **ConsistencyCheckerAgent** probes for drift
11. **SafetySanityAgent** enforces limits
12. **BaselineMemoryBridgeAgent** interfaces with RAG/chat
13. **NarrativeExplainerAgent** generates explanations
14. **ExperimentOrchestratorAgent** logs results

### Snapshot Capture

After each iteration, the system captures:
- All belief states (frozen `BeliefSnapshot` objects)
- Contradiction edges: `[(uuid, uuid, score)]`
- Support edges: high similarity, no negation
- Lineage edges: parent → child mutations
- Agent actions taken
- RL state/action if training

## Core Models

### Belief

```python
class Belief:
    id: UUID
    content: str                    # natural language
    confidence: float               # 0.0 to 1.0
    origin: OriginMetadata          # source, timestamp, last_reinforced
    tags: list[str]
    tension: float                  # contradiction pressure
    cluster_id: Optional[UUID]
    status: BeliefStatus            # active, decaying, mutated, deprecated
    parent_id: Optional[UUID]       # lineage tracking
    use_count: int
```

### Snapshot

```python
class Snapshot:
    id: UUID
    metadata: SnapshotMetadata      # iteration, timestamp
    beliefs: list[BeliefSnapshot]
    global_tension: float
    contradiction_edges: list[tuple[UUID, UUID, float]]
    support_edges: list[tuple[UUID, UUID, float]]
    lineage_edges: list[tuple[UUID, UUID]]
    agent_actions: list[dict]
    rl_state_action: Optional[dict]
```

## Formulas

All computations follow spec sections 3.4.1–3.4.10:

| Formula | Equation |
|---------|----------|
| Decay | `confidence *= decay_rate ^ hours_elapsed` |
| Contradiction | `semantic_similarity × negation_signal` |
| Tension | `max(contradiction_score)` per belief |
| Ranking | `0.4×relevance + 0.3×confidence + 0.2×recency - 0.1×tension` |

Thresholds:
- `active → decaying`: confidence < 0.3
- `decaying → deprecated`: confidence < 0.1
- Mutation trigger: tension ≥ 0.6 AND confidence < 0.5
- Resolution trigger: contradiction ≥ 0.7 AND both confidences ≥ 0.6

## RL Environment

**State (15 dimensions):**
- Confidence: mean, std, min, max
- Tension: mean, max, high_tension_count
- Counts: total_beliefs, active, decaying (normalized)
- Cluster count
- Recent: mutations, deprecations, reinforcements
- Episode progress

**Action (7 dimensions):**
- Global decay rate adjustment
- Mutation threshold adjustment
- Resolution threshold adjustment
- Deprecation threshold adjustment
- Ranking weight: relevance
- Ranking weight: confidence
- Beliefs to surface count

**Reward shaping:**
- +task_success, +consistency, +efficiency, +stability
- −contradiction_errors, −forgetting_penalty

## Storage

Abstract interfaces in `storage/base.py`:
- `BeliefStoreABC`: CRUD + embedding search + bulk update
- `SnapshotStoreABC`: save/get/list/compare snapshots

Current implementation: in-memory dicts. Snapshots are compressed via msgpack + zlib.

## Module Map

```
backend/
├── agents/                 # 15 agents + scheduler
│   ├── perception.py
│   ├── belief_creator.py
│   ├── reinforcement.py
│   ├── contradiction_auditor.py
│   ├── mutation_engineer.py
│   ├── resolution_strategist.py
│   ├── relevance_curator.py
│   ├── decay_controller.py
│   ├── baseline_memory_bridge.py
│   ├── rl_policy.py
│   ├── reward_shaper.py
│   ├── experiment_orchestrator.py
│   ├── consistency_checker.py
│   ├── narrative_explainer.py
│   ├── safety_sanity.py
│   └── scheduler.py
├── core/
│   ├── config.py           # ABESSettings
│   ├── deps.py             # dependency injection
│   ├── events.py           # event models
│   ├── models/
│   │   ├── belief.py
│   │   └── snapshot.py
│   └── bel/
│       ├── loop.py         # BeliefEcologyLoop (untested)
│       ├── decay.py
│       ├── contradiction.py
│       ├── ranking.py
│       ├── relevance.py
│       ├── rl_integration.py
│       ├── snapshot_compression.py
│       ├── snapshot_logger.py
│       └── timeline.py
├── rl/
│   ├── environment.py      # BeliefEcologyEnv
│   ├── policy.py           # MLPPolicy, EvolutionStrategy
│   └── training.py         # ESTrainer
└── storage/
    ├── base.py             # abstract interfaces
    ├── in_memory.py        # dict-based implementations
    └── snapshot_queries.py # query utilities
```

## Embeddings

- Model: `all-MiniLM-L6-v2` (384 dimensions)
- Library: sentence-transformers
- Usage: deduplication, contradiction detection, relevance scoring
- Cached per agent to avoid redundant encoding
