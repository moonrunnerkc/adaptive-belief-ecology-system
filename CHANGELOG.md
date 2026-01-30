# Changelog

All notable changes to ABES are documented here.

## [0.1.0] - 2026-01-30

### Added

**Core Models**
- `Belief` model with confidence, tension, status, tags, lineage tracking
- `Snapshot` model capturing ecology state with edge relationships
- `BeliefSnapshot` frozen belief state for snapshots
- Event models: `BeliefCreatedEvent`, `BeliefReinforcedEvent`, etc.

**15 Specialized Agents**
- `PerceptionAgent` — extracts claims from raw input
- `BeliefCreatorAgent` — creates beliefs with embedding-based deduplication
- `ReinforcementAgent` — boosts confidence on similar evidence
- `ContradictionAuditorAgent` — computes pairwise tensions via embeddings
- `MutationEngineerAgent` — proposes hedged/conditional belief variants
- `ResolutionStrategistAgent` — integrates, splits, or deprecates conflicts
- `RelevanceCuratorAgent` — ranks beliefs by weighted formula
- `DecayControllerAgent` — applies time-based confidence decay
- `BaselineMemoryBridgeAgent` — interfaces with RAG/chat history
- `RLPolicyAgent` — outputs actions from ecology state (heuristic fallback)
- `RewardShaperAgent` — computes shaped reward signals
- `ExperimentOrchestratorAgent` — runs scripted scenarios
- `ConsistencyCheckerAgent` — probes for answer drift
- `NarrativeExplainerAgent` — generates human-readable explanations
- `SafetySanityAgent` — enforces limits, vetoes dangerous actions

**Agent Scheduler**
- 14-phase execution order per spec 4.2
- Conditional execution support (`run_every_n`, enable/disable)
- `AgentProtocol` and `AgentResult` for uniform interface

**RL Layer**
- `BeliefEcologyEnv` — 15-dim state, 7-dim action, episodic
- `MLPPolicy` — pure NumPy MLP with tanh activations
- `EvolutionStrategy` — gradient-free optimizer
- `ESTrainer` — training loop with checkpointing

**Storage**
- `BeliefStoreABC` and `SnapshotStoreABC` abstract interfaces
- `InMemoryBeliefStore` and `InMemorySnapshotStore` implementations
- Snapshot compression via msgpack + zlib

**Snapshot Features**
- Contradiction edges: `(belief_a, belief_b, score)`
- Support edges: high similarity without negation
- Lineage edges: parent → child from mutations
- `Snapshot.diff()` for comparing snapshots

**Configuration**
- `ABESSettings` via pydantic-settings
- All spec formulas configurable (decay rate, thresholds, weights)

**Tests**
- 361 agent tests across 16 test files
- 50 RL tests across 3 test files
- 411 total tests passing

### Not Implemented

- HTTP API (`backend/api/` is empty)
- Frontend (`frontend/` contains placeholder stubs)
- Benchmark scenarios (`backend/benchmark/` is empty)
- Metrics collection (`backend/metrics/` is empty)
- Configuration files (`configs/` is empty)
- Cluster maintenance (merge/split per spec 3.6.2)
- Direct tests for `BeliefEcologyLoop`

---

## Format

Based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
