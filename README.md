# ABES – Adaptive Belief Ecology System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org/)
![Early Development](https://img.shields.io/badge/status-early%20development-orange)

ABES is an open-source research platform for exploring **Belief Ecology**: treating beliefs as living, evolving entities instead of static memory entries.

Most LLM memory systems use key-value stores or vector retrieval. ABES implements something different: beliefs decay over time, contradict each other, get reinforced when similar evidence arrives, and cluster around related concepts. The system uses specialized agents to manage this ecology.

**Current implementation includes:**
- Complete Belief Ecology Loop (decay, contradiction detection, relevance scoring, ranking)
- 4 specialized agents (Perception, Creation, Reinforcement, Contradiction Auditor)
- 38 passing tests with full type safety
- Production-grade error handling and deterministic caching

> **Core systems are functional.**
> Working on storage layer, remaining agents, and RL integration.

## What I'm building

A cognitively realistic long-term memory system where:

- Beliefs naturally decay unless they're reinforced (no infinite perfect recall)
- Contradictions get detected and either resolved or marked as competing hypotheses
- Ambiguous evidence causes beliefs to mutate instead of just accumulating
- The system adapts decay rates and retention based on what's actually useful
- RL learns selection policies, decay schedules, and mutation triggers from experience instead of hardcoding them
- Every state change is captured in snapshots so you can replay and debug how memory evolved

Exposed through a UI that shows beliefs moving around as a graph instead of making you guess what's happening inside.

## What's in here (planned)

| Component | What it does |
|-----------|--------------|
| **Belief Ecology Engine** | Main loop: decay, detect contradictions, mutate/resolve, re-rank beliefs |
| **15 Specialized Agents** | Perception, Creation, Reinforcement, Mutation, Resolution, RL Policy, etc. |
| **Reinforcement Learning** | Learns decay rates, thresholds, and selection weights from experience |
| **Snapshot & Time Travel** | Captures full state every iteration so you can replay belief evolution |
| **Visualization UI** | Belief graph, timeline explorer, agent activity monitor, RL metrics |
| **Benchmark Lab** | Test scenarios + comparisons against RAG, chat history, static memory |

## Repository Structure

```
backend/          - FastAPI + Pydantic (belief models, ecology loop, agents, RL)
frontend/         - Next.js 14+ with TypeScript and Tailwind
experiments/      - Test scenarios and result analysis
configs/          - Ecology parameters, RL hyperparameters, agent settings
docs/             - Architecture notes, agent specs, research documents
data/             - (future) embeddings, snapshots, experiment logs
infra/            - Docker configs, dev scripts, CI
scripts/          - Utilities and one-off tools
```

## Current Status (December 2025)

**What's implemented:**

### Core Data Models
- `Belief` model with full Pydantic v2 validation (`backend/core/models/belief.py`)
  - UUID identifiers, confidence tracking, decay metadata
  - Tag system, tension tracking, cluster assignment
  - Status lifecycle (active/decaying/deprecated)
  - Parent/child lineage for mutations
- `BeliefSnapshot` model for state capture (`backend/core/models/snapshot.py`)
- Event system for belief lifecycle tracking (`backend/core/events.py`)

### Belief Ecology Loop (BEL)
Complete implementation in `backend/core/bel/`:
- **Decay system** (`decay.py`) - time-based confidence degradation
- **Contradiction detection** (`contradiction.py`) - pairwise tension computation
- **Relevance scoring** (`relevance.py`) - context-based belief ranking
- **Ranking system** (`ranking.py`) - multi-factor belief prioritization
- **Main loop** (`loop.py`) - orchestrates decay → contradiction → relevance → rank
- **Snapshot compression** (`snapshot_compression.py`) - efficient state serialization
- **Timeline tracking** (`timeline.py`) - snapshot history management
- **Snapshot logger** (`snapshot_logger.py`) - state persistence

### Agent System
Four specialized agents implemented in `backend/agents/`:
- **PerceptionAgent** (`perception.py`) - ingests raw input and extracts belief candidates
- **BeliefCreatorAgent** (`belief_creator.py`) - creates new beliefs from candidates with duplicate detection
- **ReinforcementAgent** (`reinforcement.py`) - strengthens existing beliefs when similar evidence arrives
- **ContradictionAuditorAgent** (`contradiction_auditor.py`) - detects high-tension belief pairs using semantic similarity

All agents include:
- Production-grade error handling
- SHA256-based deterministic caching
- Lazy model loading for performance
- Comprehensive input validation

### Test Coverage
38 passing tests across 3 test files:
- `test_belief_creator.py` - 14 tests for belief creation, deduplication, UUID handling
- `test_reinforcement.py` - 12 tests for reinforcement logic, confidence capping, cooldown
- `test_contradiction_auditor.py` - 12 tests for contradiction detection, debouncing, persistence

All tests validate:
- Core functionality
- Edge cases
- Lazy loading behavior
- State persistence
- Deterministic behavior

### Configuration System
- Centralized config management (`backend/core/config.py`)
- Dependency injection setup (`backend/core/deps.py`)
- Environment-based settings with Pydantic validation

### Project Infrastructure
- Complete monorepo structure (backend/frontend/experiments/configs/docs)
- Python 3.11+ with Poetry dependency management
- pytest-based test suite with asyncio support
- Type hints throughout with modern syntax (`list[str]` not `List[str]`)
- MIT license and contribution guidelines

**What's next:**
- Storage layer implementation (belief persistence, snapshot store)
- Remaining 11 agents (mutation, resolution, RL policy, etc.)
- RL environment and training loop
- Frontend visualization (Next.js + TypeScript)
- Benchmark scenarios and baseline comparisons
- API layer (FastAPI endpoints)
- Time travel UI for snapshot replay

## Getting Started

### Prerequisites
- Python 3.11 or higher
- Poetry for Python dependency management
- Node.js 18+ and pnpm (for frontend, when ready)

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/moonrunnerkc/adaptive-belief-ecology-system.git
cd adaptive-belief-ecology-system

# Install Python dependencies
poetry install

# Activate the virtual environment
poetry shell

# Run the test suite
pytest tests/ -v

# All 38 tests should pass
```

### Project Status
The backend is functional with core models, the Belief Ecology Loop, and 4 specialized agents. The system can:
- Create and track beliefs with decay
- Detect contradictions between beliefs
- Reinforce beliefs when similar evidence arrives
- Compute relevance and rank beliefs by importance

Frontend and API endpoints are not yet implemented.

## Contributing

This is a research-driven project with a very strict specification.

I'm not accepting PRs yet – the initial implementation must match the spec exactly.

Once Phase 1+ is complete and stable, I'll welcome:

- Bug reports
- Performance improvements
- New benchmark scenarios
- UI/UX suggestions
- Research collaborations

Please open an issue first to discuss any ideas.

## License

MIT © 2025 ABES Contributors
