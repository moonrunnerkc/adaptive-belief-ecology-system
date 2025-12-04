# ABES – Adaptive Belief Ecology System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org/)
![Early Development](https://img.shields.io/badge/status-early%20development-orange)

ABES is an open-source research platform for exploring **Belief Ecology** – treating beliefs as living, evolving entities instead of static memory entries.

Most LLM memory systems use key-value stores or vector retrieval. ABES tries something different: beliefs decay over time, contradict each other, mutate when evidence is ambiguous, and cluster around related concepts. The design calls for 15 specialized agents and reinforcement learning to manage this ecology.

Right now it's mostly Pydantic models and empty function stubs – no magic brain yet.

> **Currently in very early development (Phase 0 – Project Setup & Skeleton)**.
> The codebase is mostly scaffolding and data models at this stage.

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
| **Belief Ecology Engine** | Main loop: decay → detect contradictions → mutate/resolve → re-rank beliefs |
| **15 Specialized Agents** | Perception, Creation, Reinforcement, Mutation, Resolution, RL Policy, etc. |
| **Reinforcement Learning** | Learns decay rates, thresholds, and selection weights from experience |
| **Snapshot & Time Travel** | Captures full state every iteration so you can replay belief evolution |
| **Visualization UI** | Belief graph, timeline explorer, agent activity monitor, RL metrics |
| **Benchmark Lab** | Test scenarios + comparisons against RAG, chat history, static memory |

## Repository Structure

```
backend/          → FastAPI + Pydantic (belief models, ecology loop, agents, RL)
frontend/         → Next.js 14+ with TypeScript and Tailwind
experiments/      → Test scenarios and result analysis
configs/          → Ecology parameters, RL hyperparameters, agent settings
docs/             → Architecture notes, agent specs, research documents
data/             → (future) embeddings, snapshots, experiment logs
infra/            → Docker configs, dev scripts, CI
scripts/          → Utilities and one-off tools
```

## Current Status (December 2025)

**What's done:**
- Project skeleton and directory structure
- Core belief models (`backend/core/models/belief.py`) with full Pydantic v2 validation
- Basic config files and `.gitignore`

**What's next:**
- Storage layer for belief persistence
- Belief Ecology Loop implementation
- Agent system
- Everything else from the spec

## Getting Started (When Ready)

Eventually you'll be able to run this locally:

```bash
# Backend (future)
cd backend
poetry install
uvicorn main:app --reload

# Frontend (future)
cd frontend
pnpm install
pnpm dev
```

Until then: watch this space.

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
