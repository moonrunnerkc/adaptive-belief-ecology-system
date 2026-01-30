# Author: Bradley R. Kinnard
"""
FastAPI application for ABES.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import (
    agents_router,
    bel_router,
    beliefs_router,
    clusters_router,
    snapshots_router,
)

app = FastAPI(
    title="ABES API",
    description="Adaptive Belief Ecology System API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routers
app.include_router(beliefs_router)
app.include_router(snapshots_router)
app.include_router(agents_router)
app.include_router(bel_router)
app.include_router(clusters_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "ABES API",
        "version": "0.1.0",
        "docs": "/docs",
    }


__all__ = ["app"]
