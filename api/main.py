"""FastAPI backend for the Claim Denial Prevention product.

Run locally after installing API deps:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_auth_repository, settings
from api.routes import auth, claims, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create auth tables on startup. In AWS this points to RDS PostgreSQL.
    get_auth_repository().initialize()
    yield


app = FastAPI(
    title="Claim Denial Prevention API",
    version="0.7.0",
    description="FastAPI backend for validation, risk scoring, policy-backed explanations, and remediation recommendations.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(claims.router)


@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "claim-denial-api", "docs": "/docs"}
