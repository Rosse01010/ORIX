"""
run_demo.py
───────────
Local demo script. Starts the API + workers in-process (no Docker required)
and verifies the pipeline end-to-end.

Usage:
    python run_demo.py

Requirements:
    • .env file present (copy from .env.example and adjust).
    • PostgreSQL + Redis running locally or via Docker.
    • Optional: webcam at index 0.

The script:
  1. Initialises the database.
  2. Registers a synthetic "Demo Person" with a random embedding.
  3. Starts the worker supervisor in a background thread.
  4. Starts the FastAPI server in a background thread.
  5. Connects a WebSocket client and prints received events for 30 seconds.
"""
from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import uuid
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONPATH", str(ROOT))


# ── 1. Init DB + seed data ─────────────────────────────────────────────────────

async def _seed_db() -> None:
    from app.database import init_db, AsyncSessionLocal
    from app.models import Person
    import numpy as np

    await init_db()
    print("[demo] Database initialised.")

    async with AsyncSessionLocal() as session:
        from sqlalchemy import select
        result = await session.execute(select(Person).where(Person.name == "Demo Person"))
        if not result.scalar_one_or_none():
            rng = np.random.default_rng(42)
            emb = rng.standard_normal(512).tolist()
            session.add(Person(name="Demo Person", embedding=emb))
            await session.commit()
            print("[demo] Seeded 'Demo Person' with random embedding.")
        else:
            print("[demo] 'Demo Person' already in DB.")


def seed_db() -> None:
    asyncio.run(_seed_db())


# ── 2. Start API server ────────────────────────────────────────────────────────

def _start_api() -> None:
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        log_level="warning",
        loop="asyncio",
    )


# ── 3. Start workers ───────────────────────────────────────────────────────────

def _start_workers() -> None:
    from workers.main_worker import run_supervisor
    run_supervisor()


# ── 4. WebSocket listener ──────────────────────────────────────────────────────

async def _listen_ws(duration: int = 30) -> None:
    import websockets

    uri = "ws://127.0.0.1:8000/ws/detections"
    print(f"[demo] Connecting to {uri} ...")
    try:
        async with websockets.connect(uri) as ws:
            print(f"[demo] Connected. Listening for {duration}s ...\n")
            deadline = asyncio.get_event_loop().time() + duration
            while asyncio.get_event_loop().time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(msg)
                    if data.get("type") == "ping":
                        continue
                    print(json.dumps(data, indent=2))
                except asyncio.TimeoutError:
                    pass
    except Exception as e:
        print(f"[demo] WebSocket error: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    from app.utils.logging_utils import configure_logging
    configure_logging("WARNING")

    # Seed DB
    seed_db()

    # Start API in background thread
    api_thread = threading.Thread(target=_start_api, daemon=True)
    api_thread.start()
    print("[demo] API server starting on http://127.0.0.1:8000 ...")
    time.sleep(3)

    # Start workers in background process
    mp.set_start_method("spawn", force=True)
    worker_proc = mp.Process(target=_start_workers, daemon=True)
    worker_proc.start()
    print("[demo] Worker supervisor started.")
    time.sleep(2)

    # Verify health
    import httpx
    try:
        r = httpx.get("http://127.0.0.1:8000/health", timeout=5)
        print(f"[demo] GET /health → {r.json()}")
    except Exception as e:
        print(f"[demo] Health check failed: {e}")

    try:
        r = httpx.get("http://127.0.0.1:8000/health/detailed", timeout=5)
        print(f"[demo] GET /health/detailed → {json.dumps(r.json(), indent=2)}")
    except Exception as e:
        print(f"[demo] Detailed health check failed: {e}")

    # Listen to WS events
    try:
        asyncio.run(_listen_ws(duration=30))
    except KeyboardInterrupt:
        pass

    print("[demo] Demo complete.")
    worker_proc.terminate()


if __name__ == "__main__":
    main()
