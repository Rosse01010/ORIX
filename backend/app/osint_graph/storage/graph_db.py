"""
Backward-compatibility shim — all graph DB operations are now in unified_db.py.

Import UnifiedGraphDB from storage.unified_db for the single-transaction
Graph+Vector layer that prevents split-brain sync issues.
"""
from app.osint_graph.storage.unified_db import UnifiedGraphDB as GraphDB  # noqa: F401

__all__ = ["GraphDB"]
