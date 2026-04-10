"""
Backward-compatibility shim — vector search is now in unified_db.py.

Import UnifiedGraphDB from storage.unified_db for the single-transaction
Graph+Vector layer that prevents split-brain sync issues.
"""
from app.osint_graph.storage.unified_db import UnifiedGraphDB as VectorStore  # noqa: F401

__all__ = ["VectorStore"]
