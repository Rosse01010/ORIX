"""
Backward-compatibility shim — entity linking is now in intelligence/entity_linker.py.
"""
from app.osint_graph.intelligence.entity_linker import EntityLinker  # noqa: F401

__all__ = ["EntityLinker"]
