"""Standalone retrieval plugin.

Applies STAR-based event-memory retrieval as a post-prediction step on any
standard model (GRU, Informer, etc.) without requiring the AAForecast model
surface.  The authoritative entry point is the ``RetrievalStagePlugin``
registered in ``plugin.py``.
"""
