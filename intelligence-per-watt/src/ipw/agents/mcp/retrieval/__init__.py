"""Retrieval MCP servers for benchmark integration.

This module provides retrieval capabilities (BM25, dense, grep, hybrid) for
benchmarks that need document retrieval, with full energy/power/latency profiling.

Example:
    from ipw.agents.mcp.retrieval import HybridRetrievalServer, Document

    server = HybridRetrievalServer(telemetry_collector=collector)
    docs = [Document(id="1", content="..."), Document(id="2", content="...")]
    server.index_documents(docs)

    result = server.execute("search query", top_k=5)
"""

from ipw.agents.mcp.retrieval.base import BaseRetrievalServer, Document, RetrievalResult
from ipw.agents.mcp.retrieval.grep_server import GrepRetrievalServer
from ipw.agents.mcp.retrieval.bm25_server import BM25RetrievalServer
from ipw.agents.mcp.retrieval.dense_server import DenseRetrievalServer
from ipw.agents.mcp.retrieval.hybrid_server import HybridRetrievalServer
from ipw.agents.mcp.retrieval.index_manager import IndexManager

__all__ = [
    # Base classes
    "BaseRetrievalServer",
    "Document",
    "RetrievalResult",
    # Server implementations
    "GrepRetrievalServer",
    "BM25RetrievalServer",
    "DenseRetrievalServer",
    "HybridRetrievalServer",
    # Index management
    "IndexManager",
]
