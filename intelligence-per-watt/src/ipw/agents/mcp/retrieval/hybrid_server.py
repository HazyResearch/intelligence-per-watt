"""Hybrid retrieval server combining BM25 + dense with RRF fusion.

Combines the strengths of sparse (BM25) and dense retrieval:
- BM25 for exact keyword matching
- Dense for semantic understanding
- Reciprocal Rank Fusion (RRF) for combining results

Best accuracy at the cost of higher latency (~100ms).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ipw.agents.mcp.base import MCPToolResult
from ipw.agents.mcp.retrieval.base import BaseRetrievalServer, Document, RetrievalResult
from ipw.agents.mcp.retrieval.bm25_server import BM25RetrievalServer
from ipw.agents.mcp.retrieval.dense_server import DenseRetrievalServer


class HybridRetrievalServer(BaseRetrievalServer):
    """Hybrid BM25 + dense retrieval with RRF fusion.

    RRF Formula: score(d) = sum(1 / (k + rank(d))) for each retriever

    Latency: ~100ms per query
    Cost: Zero (local inference)

    Example:
        server = HybridRetrievalServer()
        server.index_documents([
            Document(id="1", content="Machine learning automates data analysis."),
        ])
        result = server.execute("ML data patterns", top_k=5)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        telemetry_collector: Optional[Any] = None,
        event_recorder: Optional[Any] = None,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        rrf_k: int = 60,
        use_gpu: bool = False,
        gpu_device: int = 0,
    ):
        super().__init__(
            name="retrieval:hybrid",
            telemetry_collector=telemetry_collector,
            event_recorder=event_recorder,
        )
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k

        self._bm25 = BM25RetrievalServer(
            telemetry_collector=telemetry_collector,
            event_recorder=event_recorder,
        )
        self._dense = DenseRetrievalServer(
            model_name=model_name,
            telemetry_collector=telemetry_collector,
            event_recorder=event_recorder,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
        )
        self._documents: Dict[str, Document] = {}

    def index_documents(self, documents: List[Document]) -> int:
        self._documents = {doc.id: doc for doc in documents}

        bm25_count = self._bm25.index_documents(documents)
        dense_count = self._dense.index_documents(documents)

        self._document_count = min(bm25_count, dense_count)
        return self._document_count

    def clear_index(self) -> None:
        self._documents.clear()
        self._bm25.clear_index()
        self._dense.clear_index()
        self._document_count = 0

    def _rrf_fusion(
        self,
        bm25_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        bm25_ranks: Dict[str, int] = {
            r.document.id: i + 1 for i, r in enumerate(bm25_results)
        }
        dense_ranks: Dict[str, int] = {
            r.document.id: i + 1 for i, r in enumerate(dense_results)
        }

        all_doc_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())

        rrf_scores: Dict[str, float] = {}
        for doc_id in all_doc_ids:
            score = 0.0
            if doc_id in bm25_ranks:
                score += self.bm25_weight / (self.rrf_k + bm25_ranks[doc_id])
            if doc_id in dense_ranks:
                score += self.dense_weight / (self.rrf_k + dense_ranks[doc_id])
            rrf_scores[doc_id] = score

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for doc_id in sorted_ids[:top_k]:
            doc = self._documents.get(doc_id)
            if doc is None:
                continue

            highlights = []
            for r in bm25_results:
                if r.document.id == doc_id:
                    highlights.extend(r.highlights)
                    break
            for r in dense_results:
                if r.document.id == doc_id:
                    for h in r.highlights:
                        if h not in highlights:
                            highlights.append(h)
                    break

            results.append(
                RetrievalResult(
                    document=doc,
                    score=rrf_scores[doc_id],
                    highlights=highlights[:3],
                )
            )

        return results

    def _search(
        self,
        query: str,
        top_k: int = 5,
        bm25_candidates: int = 20,
        dense_candidates: int = 20,
    ) -> List[RetrievalResult]:
        bm25_results = self._bm25._search(query, top_k=bm25_candidates)
        dense_results = self._dense._search(query, top_k=dense_candidates)

        return self._rrf_fusion(bm25_results, dense_results, top_k)

    def save_index(self, path: Union[str, Path]) -> None:
        import json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._dense.save_index(path / "dense")

        meta = {
            "bm25_weight": self.bm25_weight,
            "dense_weight": self.dense_weight,
            "rrf_k": self.rrf_k,
            "document_count": self._document_count,
        }
        with open(path / "hybrid_metadata.json", "w") as f:
            json.dump(meta, f)

    def load_index(self, path: Union[str, Path]) -> None:
        import json

        path = Path(path)

        self._dense.load_index(path / "dense")
        self._documents = {doc.id: doc for doc in self._dense._documents}
        self._bm25.index_documents(list(self._documents.values()))

        with open(path / "hybrid_metadata.json") as f:
            meta = json.load(f)
        self._document_count = meta["document_count"]

    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        top_k = params.get("top_k", 5)
        bm25_candidates = params.get("bm25_candidates", 20)
        dense_candidates = params.get("dense_candidates", 20)
        include_scores = params.get("include_scores", True)
        include_metadata = params.get("include_metadata", False)

        if self._document_count == 0:
            return MCPToolResult(
                content="No documents indexed. Call index_documents() first.",
                cost_usd=0.0,
                metadata={"tool": "retrieval:hybrid", "error": "no_index"},
            )

        results = self._search(
            prompt,
            top_k=top_k,
            bm25_candidates=bm25_candidates,
            dense_candidates=dense_candidates,
        )

        content = self._format_results(
            results,
            include_scores=include_scores,
            include_metadata=include_metadata,
        )

        return MCPToolResult(
            content=content,
            cost_usd=0.0,
            metadata={
                "tool": "retrieval:hybrid",
                "query": prompt,
                "num_results": len(results),
                "top_k": top_k,
                "bm25_weight": self.bm25_weight,
                "dense_weight": self.dense_weight,
                "indexed_documents": self._document_count,
            },
        )
