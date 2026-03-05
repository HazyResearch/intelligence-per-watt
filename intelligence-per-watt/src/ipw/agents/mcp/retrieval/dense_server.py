"""Dense neural retrieval server using FAISS and sentence-transformers.

Implements semantic search using dense embeddings:
- Uses sentence-transformers for encoding
- FAISS for fast similarity search
- Supports both CPU and GPU indices
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Optional, Union

from ipw.agents.mcp.base import MCPToolResult
from ipw.agents.mcp.retrieval.base import BaseRetrievalServer, Document, RetrievalResult


class DenseRetrievalServer(BaseRetrievalServer):
    """Dense neural retrieval server using FAISS + sentence-transformers.

    Latency: ~50ms per query
    Cost: Zero (local inference)

    Example:
        server = DenseRetrievalServer(model_name="all-MiniLM-L6-v2")
        server.index_documents([
            Document(id="1", content="Machine learning automates data analysis."),
        ])
        result = server.execute("AI learns patterns from data", top_k=5)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        telemetry_collector: Optional[Any] = None,
        event_recorder: Optional[Any] = None,
        use_gpu: bool = False,
        gpu_device: int = 0,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ):
        super().__init__(
            name="retrieval:dense",
            telemetry_collector=telemetry_collector,
            event_recorder=event_recorder,
        )
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

        self._encoder: Optional[Any] = None
        self._index: Optional[Any] = None
        self._documents: List[Document] = []
        self._embedding_dim: Optional[int] = None

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for DenseRetrievalServer. "
                    "Install with: pip install sentence-transformers"
                )

            if self.use_gpu:
                device = f"cuda:{self.gpu_device}"
            else:
                device = "cpu"

            self._encoder = SentenceTransformer(self.model_name, device=device)
            self._embedding_dim = self._encoder.get_sentence_embedding_dimension()

        return self._encoder

    def _create_faiss_index(self, dimension: int) -> Any:
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for DenseRetrievalServer. "
                "Install with: pip install faiss-cpu"
            )

        if self.normalize_embeddings:
            index = faiss.IndexFlatIP(dimension)
        else:
            index = faiss.IndexFlatL2(dimension)

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, self.gpu_device, index)
            except Exception:
                pass

        return index

    def index_documents(self, documents: List[Document]) -> int:
        import numpy as np

        if not documents:
            return 0

        encoder = self._get_encoder()
        self._documents = list(documents)

        texts = [doc.content for doc in documents]
        embeddings = encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
        )

        embeddings = np.array(embeddings, dtype=np.float32)

        self._index = self._create_faiss_index(embeddings.shape[1])
        self._index.add(embeddings)

        self._document_count = len(self._documents)
        return self._document_count

    def clear_index(self) -> None:
        self._documents.clear()
        self._index = None
        self._document_count = 0

    def _search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        import numpy as np

        if self._index is None or not self._documents:
            return []

        encoder = self._get_encoder()

        query_embedding = encoder.encode(
            [query],
            normalize_embeddings=self.normalize_embeddings,
        )
        query_embedding = np.array(query_embedding, dtype=np.float32)

        k = min(top_k, len(self._documents))
        scores, indices = self._index.search(query_embedding, k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(self._documents):
                doc = self._documents[idx]
                highlights = self._generate_highlights(doc.content, query)
                results.append(
                    RetrievalResult(
                        document=doc,
                        score=float(score),
                        highlights=highlights,
                    )
                )

        return results

    def _generate_highlights(
        self, content: str, query: str, max_highlights: int = 3
    ) -> List[str]:
        sentences = re.split(r"[.!?]\s+", content)

        highlights = []
        for sentence in sentences[:max_highlights]:
            sentence = sentence.strip()
            if sentence:
                if len(sentence) > 150:
                    sentence = sentence[:150] + "..."
                highlights.append(sentence)

        return highlights

    def save_index(self, path: Union[str, Path]) -> None:
        import json

        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._index is not None:
            faiss.write_index(self._index, str(path / "index.faiss"))

        docs_data = [
            {"id": doc.id, "content": doc.content, "metadata": doc.metadata}
            for doc in self._documents
        ]
        with open(path / "documents.json", "w") as f:
            json.dump(docs_data, f)

        meta = {
            "model_name": self.model_name,
            "document_count": self._document_count,
            "embedding_dim": self._embedding_dim,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f)

    def load_index(self, path: Union[str, Path]) -> None:
        import json

        import faiss

        path = Path(path)

        self._index = faiss.read_index(str(path / "index.faiss"))

        with open(path / "documents.json") as f:
            docs_data = json.load(f)
        self._documents = [
            Document(id=d["id"], content=d["content"], metadata=d.get("metadata", {}))
            for d in docs_data
        ]

        with open(path / "metadata.json") as f:
            meta = json.load(f)
        self._document_count = meta["document_count"]
        self._embedding_dim = meta.get("embedding_dim")

    def _execute_impl(self, prompt: str, **params: Any) -> MCPToolResult:
        top_k = params.get("top_k", 5)
        include_scores = params.get("include_scores", True)
        include_metadata = params.get("include_metadata", False)

        if self._index is None:
            return MCPToolResult(
                content="No documents indexed. Call index_documents() first.",
                cost_usd=0.0,
                metadata={"tool": "retrieval:dense", "error": "no_index"},
            )

        results = self._search(prompt, top_k=top_k)

        content = self._format_results(
            results,
            include_scores=include_scores,
            include_metadata=include_metadata,
        )

        return MCPToolResult(
            content=content,
            cost_usd=0.0,
            metadata={
                "tool": "retrieval:dense",
                "query": prompt,
                "num_results": len(results),
                "top_k": top_k,
                "model": self.model_name,
                "indexed_documents": self._document_count,
            },
        )
