"""Index management and persistence for retrieval servers.

Provides utilities for:
- Saving and loading indices
- Index caching and versioning
- Corpus loading from various sources
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Type, Union

from ipw.agents.mcp.retrieval.base import BaseRetrievalServer, Document


@dataclass
class IndexMetadata:
    """Metadata for a saved index."""

    name: str
    server_type: str
    document_count: int
    corpus_hash: str
    version: str = "1.0"
    extra: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "server_type": self.server_type,
            "document_count": self.document_count,
            "corpus_hash": self.corpus_hash,
            "version": self.version,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexMetadata":
        return cls(
            name=data["name"],
            server_type=data["server_type"],
            document_count=data["document_count"],
            corpus_hash=data["corpus_hash"],
            version=data.get("version", "1.0"),
            extra=data.get("extra", {}),
        )


class IndexManager:
    """Manages index persistence and caching.

    Example:
        manager = IndexManager(cache_dir="./index_cache")

        if not manager.is_valid("my_corpus", corpus_hash):
            server = BM25RetrievalServer()
            server.index_documents(documents)
            manager.save(server, "my_corpus", corpus_hash)
        else:
            server = manager.load("my_corpus", BM25RetrievalServer)
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "./.retrieval_cache",
        auto_clean: bool = True,
        max_cache_entries: int = 50,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.auto_clean = auto_clean
        self.max_cache_entries = max_cache_entries

    def _get_index_path(self, name: str) -> Path:
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        return self.cache_dir / safe_name

    def _compute_corpus_hash(self, documents: List[Document]) -> str:
        hasher = hashlib.sha256()
        for doc in sorted(documents, key=lambda d: d.id):
            hasher.update(doc.id.encode())
            hasher.update(doc.content.encode())
        return hasher.hexdigest()[:16]

    def is_valid(self, name: str, corpus_hash: Optional[str] = None) -> bool:
        index_path = self._get_index_path(name)
        metadata_path = index_path / "index_metadata.json"

        if not metadata_path.exists():
            return False

        if corpus_hash is None:
            return True

        try:
            with open(metadata_path) as f:
                metadata = IndexMetadata.from_dict(json.load(f))
            return metadata.corpus_hash == corpus_hash
        except Exception:
            return False

    def get_metadata(self, name: str) -> Optional[IndexMetadata]:
        index_path = self._get_index_path(name)
        metadata_path = index_path / "index_metadata.json"

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path) as f:
                return IndexMetadata.from_dict(json.load(f))
        except Exception:
            return None

    def save(
        self,
        server: BaseRetrievalServer,
        name: str,
        corpus_hash: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        index_path = self._get_index_path(name)

        if index_path.exists():
            shutil.rmtree(index_path)
        index_path.mkdir(parents=True)

        if hasattr(server, "save_index"):
            server.save_index(index_path / "server_data")
        else:
            docs_data = []
            if hasattr(server, "_documents"):
                docs = server._documents
                if isinstance(docs, dict):
                    docs = list(docs.values())
                docs_data = [
                    {"id": d.id, "content": d.content, "metadata": d.metadata}
                    for d in docs
                ]
            with open(index_path / "documents.json", "w") as f:
                json.dump(docs_data, f)

        metadata = IndexMetadata(
            name=name,
            server_type=server.__class__.__name__,
            document_count=server.document_count,
            corpus_hash=corpus_hash,
            extra=extra_metadata or {},
        )
        with open(index_path / "index_metadata.json", "w") as f:
            json.dump(metadata.to_dict(), f)

        if self.auto_clean:
            self._cleanup_old_entries()

        return index_path

    def load(
        self,
        name: str,
        server_class: Type[BaseRetrievalServer],
        **server_kwargs: Any,
    ) -> Optional[BaseRetrievalServer]:
        index_path = self._get_index_path(name)
        metadata_path = index_path / "index_metadata.json"

        if not metadata_path.exists():
            return None

        try:
            server = server_class(**server_kwargs)

            if hasattr(server, "load_index") and (index_path / "server_data").exists():
                server.load_index(index_path / "server_data")
            elif (index_path / "documents.json").exists():
                with open(index_path / "documents.json") as f:
                    docs_data = json.load(f)
                documents = [
                    Document(
                        id=d["id"],
                        content=d["content"],
                        metadata=d.get("metadata", {}),
                    )
                    for d in docs_data
                ]
                server.index_documents(documents)

            return server
        except Exception as e:
            print(f"Warning: Failed to load index '{name}': {e}")
            return None

    def delete(self, name: str) -> bool:
        index_path = self._get_index_path(name)
        if index_path.exists():
            shutil.rmtree(index_path)
            return True
        return False

    def list_indices(self) -> List[IndexMetadata]:
        indices = []
        for path in self.cache_dir.iterdir():
            if path.is_dir():
                metadata_path = path / "index_metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path) as f:
                            indices.append(IndexMetadata.from_dict(json.load(f)))
                    except Exception:
                        pass
        return indices

    def _cleanup_old_entries(self) -> int:
        indices = self.list_indices()
        if len(indices) <= self.max_cache_entries:
            return 0

        paths = []
        for idx in indices:
            path = self._get_index_path(idx.name)
            if path.exists():
                mtime = path.stat().st_mtime
                paths.append((mtime, idx.name))

        paths.sort()

        to_remove = len(paths) - self.max_cache_entries
        removed = 0
        for _, name in paths[:to_remove]:
            if self.delete(name):
                removed += 1

        return removed


def load_corpus_from_jsonl(
    path: Union[str, Path],
    id_field: str = "id",
    content_field: str = "content",
    metadata_fields: Optional[List[str]] = None,
) -> Iterator[Document]:
    """Load documents from a JSONL file."""
    path = Path(path)
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)

            doc_id = data.get(id_field, str(i))
            content = data.get(content_field, "")

            metadata = {}
            if metadata_fields:
                for field in metadata_fields:
                    if field in data:
                        metadata[field] = data[field]

            if content:
                yield Document(id=str(doc_id), content=content, metadata=metadata)


def load_corpus_from_huggingface(
    dataset_name: str,
    split: str = "train",
    id_field: str = "id",
    content_field: str = "text",
    max_documents: Optional[int] = None,
) -> Iterator[Document]:
    """Load documents from a HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets is required for HuggingFace corpus loading. "
            "Install with: pip install datasets"
        )

    dataset = load_dataset(dataset_name, split=split, streaming=True)

    for i, item in enumerate(dataset):
        if max_documents and i >= max_documents:
            break

        doc_id = item.get(id_field, str(i))
        content = item.get(content_field, "")

        if content:
            yield Document(id=str(doc_id), content=content)
