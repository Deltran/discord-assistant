"""Vector store with hybrid search using ChromaDB."""

from pathlib import Path

import chromadb


class VectorMemory:
    def __init__(self, persist_dir: Path):
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name="messages",
            metadata={"hnsw:space": "cosine"},
        )
        self._id_counter = self._collection.count()

    def add(self, *, text: str, metadata: dict) -> str:
        doc_id = f"doc-{self._id_counter}"
        self._id_counter += 1
        self._collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id],
        )
        return doc_id

    def search(self, query: str, *, k: int = 5) -> list[dict]:
        if self._collection.count() == 0:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count()),
        )
        items = []
        for i in range(len(results["documents"][0])):
            items.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        return items
