import uuid
from typing import List, Dict, Any
from tqdm import tqdm

from utils.env import load_env, env
from utils.embeddings import get_embedding
from utils.opensearch import create_client
from utils.chunker import create_chunker


class VectorStoreManager:
    def __init__(
        self,
        index_name: str = "rag_docs",
        embedding_model: str = "mxbai-embed-large",
    ):
        load_env()

        self.index_name = index_name
        self.embedding_model = embedding_model

        # OpenSearch config
        self.client = create_client(
            host=env("OPENSEARCH_HOST", "localhost"),
            port=int(env("OPENSEARCH_PORT", 9200)),
            user=env("OPENSEARCH_USER", "admin"),
            password=env("OPENSEARCH_PASS", ""),
            ca_certs=env("OPENSEARCH_CA_CERT", "./root-ca.pem"),
        )

        self.ollama_url = env("OLLAMA_EMBED_URL", "http://localhost:11434/api/embed")
        self.chunker = create_chunker()

        self._ensure_index()

    # -------------------------
    # Index management
    # -------------------------

    def _ensure_index(self):
        if self.client.indices.exists(index=self.index_name):
            return

        self.client.indices.create(
            index=self.index_name,
            body={
                "settings": {"index.knn": True},
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "vector": {
                            "type": "knn_vector",
                            "dimension": 1024,
                        },
                        "source": {"type": "keyword"},
                    }
                },
            },
        )

    # -------------------------
    # Core operations
    # -------------------------

    def ingest(self, content: str, metadata: Dict[str, Any]):
        chunks = self.chunker.split_text(content)
        print(f"Splitting into {len(chunks)} chunks…")

        for chunk in tqdm(chunks, desc="Indexing"):
            vector = self._embed(chunk)
            if not vector:
                continue

            self.client.index(
                index=self.index_name,
                id=str(uuid.uuid4()),
                body={
                    "content": chunk,
                    "vector": vector,
                    **metadata,
                },
            )

    def rebuild_index(self, docs: List[Dict[str, Any]]):
        self.client.indices.delete(
            index=self.index_name,
            ignore_unavailable=True,
        )
        self._ensure_index()

        for doc in docs:
            self.ingest(doc["content"], doc["metadata"])

    def update_delta(self, docs: List[Dict[str, Any]]):
        for doc in docs:
            self.ingest(doc["content"], doc["metadata"])

    # -------------------------
    # Search
    # -------------------------

    def search(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        query_vector = self._embed(query_text)
        if not query_vector:
            return []

        body = {
            "size": limit,
            "query": {
                "knn": {
                    "vector": {
                        "vector": query_vector,
                        "k": limit,
                    }
                }
            },
        }

        response = self.client.search(
            index=self.index_name,
            body=body,
        )

        hits = response.get("hits", {}).get("hits", [])
        return [
            {
                "content": h["_source"]["content"],
                "score": h["_score"],
                "source": h["_source"].get("source", "Unknown"),
            }
            for h in hits
        ]

    # -------------------------
    # Internal helpers
    # -------------------------

    def _embed(self, text: str) -> List[float]:
        try:
            return get_embedding(
                text=text,
                model=self.embedding_model,
                url=self.ollama_url,
            )
        except Exception as e:
            print("Embedding failed:", e)
            return []
