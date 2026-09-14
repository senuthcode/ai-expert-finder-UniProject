"""
Vector Store & Retrieval Module using ChromaDB and Dense Embeddings.
"""

from typing import List, Dict, Any, Optional


class ResumeVectorIndex:
    """Manages dense semantic indexing and similarity retrieval for candidate resumes."""

    def __init__(
        self,
        collection_name: str = "resumes_collection",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.persist_directory = persist_directory
        self._collection = None

    def initialize_client(self):
        """Initializes ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )

            if self.persist_directory:
                client = chromadb.PersistentClient(path=self.persist_directory)
            else:
                client = chromadb.EphemeralClient()

            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=emb_fn,
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except ImportError:
            return False

    def add_resumes(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ):
        """Adds preprocessed resume texts to the vector index."""
        if not self._collection:
            self.initialize_client()
        self._collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(
        self,
        job_query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Queries vector index for most semantically similar resumes."""
        if not self._collection:
            self.initialize_client()

        where_clause = {"category": category_filter} if category_filter else None

        results = self._collection.query(
            query_texts=[job_query],
            n_results=top_k,
            where=where_clause
        )

        candidates = []
        if results and "ids" in results and results["ids"]:
            for i in range(len(results["ids"][0])):
                candidates.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i] if "documents" in results else "",
                    "metadata": results["metadatas"][0][i] if "metadatas" in results else {},
                    "distance": results["distances"][0][i] if "distances" in results else 0.0
                })
        return candidates
