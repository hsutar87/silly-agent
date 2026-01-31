import requests
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional

class VectorStoreManager:
    """
    Manages local vector storage using ChromaDB and Ollama for embeddings.
    No Docker required; data is stored in a local directory.
    """

    def __init__(self, storage_path: str = "./chroma_db", collection_name: str = "rag_docs"):
        """
        Initialize ChromaDB persistent client.
        
        Args:
            storage_path: Local directory to save the database.
            collection_name: Name of the collection (similar to an index).
        """
        self.client = chromadb.PersistentClient(path=storage_path)
        
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        self.ollama_url = "http://localhost:11434/api/embeddings"
        self.embedding_model = "mxbai-embed-large"  # 1024 dimensions
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )

    def _get_embedding(self, text: str) -> List[float]:
        """
        Calls Ollama API to generate embeddings.
        
        Args:
            text: The string to embed.
        Returns:
            A list of floats representing the vector.
        """
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": self.embedding_model, "prompt": text}
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            raise

    def ingest(self, content: str, metadata: Dict[str, Any]):
        """
        Chunks text, generates embeddings via Ollama, and stores in ChromaDB.
        
        Args:
            content: Raw text from the document.
            metadata: Dictionary containing source, type, etc.
        """
        chunks = self.splitter.split_text(content)
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{metadata['source']}_chunk_{i}"
            vector = self._get_embedding(chunk)
            
            ids.append(chunk_id)
            embeddings.append(vector)
            documents.append(chunk)
            metadatas.append({**metadata, "chunk_index": i})

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print(f"Successfully ingested {len(chunks)} chunks from {metadata['source']}")

    def search(self, query_text: str, source_filter: Optional[str] = None, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Performs a semantic search with optional metadata filtering.
        
        Args:
            query_text: The user's question.
            source_filter: Optional filename to restrict search.
            limit: Number of results to return.
            
        Returns:
            List of dictionaries containing content and metadata.
        """
        query_vector = self._get_embedding(query_text)
        
       
        where_clause = {"source": source_filter} if source_filter else None

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where=where_clause
        )

        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })
            
        return formatted_results