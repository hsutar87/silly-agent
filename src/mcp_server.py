from fastmcp import FastMCP
from .vector_store import VectorStoreManager

mcp = FastMCP("Silly-Knowledge-Server")
store = VectorStoreManager()

@mcp.tool()
def search_local_docs(query: str) -> str:
    """Search the local knowledge base for documents. Returns text and source filenames."""
    results = store.search(query, limit=5)
    if not results:
        return "RESULT_EMPTY: No documents matched this query."
    
    return "\n\n".join([
        f"SOURCE: {r['metadata']['source']}\nCONTENT: {r['content']}" 
        for r in results
    ])

@mcp.tool()
def list_available_files() -> str:
    """Returns a list of all filenames currently in the database."""
    data = store.collection.get(include=['metadatas'])
    sources = sorted({m.get('source') for m in data['metadatas'] if m.get('source')})
    return "Files in DB: " + ", ".join(sources)

if __name__ == "__main__":
    # Start SSE server on port 8000
    mcp.run(transport="sse", port=8000)