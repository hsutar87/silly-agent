from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunker(
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
