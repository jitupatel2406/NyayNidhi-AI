
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from app.config import QDRANT_PATH, COLLECTION_NAME


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )

    print(f"Successfully stored {len(chunks)} chunks in Qdrant.")
    return vector_store
