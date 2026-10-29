from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import QDRANT_PATH, COLLECTION_NAME

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = QdrantClient(path=QDRANT_PATH)

try:
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    question = "What is the punishment for theft under Section 303 of the Bharatiya Nyaya Sanhita?"

    documents = vector_store.similarity_search(
        question,
        k=10,
    )

    print(f"\nRetrieved {len(documents)} documents.\n")

    for i, doc in enumerate(documents, start=1):
        print(f"--- Result {i} ---")
        print(f"Page: {doc.metadata.get('page', 'Unknown')}")
        print(doc.page_content[:500])
        print()

finally:
    client.close()