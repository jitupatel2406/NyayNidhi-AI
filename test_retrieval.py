
from app.rag.chain import get_retriever

retriever = get_retriever()

question = "What is the punishment for murder under BNS?"

documents = retriever.invoke(question)

print(f"\nRetrieved {len(documents)} documents.\n")

for i, doc in enumerate(documents, start=1):
    print(f"--- Result {i} ---")
    print(f"Page: {doc.metadata.get('page', 'Unknown')}")
    print(doc.page_content[:500])
    print()
