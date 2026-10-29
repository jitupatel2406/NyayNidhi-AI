
from app.rag.chain import get_rag_chain

answer_question = get_rag_chain()

question = "What is the punishment for theft?"

result = answer_question(question)

print("\n--- NyayNidhi AI Answer ---\n")
print(result["answer"])

print("\n--- Sources ---\n")
for source in result["sources"]:
    print(f"Page: {source['page']}")
    print(source["content"])
    print()
