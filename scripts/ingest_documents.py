
from app.rag.ingestion import extract_and_split
from app.rag.vector_store import create_vector_store


def main():
    chunks = extract_and_split()
    create_vector_store(chunks)


if __name__ == "__main__":
    main()
