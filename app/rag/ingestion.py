
from pathlib import Path
import pymupdf

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = Path("data/legal_documents/BNS_2023.pdf")


def extract_and_split():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    documents = []

    with pymupdf.open(PDF_PATH) as pdf:
        print(f"Total pages: {len(pdf)}")

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text().strip()

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": PDF_PATH.name,
                            "page": page_number,
                            "law": "Bharatiya Nyaya Sanhita, 2023",
                        },
                    )
                )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    print(f"Pages containing text: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")

    for i, chunk in enumerate(chunks[:3], start=1):
        print(f"\n--- Chunk {i} ---")
        print(f"Page: {chunk.metadata['page']}")
        print(chunk.page_content[:500])

    return chunks


if __name__ == "__main__":
    extract_and_split()
