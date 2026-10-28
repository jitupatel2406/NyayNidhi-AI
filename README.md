# ⚖️ NyayNidhi AI

### AI-Powered Legal Information Assistant for Bharatiya Nyaya Sanhita (BNS), 2023

NyayNidhi AI is an AI-powered legal information chatbot designed to help people understand India's **Bharatiya Nyaya Sanhita (BNS), 2023** in simple language. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant passages from the BNS document and generate answers grounded in the available legal text.

## 🚀 Features

- **Legal Q&A:** Ask questions about the Bharatiya Nyaya Sanhita, 2023.
- **Retrieval-Augmented Generation:** Retrieves relevant legal passages before generating answers.
- **Source References:** Displays relevant source excerpts and page numbers.
- **Simple Explanations:** Presents legal information in easy-to-understand language.
- **Emergency Helplines:** Provides access to emergency assistance (112) and free legal aid information (NALSA 15100).
- **Interactive Chat Interface:** Built with Streamlit.

## 🛠️ Technology Stack

- **Language:** Python
- **Frontend:** Streamlit
- **LLM Integration:** Groq API with LangChain
- **RAG Framework:** LangChain
- **Vector Database:** Qdrant
- **Embeddings:** Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`)
- **PDF Processing:** PyMuPDF
- **Configuration:** Python dotenv

## 🏗️ Project Structure

```text
NyayNidhi-AI/
├── app/
│   ├── api/
│   ├── rag/
│   │   ├── chain.py
│   │   ├── ingestion.py
│   │   └── vector_store.py
│   ├── services/
│   ├── config.py
│   └── main.py
├── data/
│   └── legal_documents/
│       └── BNS_2023.pdf
├── scripts/
│   └── ingest_documents.py
├── streamlit_app.py
├── requirements.txt
├── test_rag.py
├── test_retrieval.py
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/jitupatel2406/NyayNidhi-AI.git
cd NyayNidhi-AI
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
QDRANT_PATH=./qdrant_data
COLLECTION_NAME=bns_legal_documents
```

Replace `your_groq_api_key` with your own Groq API key. Never commit your `.env` file or expose your API key publicly.

### 5. Ingest the legal document

Build the local vector database from the BNS PDF:

```bash
python -m scripts.ingest_documents
```

Run ingestion again whenever you need to rebuild the local vector database.

### 6. Start the application

```bash
streamlit run streamlit_app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## 🔍 How It Works

1. **Document Ingestion:** Extracts text from the BNS PDF using PyMuPDF.
2. **Text Chunking:** Splits the document into smaller passages suitable for retrieval.
3. **Embedding Generation:** Converts passages into vector embeddings using Sentence Transformers.
4. **Similarity Search:** Retrieves relevant passages from the Qdrant vector database.
5. **Answer Generation:** Sends the question and retrieved context to the Groq-powered language model through LangChain.
6. **Source Display:** Presents the generated response alongside relevant source excerpts and page references.

## 🧪 Testing

Run the retrieval test:

```bash
python test_retrieval.py
```

Run the RAG test:

```bash
python test_rag.py
```

These scripts help verify document retrieval and question-answer generation.

## ⚖️ Legal Disclaimer

NyayNidhi AI provides general legal information for educational and informational purposes only. It does not replace professional legal advice, and its responses may contain errors or omissions. Always verify important legal information against official legislation and consult a qualified lawyer for case-specific guidance.

In an emergency in India, call **112**. For information about free legal aid, visit the [National Legal Services Authority (NALSA)](https://nalsa.gov.in/legal-aid/).

## 👨‍💻 Author

**Jitu Patel**

GitHub: [@jitupatel2406](https://github.com/jitupatel2406)

---

*Built to make legal information more accessible through AI and retrieval-based question answering.*
