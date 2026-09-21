# RAG Compliance Assistant (Legal & Insurance QA)
A high-performance, multilingual **Retrieval-Augmented Generation (RAG)** compliance assistant designed for automated insurance policy and legal document Question-Answering (QA). Built with **FastAPI**, **LangChain**, **Pinecone**, and **Groq (Llama 3.1 8B)**, the system ingests multi-format documents, processes vectors into dynamically isolated namespaces, and delivers concise, context-bound responses in **English, Hindi, and Marathi**.
---
## Technical Architecture & Pipeline
```
  +-----------------------------------------------------------------------------------+
  |                                DOCUMENT INGESTION                                 |
  |  Formats: PDF (PyMuPDF), DOCX (python-docx), TXT, EML (email/BS4), MSG (extract)  |
  +-----------------------------------------------------------------------------------+
                                           |
                                           v
  +-----------------------------------------------------------------------------------+
  |                             TEXT PREPROCESSING & CHUNKING                         |
  |    - ASCII & whitespace cleaning                                                  |
  |    - RecursiveCharacterTextSplitter (chunk_size=250, chunk_overlap=50)            |
  +-----------------------------------------------------------------------------------+
                                           |
                                           v
  +-----------------------------------------------------------------------------------+
  |                           VECTOR EMBEDDING & STORAGE                              |
  |    - Model: HuggingFace sentence-transformers/all-MiniLM-L6-v2                    |
  |    - DB: Pinecone Serverless Index (Dynamically Isolated Request Namespaces)      |
  +-----------------------------------------------------------------------------------+
                                           |
                                           v
  +-----------------------------------------------------------------------------------+
  |                           MULTILINGUAL QUERY PROCESSING                           |
  |    1. Language Detection Chain (Detects: English, Hindi, Marathi)                 |
  |    2. Translation Chain (Translates non-English query -> English)                 |
  |    3. Vector Retrieval (Top-K=7 context search in Pinecone namespace)            |
  +-----------------------------------------------------------------------------------+
                                           |
                                           v
  +-----------------------------------------------------------------------------------+
  |                           LLM RAG ANSWER GENERATION                               |
  |    - Engine: Groq ChatGroq (llama-3.1-8b-instant, temp=0.2, max_tokens=180)       |
  |    - Prompt: Strict 30-40 word formal insurance language constraint              |
  |    - Target Language Re-translation (Bilingual Output: Target + English)          |
  +-----------------------------------------------------------------------------------+
```
---
## Core Features
- **Multi-Format Document Parsing**: Natively extracts clean text from `.pdf`, `.docx`, `.txt`, `.eml`, and `.msg` files.
- **Dynamic Namespace Isolation**: Generates unique session UUID namespaces (`upload_<uuid>` / `run_<uuid>`) per document upload to isolate vector spaces and avoid cross-document contamination.
- **Cross-Lingual Information Retrieval**: Native query detection and translation pipeline supporting **English**, **Hindi**, and **Marathi**.
- **Deterministic & Compliant LLM Generation**: Constrained zero-shot / few-shot prompts forced to output concise (30–40 words) formal policy statements ("shall indemnify", "subject to", etc.) or an explicit fallback (`No relevant policy information found.`).
- **Dual API Access & Web UI**: Offers both a modern Glassmorphism interactive web dashboard and a Bearer-token secured REST endpoint for programmatic integrations.
---
## Project Structure
```
.
├── main.py              # FastAPI application, route handlers, text extraction & API schemas
├── llm.py               # LangChain chains, ChatGroq configuration & translation logic
├── vectorstore.py       # Pinecone index integration, HuggingFace embeddings & batch upserting
├── requirements.txt     # Python dependencies
├── static/
│   ├── index.html       # Web UI layout
│   ├── script.js        # Frontend form handling, dynamic render & file metrics
│   └── styles.css       # Glassmorphism dark mode UI styling
└── README.md            # Technical documentation
```
### Module Responsibilities
1. **[main.py](file:///Users/samikshasolanke/Desktop/RAG-Compliance-Assistant/main.py)**
   - Initialises the **FastAPI** service and mounts static files for the Web UI.
   - Handles Bearer Authentication for protected routes via `HTTPBearer`.
   - Parses incoming files/bytes from uploaded multipart forms or HTTP URLs (`extract_text_from_*` helper methods).
   - Manages query chunking (`RecursiveCharacterTextSplitter`) and triggers Pinecone vector upserting and retrieval.
2. **[llm.py](file:///Users/samikshasolanke/Desktop/RAG-Compliance-Assistant/llm.py)**
   - Configures `ChatGroq` using `llama-3.1-8b-instant` (low temperature `0.2` / `0.1` for precision).
   - `lang_chain`: Detects question language (`English`, `Hindi`, `Marathi`).
   - `to_eng_chain`: Translates foreign query to English before vector matching.
   - `answer_chain`: Formulates strictly formatted, formal insurance answers from retrieved document context.
   - `to_user_chain` / `translate_answer`: Translates the final answer back to the query's original language, returning both English and localized versions.
3. **[vectorstore.py](file:///Users/samikshasolanke/Desktop/RAG-Compliance-Assistant/vectorstore.py)**
   - Loads `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2`.
   - Manages connection to Pinecone serverless vector database (`AWS` / `us-east-1`).
   - `embed_to_pinecone()`: Performs batch upserts (batch size = 20) with metadata (`text`) into targeted request namespaces.
   - `get_relevant_context()`: Embeds search queries and queries Pinecone for `top_k=7` context matches.
4. **[static/](file:///Users/samikshasolanke/Desktop/RAG-Compliance-Assistant/static)**
   - Provides a client-side interface allowing users to upload documents, input multi-line questions, and view formatted responses with smooth animations.
---
## Tech Stack & Dependencies
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/)
- **LLM Engine**: [Groq API](https://groq.com/) (`llama-3.1-8b-instant`)
- **Orchestration Framework**: [LangChain Core / Community](https://www.langchain.com/), `langchain-groq`
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings`
- **Vector Database**: [Pinecone](https://www.pinecone.io/) (Serverless Vector Index)
- **Document Extractors**:
  - `PyMuPDF` (`fitz`) for PDF parsing
  - `python-docx` for Word `.docx` documents
  - `extract-msg` & `beautifulsoup4` for Outlook `.msg` & `.eml` emails
- **Frontend**: HTML5, CSS3 (Vanilla Glassmorphism), JavaScript (Fetch API)
---
## API Reference
### 1. Web Interface
`GET /`
- **Description**: Renders the client web portal for document upload and interactive QA.
- **Response**: `HTMLResponse` (`static/index.html`)
---
### 2. Interactive File Upload QA
`POST /api/v1/hackrx/upload`
- **Description**: Upload a document directly with one or more questions (newline-separated).
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: File upload (`.pdf`, `.docx`, `.txt`, `.eml`, `.msg`)
  - `questions`: String (newline-separated questions)
- **Response Format**:
  ```json
  {
    "answers": [
      {
        "question": "what is the policy period?",
        "answer": "The policy period starts on January 1, 2024, and expires on December 31, 2024, subject to timely premium payment."
      },
      {
        "question": "कवर की गई राशि क्या है?",
        "answer": "**Hindi Translation:**\nबीमा राशि ₹5,00,000 तक की चिकित्सा लागतों को कवर करेगी।\n\n**English Version:**\nThe sum insured shall indemnify medical expenses up to ₹5,00,000."
      }
    ]
  }
  ```
---
### 3. Programmatic URL Document QA (Secured)
`POST /api/v1/hackrx/run`
- **Authentication**: `Bearer <AUTH_KEY>` header required.
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "documents": "https://example.com/sample_policy.pdf",
    "questions": [
      "What are the policy exclusions?",
      "दावा प्रक्रिया क्या है?"
    ]
  }
  ```
- **Response Format**:
  ```json
  {
    "answers": [
      "The policy excludes pre-existing conditions during the first 24 months of coverage.",
      "**Hindi Translation:**\nदावा 30 दिनों के भीतर प्रस्तुत किया जाना चाहिए।\n\n**English Version:**\nClaims must be submitted within 30 days of the occurrence."
    ]
  }
  ```
---
## Setup & Installation
### Prerequisites
- Python 3.9+
- Pinecone Account & Index (`384` dimensional vector index configured for `all-MiniLM-L6-v2`)
- Groq API Key
### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/SamikshaSolanke/RAG-Compliance-Assistant.git
cd RAG-Compliance-Assistant
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Environment & Key Configuration
Ensure the API keys in [vectorstore.py](file:///Users/samikshasolanke/Desktop/RAG-Compliance-Assistant/vectorstore.py), [llm.py](file:///Users/samikshasolanke/Desktop/RAG-Compliance-Assistant/llm.py), and [main.py](file:///Users/samikshasolanke/Desktop/RAG-Compliance-Assistant/main.py) are properly set or configured via environment variables:
- `PINECONE_API_KEY`: Your Pinecone API Key
- `PINECONE_INDEX`: Name of your existing Pinecone index
- `GROQ_API_KEY`: Your Groq API Key
- `AUTH_KEY`: Bearer token key for secured endpoints
### 4. Run Application Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Access the application at `http://localhost:8000` or view the automatic API documentation at `http://localhost:8000/docs`.
---
