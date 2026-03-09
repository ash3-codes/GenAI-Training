# 📘 HR Policy Assistant (RAG-Based AI System)

An AI-powered assistant that allows employees to query internal HR policy documents using natural language. The system uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant policy sections and generate grounded answers with citations.

---

## 🚀 Features

- Natural language querying of HR policies
- Accurate responses grounded in official documents
- Source citations with document name and page number
- Conversational interaction
- Acronym expansion (POSH, BGV, FNF, etc.)
- Spell correction and query normalization
- LLM-based reranking for better retrieval quality
- Streamlit chat interface
- Session-based conversation memory

---

## 🧠 System Architecture

The system follows a Retrieval-Augmented Generation (RAG) architecture:

```
User Query
     │
     ▼
Query Intelligence
     │
     ▼
Follow-up Query Rewrite
     │
     ▼
Vector Retrieval (Qdrant)
     │
     ▼
Top K Candidate Chunks
     │
     ▼
LLM Reranking
     │
     ▼
Top Relevant Chunks
     │
     ▼
Context Builder
     │
     ▼
Answer Generation (Azure OpenAI)
     │
     ▼
Response + Citations
```

---

## 🧱 Project Structure

```
hr-policy-bot/
│
├── app.py
│
├── config/
│   └── settings.py
│
├── ingestion/
│   ├── loader.py
│   ├── structure_parser.py
│   ├── chunker.py
│   └── metadata_builder.py
│
├── vectorstore/
│   ├── qdrant_client.py
│   ├── indexer.py
│   └── schema.py
│
├── retriever/
│   ├── vector_retriever.py
│   ├── gpt_reranker.py
│   └── context_builder.py
│
├── llm/
│   ├── query_intelligence.py
│   ├── followup_rewriter.py
│   └── answer_engine.py
│
├── memory/
│   └── conversation_memory.py
│
├── scripts/
│   └── run_ingestion.py
│
├── data/
│   └── policy_docs/
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd hr-policy-bot
```

### 2. Create environment

Using Conda:

```bash
conda create -n llm python=3.11
conda activate llm
```

Or using venv:

```bash
python -m venv llm
source llm/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Configuration

Create a `.env` file in the project root with the following variables:

```env
ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_KEY=your_api_key
API_VERSION=2024-02-15-preview
DEPLOYMENT_NAME=chat_model_name

DEPLOYMENT_NAME_EMBEDDING=embedding_model_name
API_VERSION_EMBEDDING=2024-02-15-preview

QDARNT_ENDPOINT=qdrant_cluster_url
QDARNT_API_KEY=qdrant_api_key
```

---

## 📄 Add Policy Documents

Place all HR policy PDFs inside the following directory:

```
data/policy_docs/
```

**Example:**

```
data/policy_docs/
    Leave Policy.pdf
    Travel Policy.pdf
    Exit Policy.pdf
```

---

## 📥 Run Document Ingestion

This step parses documents and stores embeddings in Qdrant:

```bash
python scripts/run_ingestion.py
```

---

## 🖥 Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## 💬 Example Queries

- How many sick leaves are allowed per year?
- What is the POSH policy?
- What happens during full and final settlement?
- Do interns get leave benefits?
- Explain the exit policy.

---

## 🧩 Core Components

| Component | Description |
|-----------|-------------|
| **Query Intelligence** | Handles query cleaning, acronym expansion, spell correction, and intent classification |
| **Vector Retrieval** | Uses Qdrant to perform semantic search on document embeddings |
| **LLM Reranker** | Uses an LLM to rerank retrieved chunks and select the most relevant ones |
| **Context Builder** | Constructs the final context passed to the answer generation model |
| **Answer Engine** | Generates grounded responses using Azure OpenAI |
| **Conversation Memory** | Stores session chat history to support follow-up questions |

---

## 📊 Technology Stack

| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| LLM | Azure OpenAI |
| Embeddings | Azure OpenAI |
| Vector Database | Qdrant |
| Language Framework | LangChain |
| PDF Parsing | pypdf |
| Programming Language | Python |

---

## 🔒 Security Considerations

- Documents may contain confidential company information
- Access should be restricted to internal users only
- Avoid external exposure of policy documents

---

## ⚡ Performance

Typical response latency: **1–3 seconds**, depending on retrieval and LLM inference time.

---

## 🚧 Future Improvements

- Hybrid search (vector + keyword)
- Confidence scoring
- Clarification engine for ambiguous queries
- Hallucination detection
- Document version management
- Interactive citation links
- Evaluation framework for RAG accuracy
