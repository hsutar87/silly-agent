# 💃 Silly Agent: Witty, Concise, and Dangerous

> “Information is a heavy burden; let me carry it for you. Just don’t expect me to be nice about it.” — **Silly**

Silly Agent is a local, RAG-powered AI assistant that is sharp, concise, and context-aware. It retrieves answers from your local documents using OpenSearch vectors and Ollama embeddings, and exposes tools via FastMCP.

---

## ✨ Features

- Local RAG with OpenSearch vector index
- FastMCP tool server (SSE transport)
- Witty persona with short answers
- Sliding-window conversation memory
- Ollama embeddings + LLM (local inference)

---

## 🛠 Prerequisites

- Python 3.10+
- OpenSearch running locally (HTTPS)
- Ollama installed with required models
- `uv` (optional, recommended)

---

## 🔥 OpenSearch Verification
```bash
curl -k -u <USER>:<PASS> https://localhost:9200
```
## 🧩 Ollama Models
```bash
ollama pull deepseek-r1:8b
ollama pull mxbai-embed-large
```

---

## 🚀 Installation

```bash
git clone https://github.com/hsutar87/silly-agent.git
cd silly-agent
uv sync
```

Or with pip:

```bash
pip install -e .
```

---

## 🔐 Environment Variables

Create a `.env` file:

```env
OPENSEARCH_USER=admin
OPENSEARCH_PASS=<YOUR_PASS>
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
```

Add `.env` to `.gitignore`.

---

## 🧾 CA Certificate (Ubuntu)

```bash
sudo cp /etc/opensearch/root-ca.pem ./root-ca.pem
sudo chown $USER:$USER ./root-ca.pem
```

---

## 📖 Usage

### 1. Start the MCP Server

```bash
uv run python -m src.mcp_server
```

### 2. Ingest Documents

Put PDFs/DOCX/TXT in `data/` and run:

```bash
uv run python main.py --ingest
```

### 3. Chat

```bash
uv run python main.py
```

---

## 💬 Commands

* `clear`: wipe memory
* `exit` / `quit`: exit
* Normal text: ask a question

---

## 📂 Project Structure

* `src/reader.py` — file parsing
* `src/vector_store.py` — OpenSearch + Ollama embeddings
* `src/memory.py` — conversation memory
* `src/agent.py` — agent logic
* `src/mcp_server.py` — FastMCP tools
* `main.py` — ingestion + chat loop
---


## 📝 License
This project is licensed under the [MIT License](LICENSE).
