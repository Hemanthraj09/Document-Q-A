# Document Q&A

📄 **Ask questions about your documents using AI** — A FastAPI-based chat application that integrates with LangFlow's RAG (Retrieval Augmented Generation) pipeline to provide intelligent, context-aware answers.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi&logoColor=white)
![LangFlow](https://img.shields.io/badge/LangFlow-RAG-purple)

## ✨ Features

- **📤 PDF Upload** — Extract text from PDF documents automatically
- **📝 Text Input** — Paste any text content directly
- **💬 Chat Interface** — Modern, responsive UI for asking questions
- **🔗 LangFlow Integration** — Powered by your custom RAG pipeline
- **🔄 Session Management** — Maintains conversation context

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- LangFlow running on `localhost:7860`

### 1. Clone & Install

```bash
git clone https://github.com/Hemanthraj09/Document-Q-A.git
cd Document-Q-A
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file (optional):

```env
LANGFLOW_URL=http://localhost:7860/api/v1/run/YOUR-FLOW-ID
LANGFLOW_API_KEY=your-api-key-if-needed
```

### 3. Start LangFlow

```bash
# Skip auth for local development (PowerShell)
$env:LANGFLOW_SKIP_AUTH_AUTO_LOGIN="true"; langflow run
```

### 4. Run the Server

```bash
uvicorn main:app --reload
```

### 5. Open the UI

Navigate to **[http://localhost:8000](http://localhost:8000)**

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve chat UI |
| `/chat` | POST | Send message to LangFlow |
| `/upload-pdf` | POST | Upload and extract text from PDF |
| `/health` | GET | Health check |

## 📁 Project Structure

```
Document-Q-A/
├── main.py           # FastAPI server
├── index.html        # Chat UI
├── requirements.txt  # Python dependencies
├── .env.example      # Environment template
├── .gitignore        # Git ignore rules
└── samples/          # Sample documents
```

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn
- **Frontend**: Vanilla HTML/CSS/JS
- **PDF Processing**: PyMuPDF
- **AI/RAG**: LangFlow

## 📝 License

MIT License

## 👤 Author

**Hemanth Raj** — [GitHub](https://github.com/Hemanthraj09)
