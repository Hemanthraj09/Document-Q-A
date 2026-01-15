# Document Q&A

A clean chat interface for your LangFlow RAG pipeline.

## Features

- **Simple Chat UI**: Modern, responsive chat interface
- **LangFlow Integration**: Proxies requests to your LangFlow RAG pipeline
- **Session Management**: Maintains conversation context

## Prerequisites

- Python 3.8+
- LangFlow running on `localhost:7860`

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure LangFlow URL (Optional)

Create a `.env` file if your LangFlow uses a different URL or requires an API key:

```env
LANGFLOW_URL=http://localhost:7860/api/v1/run/YOUR-FLOW-ID
LANGFLOW_API_KEY=your-api-key-if-needed
```

### 3. Start LangFlow

```bash
# Skip auth for local development
$env:LANGFLOW_SKIP_AUTH_AUTO_LOGIN="true"; langflow run
```

### 4. Run the Server

```bash
uvicorn main:app --reload
```

### 5. Open the UI

Navigate to [http://localhost:8000](http://localhost:8000)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve chat UI |
| `/chat` | POST | Send message to LangFlow |
| `/health` | GET | Health check |

## Project Structure

```
Visual Document Q&A/
├── main.py           # FastAPI server (~80 lines)
├── index.html        # Chat UI (~280 lines)
├── requirements.txt  # 4 dependencies
└── .env              # LangFlow config (optional)
```

## Author

**Hemanth Raj** - [GitHub](https://github.com/Hemanthraj09)
