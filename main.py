"""
Document Q&A - LangFlow UI
Simple FastAPI backend that proxies requests to LangFlow
"""

import os
import uuid
import tempfile
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import requests

# PDF processing
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: PyMuPDF not installed. PDF upload will not work. Run: pip install pymupdf")

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Document Q&A",
    description="Chat interface for LangFlow RAG pipeline",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangFlow configuration
LANGFLOW_URL = os.getenv(
    "LANGFLOW_URL", 
    "http://localhost:7860/api/v1/run/1af6f82e-e263-4943-87cf-a99cf1d49408"
)
LANGFLOW_API_KEY = os.getenv("LANGFLOW_API_KEY", "")


# Pydantic models
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the web UI"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Visual Document Q&A</h1><p>UI not found. Please ensure index.html exists.</p>",
            status_code=404
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "langflow_url": LANGFLOW_URL,
        "pdf_support": PDF_SUPPORT
    }


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and extract its text content"""
    
    if not PDF_SUPPORT:
        raise HTTPException(
            status_code=503, 
            detail="PDF support not available. Install PyMuPDF: pip install pymupdf"
        )
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Read PDF content
        content = await file.read()
        
        # Create temp file to process
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Extract text from PDF
            doc = fitz.open(tmp_path)
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
            
            doc.close()
            full_text = "\n\n".join(text_parts)
            
            return {
                "filename": file.filename,
                "pages": len(text_parts),
                "text": full_text,
                "chars": len(full_text)
            }
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to LangFlow and get a response"""
    
    session_id = request.session_id or str(uuid.uuid4())
    
    payload = {
        "output_type": "chat",
        "input_type": "chat",
        "input_value": request.question,
        "session_id": session_id
    }
    
    headers = {}
    if LANGFLOW_API_KEY:
        headers["x-api-key"] = LANGFLOW_API_KEY
    
    try:
        response = requests.post(LANGFLOW_URL, json=payload, headers=headers, timeout=180)
        response.raise_for_status()
        result = response.json()
        
        # Extract the answer from LangFlow response
        outputs = result.get("outputs", [])
        if outputs and len(outputs) > 0:
            output_data = outputs[0].get("outputs", [])
            if output_data:
                message = output_data[0].get("results", {}).get("message", {})
                answer = message.get("text", str(result))
            else:
                answer = str(result)
        else:
            answer = str(result)
        
        return ChatResponse(answer=answer, session_id=session_id)
        
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503, 
            detail="LangFlow is not running. Please start LangFlow on port 7860."
        )
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="LangFlow request timed out")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"LangFlow error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

