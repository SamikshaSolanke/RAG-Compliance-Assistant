# main.py

import os
import fitz  # PyMuPDF
import tempfile
import re
import requests
import email
from bs4 import BeautifulSoup
import extract_msg
from docx import Document
from typing import List
from fastapi import FastAPI, Request, Body, HTTPException, Depends
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from fastapi.security import HTTPBearer
from fastapi.openapi.utils import get_openapi
from fastapi import Security
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from vectorstore import embed_to_pinecone, get_relevant_context
from llm import get_llm_answer


security_scheme = HTTPBearer()

app = FastAPI(
    title="HackRx AI Assistant",
    description="Extract answers from uploaded documents using LLaMA 3 + Pinecone",
    version="2.0.0"
)

# Inject Bearer Auth into OpenAPI docs
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method.setdefault("security", [{"BearerAuth": []}])
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

AUTH_KEY = "84ae11520a9ef092d711f960ce31d92fc90a47b0f38aa3202fe16ac543ef9871"

# ---- Auth Dependency ----
def verify_token(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = auth.split("Bearer ")[1]
    if token != AUTH_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")

# ---- Utility Functions ----
def clean_text(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
    )
    return splitter.split_text(text)

def extract_text_from_pdf_bytes(data: bytes):
    with fitz.open(stream=data, filetype="pdf") as doc:
        return "".join([page.get_text() for page in doc])

def extract_text_from_docx_bytes(data: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp:
        temp.write(data)
        temp.flush()
        doc = Document(temp.name)
        return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_eml_bytes(data: bytes):
    msg = email.message_from_bytes(data)
    for part in msg.walk():
        content_type = part.get_content_type()
        payload = part.get_payload(decode=True)
        if payload:
            try:
                if content_type == "text/plain":
                    return payload.decode(errors="ignore")
                elif content_type == "text/html":
                    return BeautifulSoup(payload.decode(errors="ignore"), "html.parser").get_text()
            except Exception:
                continue
    return ""

def extract_text_from_msg_bytes(data: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".msg") as temp:
        temp.write(data)
        temp.flush()
        msg = extract_msg.Message(temp.name)
        return msg.body or ""

# ---- Main HackRx Endpoint ----
@app.post("/api/v1/hackrx/run")
async def hackrx_run(request: Request, _: None = Depends(verify_token), payload: dict = Body(...)):
    url = payload.get("documents")
    questions = payload.get("questions", [])

    if not url or not questions:
        return {"error": "Missing 'documents' or 'questions'."}

    try:
        # Step 1: Download file
        response = requests.get(url)
        if response.status_code != 200:
            return {"error": "Failed to download document from URL."}
        file_data = response.content
        ext = url.split(".")[-1].split("?")[0].lower()

        # Step 2: Extract text based on file type
        if ext == "pdf":
            full_text = extract_text_from_pdf_bytes(file_data)
        elif ext == "docx":
            full_text = extract_text_from_docx_bytes(file_data)
        elif ext == "eml":
            full_text = extract_text_from_eml_bytes(file_data)
        elif ext == "msg":
            full_text = extract_text_from_msg_bytes(file_data)
        else:
            return {"error": f"Unsupported file format: {ext}"}

        # Step 3: Clean, chunk, embed to Pinecone
        cleaned = clean_text(full_text)
        chunks = get_chunks(cleaned)
        metadatas = [{"text": chunk} for chunk in chunks]
        embed_to_pinecone(chunks, metadatas=metadatas, namespace="preload")


        # Step 4: Answer questions with Groq + LLaMA 3
        answers = []
        for q in questions:
            context = get_relevant_context(q)
            answer = get_llm_answer(q, context)
            answers.append(answer)

        return {"answers": answers}

    except Exception as e:
        return {"error": str(e)}