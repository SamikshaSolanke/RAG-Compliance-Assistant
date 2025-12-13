# main.py

import os
import fitz
import tempfile
import re
import requests
import email
from bs4 import BeautifulSoup
import extract_msg
from docx import Document
from uuid import uuid4
from typing import List
from fastapi import FastAPI, Request, Body, HTTPException, Depends, UploadFile, File, Form
from fastapi.security import HTTPBearer
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vectorstore import embed_to_pinecone, get_relevant_context
from llm import (
    get_llm_answer,
    detect_language,
    translate_to_english,
    translate_answer
)

app = FastAPI(
    title="HackRx AI Assistant",
    description="Insurance QA with multilingual support (English/Hindi/Marathi)",
    version="3.1.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")
AUTH_KEY = "84ae11520a9ef092d711f960ce31d92fc90a47b0f38aa3202fe16ac543ef9871"
security_scheme = HTTPBearer()

def custom_openapi():
    app.openapi_schema = None

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    schema["components"]["securitySchemes"] = {
        "BearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
    }

    for path in schema["paths"].values():
        for method in path.values():
            method["security"] = [{"BearerAuth": []}]

    app.openapi_schema = schema
    return schema
app.openapi = custom_openapi

def verify_token(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    if auth.split("Bearer ")[1] != AUTH_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")

def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=250,
        chunk_overlap=50,
    )
    return splitter.split_text(text)

def extract_text_from_pdf_bytes(data):
    with fitz.open(stream=data, filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)

def extract_text_from_docx_bytes(data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as t:
        t.write(data)
        t.flush()
        doc = Document(t.name)
        return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_txt_bytes(data):
    return data.decode("utf-8", errors="ignore")

def extract_text_from_eml_bytes(data):
    msg = email.message_from_bytes(data)
    for part in msg.walk():
        payload = part.get_payload(decode=True)
        if payload:
            if part.get_content_type() == "text/plain":
                return payload.decode(errors="ignore")
            if part.get_content_type() == "text/html":
                return BeautifulSoup(payload.decode(errors="ignore"), "html.parser").get_text()
    return ""

def extract_text_from_msg_bytes(data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".msg") as t:
        t.write(data)
        t.flush()
        msg = extract_msg.Message(t.name)
        return msg.body or ""

@app.get("/", response_class=HTMLResponse)
async def ui_home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/v1/hackrx/upload")
async def hackrx_upload(file: UploadFile = File(...), questions: str = Form(...)):

    file_bytes = await file.read()
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        full_text = extract_text_from_pdf_bytes(file_bytes)
    elif filename.endswith(".docx"):
        full_text = extract_text_from_docx_bytes(file_bytes)
    elif filename.endswith(".txt"):
        full_text = extract_text_from_txt_bytes(file_bytes)
    elif filename.endswith(".eml"):
        full_text = extract_text_from_eml_bytes(file_bytes)
    elif filename.endswith(".msg"):
        full_text = extract_text_from_msg_bytes(file_bytes)
    else:
        return {"error": "Unsupported file format"}

    namespace = f"upload_{uuid4()}"

    cleaned = clean_text(full_text)
    chunks = get_chunks(cleaned)
    embed_to_pinecone(chunks, [{"text": c} for c in chunks], namespace=namespace)

    question_list = [q.strip() for q in questions.split("\n") if q.strip()]

    outputs = []

    for q in question_list:
        lang = detect_language(q)
        en_q = translate_to_english(q) if lang != "English" else q

        ctx = get_relevant_context(en_q, namespace=namespace)
        en_ans = get_llm_answer(en_q, ctx)
        final_ans = translate_answer(en_ans, lang)

        outputs.append({"question": q, "answer": final_ans})

    return {"answers": outputs}

@app.post("/api/v1/hackrx/run")
async def hackrx_run(request: Request, _: None = Depends(verify_token), payload: dict = Body(...)):

    url = payload.get("documents")
    questions = payload.get("questions", [])

    if not url or not questions:
        return {"error": "Missing fields"}

    res = requests.get(url)
    if res.status_code != 200:
        return {"error": "Failed to download document"}

    file_bytes = res.content
    ext = url.split(".")[-1].split("?")[0].lower()

    if ext == "pdf":
        full_text = extract_text_from_pdf_bytes(file_bytes)
    elif ext == "docx":
        full_text = extract_text_from_docx_bytes(file_bytes)
    elif ext == "eml":
        full_text = extract_text_from_eml_bytes(file_bytes)
    elif ext == "msg":
        full_text = extract_text_from_msg_bytes(file_bytes)
    else:
        return {"error": f"Unsupported file format: {ext}"}

    namespace = f"run_{uuid4()}"

    cleaned = clean_text(full_text)
    chunks = get_chunks(cleaned)
    embed_to_pinecone(chunks, [{"text": c} for c in chunks], namespace=namespace)

    answers = []

    for q in questions:
        lang = detect_language(q)
        en_q = translate_to_english(q) if lang != "English" else q

        ctx = get_relevant_context(en_q, namespace=namespace)
        en_ans = get_llm_answer(en_q, ctx)

        final_ans = translate_answer(en_ans, lang)
        answers.append(final_ans)

    return {"answers": answers}
