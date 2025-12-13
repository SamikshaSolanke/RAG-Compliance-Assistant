# vectorstore.py

import os
import time
from uuid import uuid4
from typing import List
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings import HuggingFaceEmbeddings

PINECONE_API_KEY = "KEY"
PINECONE_INDEX = "NAME"
PINECONE_REGION = "us-east-1"
PINECONE_CLOUD = "aws"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = PINECONE_INDEX

if not pc.has_index(index_name):
    raise ValueError(f"Pinecone index '{index_name}' does not exist. Create it first.")

index = pc.Index(index_name)

def embed_texts(texts: List[str]) -> List[List[float]]:
    return embedding_model.embed_documents(texts)

def embed_query_text(query: str) -> List[float]:
    return embedding_model.embed_query(query)

def embed_to_pinecone(docs: List[str], metadatas: List[dict] = None, ids: List[str] = None, namespace="__default__"):

    if metadatas is None:
        metadatas = [{} for _ in docs]

    if ids is None:
        ids = [str(uuid4()) for _ in docs]

    def batch(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    for batch_docs, batch_metas, batch_ids in zip(batch(docs, 20), batch(metadatas, 20), batch(ids, 20)):
        embeddings = embed_texts(batch_docs)
        vectors = list(zip(batch_ids, embeddings, batch_metas))
        index.upsert(vectors=vectors, namespace=namespace)

def get_relevant_context(query: str, namespace="preload", top_k=7):
    query_embedding = embed_query_text(query)
    results = index.query(
        namespace=namespace,
        top_k=top_k,
        vector=query_embedding,
        include_metadata=True
    )

    chunks = [
        match["metadata"].get("text")
        for match in results.get("matches", [])
        if match.get("metadata", {}).get("text")
    ]

    return "\n".join(chunks)

def clear_namespace(namespace="__default__"):
    index.delete(namespace=namespace, delete_all=True)
