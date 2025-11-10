import os
import fitz  # PyMuPDF
import numpy as np
import camelot
import openai
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, Query
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from starlette.responses import JSONResponse
from typing import List, Optional

# FastAPI initialization
app = FastAPI()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = None
pdf_path = "merged_papers.pdf"
FAISS_INDEX_PATH = "faiss_index"

# Load OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️ Warning: OpenAI API Key is missing")

def extract_content_from_pdf():
    """Extract text from PDF file."""
    if not os.path.exists(pdf_path):
        print("❌ Error: PDF document not found.")
        return {"raw_text": "", "tables": []}

    print("🔹 Opening PDF document...")
    doc = fitz.open(pdf_path)
    raw_text = []

    for page_index, page in enumerate(doc):
        print(f"🔹 Processing page {page_index + 1}...")
        raw_text.append(page.get_text("text"))

    full_text = "\n".join(raw_text)
    return {"raw_text": full_text, "tables": []}

def create_faiss_index(data):
    """Create FAISS vector index."""
    global vector_store
    
    combined_text = data["raw_text"]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    texts = text_splitter.split_text(combined_text)
    
    print(f"Total chunks created: {len(texts)}")
    
    try:
        vector_store = FAISS.from_texts(texts, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        print("✅ FAISS index created successfully!")
    except Exception as e:
        print(f"❌ Error creating FAISS index: {e}")

def load_document():
    """Load FAISS index if available."""
    global vector_store
    
    if os.path.exists(FAISS_INDEX_PATH):
        print("🔄 Loading existing FAISS index...")
        try:
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            print("✅ FAISS index loaded successfully!")
            return
        except Exception as e:
            print(f"❌ Error loading FAISS index: {e}")
    
    print("⚠️ Processing the document...")
    extracted_data = extract_content_from_pdf()
    if extracted_data["raw_text"]:
        create_faiss_index(extracted_data)

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = []

def query_openai(context, query):
    """Query OpenAI GPT-4o."""
    if not OPENAI_API_KEY:
        return "Error: OpenAI API key is missing."
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    system_message = (
        "You are a clinician scientist in transplant infections. "
        "Answer questions based on your expertise with the publications. "
        "Make sure that all information you provide is accurate."
    )
    
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

@app.get("/")
def root():
    return {"message": "FastAPI is running! Go to /docs to test the API."}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "vector_store_loaded": vector_store is not None,
        "openai_key_set": bool(OPENAI_API_KEY)
    }

@app.post("/chat/")
def query_api_post(chat_request: ChatRequest):
    """Process user queries."""
    print(f"Received query: {chat_request.query}")
    
    try:
        if vector_store is None:
            load_document()
            if vector_store is None:
                return JSONResponse(content={"error": "FAISS index not loaded"}, status_code=500)
        
        docs = vector_store.similarity_search(chat_request.query, k=3)
        if not docs:
            return JSONResponse(content={"answer": "No relevant documents found."})
        
        context = "\n".join([doc.page_content for doc in docs])
        answer = query_openai(context, chat_request.query)
        
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        print(f"❌ Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Try to load on startup but don't fail if it doesn't work
try:
    load_document()
except Exception as e:
    print(f"⚠️ Startup load failed: {e}")
