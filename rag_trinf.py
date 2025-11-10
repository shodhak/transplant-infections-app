import os
import fitz  # PyMuPDF
import numpy as np
import openai
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI
from starlette.responses import JSONResponse
from typing import List, Optional
import faiss
from sentence_transformers import SentenceTransformer

# FastAPI initialization
app = FastAPI()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
vector_store = None
pdf_path = "merged_papers.pdf"
FAISS_INDEX_PATH = "faiss_index"

# Simple text splitter function
def split_text(text, chunk_size=1500, overlap=100):
    """Split text into chunks"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

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
    """Create FAISS vector index without LangChain."""
    global vector_store
    
    # Split text into chunks
    texts = split_text(data["raw_text"])
    print(f"Total chunks created: {len(texts)}")
    
    # Create embeddings
    embeddings = model.encode(texts)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    # Store the index and texts
    vector_store = {
        'index': index,
        'texts': texts
    }
    
    # Save index
    faiss.write_index(index, f"{FAISS_INDEX_PATH}.index")
    # Save texts
    import pickle
    with open(f"{FAISS_INDEX_PATH}.pkl", 'wb') as f:
        pickle.dump(texts, f)
    
    print("✅ FAISS index created successfully!")

def load_document():
    """Load FAISS index if available."""
    global vector_store
    
    if os.path.exists(f"{FAISS_INDEX_PATH}.index"):
        print("🔄 Loading existing FAISS index...")
        try:
            # Load index
            index = faiss.read_index(f"{FAISS_INDEX_PATH}.index")
            # Load texts
            import pickle
            with open(f"{FAISS_INDEX_PATH}.pkl", 'rb') as f:
                texts = pickle.load(f)
            
            vector_store = {
                'index': index,
                'texts': texts
            }
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
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY"))
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
        
        # Search similar texts
        query_embedding = model.encode([chat_request.query])
        D, I = vector_store['index'].search(query_embedding.astype('float32'), k=3)
        
        # Get relevant texts
        relevant_texts = [vector_store['texts'][i] for i in I[0]]
        context = "\n".join(relevant_texts)
        
        answer = query_openai(context, chat_request.query)
        
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        print(f"❌ Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Try to load on startup
try:
    load_document()
except Exception as e:
    print(f"⚠️ Startup load failed: {e}")
