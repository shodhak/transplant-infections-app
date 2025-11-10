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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from starlette.responses import JSONResponse
from typing import List, Optional

# FastAPI initialization
app = FastAPI()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = None
pdf_path = "merged_papers.pdf"
FAISS_INDEX_PATH = "faiss_index"

# Load OpenAI API Key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Don't raise an error on startup - handle it in the endpoints
if not OPENAI_API_KEY:
    print("⚠️ Warning: OpenAI API Key is missing. Set OPENAI_API_KEY environment variable.")

# Comment out BLIP model to save memory
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def extract_content_from_pdf():
    """Extract text and tables from the PDF file."""
    if not os.path.exists(pdf_path):
        print("❌ Error: PDF document not found.")
        return {"raw_text": "", "tables": [], "images": []}

    print("🔹 Opening PDF document...")
    doc = fitz.open(pdf_path)
    raw_text = []
    table_data = []

    # Extract text from each page
    for page_index, page in enumerate(doc):
        print(f"🔹 Processing text from page {page_index + 1}...")
        raw_text.append(page.get_text("text"))

    print("✅ Finished extracting text.")
    
    # Join all text
    full_text = "\n".join(raw_text)

    # Extract tables using Camelot
    try:
        print("🔹 Extracting tables...")
        tables_stream = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        tables_lattice = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
        
        tables = tables_stream + tables_lattice
        for t in tables:
            table_data.append(t.df.to_string())
        
        print(f"✅ {len(table_data)} tables extracted.")
    except Exception as e:
        print(f"⚠️ Table extraction failed: {e}")

    return {"raw_text": full_text, "tables": table_data, "images": []}

def create_faiss_index(data):
    """Create FAISS vector index for text and tables."""
    global vector_store

    # Combine text and tables
    combined_text = data["raw_text"] + "\n".join(data["tables"])

    # Chunk text for indexing
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
    """Load FAISS index if available; otherwise, process the document."""
    global vector_store

    if os.path.exists(FAISS_INDEX_PATH):
        print("🔄 Loading existing FAISS index...")
        try:
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            print("✅ FAISS index loaded successfully!")
            return  # Exit early if index loads successfully
        except Exception as e:
            print(f"❌ Error loading FAISS index: {e}")
    
    # Only process PDF if index doesn't exist or failed to load
    print("⚠️ Processing the document...")
    extracted_data = extract_content_from_pdf()
    if extracted_data:
        create_faiss_index(extracted_data)

# Define Request Model
class ChatRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = []

def query_openai(context, query):
    """Query OpenAI GPT-4o using a clinician-scientist role."""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        return "Error: OpenAI API key is missing."

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    system_message = (
        "You are a clinician scientist in transplant infections powered by a RAG (Retrieval-Augmented Generation) system. "
        "Your implementation details:\n"
        "- Knowledge base: 130+ transplant infection publications merged into a single PDF\n"
        "- Retrieval: FAISS vector database using sentence-transformers/all-MiniLM-L6-v2 embeddings\n"
        "- Chunks: 1500 characters with 100 character overlap, retrieving top 3 most similar\n"
        "- Generation: OpenAI GPT-4o model\n"
        "- Developed by: Keating Lab, NYU Langone Health\n\n"
        "Answer questions based on your expertise with the publications. "
        "Make sure that all information you provide is accurate. You can use outside information, but when you do, mention that "
        "and provide references. If asked about your implementation or how you work, explain these technical details.\n\n"
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
        return f"OpenAI API error: {e}"

# GET endpoint for simple queries
@app.get("/query/")
def query_api_get(query: str = Query(..., description="Enter your question")):
    """Handle user queries via GET request."""
    print(f"Received GET query: {query}")

    try:
        if vector_store is None:
            return JSONResponse(content={"error": "FAISS index is not loaded. Please restart the app."}, status_code=500)

        docs = vector_store.similarity_search(query, k=3)
        if not docs:
            return JSONResponse(content={"answer": "No relevant documents found."})

        context = "\n".join([doc.page_content for doc in docs])
        answer = query_openai(context, query)

        print(f"Generated answer: {answer}")
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# POST endpoint for queries with chat history
@app.post("/chat/")  # Changed to /chat/ to avoid conflict
def query_api_post(chat_request: ChatRequest):
    """Process user queries with chat history via POST request."""
    print(f"Received POST query: {chat_request.query}")
    
    try:
        if vector_store is None:
            return JSONResponse(content={"error": "FAISS index is not loaded. Please restart the app."}, status_code=500)

        docs = vector_store.similarity_search(chat_request.query, k=3)
        if not docs:
            return JSONResponse(content={"answer": "No relevant documents found."})

        context = "\n".join([doc.page_content for doc in docs])
        answer = query_openai(context, chat_request.query)
        
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

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

# Load FAISS index on startup
try:
    load_document()
except Exception as e:
    print(f"⚠️ Warning: Failed to load document on startup: {e}")
    print("Will retry on first request.")
