import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import camelot
import openai
import pandas as pd
from pydantic import BaseModel  # ✅ Import BaseModel
from fastapi import FastAPI, Query
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from starlette.responses import JSONResponse
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import List, Optional

# Make virtual environment and activate it
#python3 -m venv ~/tr_inf
#source ~/tr_inf/bin/activate

# FastAPI initialization
app = FastAPI()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = None
#pdf_path = "/Users/sj1212/Documents/RAG_transplant_infections/downloaded_papers/merged_papers.pdf"
pdf_path = "merged_papers.pdf"
#FAISS_INDEX_PATH = "/Users/sj1212/Documents/RAG_transplant_infections/faiss_index"
FAISS_INDEX_PATH = "faiss_index"

# ✅ Load OpenAI API Key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OpenAI API Key is missing. Set OPENAI_API_KEY environment variable.")

# Ensure you have Tesseract installed (for OCR)
# macOS: brew install tesseract
# Ubuntu: sudo apt install tesseract-ocr
# Windows: Install from https://github.com/tesseract-ocr/tesseract

### 🛠 Extract Text, Tables & Images from PDF

# Ensure you have Tesseract installed for OCR
# macOS: brew install tesseract
# Ubuntu: sudo apt install tesseract-ocr

# Initialize BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def generate_image_caption(image_path):
    """Generate a text description of an image using BLIP."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        caption = model.generate(**inputs)
        return processor.decode(caption[0], skip_special_tokens=True)
    except Exception as e:
        print(f"⚠️ Failed to process image {image_path}: {e}")
        return "Image could not be processed"

def extract_content_from_pdf():
    """Extract text, tables, and figures from the PDF file."""
    if not os.path.exists(pdf_path):
        print("❌ Error: PDF document not found.")
        return {"raw_text": "", "tables": [], "images": []}

    print("🔹 Opening PDF document...")
    doc = fitz.open(pdf_path)
    raw_text = []
    table_data = []
    image_descriptions = []
    image_folder = "extracted_images"
    os.makedirs(image_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Extract text from each page
    for page_index, page in enumerate(doc):
        print(f"🔹 Processing text from page {page_index + 1}...")
        raw_text.append(page.get_text("text"))

        # Check if memory consumption is high
        if page_index % 5 == 0:  # Print every 5 pages
            import psutil
            print(f"🔹 Memory Usage: {psutil.virtual_memory().percent}%")

    print("✅ Finished extracting text.")

    # Extract tables using Camelot
    try:
        print("🔹 Extracting tables...")
        tables_stream = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        tables_lattice = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")

        # ✅ Ensure this is inside the try block
        tables = tables_stream + tables_lattice  # Combine results

        for t in tables:
            table_data.append(t.df.to_string())  # Convert DataFrame to a string
    
        print(f"✅ {len(table_data)} tables extracted.")

    except Exception as e:  # ✅ Ensure except is properly indented
        print(f"⚠️ Table extraction failed: {e}")

    
### 🛠 Create FAISS Index for Text, Tables, and Figures
def create_faiss_index(data):
    """Create FAISS vector index for text, tables, and figures."""
    global vector_store

    # Combine text, tables, and image descriptions
    combined_text = data["raw_text"] + "\n".join(data["tables"]) + "\n".join(data.get("images", []))

    # Chunk text for indexing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    texts = text_splitter.split_text(combined_text)

    print(f"Total chunks created: {len(texts)}")

    try:
        vector_store = FAISS.from_texts(texts, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        print("✅ FAISS index created successfully with figures included!")
    except Exception as e:
        print(f"❌ Error creating FAISS index: {e}")


### 🛠 Load FAISS Index
def load_document():
    """Load FAISS index if available; otherwise, process the document."""
    global vector_store

    if os.path.exists(FAISS_INDEX_PATH):
        print("🔄 Loading existing FAISS index...")
        try:
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            print("✅ FAISS index loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading FAISS index: {e}")
            print("⚠️ Reprocessing the document to recreate FAISS index...")
            extracted_data = extract_content_from_pdf()
            create_faiss_index(extracted_data)
    else:
        print("⚠️ FAISS index not found, processing the document...")
        extracted_data = extract_content_from_pdf()
        if extracted_data:
            create_faiss_index(extracted_data)


### 🛠 FastAPI Query Endpoint
@app.get("/query/")
def query_api(query: str = Query(..., description="Enter your question")):
    """Handle user queries to search across text, tables, and figures."""
    print(f"Received query: {query}")

    try:
        if vector_store is None:
            return JSONResponse(content={"error": "FAISS index is not loaded. Please restart the app."}, status_code=500)

        docs = vector_store.similarity_search(query, k=3)
        if not docs:
            return JSONResponse(content={"answer": "No relevant documents found."})

        context = "\n".join([doc.page_content for doc in docs])
        answer = query_openai(context, query)  # Use the provided query_openai function

        print(f"Generated answer: {answer}")
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


### 🛠 Query OpenAI API for Answers
# Define Request Model
class ChatRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = [] # history is optional with a default empty list

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
                {"role": "system", "content": system_message},  # Assign clinician-scientist role
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        return f"OpenAI API error: {e}"

@app.post("/query/")
def query_api(chat_request: ChatRequest):
    """Process user queries with chat history."""
    answer = query_openai(chat_request.query, chat_request.history)
    return {"answer": answer}

# Load FAISS index on startup
load_document()


### 🛠 Root Endpoint for API Status
@app.get("/")
def root():
    return {"message": "FastAPI is running! Go to /docs to test the API."}
