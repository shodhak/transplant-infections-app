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
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
# Path to directory containing individual PDF documents
# Use environment variable if available, otherwise use local path
pdf_directory = os.getenv("PDF_DIRECTORY", "/Users/sj1212/Documents/RAG_transplant_infections/downloaded_papers/all_docs_for_rag")
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

def extract_content_from_pdf(pdf_path, doc_name):
    """Extract text, tables, and figures from a single PDF file."""
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF document not found: {pdf_path}")
        return None

    print(f"🔹 Processing: {doc_name}")
    doc = fitz.open(pdf_path)
    raw_text = []
    table_data = []

    # Extract text from each page
    for page_index, page in enumerate(doc):
        raw_text.append(page.get_text("text"))

    text_content = "\n".join(raw_text)
    print(f"✅ Extracted {len(raw_text)} pages from {doc_name}")

    # Extract tables using Camelot (optional, may be slow)
    try:
        tables_stream = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        tables_lattice = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
        
        # Combine tables from both methods
        for t in tables_stream:
            table_data.append(t.df.to_string())
        for t in tables_lattice:
            table_data.append(t.df.to_string())
        
        if table_data:
            print(f"✅ Extracted {len(table_data)} tables from {doc_name}")
    except Exception as e:
        pass  # Silently skip table extraction errors

    return {
        "text": text_content,
        "tables": "\n".join(table_data) if table_data else "",
        "doc_name": doc_name,
        "pdf_path": pdf_path
    }

def extract_content_from_all_pdfs():
    """Extract content from all PDFs in the directory."""
    if not os.path.exists(pdf_directory):
        print(f"❌ Error: Directory not found: {pdf_directory}")
        return []

    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_directory}")
        return []

    print(f"📚 Found {len(pdf_files)} PDF files to process...")
    
    all_documents = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        doc_name = os.path.splitext(pdf_file)[0]  # Remove .pdf extension
        
        result = extract_content_from_pdf(pdf_path, doc_name)
        if result:
            all_documents.append(result)
    
    print(f"✅ Successfully processed {len(all_documents)} documents")
    return all_documents

    
### 🛠 Create FAISS Index for Text, Tables, and Figures
def create_faiss_index(all_documents):
    """Create FAISS vector index from multiple PDF documents."""
    global vector_store

    if not all_documents:
        print("❌ No documents to index")
        return

    all_texts = []
    all_metadatas = []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)

    # Process each document
    for doc in all_documents:
        # Combine text and tables for this document
        combined_text = doc["text"]
        if doc["tables"]:
            combined_text += "\n\n" + doc["tables"]
        
        # Split into chunks
        chunks = text_splitter.split_text(combined_text)
        
        # Create metadata for each chunk with document name
        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metadatas.append({
                "document_name": doc["doc_name"],
                "source_file": doc["pdf_path"],
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
        
        print(f"✅ Created {len(chunks)} chunks from {doc['doc_name']}")

    print(f"\n📊 Total chunks created: {len(all_texts)} from {len(all_documents)} documents")

    try:
        vector_store = FAISS.from_texts(all_texts, embeddings, metadatas=all_metadatas)
        vector_store.save_local(FAISS_INDEX_PATH)
        print("✅ FAISS index created successfully with document metadata!")
    except Exception as e:
        print(f"❌ Error creating FAISS index: {e}")


### 🛠 Load FAISS Index
def load_document():
    """Load FAISS index if available; otherwise, process all PDF documents if they exist."""
    global vector_store

    if os.path.exists(FAISS_INDEX_PATH):
        print("🔄 Loading existing FAISS index...")
        try:
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            print("✅ FAISS index loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading FAISS index: {e}")
            # Only try to rebuild if PDF directory exists
            if os.path.exists(pdf_directory):
                print("⚠️ Reprocessing documents to recreate FAISS index...")
                all_documents = extract_content_from_all_pdfs()
                create_faiss_index(all_documents)
            else:
                print(f"❌ PDF directory not found at {pdf_directory}")
                print("⚠️ Cannot rebuild FAISS index without PDF files.")
                raise Exception("FAISS index is corrupted and PDF files are not available for rebuilding.")
    else:
        # Only try to build if PDF directory exists
        if os.path.exists(pdf_directory):
            print("⚠️ FAISS index not found, processing all PDF documents...")
            all_documents = extract_content_from_all_pdfs()
            if all_documents:
                create_faiss_index(all_documents)
        else:
            print(f"❌ FAISS index not found and PDF directory not available at {pdf_directory}")
            print("⚠️ Application will not function without FAISS index or PDF files.")
            raise Exception("FAISS index is missing and PDF files are not available. Please include faiss_index/ folder in deployment.")


# Define Request Model
class ChatRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = [] # history is optional with a default empty list

def query_openai(context, query, history=None):
    """Query OpenAI GPT-4o using a clinician-scientist role."""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        return "Error: OpenAI API key is missing."

    client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=45.0)  # Set 45 second timeout

    # Limit context size to avoid token limits
    max_context_length = 3000  # Adjust as needed
    if len(context) > max_context_length:
        context = context[:max_context_length] + "...[truncated]"

    system_message = (
        "You are an expert AI assistant specializing in transplant infectious diseases, designed for clinician scientists and researchers. "
        "Provide evidence-based, scientifically rigorous answers suitable for an academic medical audience. "
        "\n\nFormatting Requirements:"
        "\n- Organize responses with clear paragraphs and structure"
        "\n- Use bullet points for lists when appropriate"
        "\n- Keep responses comprehensive but concise (aim for completeness within token limit)"
        "\n- ALWAYS complete your thought - never end mid-sentence"
        "\n- If the answer would be too long, prioritize the most clinically relevant information"
        "\n\nContent Guidelines:"
        "\n- Use precise medical terminology and cite specific evidence when available"
        "\n- IMPORTANT: When referencing information from the provided sources, cite them as [Source 1], [Source 2], etc."
        "\n- Include relevant pathophysiology, clinical implications, and research context"
        "\n- Reference specific studies, guidelines, or systematic reviews from the provided literature when applicable"
        "\n- When discussing clinical management, distinguish between established guidelines and emerging evidence"
        "\n- If supplementing with outside knowledge beyond the provided context:"
        "\n  * ONLY use peer-reviewed academic literature or established clinical guidelines"
        "\n  * Explicitly state: 'Based on additional literature (not from provided sources):'"
        "\n  * Provide complete academic citations in the format: Author(s), Journal, Year, DOI/PMID when possible"
        "\n  * Example: 'Based on additional literature: Smith et al., N Engl J Med 2023; 388:123-134 (PMID: 12345678)'"
        "\n  * Never use anecdotal evidence or non-academic sources"
        "\n- For research questions, discuss study design, limitations, and clinical significance"
        "\n- Maintain an academic tone appropriate for peer discussion among transplant specialists"
        "\n\nContext from scientific publications:\n{context}"
    )

    # Build messages with chat history if provided
    messages = [{"role": "system", "content": system_message}]
    
    if history:
        # Limit chat history to last 4 messages to avoid token limits
        recent_history = history[-5:-1] if len(history) > 1 else []
        for entry in recent_history:
            if entry["role"] == "user":
                messages.append({"role": "user", "content": entry["text"]})
            elif entry["role"] == "bot":
                messages.append({"role": "assistant", "content": entry["text"]})
    
    # Add current query
    messages.append({"role": "user", "content": query})

    try:
        response = client.chat.completions.create(
            model="gpt-5-chat-latest",  # Using GPT-5 chat latest
            messages=messages,
            max_tokens=1500,  # Increased to 1500 tokens for complete responses
            temperature=0.7
        )
        return response.choices[0].message.content
    except openai.APITimeoutError:
        return "The request to OpenAI timed out. Please try again with a shorter question."
    except openai.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return f"OpenAI API error: {str(e)}"

@app.post("/query/")
def query_api(chat_request: ChatRequest):
    """Process user queries with chat history."""
    import time
    start_time = time.time()
    print(f"🔹 Received query: {chat_request.query}")

    try:
        if vector_store is None:
            return JSONResponse(content={"error": "FAISS index is not loaded. Please restart the app."}, status_code=500)

        # Check if this is a meta-question about the AI itself
        query_lower = chat_request.query.lower()
        meta_keywords = ["which model", "what model", "who are you", "what are you", "your name", "ai model", "gpt"]
        is_meta_question = any(keyword in query_lower for keyword in meta_keywords)
        
        sources = []
        if is_meta_question:
            # For meta questions, use minimal context
            context = "This is an AI assistant for transplant infections questions."
            print("🔹 Detected meta-question, using minimal context")
        else:
            # Search for relevant documents
            search_start = time.time()
            docs = vector_store.similarity_search(chat_request.query, k=5)  # Increased to 5 for better coverage
            print(f"🔹 Document search took: {time.time() - search_start:.2f}s")
            
            if not docs:
                return JSONResponse(content={"answer": "No relevant documents found.", "sources": []})
            
            # Build context and extract sources
            context_parts = []
            for i, doc in enumerate(docs, 1):
                context_parts.append(f"[Source {i}]: {doc.page_content}")
                
                # Extract metadata if available
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                
                # Get document name from metadata
                doc_name = metadata.get('document_name', 'Unknown Document')
                
                source_info = {
                    "index": i,
                    "document_name": doc_name,
                    "chunk_info": f"Chunk {metadata.get('chunk_index', '?')} of {metadata.get('total_chunks', '?')}",
                    "content_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "metadata": metadata
                }
                sources.append(source_info)
            
            context = "\n\n".join(context_parts)
        
        # Query OpenAI
        openai_start = time.time()
        answer = query_openai(context, chat_request.query, chat_request.history)
        print(f"🔹 OpenAI call took: {time.time() - openai_start:.2f}s")
        
        print(f"✅ Total processing time: {time.time() - start_time:.2f}s")
        print(f"Generated answer (first 100 chars): {answer[:100]}...")
        
        return JSONResponse(content={
            "answer": answer,
            "sources": sources
        })
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/query/")
def query_api_get(query: str = Query(..., description="Enter your question")):
    """Handle user queries via GET request (for testing in browser)."""
    print(f"Received GET query: {query}")

    try:
        if vector_store is None:
            return JSONResponse(content={"error": "FAISS index is not loaded. Please restart the app."}, status_code=500)

        docs = vector_store.similarity_search(query, k=5)
        if not docs:
            return JSONResponse(content={"answer": "No relevant documents found.", "sources": []})

        # Build context and extract sources
        context_parts = []
        sources = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Source {i}]: {doc.page_content}")
            
            # Extract metadata if available
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            
            # Get document name from metadata
            doc_name = metadata.get('document_name', 'Unknown Document')
            
            source_info = {
                "index": i,
                "document_name": doc_name,
                "chunk_info": f"Chunk {metadata.get('chunk_index', '?')} of {metadata.get('total_chunks', '?')}",
                "content_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "metadata": metadata
            }
            sources.append(source_info)
        
        context = "\n\n".join(context_parts)
        answer = query_openai(context, query)

        print(f"Generated answer: {answer}")
        return JSONResponse(content={"answer": answer, "sources": sources})
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Load FAISS index on startup
load_document()


### 🛠 Root Endpoint for API Status
@app.get("/")
def root():
    return {"message": "FastAPI is running! Go to /docs to test the API."}
