# Transplant Infections AI: Intelligent Research Assistant for Clinicians & Scientists

An AI-powered research assistant leveraging Retrieval-Augmented Generation (RAG) to help clinician scientists and researchers efficiently access and synthesize transplant infectious disease literature.

## The Problem

Clinicians and researchers working in transplant infectious diseases face a challenge: staying current with a growing body of literature while making evidence-based decisions in time-sensitive clinical scenarios. Traditional literature searches are time-consuming and may miss connections across studies. When facing complex clinical questions—such as optimal CMV prophylaxis strategies, emerging resistant organisms, or risk stratification for opportunistic infections—synthesizing evidence from hundreds of papers is difficult and time-intensive.

## The Solution

This application provides a more efficient way to access and synthesize transplant infectious disease knowledge by:

**1. Comprehensive Knowledge Base**
- Indexes 226 peer-reviewed research publications on transplant infections
- Processes documents into 14,085 semantically meaningful chunks
- Maintains complete document provenance for every piece of information
- Covers viral, bacterial, fungal, and parasitic infections in solid organ and stem cell transplant recipients

**2. Intelligent Information Retrieval**
- Uses vector embeddings (HuggingFace's sentence-transformers) to understand semantic meaning, not just keywords
- FAISS vector database enables fast retrieval across the entire corpus
- Finds relevant information even when query terminology differs from source documents
- Retrieves context-rich passages rather than isolated facts

**3. AI-Powered Synthesis & Analysis**
- OpenAI's GPT-5 model trained to think like a clinician scientist
- Synthesizes information across multiple sources to answer complex questions
- Distinguishes between established guidelines and emerging evidence
- Provides academic citations with document sources for every claim
- Maintains rigorous evidence-based approach appropriate for peer discussion

**4. Clinical & Research Applications**
- **Literature Review**: Rapidly survey evidence on specific topics (e.g., "BK virus nephropathy treatment approaches post-kidney transplant")
- **Clinical Decision Support**: Access synthesized evidence for complex cases in real-time
- **Hypothesis Generation**: Identify knowledge gaps and research opportunities by exploring connections across studies
- **Grant Writing**: Quickly gather background evidence and identify key citations for proposals
- **Teaching**: Generate evidence-based explanations for trainees with full source attribution

## How It Works: The Science Behind the System

**Architecture Overview:**

```
Query → Embedding → Vector Search → Context Retrieval → LLM Synthesis → Cited Answer
```

**1. Document Processing & Indexing**
- PDFs are parsed using PyMuPDF, extracting text, tables, and structured data
- Content is split into semantically coherent chunks (1,500 characters with 100-character overlap)
- Each chunk is converted to a 384-dimensional vector embedding using sentence-transformers/all-MiniLM-L6-v2
- Vectors are indexed using FAISS (Facebook AI Similarity Search) for efficient nearest-neighbor retrieval
- Metadata preserved: document name, source file, chunk position, enabling full traceability

**2. Query Processing**
- User query is embedded using the same transformer model (semantic consistency)
- Vector similarity search identifies the most relevant chunks across the entire corpus
- Context is ranked by semantic similarity, not simple keyword matching
- Retrieves comprehensive context while maintaining focus on relevance

**3. AI-Powered Answer Generation**
- Retrieved context is provided to GPT-5 with a specialized system prompt
- Model is instructed to reason like a clinician scientist with expertise in transplant infectious diseases
- Emphasis on evidence-based responses with clear distinction between established knowledge and emerging data
- Responses include inline citations ([Source 1], [Source 2]) linked to specific documents
- When supplementing with external knowledge, explicit academic citations are required

**4. Source Attribution & Transparency**
- Every response includes expandable source references
- Shows exact document name, chunk location, and relevant passage
- Enables verification and deeper exploration of primary sources
- Maintains scientific rigor and accountability

## Technical Innovation

**Retrieval-Augmented Generation (RAG):**
Unlike pure language models that rely solely on training data (prone to hallucination and outdated information), RAG grounds responses in your specific document corpus. This means:
- Answers are based on actual published research, not model confabulation
- Information is current to your indexed literature
- Full transparency with source attribution
- Reduced risk of false or fabricated information

**Vector Embeddings & Semantic Search:**
Traditional keyword search fails when synonyms, related concepts, or paraphrased queries are used. Semantic embeddings capture meaning:
- "Cytomegalovirus prophylaxis" and "CMV prevention strategies" retrieve similar results
- Understands context: "rejection" in transplant vs. statistical contexts
- Finds conceptually related information even without exact term matches

**Optimized for Scientific Discourse:**
- System prompt engineered for academic rigor
- Maintains appropriate terminology and evidence levels
- Balances comprehensiveness with conciseness
- Structures responses with clear organization and clinical relevance

## Local Development

```bash
# Create virtual environment
python3 -m venv rag_trinf
source rag_trinf/bin/activate  # On Windows: rag_trinf\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variable
export OPENAI_API_KEY="your-key-here"

# Run backend
uvicorn rag_trinf:app --reload --host 0.0.0.0 --port 8000

# Run frontend (in separate terminal)
streamlit run app.py
```

## Technology Stack

**Backend:**
- FastAPI: High-performance async API framework
- LangChain: RAG orchestration and document processing
- FAISS: Vector similarity search (optimized C++ implementation)
- HuggingFace Transformers: Sentence embeddings
- PyMuPDF & Camelot: PDF parsing with table extraction

**AI & ML:**
- OpenAI GPT-5: State-of-the-art language model
- sentence-transformers/all-MiniLM-L6-v2: Efficient semantic embeddings
- Vector dimensionality: 384 (balance between accuracy and performance)

**Frontend:**
- Streamlit: Interactive web interface for researchers
- Real-time query processing with source visualization

## Impact & Future Directions

This application represents a new paradigm in scientific literature interaction—moving from manual search and synthesis to AI-augmented knowledge discovery. For clinician scientists, it compresses hours of literature review into minutes, while maintaining the rigor and evidence-based approach essential to medical research.

**Potential Extensions:**
- Expand to broader infectious disease or transplant medicine literature
- Integrate real-time PubMed updates for continuous knowledge augmentation
- Add comparative analysis capabilities across studies
- Implement evidence-level scoring and study quality assessment
- Multi-modal inputs: integrate clinical data, images, or lab values for contextualized queries

---

## About

**App Developed By:** Shreyas Joshi  
**Literature Corpus Curated By:** Frank Liu, Berk Maden, and Shreyas Joshi  
**Institution:** Keating Lab, NYU Langone Health  
**Contact:** shreyas.joshi@nyulangone.org (bug reports, suggestions, literature updates welcome)
