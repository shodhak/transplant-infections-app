# Transplant Infections AI Application

AI-powered RAG application for transplant infectious disease research.

## Railway Deployment

### Prerequisites
1. Railway account
2. OpenAI API key

### Steps to Deploy

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for Railway deployment"
   git push origin main
   ```

2. **Deploy on Railway**
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository: `transplant-infections-app`

3. **Set Environment Variables**
   In Railway project settings, add:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `PORT`: (Railway will set this automatically)

4. **Important Notes**
   - The FAISS index will be rebuilt on first deployment (takes 10-15 minutes)
   - Make sure all PDF files are in the correct directory path
   - The app requires ~2GB RAM minimum

### Local Development

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

## Features
- RAG-based question answering
- Document source attribution
- Academic citation support
- 226 research publications indexed
- Real-time query processing

## Tech Stack
- FastAPI
- Streamlit
- OpenAI GPT-5
- FAISS vector database
- LangChain
- HuggingFace embeddings
