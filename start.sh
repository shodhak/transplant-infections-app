#!/bin/bash
# Railway startup script

# Start the FastAPI backend
uvicorn rag_trinf:app --host 0.0.0.0 --port $PORT
