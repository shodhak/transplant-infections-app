import streamlit as st
import requests
import time
from datetime import datetime

# Set page title and layout
st.set_page_config(
    page_title="Transplant Infections AI Chat",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Chat UI
st.markdown("""
    <style>
    /* General Background */
    body {
        background-color: #F8F8F8;
        color: #000000 !important;
    }
    
    .stApp {
        background-color: #F8F8F8;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        color: #000000 !important;
    }

    /* Input Box */
    .stTextInput > div > div > input {
        border: 2px solid #008CBA;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        color: #000000 !important;
        background-color: #ffffff;
    }

    /* Buttons */
    .stButton > button {
        background-color: #008CBA;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 20px;
        font-size: 16px;
        cursor: pointer;
        transition: background 0.3s;
    }

    .stButton > button:hover {
        background-color: #005F73;
    }

    /* Chat Message Box */
    .message-box {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 12px;
        font-size: 16px;
        line-height: 1.5;
        color: #000000 !important;
    }

    /* User Message */
    .user-message {
        background: linear-gradient(135deg, #0077cc 0%, #005fa3 100%);
        color: white !important;
        margin-left: 20%;
        text-align: left;
    }

    /* Bot Message */
    .bot-message {
        background-color: #f0f0f0;
        color: #000000 !important;
        text-align: left;
        margin-right: 20%;
        border: 1px solid #e0e0e0;
    }

    /* Timestamp */
    .timestamp {
        font-size: 12px;
        color: #666;
        margin-top: 5px;
    }

    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #008CBA;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Tabs */
    div[data-baseweb="tab-list"] button {
        font-size: 22px !important;
        font-weight: bold !important;
        padding: 12px 20px;
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title Section with better styling
st.markdown("""
<div style='text-align: center; margin-bottom: 30px;'>
    <h1 style='color: #008CBA; margin-bottom: 10px;'>💬 Transplant Infections AI Chat</h1>
    <p style='color: #666; font-size: 18px;'>RAG-powered clinical decision support for transplant infections</p>
</div>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "https://transplant-infections-app-production.up.railway.app"

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query" not in st.session_state:
    st.session_state.query = ""
if "api_status" not in st.session_state:
    st.session_state.api_status = None

# Helper Functions
def format_timestamp(timestamp=None):
    """Format timestamp for display"""
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%I:%M %p")

def check_api_status():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def clear_chat():
    """Clear chat history"""
    st.session_state.chat_history = []

def clear_input():
    """Clear input field"""
    st.session_state.query = ""

# Sidebar with settings and info
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # API Status indicator
    if st.button("🔄 Check API Status"):
        with st.spinner("Checking..."):
            st.session_state.api_status = check_api_status()
    
    if st.session_state.api_status is not None:
        if st.session_state.api_status:
            st.success("✅ API is online")
        else:
            st.error("❌ API is offline")
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", type="secondary"):
        clear_chat()
        st.rerun()
    
    st.markdown("---")
    
    # App Info
    st.markdown("### 📊 System Info")
    st.info("""
    **Model:** OpenAI GPT-4o
    **Embeddings:** all-MiniLM-L6-v2
    **Vector DB:** FAISS
    **Documents:** 130+ publications
    """)
    
    st.markdown("---")
    
    # Contact Info
    st.markdown("### 📧 Contact")
    st.markdown("""
    **Developed by:** Keating Lab  
    **Institution:** NYU Langone Health  
    **Email:** shreyas.joshi@nyulangone.org
    """)

# Create tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Publications", "ℹ️ About"])

# TAB 1: CHAT INTERFACE
with tab1:
    # Chat history container
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_history:
            # Welcome message
            st.markdown("""
            <div style='text-align: center; padding: 40px; color: #666;'>
                <h3>Welcome to Transplant Infections AI Chat</h3>
                <p>Ask any question about transplant infections and I'll search through 130+ medical publications to help you.</p>
                <p style='margin-top: 20px;'><b>Example questions:</b></p>
                <ul style='text-align: left; max-width: 600px; margin: 0 auto;'>
                    <li>What are the common fungal infections in kidney transplant recipients?</li>
                    <li>What is the recommended prophylaxis for CMV?</li>
                    <li>How should we manage nocardia infections?</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display chat history
            for i, entry in enumerate(st.session_state.chat_history):
                role, text = entry["role"], entry["text"]
                timestamp = entry.get("timestamp", "")
                
                if role == "user":
                    st.markdown(f"""
                    <div class='message-box user-message'>
                        <b>You</b>
                        <div>{text}</div>
                        <div class='timestamp'>{timestamp}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='message-box bot-message'>
                        <b>🤖 AI Assistant</b>
                        <div>{text}</div>
                        <div class='timestamp'>{timestamp}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Input section at the bottom
    st.markdown("---")
    
    # Create form for input
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        
        with col1:
            query = st.text_input(
                "Ask your question about transplant infections:",
                key="query_input",
                placeholder="Type your question here..."
            )
        
        with col2:
            submit_button = st.form_submit_button("Send 📤", use_container_width=True)
    
    # Process query
    if submit_button and query.strip():
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "text": query,
            "timestamp": format_timestamp()
        })
        
        # Show typing indicator
        with st.spinner("🔍 Searching through medical literature and generating response..."):
            try:
                start_time = time.time()
                
                # Send request to API
                payload = {"query": query, "history": st.session_state.chat_history}
                response = requests.post(f"{API_BASE_URL}/chat/", json=payload, timeout=30)
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "No response received.")
                    response_time = time.time() - start_time
                    
                    # Add response to history
                    st.session_state.chat_history.append({
                        "role": "bot",
                        "text": answer,
                        "timestamp": format_timestamp()
                    })
                    
                    # Show response time
                    st.success(f"✅ Response generated in {response_time:.1f} seconds")
                    st.rerun()
                else:
                    st.error(f"❌ API Error {response.status_code}: {response.text}")
                    st.info("Please check if the API server is running and try again.")
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The server might be processing a large query. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to the API server. Please ensure it's running.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

# TAB 2: PUBLICATIONS LIST
with tab2:
    st.markdown("### 📚 Medical Literature Corpus")
    st.markdown("Our AI assistant searches through the following 130+ transplant infection publications:")
    
    # Create columns for better display
    col1, col2 = st.columns(2)
    
    # Full list of publications
    publications = [
        "Multidrug-resistant bacteria in solid organ transplant recipients",
        "Multidrug-Resistant Gram-Negative Bacteria Infections in Solid Organ Transplantation",
        # ... (keeping the full list from your original code)
    ]
    
    # Split publications into two columns
    mid_point = len(publications) // 2
    
    with col1:
        for pub in publications[:mid_point]:
            st.markdown(f"• {pub}")
    
    with col2:
        for pub in publications[mid_point:]:
            st.markdown(f"• {pub}")
    
    # Download button for publication list
    publications_text = "\n".join(publications)
    st.download_button(
        label="📥 Download Publication List",
        data=publications_text,
        file_name="transplant_infections_publications.txt",
        mime="text/plain"
    )

# TAB 3: ABOUT
with tab3:
    st.markdown("""
    ### 🏥 About Transplant Infections AI Chat
    
    This AI-powered application uses **Retrieval-Augmented Generation (RAG)** to provide evidence-based answers 
    to questions about transplant infections.
    
    #### 🔍 How it Works
    
    1. **Document Processing**: 130+ medical publications are processed and indexed
    2. **Semantic Search**: Your question is matched with relevant content using vector embeddings
    3. **AI Generation**: GPT-4o generates a comprehensive answer based on the retrieved content
    4. **Clinical Context**: The AI is prompted to act as a clinician scientist specializing in transplant infections
    
    #### 🛠️ Technical Stack
    
    - **Language Model**: OpenAI GPT-4o
    - **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
    - **Vector Database**: FAISS (Facebook AI Similarity Search)
    - **Backend**: FastAPI + Python
    - **Frontend**: Streamlit
    - **Deployment**: Railway
    
    #### 👥 Team
    
    - **App Development**: Shreyas Joshi
    - **Literature Curation**: Frank Liu, Berk Maden, Shreyas Joshi
    - **Institution**: Keating Lab, NYU Langone Health
    
    #### 📞 Support
    
    For bug reports, suggestions, or literature updates:
    - Email: shreyas.joshi@nyulangone.org
    - GitHub: [View Repository](https://github.com/shodhak/transplant-infections-app)
    
    #### ⚠️ Disclaimer
    
    This tool is designed to assist healthcare professionals and researchers. It should not replace clinical 
    judgment or be used as the sole basis for medical decisions. Always verify critical information and 
    consult current guidelines and literature.
    """)
