import streamlit as st
import fitz
from qdrant_client import QdrantClient, models
import hashlib
from datetime import datetime
import ollama
import requests
import json
import re

# =============================
# CONFIG
# =============================
CHAT_MODEL = "openai/gpt-oss-20b"
EMBED_MODEL = "qwen3-embedding:8b"
VECTOR_SIZE = 4096
QDRANT_COLLECTION = "research_papers"
LMSTUDIO_BASE = "http://127.0.0.1:4040/v1"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Academic RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# CUSTOM CSS
# =============================
def load_css():
    st.markdown("""
    <style>
        /* Typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+Pro:wght@400;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        h1, h2, h3 {
            font-family: 'Source Serif Pro', serif;
            color: #1a1a2e;
        }
        
        /* Main container */
        .block-container {
            padding-top: 2rem;
            max-width: 1200px;
        }
        
        /* Header styling */
        .main-header {
            font-size: 1.75rem;
            font-weight: 600;
            color: #1a1a2e;
            margin-bottom: 0.25rem;
            font-family: 'Source Serif Pro', serif;
        }
        
        .sub-header {
            font-size: 0.95rem;
            color: #5a6778;
            margin-bottom: 1.5rem;
            line-height: 1.5;
        }
        
        /* Paper cards */
        .paper-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
            transition: border-color 0.2s;
        }
        
        .paper-card:hover {
            border-color: #94a3b8;
        }
        
        .paper-title {
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 0.375rem;
            font-size: 0.95rem;
            line-height: 1.4;
        }
        
        .paper-meta {
            font-size: 0.8rem;
            color: #64748b;
        }
        
        /* Source citations */
        .source-container {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .source-header {
            display: flex;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        
        .source-number {
            background: #3b82f6;
            color: white;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.125rem 0.5rem;
            border-radius: 4px;
            margin-right: 0.5rem;
        }
        
        .source-title {
            font-weight: 600;
            color: #1e293b;
            font-size: 0.9rem;
        }
        
        .source-file {
            font-size: 0.8rem;
            color: #64748b;
            margin-bottom: 0.5rem;
        }
        
        .source-content {
            font-size: 0.85rem;
            color: #475569;
            line-height: 1.6;
            padding: 0.75rem;
            background: white;
            border-radius: 4px;
            border: 1px solid #e2e8f0;
        }
        
        /* Chat messages */
        .chat-user {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 6px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .chat-assistant {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        /* Sidebar */
        .css-1d391kg {
            padding-top: 1rem;
        }
        
        .sidebar-header {
            font-size: 0.85rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.75rem;
        }
        
        /* Status badge */
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        
        .status-online {
            background: #dcfce7;
            color: #166534;
        }
        
        .status-offline {
            background: #fee2e2;
            color: #991b1b;
        }
        
        /* Buttons */
        .stButton > button {
            font-size: 0.875rem;
            font-weight: 500;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            transition: all 0.2s;
        }
        
        /* Hide default streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* Expander styling */
        .streamlit-expanderHeader {
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        /* Divider */
        .divider {
            height: 1px;
            background: #e2e8f0;
            margin: 1.5rem 0;
        }
        
        /* Metrics */
        .metric-container {
            background: #f8fafc;
            border-radius: 6px;
            padding: 1rem;
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1e293b;
        }
        
        .metric-label {
            font-size: 0.8rem;
            color: #64748b;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# =============================
# QDRANT CLIENT
# =============================
@st.cache_resource
def get_qdrant_client():
    client = QdrantClient("localhost", port=6333)
    try:
        client.get_collection(QDRANT_COLLECTION)
    except Exception:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )
    return client

# =============================
# PDF PROCESSING
# =============================
def extract_pdf_text(file):
    """Extract text content from a PDF file."""
    try:
        file.seek(0)
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        return "\n".join(text_parts)
    except Exception as e:
        st.error(f"Failed to process PDF: {e}")
        return None

def extract_metadata(file, filename):
    """Extract metadata and text from a PDF file."""
    file.seek(0)
    text = extract_pdf_text(file)
    
    meta = {
        "filename": filename,
        "upload_time": datetime.now().isoformat(),
        "file_size": len(file.getvalue())
    }
    
    if text:
        # Try to extract title from first few lines
        lines = text.split("\n")[:15]
        for line in lines:
            clean_line = line.strip()
            # Look for a substantial line that could be a title
            if len(clean_line) > 20 and len(clean_line) < 200:
                if not clean_line.lower().startswith(('abstract', 'keywords', 'introduction')):
                    meta["potential_title"] = clean_line
                    break
    
    return meta, text

def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks for better context."""
    if not text:
        return []
    
    # Clean the text
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings
            last_period = text.rfind('. ', start + chunk_size - 200, end)
            last_newline = text.rfind('\n', start + chunk_size - 200, end)
            break_point = max(last_period, last_newline)
            
            if break_point > start:
                end = break_point + 1
        
        chunk = text[start:end].strip()
        if len(chunk) > 50:  # Only add meaningful chunks
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks

# =============================
# EMBEDDINGS
# =============================
def generate_embedding(text):
    """Generate embedding for a text using Ollama."""
    try:
        result = ollama.embed(model=EMBED_MODEL, input=text)
        return result["embeddings"][0]
    except Exception as e:
        return None

# =============================
# SEMANTIC SEARCH
# =============================
def semantic_search(query, limit=5):
    """Search for relevant document chunks."""
    embedding = generate_embedding(query)
    if embedding is None:
        return []
    
    client = get_qdrant_client()
    
    try:
        results = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=embedding,
            limit=limit,
            with_payload=True
        )
        return results.points
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

def get_collection_stats():
    """Get statistics about the vector collection."""
    try:
        client = get_qdrant_client()
        info = client.get_collection(QDRANT_COLLECTION)
        return {
            "points": info.points_count,
            "vectors": info.vectors_count
        }
    except:
        return {"points": 0, "vectors": 0}

def get_unique_papers():
    """Get list of unique papers in the collection."""
    try:
        client = get_qdrant_client()
        # Scroll through all points to get unique filenames
        results = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=1000,
            with_payload=True
        )
        
        papers = {}
        for point in results[0]:
            filename = point.payload.get("filename", "Unknown")
            if filename not in papers:
                papers[filename] = {
                    "title": point.payload.get("potential_title", filename),
                    "filename": filename,
                    "upload_time": point.payload.get("upload_time", ""),
                    "chunks": 0
                }
            papers[filename]["chunks"] += 1
        
        return list(papers.values())
    except:
        return []

def delete_paper(filename):
    """Delete all chunks from a specific paper."""
    try:
        client = get_qdrant_client()
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                )
            )
        )
        return True
    except Exception as e:
        st.error(f"Failed to delete: {e}")
        return False

# =============================
# CONTEXT BUILDING
# =============================
def build_context(results):
    """Build context string from search results."""
    if not results:
        return "", []
    
    context_parts = []
    sources = []
    
    for i, r in enumerate(results, 1):
        payload = r.payload if hasattr(r, 'payload') else r.get('payload', {})
        chunk_text = payload.get('chunk_text', '')
        title = payload.get('potential_title', 'Untitled')
        filename = payload.get('filename', 'Unknown')
        score = r.score if hasattr(r, 'score') else 0
        
        context_parts.append(f"[Source {i}] From: {title}\n{chunk_text}")
        sources.append({
            "number": i,
            "title": title,
            "filename": filename,
            "content": chunk_text,
            "score": score
        })
    
    return "\n\n---\n\n".join(context_parts), sources

# =============================
# CHAT STREAMING
# =============================
def stream_chat(messages):
    """Stream chat responses from LM Studio."""
    try:
        response = requests.post(
            f"{LMSTUDIO_BASE}/chat/completions",
            headers={
                "Authorization": "Bearer lm-studio",
                "Content-Type": "application/json"
            },
            json={
                "model": CHAT_MODEL,
                "messages": messages,
                "stream": True,
                "temperature": 0.7
            },
            stream=True,
            timeout=180
        )
        
        for line in response.iter_lines():
            if line:
                line = line.decode()
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"]
                        if "content" in delta:
                            yield delta["content"]
                    except:
                        pass
    except requests.exceptions.ConnectionError:
        yield "Error: Cannot connect to LM Studio. Please ensure it is running on port 4040."
    except Exception as e:
        yield f"Error: {str(e)}"

def check_lmstudio_status():
    """Check if LM Studio is running."""
    try:
        response = requests.get(f"{LMSTUDIO_BASE}/models", timeout=2)
        return response.status_code == 200
    except:
        return False

def check_ollama_status():
    """Check if Ollama is running."""
    try:
        ollama.list()
        return True
    except:
        return False

# =============================
# UI COMPONENTS
# =============================
def render_source_card(source):
    """Render a source citation card."""
    st.markdown(f"""
    <div class="source-container">
        <div class="source-header">
            <span class="source-number">{source['number']}</span>
            <span class="source-title">{source['title'][:80]}{'...' if len(source['title']) > 80 else ''}</span>
        </div>
        <div class="source-file">{source['filename']} | Relevance: {source['score']:.2f}</div>
        <div class="source-content">{source['content'][:400]}{'...' if len(source['content']) > 400 else ''}</div>
    </div>
    """, unsafe_allow_html=True)

def render_paper_card(paper):
    """Render a paper card in the library."""
    title = paper['title'][:100] + ('...' if len(paper['title']) > 100 else '')
    st.markdown(f"""
    <div class="paper-card">
        <div class="paper-title">{title}</div>
        <div class="paper-meta">{paper['filename']} | {paper['chunks']} chunks</div>
    </div>
    """, unsafe_allow_html=True)

# =============================
# MAIN APPLICATION
# =============================
def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_sources" not in st.session_state:
        st.session_state.current_sources = []
    
    # =============================
    # SIDEBAR
    # =============================
    with st.sidebar:
        st.markdown('<p class="sidebar-header">System Status</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            ollama_ok = check_ollama_status()
            status_class = "status-online" if ollama_ok else "status-offline"
            status_text = "Online" if ollama_ok else "Offline"
            st.markdown(f'<span class="status-badge {status_class}">Ollama: {status_text}</span>', unsafe_allow_html=True)
        
        with col2:
            lm_ok = check_lmstudio_status()
            status_class = "status-online" if lm_ok else "status-offline"
            status_text = "Online" if lm_ok else "Offline"
            st.markdown(f'<span class="status-badge {status_class}">LM Studio: {status_text}</span>', unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Collection stats
        st.markdown('<p class="sidebar-header">Collection</p>', unsafe_allow_html=True)
        stats = get_collection_stats()
        papers = get_unique_papers()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Papers", len(papers))
        with col2:
            st.metric("Chunks", stats["points"])
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Paper library
        st.markdown('<p class="sidebar-header">Paper Library</p>', unsafe_allow_html=True)
        
        if papers:
            for paper in papers:
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{paper['title'][:40]}...**" if len(paper['title']) > 40 else f"**{paper['title']}**")
                        st.caption(f"{paper['chunks']} chunks")
                    with col2:
                        if st.button("X", key=f"del_{paper['filename']}", help="Remove paper"):
                            delete_paper(paper['filename'])
                            st.rerun()
        else:
            st.caption("No papers uploaded yet")
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Settings
        st.markdown('<p class="sidebar-header">Settings</p>', unsafe_allow_html=True)
        
        num_sources = st.slider("Sources to retrieve", 1, 10, 5)
        
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_sources = []
            st.rerun()
        
        if st.button("Reset Vector Database", use_container_width=True, type="secondary"):
            client = get_qdrant_client()
            client.delete_collection(QDRANT_COLLECTION)
            st.cache_resource.clear()
            st.success("Database cleared. Please refresh the page.")
    
    # =============================
    # MAIN CONTENT
    # =============================
    st.markdown('<h1 class="main-header">Academic Research Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload research papers to build your knowledge base, then ask questions to get answers grounded in your documents.</p>', unsafe_allow_html=True)
    
    # Tabs
    tab_upload, tab_chat = st.tabs(["Upload Papers", "Research Chat"])
    
    # =============================
    # UPLOAD TAB
    # =============================
    with tab_upload:
        st.markdown("### Add Papers to Your Collection")
        st.markdown("Upload PDF files of research papers. They will be processed and indexed for semantic search.")
        
        uploaded_files = st.file_uploader(
            "Select PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            process_button = st.button("Process and Index Papers", type="primary", use_container_width=True)
            
            if process_button:
                client = get_qdrant_client()
                
                for file in uploaded_files:
                    with st.status(f"Processing: {file.name}", expanded=True) as status:
                        # Extract text and metadata
                        st.write("Extracting text...")
                        meta, text = extract_metadata(file, file.name)
                        
                        if not text:
                            status.update(label=f"Failed: {file.name}", state="error")
                            continue
                        
                        # Create chunks
                        st.write("Creating chunks...")
                        chunks = create_chunks(text)
                        st.write(f"Created {len(chunks)} chunks")
                        
                        # Generate embeddings and store
                        st.write("Generating embeddings...")
                        points = []
                        progress = st.progress(0)
                        
                        for i, chunk in enumerate(chunks):
                            embedding = generate_embedding(chunk)
                            if embedding is not None:
                                point_id = hashlib.md5(f"{file.name}_{i}_{datetime.now().isoformat()}".encode()).hexdigest()
                                points.append(
                                    models.PointStruct(
                                        id=point_id,
                                        vector=embedding,
                                        payload={
                                            **meta,
                                            "chunk_text": chunk,
                                            "chunk_index": i
                                        }
                                    )
                                )
                            progress.progress((i + 1) / len(chunks))
                        
                        if points:
                            st.write("Storing in vector database...")
                            client.upsert(QDRANT_COLLECTION, points)
                            status.update(label=f"Completed: {file.name} ({len(points)} chunks)", state="complete")
                        else:
                            status.update(label=f"Failed: {file.name} (no embeddings)", state="error")
                
                st.success("Processing complete. Switch to the Research Chat tab to start asking questions.")
                st.cache_resource.clear()
    
    # =============================
    # CHAT TAB
    # =============================
    with tab_chat:
        # Check if there are papers
        if not get_unique_papers():
            st.info("Upload some papers first to start chatting.")
            return
        
        # Layout: Chat on left, sources on right
        chat_col, sources_col = st.columns([3, 2])
        
        with chat_col:
            st.markdown("### Chat")
            
            # Display chat history
            chat_container = st.container()
            
            with chat_container:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about your papers..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                
                # Search for relevant context
                with st.spinner("Searching papers..."):
                    results = semantic_search(prompt, limit=num_sources)
                    context, sources = build_context(results)
                    st.session_state.current_sources = sources
                
                # Build messages for LLM
                system_prompt = """You are a research assistant helping with academic paper analysis. 
                
Your task is to answer questions based ONLY on the provided context from research papers. Follow these guidelines:

1. Base your answers strictly on the provided sources
2. Cite sources using [Source N] notation
3. If the context doesn't contain enough information, say so clearly
4. Preserve technical terminology and mathematical notation
5. Use LaTeX notation for equations: inline $equation$ or display $$equation$$
6. Structure longer answers with clear headings and bullet points
7. Be precise and academic in tone
8. Answer in any language you're spoken to, otherwise default to English

Context from papers:
""" + context
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    *st.session_state.messages
                ]
                
                # Stream response
                with chat_container:
                    with st.chat_message("assistant"):
                        response_placeholder = st.empty()
                        full_response = ""
                        
                        for token in stream_chat(messages):
                            full_response += token
                            response_placeholder.markdown(full_response + "▌")
                        
                        response_placeholder.markdown(full_response)
                
                # Save assistant message
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        with sources_col:
            st.markdown("### Sources")
            
            if st.session_state.current_sources:
                for source in st.session_state.current_sources:
                    with st.expander(f"[{source['number']}] {source['title'][:50]}...", expanded=False):
                        st.caption(f"File: {source['filename']}")
                        st.caption(f"Relevance: {source['score']:.3f}")
                        st.markdown("---")
                        st.markdown(source['content'])
            else:
                st.caption("Sources will appear here after you ask a question.")

if __name__ == "__main__":
    main()