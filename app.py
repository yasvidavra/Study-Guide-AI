# =============================================================================
# AI Academic Assistant — app.py  (Final Version)
# Features: RAG pipeline, FAISS, Groq LLM, SQLite, Search Filter,
#           Metric Dashboard, Production Logging, Session-state guards.
# Security: All API keys loaded from .env via python-dotenv (never hardcoded).
# =============================================================================

import logging
import os
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Logging — file + console, set up once before anything else
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("logs/app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("AcademicAssistant")
logger.info("Application started")

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
load_dotenv()
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")   # reserved for future use

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Academic Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Guard: stop early if API key is missing
if not GROQ_API_KEY:
    logger.critical("GROQ_API_KEY is not set. Application cannot start.")
    st.error(
        "⛔ **GROQ_API_KEY is missing.**  \n"
        "Create a `.env` file in the project root and add:  \n"
        "```\nGROQ_API_KEY=your_key_here\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Custom CSS — professional, student-friendly styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Global font & background ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    border: 1px solid #3a8fd1;
    border-radius: 12px;
    padding: 16px 20px;
    color: white;
}
[data-testid="metric-container"] label {
    color: #a8d4f5 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

/* ── Section headings ── */
h3 { color: #1e3a5f; border-bottom: 2px solid #e8f0fe; padding-bottom: 6px; }

/* ── Info / success boxes ── */
.stAlert { border-radius: 10px; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background-color: #f0f4ff;
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — Developer Tools (collapsed by default, not in the main UI flow)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛠️ Developer Tools")
    st.caption("Internal diagnostic panel")
    st.divider()

    with st.expander("📋 Recent Log Entries", expanded=False):
        log_path = "logs/app.log"
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            last_lines = lines[-20:] if len(lines) > 20 else lines
            st.code("".join(last_lines), language="text")
            st.caption(f"Showing {len(last_lines)} of {len(lines)} total entries")
        else:
            st.info("No log file yet. Upload a PDF to generate events.")

    st.divider()
    st.caption("📁 `logs/app.log`  |  Level: INFO+")

# ---------------------------------------------------------------------------
# SQLite — Database helpers (full CRUD)
# ---------------------------------------------------------------------------

def create_database():
    """Create the notes table if it does not already exist."""
    conn = sqlite3.connect("academic_assistant.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT UNIQUE,
            total_pages INTEGER,
            upload_date TEXT,
            summary     TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialised — notes table ready")


def save_note(filename: str, total_pages: int, upload_date: str, summary: str):
    """Insert a note only if filename does not already exist (duplicate-safe)."""
    conn = sqlite3.connect("academic_assistant.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM notes WHERE filename = ?", (filename,))
    if cursor.fetchone() is None:
        cursor.execute(
            "INSERT INTO notes (filename, total_pages, upload_date, summary) VALUES (?,?,?,?)",
            (filename, total_pages, upload_date, summary),
        )
        conn.commit()
        logger.info("DB INSERT — '%s', pages=%d", filename, total_pages)
    else:
        logger.debug("DB SKIP — '%s' already exists", filename)
    conn.close()


def load_all_notes() -> pd.DataFrame:
    """Return all saved notes sorted by upload_date descending."""
    conn = sqlite3.connect("academic_assistant.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM notes ORDER BY upload_date DESC")
    records = cursor.fetchall()
    conn.close()
    return pd.DataFrame(
        records,
        columns=["ID", "Filename", "Pages", "Upload Time", "Summary"],
    )


def search_notes(query: str) -> pd.DataFrame:
    """Return notes matching filename query (case-insensitive LIKE)."""
    conn = sqlite3.connect("academic_assistant.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM notes WHERE LOWER(filename) LIKE LOWER(?) ORDER BY upload_date DESC",
        (f"%{query}%",),
    )
    records = cursor.fetchall()
    conn.close()
    logger.info("Search — query='%s', hits=%d", query, len(records))
    return pd.DataFrame(
        records,
        columns=["ID", "Filename", "Pages", "Upload Time", "Summary"],
    )


def count_notes() -> int:
    """Return total number of saved notes."""
    conn = sqlite3.connect("academic_assistant.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM notes")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def delete_note(note_id: int) -> bool:
    """Delete a note by ID. Returns True if deleted."""
    conn = sqlite3.connect("academic_assistant.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    if deleted:
        logger.info("DB DELETE — note id=%d removed", note_id)
    else:
        logger.warning("DB DELETE — note id=%d not found", note_id)
    return deleted


# ---------------------------------------------------------------------------
# Initialise DB on every startup
# ---------------------------------------------------------------------------
create_database()

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
_state_defaults = {
    "show_notes": False,
    "summary_result": None,
    "quiz_result": None,
    "qa_result": None,
    "qa_question": None,
    "total_pages": 0,
    "num_chunks": 0,
    "num_docs": 0,
}
for key, val in _state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ===========================================================================
# ── PAGE HEADER ─────────────────────────────────────────────────────────────
# ===========================================================================
st.markdown("""
<div style='text-align:center; padding: 1.5rem 0 0.5rem 0;'>
    <h1 style='font-size:2.6rem; color:#1e3a5f; margin-bottom:0;'>
        🎓 AI Academic Assistant
    </h1>
    <p style='color:#5a7fa8; font-size:1.05rem; margin-top:0.4rem;'>
        Upload your PDF notes · Get AI-powered summaries, quizzes & answers
    </p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ===========================================================================
# ── DASHBOARD METRIC CARDS ──────────────────────────────────────────────────
# ===========================================================================
saved_notes_count = count_notes()

m1, m2, m3, m4 = st.columns(4)
m1.metric("📄 PDFs Uploaded",   len(st.session_state.get("uploaded_names", [])) or "—")
m2.metric("📑 Total Pages",     st.session_state.total_pages or "—")
m3.metric("🧩 Chunks Indexed",  st.session_state.num_docs or "—")
m4.metric("🗄️ Saved Notes",    saved_notes_count)

st.divider()

# ===========================================================================
# ── SAVED NOTES PANEL ───────────────────────────────────────────────────────
# ===========================================================================
st.markdown("### 📚 Saved Notes")

col_btn, col_search, col_spacer = st.columns([1, 2, 1])

with col_btn:
    if st.button("📚 View All Notes", key="btn_view_notes", use_container_width=True):
        st.session_state.show_notes = not st.session_state.show_notes

with col_search:
    search_query = st.text_input(
        "🔍 Search by filename",
        placeholder="e.g. COA, OOP, maths…",
        key="search_input",
        label_visibility="collapsed",
    )

# ── Show search results ──────────────────────────────────────────────────────
if search_query:
    df_search = search_notes(search_query)
    if df_search.empty:
        st.warning(f"No notes found matching **'{search_query}'**.")
    else:
        st.success(f"🔍 Found **{len(df_search)}** record(s) matching *'{search_query}'*")
        st.dataframe(
            df_search,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ID":          st.column_config.NumberColumn("ID", width="small"),
                "Filename":    st.column_config.TextColumn("📄 Filename"),
                "Pages":       st.column_config.NumberColumn("Pages", width="small"),
                "Upload Time": st.column_config.TextColumn("🕐 Uploaded"),
                "Summary":     st.column_config.TextColumn("📝 Summary", width="large"),
            },
        )

# ── Show all notes ───────────────────────────────────────────────────────────
elif st.session_state.show_notes:
    df_notes = load_all_notes()
    if df_notes.empty:
        st.info("No notes saved yet. Upload a PDF to get started.")
    else:
        st.dataframe(
            df_notes,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ID":          st.column_config.NumberColumn("ID", width="small"),
                "Filename":    st.column_config.TextColumn("📄 Filename"),
                "Pages":       st.column_config.NumberColumn("Pages", width="small"),
                "Upload Time": st.column_config.TextColumn("🕐 Uploaded"),
                "Summary":     st.column_config.TextColumn("📝 Summary", width="large"),
            },
        )
        st.caption(f"📊 {len(df_notes)} records · Sorted by most recent upload")

        # Delete a note — inside an expander so it doesn't clutter the UI
        with st.expander("🗑️ Delete a Note by ID", expanded=False):
            dcol1, dcol2 = st.columns([1, 1])
            with dcol1:
                del_id = st.number_input(
                    "Note ID to delete", min_value=1, step=1, key="del_note_id"
                )
            with dcol2:
                st.write("")   # vertical alignment spacer
                st.write("")
                if st.button("🗑️ Delete", key="btn_delete_note", type="secondary"):
                    if delete_note(int(del_id)):
                        st.success(f"✅ Note ID {int(del_id)} deleted.")
                        st.rerun()
                    else:
                        st.error(f"❌ No note found with ID {int(del_id)}.")

st.divider()

# ===========================================================================
# ── PDF UPLOAD SECTION ───────────────────────────────────────────────────────
# ===========================================================================
st.markdown("### 📂 Upload Your PDF Notes")

uploaded_files = st.file_uploader(
    "Drag & drop PDF files here, or click to browse",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader",
    label_visibility="collapsed",
)

# ===========================================================================
# ── ALL PDF-DEPENDENT LOGIC ──────────────────────────────────────────────────
# ===========================================================================
if uploaded_files:
    logger.info("Upload event — %d file(s): %s",
                len(uploaded_files), [f.name for f in uploaded_files])

    # Store uploaded names for metric display
    st.session_state.uploaded_names = [f.name for f in uploaded_files]

    st.success(f"✅ **{len(uploaded_files)} PDF(s)** uploaded and ready to process!")

    # ── Step 1: Extract text ─────────────────────────────────────────────────
    with st.spinner("📖 Reading and extracting text from PDFs…"):
        total_pages = 0
        full_text   = ""
        upload_date = datetime.now().strftime("%d-%m-%Y %H:%M")

        for pdf in uploaded_files:
            try:
                reader = PdfReader(pdf)
                total_pages += len(reader.pages)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text
                logger.info("Extracted '%s' — %d pages", pdf.name, len(reader.pages))
            except Exception as exc:
                logger.error("Failed to read '%s': %s", pdf.name, exc, exc_info=True)
                st.error(f"Could not read **{pdf.name}**: {exc}")

        st.session_state.total_pages = total_pages

    # ── Step 2: Save to SQLite ───────────────────────────────────────────────
    for pdf in uploaded_files:
        save_note(
            filename=pdf.name,
            total_pages=total_pages,
            upload_date=upload_date,
            summary="Summary not generated yet",
        )

    # ── Step 3: Chunk text ───────────────────────────────────────────────────
    with st.spinner("🧩 Splitting text into chunks…"):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10_000,
            chunk_overlap=500,
        )
        chunks = text_splitter.split_text(full_text)
        logger.info("Text split into %d chunks", len(chunks))

    # ── Step 4: FAISS vector store ───────────────────────────────────────────
    with st.spinner("⚡ Building FAISS vector database…"):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        documents = [Document(page_content=chunk) for chunk in chunks[:15]]
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})
        st.session_state.num_docs = len(documents)
        logger.info("FAISS built — %d docs indexed", len(documents))

    # ── Processing summary (compact, not debug-level noise) ──────────────────
    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("📑 Pages Extracted",  total_pages)
    pc2.metric("🔤 Characters",       f"{len(full_text):,}")
    pc3.metric("📦 Chunks Indexed",   len(documents))

    # ── Step 5: Initialise LLM ───────────────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

    summary_context = "".join(
        doc.page_content[:3_000] + "\n\n" for doc in documents[:2]
    )
    qa_context = "".join(doc.page_content + "\n\n" for doc in documents)

    st.divider()

    # ========================================================================
    # ── AI STUDY TOOLS ───────────────────────────────────────────────────────
    # ========================================================================
    st.markdown("### 🚀 AI Study Tools")
    st.caption("Choose a tool below to interact with your uploaded notes.")

    tool_col1, tool_col2 = st.columns(2)
    with tool_col1:
        summary_button = st.button(
            "📝 Generate Summary", key="btn_summary",
            use_container_width=True, type="primary"
        )
    with tool_col2:
        quiz_button = st.button(
            "🎯 Generate Quiz", key="btn_quiz",
            use_container_width=True, type="primary"
        )

    st.write("")  # spacing
    question = st.text_input(
        "💬 Ask a Question from Your Notes",
        placeholder="e.g. What is cache memory? Explain OOPS concepts…",
        key="question_input",
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    if summary_button:
        logger.info("Summary requested")
        with st.spinner("✍️ Generating summary…"):
            try:
                resp = llm.invoke(
                    f"Summarize these notes clearly in simple bullet points.\n\n{summary_context}"
                )
                st.session_state.summary_result = resp.content
                logger.info("Summary done — %d chars", len(resp.content))
            except Exception as exc:
                logger.error("Summary failed: %s", exc, exc_info=True)
                st.error(f"Summary failed: {exc}")

    if st.session_state.summary_result:
        with st.container():
            st.markdown("#### 📝 Notes Summary")
            st.markdown(st.session_state.summary_result)
            st.download_button(
                "⬇️ Download Summary (.txt)",
                data=st.session_state.summary_result,
                file_name="study_summary.txt",
                mime="text/plain",
                key="dl_summary",
            )

    # ── Quiz ──────────────────────────────────────────────────────────────────
    if quiz_button:
        logger.info("Quiz requested")
        with st.spinner("🎯 Generating quiz questions…"):
            try:
                quiz_prompt = (
                    "Create 5 MCQ questions from these notes.\n\n"
                    "For each question provide:\nQuestion\nA) B) C) D)\nCorrect Answer:\n\n"
                    f"Notes:\n{summary_context}"
                )
                resp = llm.invoke(quiz_prompt)
                st.session_state.quiz_result = resp.content
                logger.info("Quiz done — %d chars", len(resp.content))
            except Exception as exc:
                logger.error("Quiz failed: %s", exc, exc_info=True)
                st.error(f"Quiz failed: {exc}")

    if st.session_state.quiz_result:
        with st.container():
            st.markdown("#### 🎯 AI Quiz")
            st.markdown(st.session_state.quiz_result)

    # ── Q&A ───────────────────────────────────────────────────────────────────
    if question and question != st.session_state.qa_question:
        st.session_state.qa_question = question
        logger.info("Question: '%s'", question)
        try:
            with st.spinner("🤔 Finding the answer…"):
                qa_prompt = (
                    "You are an academic tutor. Answer from the notes below in a "
                    "detailed, student-friendly way using bullet points where helpful.\n\n"
                    "If the answer is not in the notes, say: "
                    "'I could not find the answer in the uploaded notes.'\n\n"
                    f"NOTES:\n{qa_context}\n\nQUESTION:\n{question}"
                )
                resp = llm.invoke(qa_prompt)
                st.session_state.qa_result = resp.content
                logger.info("Answer done — %d chars", len(resp.content))
        except Exception as exc:
            logger.error("Q&A failed: %s", exc, exc_info=True)
            st.error(f"Error: {exc}")
            st.session_state.qa_result = None

    if st.session_state.qa_result and st.session_state.qa_question:
        with st.container():
            st.markdown("#### 💡 Answer")
            st.markdown(st.session_state.qa_result)

else:
    # ── Empty state — friendly onboarding ────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding:2rem; background:#f8faff;
                border-radius:12px; border:2px dashed #c8d8f0; margin-top:1rem;'>
        <h3 style='color:#5a7fa8;'>👆 Upload your PDF notes to get started</h3>
        <p style='color:#8eaacc;'>
            Supported format: <strong>PDF</strong> &nbsp;·&nbsp;
            Multiple files supported &nbsp;·&nbsp;
            Your notes stay private
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown("""
<div style='text-align:center; color:#9ab0cc; font-size:0.82rem; padding-bottom:1rem;'>
    🎓 AI Academic Assistant &nbsp;·&nbsp;
    Powered by <strong>LangChain · FAISS · Groq LLaMA 3.1 · HuggingFace</strong>
    &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
