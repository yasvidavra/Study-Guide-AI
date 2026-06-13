# =============================================================================
# AI Academic Assistant — app.py
# Features: RAG pipeline, FAISS, Groq LLM, SQLite, Search Filter,
#           Production-grade logging, Session-state guards.
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
# Logging configuration
# Writes to both logs/app.log (file) and the console.
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)   # create logs/ folder if it doesn't exist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("logs/app.log", encoding="utf-8"),
        logging.StreamHandler(),          # also print to terminal
    ],
)
logger = logging.getLogger("AcademicAssistant")
logger.info("Application started")

# ---------------------------------------------------------------------------
# Load environment variables from .env file (must be in project root)
# ---------------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # reserved for future use

# Fail fast with a clear message if the required key is missing
if not GROQ_API_KEY:
    logger.critical("GROQ_API_KEY is not set. Application cannot start.")
    st.error(
        "⛔ GROQ_API_KEY is not set. "
        "Create a `.env` file in the project root and add:\n\n"
        "```\nGROQ_API_KEY=your_key_here\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Academic Assistant",
    page_icon="📚",
    layout="wide",
)

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def create_database():
    """Create the notes table if it does not already exist."""
    conn = sqlite3.connect("academic_assistant.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT,
            total_pages INTEGER,
            upload_date TEXT,
            summary     TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialised — notes table ready")


def save_note(filename: str, total_pages: int, upload_date: str, summary: str):
    """Insert a note only if a record with the same filename does not exist
    (prevents duplicate inserts on every Streamlit rerun)."""
    conn = sqlite3.connect("academic_assistant.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM notes WHERE filename = ?", (filename,))
    existing = cursor.fetchone()
    if existing is None:
        cursor.execute(
            """
            INSERT INTO notes (filename, total_pages, upload_date, summary)
            VALUES (?, ?, ?, ?)
            """,
            (filename, total_pages, upload_date, summary),
        )
        conn.commit()
        logger.info("DB INSERT — filename='%s', pages=%d, date='%s'",
                    filename, total_pages, upload_date)
    else:
        logger.debug("DB SKIP — '%s' already exists (id=%d)", filename, existing[0])
    conn.close()


def load_all_notes() -> pd.DataFrame:
    """Return all saved notes as a pandas DataFrame."""
    conn = sqlite3.connect("academic_assistant.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM notes")
    records = cursor.fetchall()
    conn.close()
    df = pd.DataFrame(
        records,
        columns=["ID", "Filename", "Pages", "Upload Time", "Summary"],
    )
    return df


def search_notes(query: str) -> pd.DataFrame:
    """Return notes whose filename contains the search query (case-insensitive)."""
    conn = sqlite3.connect("academic_assistant.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM notes WHERE LOWER(filename) LIKE LOWER(?)",
        (f"%{query}%",),
    )
    records = cursor.fetchall()
    conn.close()
    logger.info("Search notes — query='%s', results=%d", query, len(records))
    df = pd.DataFrame(
        records,
        columns=["ID", "Filename", "Pages", "Upload Time", "Summary"],
    )
    return df


def delete_note(note_id: int) -> bool:
    """Delete a note by its ID. Returns True if a row was deleted, False otherwise."""
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
# Initialise database on startup
# ---------------------------------------------------------------------------
create_database()

# ---------------------------------------------------------------------------
# Session-state initialisation
# Flags prevent features from re-executing on unrelated widget interactions.
# ---------------------------------------------------------------------------
for key in ("show_notes", "summary_result", "quiz_result", "qa_result", "qa_question"):
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("🎓 AI Academic Assistant")
st.markdown("""
### 🤖 Your Personal AI Study Partner

📚 Upload PDF notes • 💬 Ask questions • ⚡ Get instant answers

🧠 Powered by RAG + AI to help you learn smarter, faster, and better.
""")

# ---------------------------------------------------------------------------
# View Saved Notes + Search Filter
# Both panels are independent of PDF upload — always visible.
# ---------------------------------------------------------------------------
col_view, col_search = st.columns([1, 2])

with col_view:
    if st.button("📚 View Saved Notes", key="btn_view_notes"):
        st.session_state.show_notes = True

with col_search:
    search_query = st.text_input(
        "🔍 Search saved notes by filename",
        placeholder="e.g. COA, OOP, maths...",
        key="search_input",
    )

# ── View all notes ────────────────────────────────────────────────────────
if st.session_state.show_notes and not search_query:
    st.subheader("📚 All Saved Notes")
    df_notes = load_all_notes()
    if df_notes.empty:
        st.info("No notes saved yet. Upload a PDF to get started.")
    else:
        st.dataframe(df_notes, use_container_width=True)
        st.caption(f"Total records: {len(df_notes)}")

        # ── Delete a note by ID (completes CRUD) ──────────────────────────
        with st.expander("🗑️ Delete a Note"):
            del_id = st.number_input(
                "Enter Note ID to delete",
                min_value=1, step=1, key="del_note_id"
            )
            if st.button("Delete Note", key="btn_delete_note"):
                if delete_note(int(del_id)):
                    st.success(f"✅ Note ID {int(del_id)} deleted successfully.")
                    st.rerun()
                else:
                    st.error(f"❌ No note found with ID {int(del_id)}.")

# ── Search results ────────────────────────────────────────────────────────
if search_query:
    st.subheader(f"🔍 Search Results for: *{search_query}*")
    df_search = search_notes(search_query)
    if df_search.empty:
        st.warning(f"No notes found matching **'{search_query}'**.")
    else:
        st.success(f"Found **{len(df_search)}** matching record(s).")
        st.dataframe(df_search, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# PDF Upload widget — always rendered so it never disappears on rerun
# ---------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "📂 Upload PDF Notes",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader",
)

# ---------------------------------------------------------------------------
# All PDF-dependent logic lives inside this block
# ---------------------------------------------------------------------------
if uploaded_files:
    logger.info("PDF upload event — %d file(s): %s",
                len(uploaded_files), [f.name for f in uploaded_files])
    st.success(f"✅ {len(uploaded_files)} PDF(s) uploaded successfully!")

    # ------------------------------------------------------------------
    # Step 1 — Read pages & extract text (single pass per PDF)
    # ------------------------------------------------------------------
    total_pages = 0
    full_text = ""
    upload_date = datetime.now().strftime("%d-%m-%Y %H:%M")

    for pdf in uploaded_files:
        try:
            reader = PdfReader(pdf)
            total_pages += len(reader.pages)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text
            logger.info("Extracted text from '%s' (%d pages)", pdf.name, len(reader.pages))
        except Exception as exc:
            logger.error("Failed to read PDF '%s': %s", pdf.name, exc, exc_info=True)
            st.error(f"Could not read {pdf.name}: {exc}")

    st.write(f"📄 Total Pages Loaded: **{total_pages}**")
    st.write(f"🔤 Total Characters Extracted: **{len(full_text)}**")

    # ------------------------------------------------------------------
    # Step 2 — Save each PDF to SQLite exactly once (duplicate-safe)
    # ------------------------------------------------------------------
    for pdf in uploaded_files:
        save_note(
            filename=pdf.name,
            total_pages=total_pages,
            upload_date=upload_date,
            summary="Summary not generated yet",
        )

    # ------------------------------------------------------------------
    # Step 3 — Chunk text
    # ------------------------------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10_000,
        chunk_overlap=500,
    )
    chunks = text_splitter.split_text(full_text)
    st.write(f"🧩 Total Chunks Created: **{len(chunks)}**")
    logger.info("Text split into %d chunks", len(chunks))

    # ------------------------------------------------------------------
    # Step 4 — Build FAISS vector store (limit to 15 chunks to avoid quota)
    # ------------------------------------------------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    documents = [Document(page_content=chunk) for chunk in chunks[:15]]
    st.write(f"📦 Chunks Sent to FAISS: **{len(documents)}**")

    vectorstore = FAISS.from_documents(documents, embeddings)
    st.write("✅ FAISS Vector Database Created Successfully")
    logger.info("FAISS vector store built — %d documents indexed", len(documents))

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    st.write("✅ Retriever Created Successfully")

    # ------------------------------------------------------------------
    # Step 5 — Initialise LLM (once per upload block)
    # ------------------------------------------------------------------
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
    )

    # Context used by both Summary and Quiz features
    summary_context = ""
    for doc in documents[:2]:
        summary_context += doc.page_content[:3_000] + "\n\n"

    # Context used by Q&A feature
    qa_context = ""
    for doc in documents:
        qa_context += doc.page_content + "\n\n"

    st.divider()

    # ------------------------------------------------------------------
    # Step 6 — AI Study Tools UI
    # ------------------------------------------------------------------
    st.markdown("### 🚀 AI Study Tools")

    col1, col2 = st.columns(2)
    with col1:
        summary_button = st.button("📝 Generate Summary", key="btn_summary")
    with col2:
        quiz_button = st.button("🎯 Generate Quiz", key="btn_quiz")

    question = st.text_input(
        "💬 Ask a Question from Your Notes",
        key="question_input",
    )

    # ------------------------------------------------------------------
    # Summary Feature — executes exactly once per button click
    # Result stored in session_state to survive reruns.
    # ------------------------------------------------------------------
    if summary_button:
        logger.info("Summary generation requested")
        with st.spinner("Generating Summary..."):
            try:
                summary_prompt = f"""
Summarize these notes in simple bullet points.

Notes:
{summary_context}
"""
                summary_response = llm.invoke(summary_prompt)
                st.session_state.summary_result = summary_response.content
                logger.info("Summary generated successfully (%d chars)",
                            len(summary_response.content))
            except Exception as exc:
                logger.error("Summary generation failed: %s", exc, exc_info=True)
                st.error(f"Summary failed: {exc}")

    if st.session_state.summary_result:
        st.subheader("📝 Notes Summary")
        st.write(st.session_state.summary_result)
        st.download_button(
            label="⬇️ Download Summary",
            data=st.session_state.summary_result,
            file_name="study_summary.txt",
            mime="text/plain",
            key="dl_summary",
        )

    # ------------------------------------------------------------------
    # Quiz Feature — executes exactly once per button click
    # ------------------------------------------------------------------
    if quiz_button:
        logger.info("Quiz generation requested")
        with st.spinner("Generating Quiz..."):
            try:
                quiz_prompt = f"""
Create 5 MCQ questions from these notes.

For each question provide:
Question
A)
B)
C)
D)

Correct Answer:

Notes:
{summary_context}
"""
                quiz_response = llm.invoke(quiz_prompt)
                st.session_state.quiz_result = quiz_response.content
                logger.info("Quiz generated successfully (%d chars)",
                            len(quiz_response.content))
            except Exception as exc:
                logger.error("Quiz generation failed: %s", exc, exc_info=True)
                st.error(f"Quiz failed: {exc}")

    if st.session_state.quiz_result:
        st.subheader("🎯 AI Quiz")
        st.write(st.session_state.quiz_result)

    # ------------------------------------------------------------------
    # Q&A Feature — executes only when a new question is submitted
    # Tracks last question in session_state to avoid duplicate answers.
    # ------------------------------------------------------------------
    if question and question != st.session_state.qa_question:
        st.session_state.qa_question = question
        logger.info("User question: '%s'", question)
        try:
            with st.spinner("Thinking..."):
                qa_prompt = f"""
You are an academic tutor.

Answer from the provided notes in a detailed and student-friendly way.

If the topic exists in the notes:
- Explain it properly
- Use bullet points when helpful
- Give important details and features
- Keep the explanation easy to understand

If the answer is not present in the notes, say:
"I could not find the answer in the uploaded notes."

NOTES:
{qa_context}

QUESTION:
{question}
"""
                response = llm.invoke(qa_prompt)
                st.session_state.qa_result = response.content
                logger.info("Answer generated for question: '%s' (%d chars)",
                            question, len(response.content))
        except Exception as exc:
            logger.error("Q&A failed for question '%s': %s", question, exc, exc_info=True)
            st.error(f"Error: {exc}")
            st.session_state.qa_result = None

    if st.session_state.qa_result and st.session_state.qa_question:
        st.subheader("💡 Answer")
        st.write(st.session_state.qa_result)

else:
    # Friendly prompt shown when no files are uploaded yet
    st.info("👆 Please upload one or more PDF files to get started.")
