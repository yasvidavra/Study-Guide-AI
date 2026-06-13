# =============================================================================
# AI Academic Assistant — app.py
# Fixed: NameErrors, indentation/scope bugs, duplicate executions,
#        View Saved Notes layout, SQLite duplicate inserts, pandas display.
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
# Load environment variables from .env file (must be in project root)
# ---------------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # reserved for future use

# Fail fast with a clear message if the required key is missing
if not GROQ_API_KEY:
    st.error(
        "⛔ GROQ_API_KEY is not set. "
        "Create a `.env` file in the project root and add:\n\n"
        "```\nGROQ_API_KEY=your_key_here\n```"
    )
    st.stop()

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
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("AcademicAssistant")
logger.info("Application started")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Academic Assistant",
    page_icon="📚",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — Log Viewer (shows last 20 lines of logs/app.log)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🛠️ Developer Tools")
    st.markdown("---")
    if st.button("📋 View Recent Logs", key="btn_view_logs"):
        log_path = "logs/app.log"
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            last_lines = lines[-20:] if len(lines) > 20 else lines
            st.text_area(
                "Last 20 log entries",
                value="".join(last_lines),
                height=300,
                key="log_viewer",
            )
        else:
            st.info("No log file found yet. Upload a PDF to generate events.")
    st.markdown("---")
    st.caption("📁 Log file: `logs/app.log`")
    st.caption("🔍 Level: INFO and above")


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


def save_note(filename: str, total_pages: int, upload_date: str, summary: str):
    """Insert a note only if a record with the same filename does not exist
    (prevents duplicate inserts on every Streamlit rerun)."""
    conn = sqlite3.connect("academic_assistant.db")
    cursor = conn.cursor()
    # Check for an existing record with this filename
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
# View Saved Notes (independent of PDF upload — always visible)
# Uses session_state so clicking other buttons does not collapse the table.
# ---------------------------------------------------------------------------
if st.button("📚 View Saved Notes"):
    st.session_state.show_notes = True

if st.session_state.show_notes:
    st.subheader("📚 Stored Notes")
    df_notes = load_all_notes()
    if df_notes.empty:
        st.info("No notes saved yet. Upload a PDF to get started.")
    else:
        st.dataframe(df_notes, use_container_width=True)

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
    st.success(f"✅ {len(uploaded_files)} PDF(s) uploaded successfully!")

    # ------------------------------------------------------------------
    # Step 1 — Read pages & extract text (single pass per PDF)
    # ------------------------------------------------------------------
    total_pages = 0
    full_text = ""
    upload_date = datetime.now().strftime("%d-%m-%Y %H:%M")

    for pdf in uploaded_files:
        reader = PdfReader(pdf)
        total_pages += len(reader.pages)

        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text

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
        with st.spinner("Generating Summary..."):
            summary_prompt = f"""
Summarize these notes in simple bullet points.

Notes:
{summary_context}
"""
            summary_response = llm.invoke(summary_prompt)
            st.session_state.summary_result = summary_response.content

    if st.session_state.summary_result:
        st.subheader("📝 Notes Summary")
        st.write(st.session_state.summary_result)

    # ------------------------------------------------------------------
    # Quiz Feature — executes exactly once per button click
    # ------------------------------------------------------------------
    if quiz_button:
        with st.spinner("Generating Quiz..."):
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

    if st.session_state.quiz_result:
        st.subheader("🎯 AI Quiz")
        st.write(st.session_state.quiz_result)

    # ------------------------------------------------------------------
    # Q&A Feature — executes only when a new question is submitted
    # Tracks last question in session_state to avoid duplicate answers.
    # ------------------------------------------------------------------
    if question and question != st.session_state.qa_question:
        st.session_state.qa_question = question
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
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.qa_result = None

    if st.session_state.qa_result and st.session_state.qa_question:
        st.subheader("💡 Answer")
        st.write(st.session_state.qa_result)

else:
    # Friendly prompt shown when no files are uploaded yet
    st.info("👆 Please upload one or more PDF files to get started.")
