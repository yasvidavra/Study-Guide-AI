# 🎓 AI Academic Assistant

> **An intelligent study companion powered by Retrieval-Augmented Generation (RAG), LLMs, and vector search.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.1-orange)](https://groq.com/)
[![FAISS](https://img.shields.io/badge/FAISS-VectorDB-purple)](https://faiss.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 Project Overview

The **AI Academic Assistant** is a Streamlit-based web application that allows students to upload their PDF study notes and interact with them using artificial intelligence. Using a **Retrieval-Augmented Generation (RAG)** pipeline, the application extracts knowledge from uploaded PDFs, stores it in a FAISS vector database, and enables:

- Intelligent **question answering** grounded in the notes
- Automatic **summary generation** in bullet-point format
- AI-generated **multiple-choice quiz** questions for self-testing
- Persistent **SQLite storage** of all uploaded PDF metadata

This project demonstrates end-to-end integration of modern AI tools including **HuggingFace embeddings**, **FAISS vector search**, and **Groq's LLaMA 3.1** large language model.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📂 **PDF Upload** | Upload one or multiple PDF files simultaneously |
| 🔤 **Text Extraction** | Automated extraction of all text content from PDFs |
| 🧩 **Text Chunking** | Splits large documents into overlapping chunks for accurate retrieval |
| 🤖 **HuggingFace Embeddings** | `all-MiniLM-L6-v2` model converts chunks to semantic vectors |
| ⚡ **FAISS Vector Search** | Lightning-fast similarity search across embedded document chunks |
| 📝 **AI Summary** | Generates structured bullet-point summaries of uploaded notes |
| 🎯 **AI Quiz Generator** | Creates 5 MCQ questions with correct answers from the notes |
| 💬 **Question Answering** | Ask any question — the AI answers strictly from your notes |
| 🗄️ **SQLite Database** | Saves PDF metadata (filename, pages, upload date, summary) persistently |
| 📊 **View Saved Notes** | Browse all previously uploaded PDFs in a formatted DataFrame table |

---

## 🛠️ Technologies Used

### Backend & AI
| Technology | Purpose |
|---|---|
| [LangChain](https://www.langchain.com/) | RAG pipeline orchestration |
| [Groq API (LLaMA 3.1 8B)](https://groq.com/) | Large Language Model for generation |
| [HuggingFace Sentence Transformers](https://huggingface.co/) | Text embedding model |
| [FAISS](https://faiss.ai/) | High-performance vector similarity search |
| [PyPDF](https://pypdf.readthedocs.io/) | PDF text extraction |
| [SQLite3](https://www.sqlite.org/) | Lightweight embedded relational database |

### Frontend & UI
| Technology | Purpose |
|---|---|
| [Streamlit](https://streamlit.io/) | Interactive web application framework |
| [Pandas](https://pandas.pydata.org/) | Tabular data display for saved notes |

---

## 📁 Project Structure

```
Agentic_AI_Project/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .gitignore                # Files excluded from version control
├── .env                      # API keys (NOT committed to Git)
├── README.md                 # Project documentation
│
├── academic_assistant.db     # SQLite database (auto-created, NOT committed)
│
├── data/                     # Sample or reference data
├── notebooks/                # Jupyter notebooks for experiments
├── report/                   # Assessment report documents
└── screenshots/              # Application screenshots for documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- A free [Groq API key](https://console.groq.com/)
- Git installed

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-academic-assistant.git
cd ai-academic-assistant
```

### Step 2 — Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Configure API Keys
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

> ⚠️ **Never commit your `.env` file.** It is included in `.gitignore`.

### Step 5 — Run the Application
```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

---

## 🖼️ Screenshots

> *Add screenshots to the `screenshots/` folder and update the paths below.*

| Upload & Process | Summary | Quiz | Q&A |
|---|---|---|---|
| ![Upload](screenshots/upload.png) | ![Summary](screenshots/summary.png) | ![Quiz](screenshots/quiz.png) | ![QA](screenshots/qa.png) |

---

## 🚀 How to Use

1. **Launch** the app with `streamlit run app.py`
2. **Upload** one or more PDF files using the file uploader
3. Wait for the FAISS vector database to be built (shown in status messages)
4. Use the **AI Study Tools**:
   - Click **📝 Generate Summary** for bullet-point notes
   - Click **🎯 Generate Quiz** for 5 MCQ practice questions
   - Type a question in the **💬 Ask a Question** box for instant answers
5. Click **📚 View Saved Notes** to browse your upload history

---

## 🔮 Future Enhancements

- [ ] **Multi-language Support** — Answer questions in different languages
- [ ] **Flashcard Generator** — Auto-create study flashcards from notes
- [ ] **Progress Tracker** — Track quiz scores and study sessions over time
- [ ] **PDF Annotation Export** — Export AI-generated summaries back to PDF
- [ ] **Voice Input** — Ask questions using microphone via speech-to-text
- [ ] **Topic Clustering** — Auto-group uploaded PDFs by subject/topic
- [ ] **Difficulty Levels** — Easy / Medium / Hard quiz modes

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👩‍💻 Author

**Kavya Vaghela**
- 📧 Academic Assessment Project — AI/ML Module
- 🔗 [GitHub](https://github.com/KavyaVaghela)

---

> *"Study smarter, not harder — let AI do the heavy lifting."* 🧠
