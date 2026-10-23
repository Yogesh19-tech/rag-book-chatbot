# AI Book Chatbot — RAG-Powered Local Chatbot

A complete web application that lets you upload any `.txt` book and chat with it using AI.
Built with Flask, ChromaDB, Hugging Face embeddings, and the Qwen2.5-1.5B-Instruct LLM running fully locally on your laptop.

---

## Features

- Upload any `.txt` book and ask questions about it
- Answers come **only** from the uploaded book — no hallucinations or outside knowledge
- Semantic RAG pipeline: Hugging Face embeddings + ChromaDB vector search
- Lightweight local LLM: Qwen2.5-1.5B-Instruct (works on 8 GB RAM)
- User signup and login with hashed passwords
- Per-user chat history stored in MySQL
- Professional UI with light/dark mode (CSS only, no JavaScript)
- CSS-only three-dot menu (no JavaScript)
- Fully offline after first model download

---

## Technology Stack

| Layer         | Technology                              |
|---------------|------------------------------------------|
| Backend       | Python, Flask                            |
| Frontend      | HTML, CSS (no JavaScript)               |
| Database      | MySQL                                    |
| Embeddings    | sentence-transformers/all-MiniLM-L6-v2  |
| Vector DB     | ChromaDB (local, persistent)            |
| LLM           | Qwen/Qwen2.5-1.5B-Instruct              |
| Auth          | Flask sessions, werkzeug password hash  |
| Env config    | python-dotenv                           |

---

## LLM: Qwen2.5-1.5B-Instruct

**Model:** `Qwen/Qwen2.5-1.5B-Instruct`

**Why this model?**
- Target hardware has only 8 GB RAM
- 1.5B parameters — loads in ~2–3 GB RAM in float16
- Strong instruction-following quality for its size
- Runs fully locally — no API key, no internet needed after download
- Qwen2.5 series has excellent reading comprehension for RAG tasks

The model is downloaded automatically from Hugging Face on first run (~3 GB).

---

## Embedding Model

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

- Fast, lightweight 384-dimensional dense embeddings
- Optimized for semantic similarity search
- Downloaded automatically from Hugging Face on first run (~90 MB)
- Used to convert book chunks and user questions into vectors for similarity search

---

## RAG Architecture

```
TXT Book
   ↓
Load Text (book_manager.py)
   ↓
Clean Text (normalize, strip, remove non-printable)
   ↓
Semantic Chunking (~500 chars, 50 char overlap)
   ↓
Hugging Face Embeddings (all-MiniLM-L6-v2)
   ↓
Store in ChromaDB (local vector database)
   ↓
User Question
   ↓
Question Embedding
   ↓
Cosine Similarity Search in ChromaDB
   ↓
Relevance Threshold Check (cosine distance < 0.75)
   ↓
Retrieve Top-3 Relevant Book Chunks
   ↓
Build Strict RAG Prompt (context + question)
   ↓
Qwen2.5-1.5B-Instruct
   ↓
Answer ONLY from Book
   ↓
Save to MySQL chat_history
```

---

## Chunking Strategy

- **Method:** Sentence-boundary-aware text chunking
- **Chunk size:** ~500 characters
- **Overlap:** ~50 characters between consecutive chunks
- **Why these values:** Small enough for 8 GB RAM; large enough to preserve sentence context. The overlap prevents information loss at chunk boundaries.
- Book content is re-indexed fresh each time a new book is uploaded.

---

## Relevance Threshold

ChromaDB returns cosine distances (0 = identical, 1 = completely different).

If all retrieved chunks have a distance > 0.75, the question is considered out-of-scope and the chatbot responds:

```
There is no information about this in the uploaded book.
```

This prevents the LLM from hallucinating answers to questions not covered by the book.

---

## Project Structure

```
book_rag_chatbot/
│
├── app.py                  ← Flask app, all routes
├── database.py             ← MySQL connection, tables, queries
├── rag_chatbot.py          ← Full RAG pipeline (embeddings, ChromaDB, LLM)
├── book_manager.py         ← TXT file upload, validation, text cleaning
│
├── requirements.txt        ← Python dependencies
├── .env                    ← Your local credentials (not committed)
├── .env.example            ← Template for .env
├── .gitignore
├── README.md
│
├── book_uploads/           ← Uploaded TXT files stored here
│   └── .gitkeep
│
├── chroma_db/              ← ChromaDB vector database (auto-created)
│
├── templates/
│   ├── login.html
│   ├── signup.html
│   ├── chatbot.html
│   └── history.html
│
└── static/
    └── css/
        └── style.css
```

---

## Installation — Step by Step

### Prerequisites

- Python 3.10 or 3.11 (recommended)
- MySQL Server installed and running
- Git (optional)

---

### Step 1 — Clone or copy the project

```
Place the book_rag_chatbot folder on your Desktop or any location.
```

---

### Step 2 — Open terminal in project folder

On Windows (PowerShell):
```powershell
cd "C:\Users\YourName\Desktop\book_rag_chatbot"
```

---

### Step 3 — Create virtual environment

```powershell
python -m venv venv
```

---

### Step 4 — Activate virtual environment

```powershell
venv\Scripts\activate
```

You should see `(venv)` at the start of your prompt.

---

### Step 5 — Install dependencies

```powershell
pip install -r requirements.txt
```

This installs Flask, MySQL connector, ChromaDB, sentence-transformers, transformers, torch, and all other required packages.

Note: PyTorch download is large (~2 GB). The first install will take time depending on your internet speed.

---

### Step 6 — Install PyTorch (if not installed by requirements)

If torch is not installed correctly, install it manually:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For CPU-only inference (recommended for 8 GB RAM laptops without a dedicated GPU).

---

## MySQL Setup

### Step 1 — Start MySQL

Make sure MySQL Server is running on your machine.

---

### Step 2 — Create the database

Open MySQL Workbench or the MySQL command line and run:

```sql
CREATE DATABASE book_rag_chatbot;
```

---

### Step 3 — Note your credentials

You need:
- MySQL host (usually `localhost`)
- MySQL username (usually `root`)
- MySQL password (whatever you set during MySQL installation)
- Database name: `book_rag_chatbot`

The application will automatically create the `users` and `chat_history` tables when it first starts.

---

## Configure Environment Variables

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and fill in your actual values:

```
SECRET_KEY=paste_a_long_random_string_here
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_actual_mysql_password
MYSQL_DATABASE=book_rag_chatbot
```

To generate a strong secret key in Python:

```python
import secrets
print(secrets.token_hex(32))
```

---

## Run the Application

```powershell
python app.py
```

Then open your browser and go to:

```
http://localhost:5000
```

On first run, the application will:
1. Connect to MySQL and create tables automatically
2. Be ready for signup and login

When you first upload a book and ask a question, it will download the AI models (one-time, ~3 GB total). Subsequent runs will use the cached models.

---

## How to Use

1. Open `http://localhost:5000`
2. Click **Create Account** and sign up
3. Log in with your email and password
4. On the chatbot page, click **Choose File** and select a `.txt` book
5. Click **Upload & Index Book** — wait for indexing to complete
6. Type your question in the input box and click **Send**
7. The AI will answer based only on the uploaded book
8. Click **⋮** (three dots) → **Chat History** to view past conversations
9. Click **⋮** → **Logout** to sign out

---

## Supported File Format

| Format  | Supported |
|---------|-----------|
| `.txt`  | ✅ Yes    |
| `.pdf`  | ❌ No     |
| `.docx` | ❌ No     |
| `.jpg`  | ❌ No     |
| `.png`  | ❌ No     |

Only plain text `.txt` files are accepted. No OCR is implemented.

---

## Setting Up on Another Laptop

1. Copy the entire `book_rag_chatbot` folder to the new laptop
2. Install Python 3.10 or 3.11
3. Install MySQL Server
4. Create the `book_rag_chatbot` database in MySQL
5. Follow the installation steps above (venv → pip install → configure .env)
6. Run `python app.py`
7. The AI models will download automatically on first use

The `chroma_db/` folder and `book_uploads/` folder do not need to be copied — they are created fresh.

---

## MySQL Tables

### users
| Column     | Type         | Description                    |
|------------|--------------|--------------------------------|
| id         | INT PK AUTO  | User ID                        |
| name       | VARCHAR(150) | Display name                   |
| email      | VARCHAR(255) | Email (unique)                 |
| password   | VARCHAR(512) | Werkzeug hashed password       |
| created_at | DATETIME     | Registration timestamp         |

### chat_history
| Column     | Type         | Description                     |
|------------|--------------|----------------------------------|
| id         | INT PK AUTO  | Record ID                        |
| user_id    | INT FK       | References users.id              |
| question   | TEXT         | User's question                  |
| answer     | TEXT         | AI-generated answer              |
| created_at | DATETIME     | Conversation timestamp           |

---

## Limitations

- Only `.txt` files are supported — no PDF or image-based books
- First run requires internet to download models (~3 GB)
- Answer generation speed depends on your CPU — expect 15–60 seconds per answer on 8 GB RAM without GPU
- Very large books (50,000+ words) may take 1–2 minutes to index on first upload
- The chatbot only knows what is in the currently uploaded book — uploading a new book replaces the old index

---

## Troubleshooting

**MySQL connection error:**
- Check that MySQL is running
- Verify your `.env` credentials are correct
- Make sure the `book_rag_chatbot` database exists

**Model download stuck:**
- Check internet connection
- Hugging Face models are cached at `C:\Users\YourName\.cache\huggingface\`
- If download was interrupted, delete the partial cache and restart

**Out of memory error:**
- Close other applications to free RAM
- The model requires ~2–3 GB RAM in float16 mode
- Make sure no other large applications are running

**ChromaDB error on startup:**
- Delete the `chroma_db/` folder and restart — it will be recreated
- This clears the book index; re-upload your book after restart

**Slow response:**
- This is normal for CPU-only inference with a 1.5B model
- First response after startup is slower (model loading)
- Subsequent responses are faster

**Port 5000 already in use:**
```powershell
# Change port in app.py last line:
app.run(debug=True, host="0.0.0.0", port=5001)
```

---

## Security Notes

- Passwords are hashed using `werkzeug.security.generate_password_hash` — never stored as plain text
- All SQL queries use parameterized statements — protected against SQL injection
- Flask `SECRET_KEY` is loaded from `.env` — never hardcoded
- The `.env` file is in `.gitignore` — credentials are never committed to git
- Uploaded filenames are sanitized with `werkzeug.utils.secure_filename`
- Users can only see their own chat history
