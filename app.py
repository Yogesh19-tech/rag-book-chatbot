# ==========================================================
# app.py
# Main Flask application for the Book RAG Chatbot.
#
# Routes:
#   GET  /              → redirect to /login
#   GET  /login         → login page
#   POST /login         → process login
#   GET  /signup        → signup page
#   POST /signup        → process signup
#   GET  /logout        → clear session, redirect to login
#   GET  /chatbot       → chatbot page (login required)
#   POST /upload-book   → upload & index TXT book (login required)
#   POST /ask           → ask a question via RAG (login required)
#   GET  /history       → chat history page (login required)
#
# Security:
#   - Passwords hashed with werkzeug
#   - Flask sessions for auth state
#   - Parameterized SQL queries (in database.py)
#   - Secure filename handling (in book_manager.py)
#   - All credentials loaded from .env
# ==========================================================

import os
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash
)
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# Local modules
import database
import book_manager
import rag_chatbot

# -------------------------------------------------------
# Load environment variables from .env
# -------------------------------------------------------
load_dotenv()

# -------------------------------------------------------
# Flask app initialisation
# -------------------------------------------------------
app = Flask(__name__)

# Secret key is loaded from .env — never hardcoded
app.secret_key = os.getenv("SECRET_KEY", "fallback_secret_change_me")

# Maximum upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ==========================================================
# INITIALISE DATABASE ON STARTUP
# ==========================================================

with app.app_context():
    database.init_db()


# ==========================================================
# HELPER — require login decorator
# ==========================================================

def login_required(f):
    """
    Decorator that redirects unauthenticated users to /login.
    Wrap any protected route with @login_required.
    """
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated


# ==========================================================
# ROOT
# ==========================================================

@app.route("/")
def index():
    """Redirect root URL to login page."""
    return redirect(url_for("login"))


# ==========================================================
# LOGIN
# ==========================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    """
    GET  : show the login form
    POST : validate credentials and start session
    """
    # Already logged in — go to chatbot
    if "user_id" in session:
        return redirect(url_for("chatbot"))

    if request.method == "POST":
        email    = request.form.get("email",    "").strip()
        password = request.form.get("password", "").strip()

        # Basic input validation
        if not email or not password:
            flash("Email and password are required.", "error")
            return render_template("login.html")

        # Look up user in MySQL
        user = database.get_user_by_email(email)

        if user is None:
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        # Verify hashed password
        if not check_password_hash(user["password"], password):
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        # Store user info in Flask session
        session["user_id"]   = user["id"]
        session["user_name"] = user["name"]
        session["user_email"] = user["email"]

        return redirect(url_for("chatbot"))

    return render_template("login.html")


# ==========================================================
# SIGNUP
# ==========================================================

@app.route("/signup", methods=["GET", "POST"])
def signup():
    """
    GET  : show the signup form
    POST : validate input, hash password, store user in MySQL
    """
    if "user_id" in session:
        return redirect(url_for("chatbot"))

    if request.method == "POST":
        name             = request.form.get("name",             "").strip()
        email            = request.form.get("email",            "").strip()
        password         = request.form.get("password",         "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        # --- Validation ---
        if not name or not email or not password or not confirm_password:
            flash("All fields are required.", "error")
            return render_template("signup.html")

        if len(name) < 2:
            flash("Name must be at least 2 characters.", "error")
            return render_template("signup.html")

        if "@" not in email or "." not in email.split("@")[-1]:
            flash("Please enter a valid email address.", "error")
            return render_template("signup.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("signup.html")

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("signup.html")

        # Hash the password — NEVER store plain text
        hashed = generate_password_hash(password)

        # Insert user into MySQL
        success = database.insert_user(name, email, hashed)

        if not success:
            flash("An account with this email already exists.", "error")
            return render_template("signup.html")

        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


# ==========================================================
# LOGOUT
# ==========================================================

@app.route("/logout")
def logout():
    """
    Clear the Flask session and redirect to login.
    Available via the three-dot menu on the chatbot page.
    """
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ==========================================================
# CHATBOT
# ==========================================================

@app.route("/chatbot", methods=["GET"])
@login_required
def chatbot():
    """
    Main chatbot page.
    Shows the chat interface, upload section, and conversation history
    for the current session.
    """
    # Fetch this user's recent chat messages for display on the page
    user_id  = session["user_id"]
    messages = database.get_chat_history(user_id)

    # Reverse so oldest messages appear at the top
    messages = list(reversed(messages))

    # Check if a book has been indexed already
    book_indexed = rag_chatbot.book_is_indexed()

    # Get list of uploaded books for display
    uploaded_books = book_manager.list_uploaded_books()
    current_book   = uploaded_books[0] if uploaded_books else None

    return render_template(
        "chatbot.html",
        user_name    = session.get("user_name", "User"),
        messages     = messages,
        book_indexed = book_indexed,
        current_book = current_book
    )


# ==========================================================
# UPLOAD BOOK
# ==========================================================

@app.route("/upload-book", methods=["POST"])
@login_required
def upload_book():
    """
    Handle TXT book file upload.

    Steps:
    1. Validate file extension (only .txt allowed)
    2. Save file securely to book_uploads/
    3. Read and clean the text
    4. Index into ChromaDB via RAG pipeline
    5. Redirect back to chatbot with status message
    """
    file = request.files.get("book_file")

    if file is None or file.filename == "":
        flash("No file selected. Please choose a TXT file to upload.", "error")
        return redirect(url_for("chatbot"))

    # Step 1 & 2: Validate + save (book_manager handles extension check)
    success, result = book_manager.save_book_file(file)

    if not success:
        # result is the error message here
        flash(result, "error")
        return redirect(url_for("chatbot"))

    # result is the saved filepath here
    saved_path = result

    # Step 3: Read and clean text
    ok, text_or_error = book_manager.get_book_text(saved_path)

    if not ok:
        flash(f"Could not read the book: {text_or_error}", "error")
        return redirect(url_for("chatbot"))

    # Step 4: Index into ChromaDB
    indexed_ok, info = rag_chatbot.process_and_index_book(text_or_error)

    if not indexed_ok:
        flash(f"Failed to process book: {info}", "error")
        return redirect(url_for("chatbot"))

    flash(
        f"Book uploaded and indexed successfully! "
        f"{info} text chunks are ready. You can now ask questions.",
        "success"
    )
    return redirect(url_for("chatbot"))


# ==========================================================
# ASK QUESTION
# ==========================================================

@app.route("/ask", methods=["POST"])
@login_required
def ask():
    """
    Handle a user question through the RAG pipeline.

    Steps:
    1. Validate question input
    2. Check a book is indexed
    3. Run RAG pipeline → get answer
    4. Save question + answer to MySQL chat_history
    5. Redirect back to chatbot page
    """
    question = request.form.get("question", "").strip()
    user_id  = session["user_id"]

    if not question:
        flash("Please enter a question before clicking Send.", "error")
        return redirect(url_for("chatbot"))

    if len(question) > 1000:
        flash("Question is too long. Please keep it under 1000 characters.", "error")
        return redirect(url_for("chatbot"))

    # Check book is indexed
    if not rag_chatbot.book_is_indexed():
        flash("No book uploaded yet. Please upload a TXT book first.", "error")
        return redirect(url_for("chatbot"))

    # Run RAG pipeline
    try:
        answer = rag_chatbot.answer_question(question)
    except Exception as e:
        answer = f"An error occurred while generating the answer: {e}"

    # Save to MySQL — every question/answer is persisted per user
    database.save_chat(user_id, question, answer)

    return redirect(url_for("chatbot"))


# ==========================================================
# CHAT HISTORY PAGE
# ==========================================================

@app.route("/history")
@login_required
def history():
    """
    Display the full chat history for the logged-in user.
    Fetches from MySQL — users only see their own conversations.
    """
    user_id  = session["user_id"]
    messages = database.get_chat_history(user_id)

    return render_template(
        "history.html",
        user_name = session.get("user_name", "User"),
        messages  = messages
    )


# ==========================================================
# ERROR HANDLERS
# ==========================================================

@app.errorhandler(404)
def not_found(e):
    return render_template("login.html"), 404


@app.errorhandler(413)
def file_too_large(e):
    flash("File is too large. Maximum upload size is 10 MB.", "error")
    return redirect(url_for("chatbot"))


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
