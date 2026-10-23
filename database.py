# ==========================================================
# database.py
# Handles all MySQL database operations for the Book RAG Chatbot.
# Responsibilities:
#   - MySQL connection management
#   - Table creation (users, chat_history)
#   - User insert and lookup
#   - Chat history save and retrieve
# ==========================================================

import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_connection():
    """
    Create and return a MySQL database connection.
    Credentials are loaded from environment variables — never hardcoded.
    """
    try:
        connection = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", "root"),
            database=os.getenv("MYSQL_DATABASE", "book_rag_chatbot")
        )
        return connection
    except Error as e:
        print(f"[DATABASE ERROR] Could not connect to MySQL: {e}")
        return None


def init_db():
    """
    Initialize the database by creating required tables if they do not exist.
    Called once when the Flask app starts.

    Tables created:
      - users        : stores registered user accounts
      - chat_history : stores each question/answer pair per user
    """
    connection = get_connection()
    if connection is None:
        print("[DATABASE ERROR] Cannot initialize DB — no connection.")
        return

    cursor = connection.cursor()

    # -------------------------------------------------------
    # users table
    # Stores all registered user accounts.
    # Passwords are stored as bcrypt/werkzeug hashes — never plain text.
    # -------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INT AUTO_INCREMENT PRIMARY KEY,
            name       VARCHAR(150)  NOT NULL,
            email      VARCHAR(255)  NOT NULL UNIQUE,
            password   VARCHAR(512)  NOT NULL,
            created_at DATETIME      DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # -------------------------------------------------------
    # chat_history table
    # Stores every question/answer exchange for every user.
    # Foreign key: chat_history.user_id → users.id
    # Each user can only see their own history.
    # -------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id         INT AUTO_INCREMENT PRIMARY KEY,
            user_id    INT           NOT NULL,
            question   TEXT          NOT NULL,
            answer     TEXT          NOT NULL,
            created_at DATETIME      DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    connection.commit()
    cursor.close()
    connection.close()
    print("[DATABASE] Tables initialized successfully.")


# ==========================================================
# USER FUNCTIONS
# ==========================================================

def insert_user(name, email, hashed_password):
    """
    Insert a new user into the users table.

    Args:
        name            : user's display name
        email           : user's email (must be unique)
        hashed_password : werkzeug-hashed password string

    Returns:
        True  if insertion succeeded
        False if email already exists or a DB error occurred
    """
    connection = get_connection()
    if connection is None:
        return False

    try:
        cursor = connection.cursor()
        # Parameterized query — prevents SQL injection
        cursor.execute(
            "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
            (name, email, hashed_password)
        )
        connection.commit()
        cursor.close()
        connection.close()
        return True
    except mysql.connector.IntegrityError:
        # Duplicate email — UNIQUE constraint violation
        connection.close()
        return False
    except Error as e:
        print(f"[DATABASE ERROR] insert_user: {e}")
        connection.close()
        return False


def get_user_by_email(email):
    """
    Fetch a user record by email address.

    Args:
        email : the email to look up

    Returns:
        A dict with keys {id, name, email, password, created_at}
        or None if not found / DB error
    """
    connection = get_connection()
    if connection is None:
        return None

    try:
        cursor = connection.cursor(dictionary=True)
        # Parameterized query — prevents SQL injection
        cursor.execute(
            "SELECT id, name, email, password, created_at FROM users WHERE email = %s",
            (email,)
        )
        user = cursor.fetchone()
        cursor.close()
        connection.close()
        return user
    except Error as e:
        print(f"[DATABASE ERROR] get_user_by_email: {e}")
        connection.close()
        return None


def get_user_by_id(user_id):
    """
    Fetch a user record by user ID.

    Args:
        user_id : integer primary key

    Returns:
        A dict with keys {id, name, email, created_at}
        or None if not found / DB error
    """
    connection = get_connection()
    if connection is None:
        return None

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, name, email, created_at FROM users WHERE id = %s",
            (user_id,)
        )
        user = cursor.fetchone()
        cursor.close()
        connection.close()
        return user
    except Error as e:
        print(f"[DATABASE ERROR] get_user_by_id: {e}")
        connection.close()
        return None


# ==========================================================
# CHAT HISTORY FUNCTIONS
# ==========================================================

def save_chat(user_id, question, answer):
    """
    Save a question/answer pair to the chat_history table.

    Args:
        user_id  : the logged-in user's ID
        question : the user's question string
        answer   : the AI-generated answer string

    Returns:
        True on success, False on failure
    """
    connection = get_connection()
    if connection is None:
        return False

    try:
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO chat_history (user_id, question, answer) VALUES (%s, %s, %s)",
            (user_id, question, answer)
        )
        connection.commit()
        cursor.close()
        connection.close()
        return True
    except Error as e:
        print(f"[DATABASE ERROR] save_chat: {e}")
        connection.close()
        return False


def get_chat_history(user_id):
    """
    Retrieve all chat history for a specific user, newest first.

    Args:
        user_id : the logged-in user's ID

    Returns:
        A list of dicts: [{id, user_id, question, answer, created_at}, ...]
        Empty list on failure or no records.
    """
    connection = get_connection()
    if connection is None:
        return []

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, user_id, question, answer, created_at
            FROM chat_history
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,)
        )
        history = cursor.fetchall()
        cursor.close()
        connection.close()
        return history
    except Error as e:
        print(f"[DATABASE ERROR] get_chat_history: {e}")
        connection.close()
        return []
