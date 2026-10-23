# ==========================================================
# book_manager.py
# Handles all TXT book upload and processing logic.
#
# Responsibilities:
#   - File extension validation (only .txt allowed)
#   - Secure file saving to book_uploads/
#   - Reading TXT file content
#   - Text cleaning and normalization
#   - Preparing clean text for the RAG pipeline
#
# IMPORTANT:
#   Only .txt files are accepted.
#   PDF, images, DOCX, and all other formats are rejected.
#   No OCR is implemented.
# ==========================================================

import os
import re
from werkzeug.utils import secure_filename

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

# Folder where uploaded TXT books are saved
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "book_uploads")

# Only plain text files are allowed
ALLOWED_EXTENSIONS = {"txt"}

# Maximum upload file size: 10 MB (reasonable for text files)
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def allowed_file(filename):
    """
    Check if the uploaded file has a .txt extension.

    Args:
        filename : original filename from the upload

    Returns:
        True  if filename ends with .txt
        False otherwise (pdf, jpg, png, docx, etc. all rejected)
    """
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def save_book_file(file_storage):
    """
    Validate and save an uploaded TXT book file.

    Args:
        file_storage : Flask FileStorage object from request.files

    Returns:
        (True, saved_filepath)  on success
        (False, error_message)  on failure
    """
    if file_storage is None or file_storage.filename == "":
        return False, "No file selected."

    original_name = file_storage.filename

    # Reject non-.txt files immediately
    if not allowed_file(original_name):
        return (
            False,
            "Only TXT book files are supported. "
            "Image-based books are not supported."
        )

    # Sanitize filename to prevent directory traversal attacks
    safe_name = secure_filename(original_name)
    if not safe_name:
        safe_name = "uploaded_book.txt"

    # Ensure upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    save_path = os.path.join(UPLOAD_FOLDER, safe_name)

    # Read file content to validate it is non-empty and check size
    file_storage.seek(0, 2)          # Seek to end
    file_size = file_storage.tell()  # Get file size
    file_storage.seek(0)             # Reset to start

    if file_size == 0:
        return False, "The uploaded TXT file is empty."

    if file_size > MAX_FILE_SIZE_BYTES:
        return False, f"File is too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB."

    # Save the file
    file_storage.save(save_path)
    return True, save_path


def read_and_clean_text(filepath):
    """
    Read a TXT file and return cleaned plain text.

    Cleaning steps:
      1. Decode UTF-8 with fallback to latin-1
      2. Normalize line endings
      3. Remove excessive blank lines (keep max 2 consecutive)
      4. Strip leading/trailing whitespace per line
      5. Remove non-printable characters

    Args:
        filepath : absolute path to the saved TXT file

    Returns:
        (True, cleaned_text_string)  on success
        (False, error_message)       on failure
    """
    if not os.path.exists(filepath):
        return False, "Book file not found."

    # Attempt UTF-8 first, fall back to latin-1
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, "r", encoding="latin-1") as f:
                raw_text = f.read()
        except Exception as e:
            return False, f"Could not read the file: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"

    if not raw_text.strip():
        return False, "The TXT file appears to be empty after reading."

    cleaned = _clean_text(raw_text)

    if len(cleaned.strip()) < 50:
        return False, "The TXT file contains too little readable text."

    return True, cleaned


def _clean_text(text):
    """
    Internal text cleaning function.

    Args:
        text : raw string read from the TXT file

    Returns:
        Cleaned string ready for semantic chunking
    """
    # Normalize Windows/Mac line endings to Unix
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove non-printable characters except newline and tab
    text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", " ", text)

    # Strip each line
    lines = [line.strip() for line in text.split("\n")]

    # Collapse sequences of more than 2 blank lines into 2 blank lines
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append("")
        else:
            blank_count = 0
            cleaned_lines.append(line)

    # Collapse multiple spaces within lines
    result_lines = []
    for line in cleaned_lines:
        line = re.sub(r" {2,}", " ", line)
        result_lines.append(line)

    return "\n".join(result_lines).strip()


def get_book_text(filepath):
    """
    Convenience wrapper: read and return cleaned text from a saved TXT file.
    Used by the RAG pipeline after a book is uploaded.

    Args:
        filepath : absolute path to the TXT file

    Returns:
        (True, cleaned_text)   on success
        (False, error_message) on failure
    """
    return read_and_clean_text(filepath)


def list_uploaded_books():
    """
    Return a list of all .txt filenames currently in the upload folder.
    Useful for showing the user what book is currently loaded.

    Returns:
        List of filename strings, e.g. ["book.txt", "novel.txt"]
    """
    if not os.path.exists(UPLOAD_FOLDER):
        return []
    return [
        f for f in os.listdir(UPLOAD_FOLDER)
        if f.lower().endswith(".txt")
    ]
