# rag-book-chatbot
A Retrieval-Augmented Generation (RAG) chatbot that answers questions from book text files using Hugging Face embeddings, ChromaDB, Python, and Flask.

# RAG Book Chatbot

A web-based Retrieval-Augmented Generation (RAG) chatbot that answers user questions using information retrieved from book text files.

## About the Project

The RAG Book Chatbot allows users to ask questions about the content of `.txt` book files. It retrieves relevant information using semantic similarity search and generates context-based responses. If the requested information is not available in the provided books, the chatbot informs the user that the information is unavailable instead of providing an unsupported answer.

## Technologies Used

- **Programming Language:** Python
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Embedding Model:** Hugging Face Sentence Transformers
- **Vector Database:** ChromaDB
- **Database:** MySQL
- **Knowledge Source:** Text files (`.txt`)

## How It Works

1. **Text Processing:** Reads book text files and splits their content into smaller chunks.
2. **Embedding Generation:** Converts text chunks into numerical embeddings using Hugging Face Sentence Transformers.
3. **Vector Storage:** Stores the embeddings and associated text in ChromaDB.
4. **Semantic Search:** Converts the user's question into an embedding and retrieves relevant text chunks.
5. **Answer Generation:** Uses the retrieved context to provide an answer relevant to the question.
6. **Fallback Response:** If relevant information is unavailable in the book content, the chatbot informs the user and requests a valid, book-related question.
7. **User Management:** Supports user registration, login, and chat history management using MySQL.

## Key Features

- Question answering based on book content
- Semantic search using text embeddings
- Vector storage and retrieval with ChromaDB
- Context-based responses
- Fallback for unavailable information
- User registration and login
- Chat history management
- Web interface built with HTML, CSS, and JavaScript

## Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/rag-book-chatbot.git
   cd rag-book-chatbot
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   Windows:

   ```bash
   venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure the MySQL database and update the database settings according to your local environment.

5. Run the Flask application using the entry-point file in the project:

   ```bash
   python app.py
   ```

6. Open the local URL displayed in the terminal in your browser.

## Usage

1. Register or log in to the application.
2. Ask a question related to the available book text files.
3. The chatbot retrieves relevant information and provides a context-based answer.
4. If the information is unavailable in the books, the chatbot returns a fallback response.

## Learning Outcomes

This project provided practical experience with RAG architecture, text chunking, embedding generation, vector databases, semantic search, Flask backend development, frontend integration, and MySQL database management.

## Future Improvements

- Support for PDF and DOCX documents
- Improved retrieval accuracy
- Source references for generated answers
- Deployment to a cloud platform

## Author

Yogesh A.

**Portfolio:** https://ai-developer-yogesh.vercel.app/
