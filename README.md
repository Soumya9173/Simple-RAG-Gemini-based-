# RAG Notes-Based Study Assistant

## Problem Statement
Students often have to read through long documents, lecture notes, or text files to find specific information or summaries. Reading everything manually is time-consuming. This project solves this problem by providing a Notes-based Study Assistant using a Retrieval-Augmented Generation (RAG) system. Users can upload their text notes, and the system allows them to ask natural language questions about the material. The AI then retrieves the most relevant information and formulates an accurate answer based *only* on the provided notes.

## Tools/Technologies Used
- **Programming Language**: Python
- **GUI Framework**: Tkinter (Standard Python Library)
- **Vector Database**: FAISS (Facebook AI Similarity Search) - `faiss-cpu`
- **AI API**: Google Gemini API (`google-generativeai`)
  - **Embedding Model**: `models/gemini-embedding-001` (Converts text chunks into vectors)
  - **Chat Model**: `models/gemini-2.5-flash` (Generates the final answer based on context)
- **Data Manipulation**: NumPy

## How the RAG System Works
1. **Document Loading**: The system reads a `.txt` file provided by the user.
2. **Chunking**: The text is split into smaller, overlapping chunks (e.g., 500 characters with 100 character overlap) to ensure we capture context without exceeding token limits.
3. **Embedding**: Each chunk is sent to the Gemini API (`gemini-embedding-001`) to generate a vector representation (a numerical array capturing semantic meaning).
4. **Vector Storage**: These vectors are inserted into a FAISS index. FAISS allows for highly efficient similarity searches.
5. **Retrieval**: When a user asks a question, the question is also converted into an embedding. FAISS then finds the "Top K" (e.g., top 3) most similar text chunks to the question's embedding using cosine similarity.
6. **Generation**: The retrieved context chunks and the original question are combined into a prompt. This prompt strictly instructs the Gemini Chat Model (`gemini-1.5-flash`) to answer the question using *only* the provided context. The model then returns a grounded, accurate response.

## Sample Inputs and Outputs

**Input Document (`sample.txt`)**:
```text
Python is a popular programming language.
It is known for simple syntax and readability.

RAG means Retrieval-Augmented Generation.
In a basic RAG pipeline:
1. You load documents.
2. You split them into chunks.
3. You convert chunks to embeddings.
4. You store embeddings in a vector database (like FAISS).
5. For a question, you retrieve relevant chunks.
6. You send retrieved context to an LLM.
7. The LLM generates a grounded answer.
```

**Interaction 1**:
- **Question**: What does RAG stand for?
- **Retrieved Context**: "RAG means Retrieval-Augmented Generation..."
- **System Output**: RAG stands for Retrieval-Augmented Generation.

**Interaction 2**:
- **Question**: What are the steps in a basic pipeline?
- **Retrieved Context**: "In a basic RAG pipeline: 1. You load documents. 2. You split them into chunks..."
- **System Output**: The steps in a basic RAG pipeline are:
  1. Load documents.
  2. Split them into chunks.
  3. Convert chunks to embeddings.
  4. Store embeddings in a vector database.
  5. Retrieve relevant chunks for a question.
  6. Send retrieved context to an LLM.
  7. The LLM generates a grounded answer.

**Interaction 3 (Out of Scope)**:
- **Question**: What is the capital of France?
- **System Output**: I don't know based on the provided document.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Get a Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/app/apikey).
3. Run the GUI application: `python rag_frontend.py`
4. Enter your API Key in the application, select your text file, click "Build Index", and start asking questions!
