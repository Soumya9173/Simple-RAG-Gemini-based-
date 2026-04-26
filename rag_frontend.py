"""
Simple RAG Frontend (Tkinter)

What this does:
- Pick a .txt file
- Build FAISS index from file chunks
- Ask questions
- Show Gemini answer

Run:
    python rag_frontend.py

Requirements:
- A Gemini API Key
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

from rag_simple import ask_llm
from rag_simple import build_faiss_index
from rag_simple import chunk_text
from rag_simple import embed_texts
from rag_simple import load_text
from rag_simple import retrieve_chunks
from rag_simple import setup_gemini


class RagApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Simple RAG Frontend")
        self.root.geometry("800x620")

        # Stored state after indexing
        self.chunks: list[str] = []
        self.index = None

        self._build_ui()

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=12, pady=10)

        tk.Label(top, text="Text file:").pack(side="left")

        self.file_var = tk.StringVar()
        file_entry = tk.Entry(top, textvariable=self.file_var, width=70)
        file_entry.pack(side="left", padx=8)

        tk.Button(top, text="Browse", command=self.pick_file).pack(side="left", padx=4)
        tk.Button(top, text="Build Index", command=self.build_index).pack(side="left", padx=4)

        self.status_var = tk.StringVar(value="Select a text file, then click Build Index.")
        status_label = tk.Label(self.root, textvariable=self.status_var, anchor="w")
        status_label.pack(fill="x", padx=12, pady=(0, 8))

        question_frame = tk.Frame(self.root)
        question_frame.pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(question_frame, text="Question:").pack(side="left")
        self.question_var = tk.StringVar()
        question_entry = tk.Entry(question_frame, textvariable=self.question_var, width=70)
        question_entry.pack(side="left", padx=8)
        tk.Button(question_frame, text="Ask", command=self.ask_question).pack(side="left")

        tk.Label(self.root, text="Answer:").pack(anchor="w", padx=12)
        self.answer_box = scrolledtext.ScrolledText(self.root, wrap="word", height=10)
        self.answer_box.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        tk.Label(self.root, text="Retrieved Chunks (for debugging):").pack(anchor="w", padx=12)
        self.context_box = scrolledtext.ScrolledText(self.root, wrap="word", height=10)
        self.context_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def pick_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select text file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self.file_var.set(path)

    def build_index(self) -> None:
        file_path = self.file_var.get().strip()
        if not file_path:
            messagebox.showerror("Error", "Please select a text file.")
            return
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"File not found:\n{file_path}")
            return

        self.status_var.set("Loading file and building index...")
        self.root.update_idletasks()

        try:
            setup_gemini()
            text = load_text(file_path)
            self.chunks = chunk_text(text)
            if not self.chunks:
                raise ValueError("No chunks created from the file.")

            vectors = embed_texts(self.chunks)
            self.index = build_faiss_index(vectors)
        except Exception as exc:
            messagebox.showerror(
                "Index Build Error",
                f"Could not build index.\n\nDetails: {exc}\n\n"
                "Check that your Gemini API Key is valid and you have an internet connection.",
            )
            self.status_var.set("Index build failed.")
            return

        self.status_var.set(f"Index ready. Loaded {len(self.chunks)} chunks.")

    def ask_question(self) -> None:
        if self.index is None or not self.chunks:
            messagebox.showwarning("Warning", "Build the index first.")
            return

        question = self.question_var.get().strip()
        if not question:
            messagebox.showwarning("Warning", "Please type a question.")
            return

        self.status_var.set("Retrieving context and generating answer...")
        self.root.update_idletasks()

        try:
            setup_gemini()
            context_chunks = retrieve_chunks(self.index, self.chunks, question)
            answer = ask_llm(question, context_chunks)
        except Exception as exc:
            messagebox.showerror(
                "RAG Error",
                f"Could not answer question.\n\nDetails: {exc}\n\n"
                "Check that your Gemini API Key is valid and you have an internet connection.",
            )
            self.status_var.set("Question failed.")
            return

        self.answer_box.delete("1.0", tk.END)
        self.answer_box.insert(tk.END, answer)

        self.context_box.delete("1.0", tk.END)
        for i, chunk in enumerate(context_chunks, start=1):
            self.context_box.insert(tk.END, f"Chunk {i}:\n{chunk}\n\n{'-' * 50}\n\n")

        self.status_var.set("Done.")


def main() -> None:
    root = tk.Tk()
    app = RagApp(root)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()