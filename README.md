# 🔎 Semantic Search (Human-Friendly Mini RAG)

A mini Retrieval-Augmented Generation (RAG) app that allows you to upload text or CSV, store embeddings in ChromaDB, and ask questions in a **GPT-like, human-friendly way**.

## Features
- Upload text (.txt) or CSV files
- Convert text into embeddings using **sentence-transformers**
- Store embeddings in **ChromaDB** (vector database)
- Ask natural language questions
- Get human-friendly, paragraph-style answers
- Handles unknown questions gracefully: "I’m sorry, I don’t have information about that."

## Tech Stack
- **Python 3**
- **Streamlit** for web UI
- **sentence-transformers** for embeddings
- **ChromaDB** as vector store
- **pandas** for CSV handling

## Installation
1. Clone the repo:

```bash
git clone https://github.com/0samaHaider/semantic-search-human-rag.git
cd semantic-search-human-rag
