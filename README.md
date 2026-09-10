# Chatbot-with-RAG
A Streamlit-based chatbot implementing Retrieval-Augmented Generation (RAG) over local PDF documents using LangChain and Groq.

## Overview

The application ingests a local PDF, splits the text into semantic chunks, generates vector embeddings using a local HuggingFace model, and indexes them into an in-memory vector store. When a user submits a query, relevant document contexts are retrieved and supplied alongside the prompt to an LLM hosted on Groq for grounded responses.

## Architecture

* **UI Layer:** Streamlit
* **Document Ingestion:** PyPDFLoader
* **Text Chunking:** RecursiveCharacterTextSplitter (chunk size: 1000, overlap: 100)
* **Embeddings:** HuggingFace `all-MiniLM-L12-v2` via `sentence-transformers`
* **Vector Store:** LangChain `InMemoryVectorStore`
* **Inference:** Groq API (`openai/gpt-oss-20b`)

## Setup Instructions

### Prerequisites

* Python 3.11
* Groq API Key

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/WaryamMangi/Chatbot-with-RAG.git](https://github.com/WaryamMangi/Chatbot-with-RAG.git)
   cd Chatbot-with-RAG
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
3. Install required packages:
   ```bash
   pip install streamlit langchain-groq langchain-community langchain-text-splitters pypdf sentence-transformers python-dotenv
5. Configure environment variables:
   Create a .env file in the project root:
   ```bash
   GROQ_API_KEY="your_groq_api_key_here"
6. Add document:
   Place the target document in the root directory and name it reflexion.pdf.

### Execution
Run the application:
streamlit run phase_3.py
