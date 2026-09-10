import os
import warnings
import logging
import streamlit as st

# Load the environment variables
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

# Disable warnings and info logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title('Ask Chatbot!')

# Setup a session state variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

@st.cache_resource
def get_vectorstore():
    pdf_name = "./reflexion.pdf"
    
    # 1. Load the PDF
    loader = PyPDFLoader(pdf_name)
    docs = loader.load()
    
    # 2. Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # 3. Embed and store the chunks in memory
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')
    vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

prompt = st.chat_input('Pass your prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content': prompt})
    
    # Using the guaranteed active model from your Phase 2 file
    model="openai/gpt-oss-20b"

    groq_chat = ChatGroq(
            groq_api_key=os.environ.get("GROQ_API_KEY"), 
            model_name=model
    )

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load document")
            st.stop()
      
        # --- PURE RAG LOGIC ---
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
        relevant_docs = retriever.invoke(prompt)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        augmented_prompt = f"Answer the user's question based strictly on the following context:\n\nContext:\n{context}\n\nQuestion: {prompt}"
        
        ai_message = groq_chat.invoke(augmented_prompt)
        response = ai_message.content
        
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append(
            {'role':'assistant', 'content':response})
    except Exception as e:
        st.error(f"Error: {str(e)}")
