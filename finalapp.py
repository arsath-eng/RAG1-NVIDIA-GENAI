import streamlit as st
import os
import time
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")

# Function to create or get vector store
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./docs")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )
    return st.session_state.vectors

# Streamlit UI
st.title("NVIDIA NIM DEMO")

# Button to create vector store
if st.button("Create Document Embeddings"):
    with st.spinner("Creating embeddings..."):
        vector_store = vector_embedding()
    st.success("FAISS vector store is ready")

# Input for user question
prompt1 = st.text_input("Enter your question about the documents:")

# Button to process the question
if st.button("Get Answer") and prompt1:
    if "vectors" not in st.session_state:
        st.error("Please create document embeddings first.")
    else:
        with st.spinner("Thinking..."):
            prompt = ChatPromptTemplate.from_template("""
            Answer the question correctly
            <context>
            {context}
            </context>
            Question: {input}
            """)
            
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            end = time.process_time()
            
            st.write("Answer:", response['answer'])
            st.write(f"Response time: {end - start:.2f} seconds")

            with st.expander("Documents similarity search"):
                for i, doc in enumerate(response["context"]):
                    st.write(f"Document {i + 1}:")
                    st.write(doc.page_content)
                    st.write("------------------")