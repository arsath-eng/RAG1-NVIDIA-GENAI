import streamlit as st
import os
import time
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
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
def vector_embedding(uploaded_files):
    if "vectors" not in st.session_state or uploaded_files:
        st.session_state.embeddings = NVIDIAEmbeddings()
        documents = []
        for file in uploaded_files:
            temp_file_path = f"temp_{file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(file.getbuffer())
            loader = PyPDFLoader(temp_file_path)
            documents.extend(loader.load())
            os.remove(temp_file_path)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        final_documents = text_splitter.split_documents(documents)
        st.session_state.vectors = FAISS.from_documents(
            final_documents,
            st.session_state.embeddings
        )
    return st.session_state.vectors

# Streamlit UI
st.title("RAG WITH EXTERNAL DOCS BY USING NVIDIA NIM ")

# File uploader for PDFs
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Button to create vector store
if st.button("Create Document Embeddings"):
    if not uploaded_files:
        st.error("Please upload PDF files first.")
    else:
        with st.spinner("Creating embeddings..."):
            vector_store = vector_embedding(uploaded_files)
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
            You are an intelligent assistant designed to analyze and explain content from a given link. Your task is to provide a comprehensive overview and detailed explanation of the content in a structured manner. Follow these steps:

            1. Overview and Fundamentals:
            - Provide a brief overview of the main topic or subject matter of the link.
            - Explain the fundamental concepts or principles related to the topic.
            - Identify and define key terms or keywords that are crucial to understanding the content.
            - List the main topics or sections covered in the link.

            2. Detailed Topic Explanation:
            - For each main topic or section identified, provide a detailed explanation.
            - Include relevant examples, if available, to illustrate key points.
            - Highlight any important sub-topics or related concepts within each main topic.
            - Ensure that the explanation is thorough and covers all significant aspects mentioned in the link.

            3. Common Functionality or Applications:
            - Discuss common uses, applications, or real-world relevance of the main topic.
            - If applicable, mention any tools, technologies, or methodologies associated with the topic.
            - Highlight any current trends or future prospects related to the subject matter.

            4. Answering Questions:
            - When answering specific questions about the content, refer back to the information you've analyzed.
            - Provide clear, concise, and accurate responses based on the content of the link.
            - If a question goes beyond the scope of the link's content, indicate this and provide the best possible answer based on the available information.

            Remember to maintain a logical flow in your explanations, ensuring that complex ideas are broken down into understandable segments. Your goal is to make the content accessible and comprehensible to users with varying levels of familiarity with the subject matter.
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