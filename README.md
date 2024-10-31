# 📚 RAG Applications with NVIDIA NeMo and Streamlit 
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA%20AI-Enabled-76B900.svg)](https://www.nvidia.com/en-us/ai-data-science/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<p align="center">
  <img src="screenshots/3.png" alt="Project Banner" width="800"/>
</p>

## 🌟 Overview

A sophisticated implementation of Retrieval Augmented Generation (RAG) leveraging NVIDIA's AI endpoints and Streamlit. This application transforms document analysis and question-answering through state-of-the-art language models and efficient vector search capabilities.

### 🎯 Core Features
- 📄 Multi-PDF Document Processing
- 🔍 Advanced Text Chunking System
- 💾 FAISS Vector Store Integration
- ⚡ NVIDIA NIM Endpoints
- 🤖 Llama 3.1 405B Model Support
- ⏱️ Real-time Performance Metrics
- 📊 Similarity Search Visualization


## 🛠️ Technical Architecture

### Component Stack
```mermaid
graph TD
    A[PDF Documents] --> B[Document Processor]
    B --> C[Text Chunker]
    C --> D[NVIDIA Embeddings]
    D --> E[FAISS Vector Store]
    F[User Query] --> G[Query Processor]
    G --> E
    E --> H[NVIDIA LLM]
    H --> I[Response Generator]
```

### System Requirements
- Python 3.8+
- 8GB RAM minimum
- NVIDIA API access
- Internet connectivity
- PDF processing capabilities

## 📦 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/arsath-eng/RAG1-NVIDIA-GENAI.git
cd RAG1-NVIDIA-GENAI
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# For Windows
.\venv\Scripts\activate
# For Unix/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
```bash
# Create environment file
touch .env

# Add required credentials
NVIDIA_API_KEY=your_api_key_here
```


## 💻 Usage Guide

### Application Launch
```bash
streamlit run app.py
```

### Document Processing Workflow

1. **Document Upload**
   - Support for multiple PDF files
   - Automatic text extraction
   - Progress tracking
   

2. **Embedding Creation**
   - Click "Create Document Embeddings"
   - Automatic chunking and processing
   - Vector store initialization
   

3. **Query Processing**
   - Enter questions about documents
   - Real-time response generation
   - View similarity search results
   
   <p align="center">
    <img src="screenshots/Rag1.png" alt="Project Banner" width="800"/>
  </p>
  <p align="center">
  <img src="screenshots/Rag2.png" alt="Project Banner" width="800"/>
</p>
<p align="center">
  <img src="screenshots/3.png" alt="Project Banner" width="800"/>
</p>
<p align="center">
  <img src="screenshots/4.png" alt="Project Banner" width="800"/>
</p>
<p align="center">
  <img src="screenshots/5.png" alt="Project Banner" width="800"/>
</p>
<p align="center">
  <img src="screenshots/6.png" alt="Project Banner" width="800"/>
</p>
## ⚙️ Configuration Options

### Text Splitting Parameters
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,  # Adjust for document length
    chunk_overlap=50  # Modify for context preservation
)
```

### Model Configuration
```python
llm = ChatNVIDIA(
    model="meta/llama-3.1-405b-instruct",
    temperature=0.7,  # Adjust for response creativity
    max_tokens=512    # Modify for response length
)
```

## 📈 Performance Optimization

### Vector Store Tuning
- Optimal chunk size selection
- Embedding dimension management
- Index optimization techniques

### Response Time Improvements
- Query preprocessing
- Cache implementation
- Batch processing capabilities


## 🔍 Advanced Features

### 1. Intelligent Chunking
- Content-aware text splitting
- Semantic boundary preservation
- Overlap optimization

### 2. Vector Search Enhancement
- Nearest neighbor search
- Similarity threshold tuning
- Result ranking optimization

### 3. Response Generation
- Context-aware answers
- Source attribution
- Confidence scoring

## 🚀 Best Practices

### Document Preparation
- Use clear, well-formatted PDFs
- Ensure text is extractable
- Optimize document size

### Query Formulation
- Be specific and clear
- Include relevant context
- Use natural language

### System Optimization
- Monitor memory usage
- Regular cache clearing
- Performance tracking

## 🛠️ Troubleshooting

### Common Issues
1. **PDF Processing Errors**
   - Solution: Check PDF format compatibility
   - Verify text extraction capabilities

2. **Memory Issues**
   - Solution: Adjust chunk size
   - Implement batch processing

3. **API Connection**
   - Solution: Verify credentials
   - Check network connectivity

## 🤝 Contributing

### Development Workflow
1. Fork repository
2. Create feature branch
```bash
git checkout -b feature/YourFeature
```
3. Commit changes
```bash
git commit -m 'Add YourFeature'
```
4. Push to branch
```bash
git push origin feature/YourFeature
```
5. Submit Pull Request

### Code Standards
- PEP 8 compliance
- Documentation requirements
- Test coverage expectations

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- NVIDIA AI Team
- Streamlit Community
- LangChain Contributors
- Meta AI Research

## 📞 Support & Contact

- Create an Issue
- Join Discussions
- Review Documentation


---

<div align="center">

**[Documentation](docs/) • [Report Bug](issues) • [Request Feature](issues/new)**

[![Stars](https://img.shields.io/github/stars/arsath-eng/RAG1-NVIDIA-GENAI?style=social)](https://github.com/arsath-eng/RAG1-NVIDIA-GENAI)

*Made with ❤️ by [@arsath-eng](https://github.com/arsath-eng)*

</div>
