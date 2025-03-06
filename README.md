# Istanbul-Travel-Assistant

## 💎 Istanbul Travel Assistant-Chatbot Value Proposition

This system delivers exceptional value as an Istanbul travel assistant:

- **Accurate Local Knowledge**: Provides reliable, factual information about Istanbul's attractions, neighborhoods, history, and culture from trusted guidebook sources
- **Reduced Hallucinations**: Multiple validation layers ensure travelers receive accurate information, preventing misinformation that could disrupt travel plans
- **Complementary Information Sources**: Seamlessly combines curated guidebook content with up-to-date web information about events, restaurant openings, or temporary closures
- **Natural Language Understanding**: Travelers can ask questions in casual, conversational language rather than using specific search terms
- **Travel Planning Efficiency**: Saves travelers hours of research by providing immediate, targeted information about Istanbul's offerings
- **Contextual Awareness**: Understands the relationships between attractions, neighborhoods, and logistics that might not be obvious in disjointed search results
- **Personalized Recommendations**: Can adapt responses based on specific interests, time constraints, or preferences mentioned in queries

## Adaptive-Self-Reflecting-RAG

An innovative Retrieval-Augmented Generation (RAG) system with self-reflection capabilities that intelligently analyzes queries, routes them to appropriate data sources, and validates outputs through multiple quality control mechanisms.

## 🚀 Features

- **Intelligent Query Analysis**: Routes queries between vectorstore and web search based on relevance to indexed documents
- **Self-Reflection Pipeline**: 
  - Document relevance verification
  - Hallucination detection
  - Answer quality validation
- **Query Optimization**: Automatically rewrites questions when initial retrievals are inadequate
- **Multi-Source Retrieval**:
  - Local vectorstore knowledge (Istanbul Guide documents)
  - Web search integration for up-to-date information
- **Extensible Architecture**: Support for adding additional routing paths and data sources
- **LLM Integration**: Powered by Groq's Mistral-Saba-24B model
- **Production-Ready API**: FastAPI backend for seamless integration

## 📋 System Architecture

 ![download](https://github.com/user-attachments/assets/d687deb7-a460-4ebb-8876-c24a3c1702bb)


The system follows a flow:
1. **Query Analysis**: Determines if the query relates to indexed documents
2. **Routing Decision**:
   - If related to index → RAG pipeline with self-reflection
   - If unrelated to index → Web search
   - Optional routes can be added for specialized queries
3. **RAG + Self-Reflection Pipeline**:
   - Retrieval from vectorstore
   - Document relevance grading
   - Answer generation
   - Hallucination detection
   - Answer quality verification
   - Query rewriting when needed

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Shrouk-Adel/Istanbul-Travel-Assistant.git
cd Istanbul-Travel-Assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

## 📦 Dependencies

```
langchain
langchain-core
langchain-groq
langchain-community
langchain-ollama
fastapi
pydantic
python-dotenv
langgraph
chromadb
tiktoken
tavily-python
uvicorn
```

## 🔑 Environment Variables

Create a `.env` file with:
```
LangChain_API_Key=your_langchain_api_key
Groq_API_Key=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## 📚 Dataset

The system currently uses Istanbul City Guide PDF data. To use different documents:

1. Place PDF files in the `./Data` directory
2. Update the file paths in the configuration section:
```python
urls = [
    r"path/to/your/document1.pdf",
    r"path/to/your/document2.pdf"
]
```

## 🚀 Usage

### Start the server

```bash
python App.py
```

### Query the API

```bash
curl -X POST "http://localhost:8001/query" -H "Content-Type: application/json" -d '{"question":"What are popular tourist attractions in Istanbul?"}'
```

 

## ⚙️ How It Works

1. **Query Routing**: The system analyzes if the query relates to indexed documents
2. **Document Retrieval**: Fetches relevant chunks using similarity search
3. **Document Grading**: Filters out irrelevant documents
4. **Dynamic Query Rewriting**: Reformulates queries when retrievals are poor
5. **Answer Generation**: Creates responses based on relevant documents
6. **Quality Control**: Verifies answers are grounded in facts and address the question

## 📈 Performance Considerations

- Multiple LLM calls may impact response time; consider implementing caching
- Adjust retrieval parameters (k, chunk size, overlap) based on your specific use case
- For production, consider implementing rate limiting and authentication

## 🔍 Future Enhancements

- [ ] Add more routing paths for specialized query types
- [ ] Implement answer caching for common queries
- [ ] Add support for more document types (beyond PDFs)
- [ ] Enhance the query rewriting with few-shot examples
- [ ] Add conversation history tracking for context-aware responses

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
 
