# Research Companion App

A full-stack application that leverages advanced NLP and information retrieval techniques to help researchers explore academic papers, answer research questions, and generate related work summaries.

## Features

- **Semantic Search**: Find relevant research papers using FAISS-based similarity search with state-of-the-art embedding models
- **Question Answering**: Get precise answers to research questions based on your paper collection using sequence-to-sequence models
- **Related Work Generation**: Automatically generate related work sections for your research based on paper metadata and content
- **Hybrid Retrieval**: Combines semantic search (FAISS) and BM25 keyword matching for improved relevance
- **RESTful API**: Clean FastAPI backend for integration with various frontend clients

## Tech Stack

### Backend
- **Framework**: FastAPI with CORS support
- **Language Models**: Transformers (HuggingFace), PEFT for efficient fine-tuning
- **Semantic Search**: FAISS + Sentence Transformers
- **Keyword Search**: BM25Okapi
- **ML Framework**: PyTorch
- **Server**: Uvicorn

### Frontend
- **Framework**: React 19
- **HTTP Client**: Axios
- **Markdown Rendering**: react-markdown
- **Code Highlighting**: react-syntax-highlighter
- **Testing**: Jest + React Testing Library

## Project Structure

```
Research_Companion_App/
├── Backend/
│   ├── main.py                      # FastAPI application
│   ├── requirements.txt             # Python dependencies
│   ├── faiss.index                  # Serialized FAISS index
│   ├── all_chunks.json              # Paper chunks for retrieval
│   ├── processed_papers.json        # Metadata and processing info
│   ├── deployment_config.json       # Configuration settings
│   ├── model/                       # Fine-tuned LoRA model weights
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── tokenizer files...
│   └── faiss/                       # FAISS library directory
│
├── frontend/
│   ├── public/                      # Static assets
│   ├── src/
│   │   ├── components/
│   │   │   ├── QuestionForm.js      # Question input component
│   │   │   ├── AnswerDisplay.js     # Answer output component
│   │   │   ├── PapersList.js        # Papers list display
│   │   │   └── RelatedWorkGenerator.js
│   │   ├── services/
│   │   │   └── api.js               # Backend API calls
│   │   └── App.js                   # Main application component
│   └── package.json
│
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- 8GB+ RAM (for FAISS and model loading)

### Backend Setup

1. Navigate to the Backend directory:
```bash
cd Backend
```

2. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node dependencies:
```bash
npm install
```

## Running the Application

### Start the Backend Server

```bash
cd Backend
python main.py
```

The API server will start on `http://localhost:8000` (by default).

### Start the Frontend Application

```bash
cd frontend
npm start
```

The React app will open on `http://localhost:3000` in your browser.

## API Endpoints

The FastAPI backend provides the following main endpoints:

- **POST /answer** - Get answers to research questions
- **POST /search** - Semantic search for relevant papers
- **GET /papers** - Retrieve list of processed papers
- **POST /related-work** - Generate related work section

For detailed API documentation, visit `http://localhost:8000/docs` (Swagger UI) when the backend is running.

## How It Works

1. **Paper Ingestion**: Research papers are processed and chunked into semantic units
2. **Embedding Generation**: Each chunk is converted to embeddings using Sentence Transformers
3. **Index Creation**: Embeddings are indexed using FAISS for fast similarity search
4. **Query Processing**: User questions are embedded and matched against the paper corpus using:
   - FAISS semantic similarity
   - BM25 keyword relevance
5. **Answer Generation**: Retrieved context is fed to a fine-tuned sequence-to-sequence model for answer generation

## Configuration

Edit `deployment_config.json` in the Backend directory to configure:
- Model paths and parameters
- FAISS index settings
- Search result thresholds
- API server settings

## Development

### Running Tests

```bash
# Backend tests
cd Backend
pytest

# Frontend tests
cd ../frontend
npm test
```

### Building for Production

```bash
# Frontend build
cd frontend
npm run build
```

This creates an optimized production build in the `build/` directory.

## Future Enhancements

- [ ] Document upload and processing pipeline
- [ ] User authentication and paper collections
- [ ] Fine-grained citation tracking
- [ ] Multi-language support
- [ ] Advanced filtering and sorting

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or feedback, please open an issue on the project repository.
