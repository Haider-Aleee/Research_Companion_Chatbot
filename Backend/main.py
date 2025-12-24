#==========================================================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

from sentence_transformers import SentenceTransformer
import faiss  # ← Changed from chromadb
import numpy as np  # ← Added for FAISS operations
import pickle
import json
import os
from pathlib import Path
import uvicorn
from rank_bm25 import BM25Okapi
import re

# ============================================================================
# BM25 Retriever Wrapper
# ============================================================================

class BM25Retriever:
    """Wrapper for BM25Okapi that provides a search interface compatible with the existing code"""
    
    def __init__(self, bm25_model, chunks_data):
        self.bm25_model = bm25_model
        self.chunks_data = chunks_data
        # Create a mapping from chunk_id to index
        self.chunk_id_to_idx = {chunk['chunk_id']: i for i, chunk in enumerate(chunks_data)}
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks using BM25"""
        # Tokenize query
        query_tokens = re.findall(r'\w+', query.lower())
        
        # Get BM25 scores
        scores = self.bm25_model.get_scores(query_tokens)
        
        # Get top k indices
        top_indices = scores.argsort()[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            if idx < len(self.chunks_data):
                chunk = self.chunks_data[idx]
                results.append({
                    'chunk_id': chunk['chunk_id'],
                    'text': chunk['text'],
                    'metadata': {
                        'paper_id': chunk['paper_id'],
                        'section': chunk.get('section', ''),
                        'strategy': chunk.get('strategy', 'section')
                    },
                    'score': float(scores[idx])
                })
        
        return results

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Model paths
    MODEL_PATH = "./model"
    FAISS_INDEX_PATH = "./faiss.index"  # ← Changed from VECTOR_DB_PATH
    CHUNKS_PKL_PATH = "./chunks.pkl"     # ← Added for FAISS chunks
    BM25_PATH = "./bm25_section.pkl"
    PAPERS_DATA_PATH = "./processed_papers.json"
    CHUNKS_DATA_PATH = "./all_chunks.json"
    
    # Model names
    BASE_MODEL_NAME = "google/flan-t5-base"
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    # API settings
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
    TOP_K = 5
    MAX_ANSWER_LENGTH = 256
    EMBEDDING_DIMENSION = 384  # ← Added for FAISS (all-MiniLM-L6-v2)

config = Config()
# ============================================================================
# Pydantic Models
# ============================================================================

class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    strategy: Optional[str] = "section"  # "fixed" or "section"

class Citation(BaseModel):
    index: int
    paper_id: str
    section: str
    chunk_id: str
    relevance_score: float

class AnswerResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    processing_time: float

class RelatedWorkRequest(BaseModel):
    topic: str
    num_citations: Optional[int] = 5

class RelatedWorkResponse(BaseModel):
    paragraph: str
    citations: List[Dict]

class PaperInfo(BaseModel):
    paper_id: str
    num_sections: int
    num_chunks: int
    sections: List[str]

# ============================================================================
# Load Models and Data (on startup)
# ============================================================================

class ModelManager:
    """Singleton to manage loaded models"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        print("🚀 Loading models and data...")
        
        # Load deployment config
        with open('deployment_config.json', 'r') as f:
            self.deployment_config = json.load(f)
        
        # Load tokenizer and model
        print("📦 Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
        
        # Force CPU-only, float32
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
        config.BASE_MODEL_NAME,
        device_map="cpu",
        torch_dtype=torch.float32
)
        self.model = PeftModel.from_pretrained(base_model, config.MODEL_PATH)
        self.model.to("cpu")
        self.model.eval()
        
        print("✓ Model loaded")
        
        # Load embedding model
        print("📦 Loading embedding model...")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        print("✓ Embedding model loaded")
        
        # Load or create vector database (FAISS)
        print("📦 Loading vector database...")
        self.faiss_index = None
        self.faiss_chunks_metadata = []
        
        if os.path.exists(config.FAISS_INDEX_PATH) and os.path.exists(config.CHUNKS_PKL_PATH):
            try:
                self.faiss_index = faiss.read_index(config.FAISS_INDEX_PATH)
                with open(config.CHUNKS_PKL_PATH, 'rb') as f:
                    self.faiss_chunks_metadata = pickle.load(f)
                print("✓ Vector database loaded")
            except Exception as e:
                print(f"⚠ Error loading FAISS index: {e}, will rebuild...")
                self.faiss_index = None
        else:
            print("⚠ Vector DB not found, will rebuild from chunks...")
            self.faiss_index = None
        
        # Load BM25
        print("📦 Loading BM25 index...")
        self.bm25_retriever = None
        if os.path.exists(config.BM25_PATH):
            try:
                # Try to load the pickle file
                with open(config.BM25_PATH, 'rb') as f:
                    loaded_obj = pickle.load(f)
                    # Check if it's already a BM25Retriever instance
                    if isinstance(loaded_obj, BM25Retriever):
                        self.bm25_retriever = loaded_obj
                    else:
                        # If it's a BM25Okapi model, wrap it
                        # We'll need to rebuild it properly
                        print("⚠ BM25 pickle format changed, will rebuild...")
                        self.bm25_retriever = None
            except (AttributeError, ImportError, pickle.UnpicklingError) as e:
                print(f"⚠ Error loading BM25 index: {e}, will rebuild...")
                self.bm25_retriever = None
        else:
            print("⚠ BM25 index not found, will rebuild...")
            self.bm25_retriever = None
        
        # Load papers data
        print("📦 Loading papers data...")
        with open(config.PAPERS_DATA_PATH, 'r') as f:
            self.papers_data = json.load(f)
        
        with open(config.CHUNKS_DATA_PATH, 'r') as f:
            self.chunks_data = json.load(f)
        
        print(f"✓ Loaded {len(self.papers_data)} papers")
        print(f"✓ Loaded {len(self.chunks_data)} chunks")
        
        # Build vector DB if needed
        if self.faiss_index is None:
            self._rebuild_vector_db()
        
        # Build BM25 if needed
        if self.bm25_retriever is None:
            self._rebuild_bm25()
        
        self._initialized = True
        print("✅ All models loaded successfully!")
    
    def _rebuild_vector_db(self):
        """Rebuild vector database from chunks using FAISS"""
        print("🔨 Rebuilding vector database...")
        
        # Get section chunks
        section_chunks = [c for c in self.chunks_data if c['strategy'] == 'section']
        
        # Generate embeddings
        texts = [chunk['text'] for chunk in section_chunks]
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index (Inner Product for cosine similarity after normalization)
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # Add embeddings to index
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.faiss_chunks_metadata = []
        for i, chunk in enumerate(section_chunks):
            self.faiss_chunks_metadata.append({
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'paper_id': chunk['paper_id'],
                'section': chunk['section'],
                'strategy': chunk['strategy']
            })
        
        # Save index and metadata
        faiss.write_index(self.faiss_index, config.FAISS_INDEX_PATH)
        with open(config.CHUNKS_PKL_PATH, 'wb') as f:
            pickle.dump(self.faiss_chunks_metadata, f)
        
        print("✓ Vector database rebuilt")
    
    def _rebuild_bm25(self):
        """Rebuild BM25 index from chunks"""
        print("🔨 Rebuilding BM25 index...")
        
        # Get section chunks
        section_chunks = [c for c in self.chunks_data if c.get('strategy') == 'section']
        
        # Tokenize texts for BM25
        tokenized_texts = []
        for chunk in section_chunks:
            text = chunk['text']
            tokens = re.findall(r'\w+', text.lower())
            tokenized_texts.append(tokens)
        
        # Create BM25 model
        bm25_model = BM25Okapi(tokenized_texts)
        
        # Wrap in BM25Retriever
        self.bm25_retriever = BM25Retriever(bm25_model, section_chunks)
        
        # Save to pickle
        with open(config.BM25_PATH, 'wb') as f:
            pickle.dump(self.bm25_retriever, f)
        
        print("✓ BM25 index rebuilt")

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Research Paper QA API",
    description="Answer questions about research papers with citations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager (lazy loading)
model_manager = None

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global model_manager
    model_manager = ModelManager()

# ============================================================================
# Helper Functions
# ============================================================================

def hybrid_search(query: str, top_k: int = 5) -> List[Dict]:
    """Perform hybrid search (vector + BM25)"""
    
    # Vector search using FAISS
    query_embedding = model_manager.embedding_model.encode([query])[0]
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_embedding)  # Normalize for cosine similarity
    
    # Search in FAISS index
    vector_results = []
    if model_manager.faiss_index and model_manager.faiss_index.ntotal > 0:
        k = min(top_k * 2, model_manager.faiss_index.ntotal)
        distances, indices = model_manager.faiss_index.search(query_embedding, k)
        
        # Get vector results
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(model_manager.faiss_chunks_metadata):
                metadata = model_manager.faiss_chunks_metadata[idx]
                vector_results.append({
                    'chunk_id': metadata['chunk_id'],
                    'text': metadata['text'],
                    'metadata': {
                        'paper_id': metadata['paper_id'],
                        'section': metadata['section'],
                        'strategy': metadata['strategy']
                    },
                    'score': float(distances[0][i])  # Already similarity (Inner Product after normalization)
                })
    
    # BM25 search
    bm25_results = model_manager.bm25_retriever.search(query, top_k=top_k * 2)
    
    # Combine and deduplicate
    combined = {}
    
    # Add vector results
    for result in vector_results:
        chunk_id = result['chunk_id']
        combined[chunk_id] = {
            'chunk_id': chunk_id,
            'text': result['text'],
            'metadata': result['metadata'],
            'score': 0.5 * result['score']  # Normalize similarity score
        }
    
    # Add BM25 results
    for result in bm25_results:
        chunk_id = result['chunk_id']
        if chunk_id in combined:
            combined[chunk_id]['score'] += 0.5 * (result['score'] / 10)  # Normalize
        else:
            combined[chunk_id] = {
                'chunk_id': chunk_id,
                'text': result['text'],
                'metadata': result['metadata'],
                'score': 0.5 * (result['score'] / 10)
            }
    
    # Sort and return top k
    results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
    return results[:top_k]

def generate_answer(question: str, context: str) -> str:
    """Generate answer using fine-tuned model"""
    
    # Prepare input
    input_text = f"question: {question} context: {context}"
    
    inputs = model_manager.tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(model_manager.model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model_manager.model.generate(
            **inputs,
            max_length=config.MAX_ANSWER_LENGTH,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    
    answer = model_manager.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "version": "1.0.0",
        "model": config.BASE_MODEL_NAME,
        "num_papers": len(model_manager.papers_data) if model_manager else 0
    }

@app.get("/papers", response_model=List[PaperInfo])
async def get_papers():
    """Get list of available papers"""
    papers_info = []
    
    for paper in model_manager.papers_data:
        # Count chunks for this paper
        paper_chunks = [c for c in model_manager.chunks_data if c['paper_id'] == paper['paper_id']]
        
        papers_info.append(PaperInfo(
            paper_id=paper['paper_id'],
            num_sections=len(paper['sections']),
            num_chunks=len(paper_chunks),
            sections=list(paper['sections'].keys())
        ))
    
    return papers_info

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Answer a question about research papers
    
    Example:
    ```
    POST /ask
    {
        "question": "What are the main contributions?",
        "top_k": 5
    }
    ```
    """
    import time
    start_time = time.time()
    
    try:
        # Retrieve relevant chunks
        retrieved_chunks = hybrid_search(request.question, request.top_k)
        
        if not retrieved_chunks:
            raise HTTPException(status_code=404, detail="No relevant information found")
        
        # Prepare context
        context_parts = []
        citations = []
        
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(f"[{i+1}] {chunk['text']}")
            
            citations.append(Citation(
                index=i + 1,
                paper_id=chunk['metadata']['paper_id'],
                section=chunk['metadata']['section'],
                chunk_id=chunk['chunk_id'],
                relevance_score=chunk['score']
            ))
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        answer = generate_answer(request.question, context)
        
        processing_time = time.time() - start_time
        
        return AnswerResponse(
            question=request.question,
            answer=answer,
            citations=citations,
            processing_time=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/related-work", response_model=RelatedWorkResponse)
async def generate_related_work(request: RelatedWorkRequest):
    """
    Generate a related work paragraph with citations
    
    Example:
    ```
    POST /related-work
    {
        "topic": "transformer architectures",
        "num_citations": 5
    }
    ```
    """
    try:
        # Retrieve relevant chunks
        query = f"Related work and previous research on {request.topic}"
        retrieved_chunks = hybrid_search(query, request.num_citations * 2)
        
        # Group by paper
        papers_mentioned = {}
        for chunk in retrieved_chunks:
            paper_id = chunk['metadata']['paper_id']
            if paper_id not in papers_mentioned:
                papers_mentioned[paper_id] = []
            papers_mentioned[paper_id].append(chunk['text'])
        
        # Limit to requested number
        papers_to_cite = list(papers_mentioned.keys())[:request.num_citations]
        
        # Generate paragraph
        paragraph_parts = [f"Recent work on {request.topic} has explored various approaches."]
        
        for i, paper_id in enumerate(papers_to_cite, 1):
            summary = papers_mentioned[paper_id][0][:150]
            paragraph_parts.append(f"{summary}... [{paper_id}]")
        
        related_work = " ".join(paragraph_parts)
        
        return RelatedWorkResponse(
            paragraph=related_work,
            citations=[
                {'index': i+1, 'paper_id': pid}
                for i, pid in enumerate(papers_to_cite)
            ]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_chunks(query: str, top_k: int = 5):
    """
    Search for relevant chunks
    
    Example: GET /search?query=transformer&top_k=5
    """
    try:
        results = hybrid_search(query, top_k)
        return {
            "query": query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Remove in production
    )