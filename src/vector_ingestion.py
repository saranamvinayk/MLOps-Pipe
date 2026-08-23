import os
import json
import argparse
import logging
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ingest_vectors(input_jsonl: str, persist_dir: str):
    logging.info(f"Starting vector ingestion from {input_jsonl}...")
    
    try:
        # 1. Load the chunked data from Day 6
        documents = []
        with open(input_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                # Reconstruct LangChain Document objects
                doc = Document(page_content=record['text'], metadata=record['metadata'])
                documents.append(doc)
        
        logging.info(f"Loaded {len(documents)} chunks from JSONL.")
        
        # 2. Initialize the Open-Source Embedding Model
        # all-MiniLM-L6-v2 is small, fast, and completely free to run locally
        logging.info("Initializing HuggingFace Embeddings (all-MiniLM-L6-v2)...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 3. Create and Populate the Vector Database
        # This will embed the text and save the database to the specified directory
        logging.info("Embedding documents and upserting into ChromaDB... this may take a moment.")
        
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=persist_dir,
            collection_name="enterprise_rag_docs"
        )
        
        logging.info(f"Successfully created Vector DB at {persist_dir}/")
        
        # 4. Perform a quick Sanity Check (Semantic Search)
        query = "What is the main topic of this document?"
        logging.info(f"Running test query: '{query}'")
        
        # k=2 returns the top 2 most semantically similar chunks
        results = vector_store.similarity_search_with_score(query, k=2)
        
        for idx, (doc, score) in enumerate(results):
            # Lower score = closer distance (more similar) in default L2 space
            logging.info(f"Result {idx+1} (Score: {score:.4f}): {doc.page_content[:100]}...")
            
    except Exception as e:
        logging.error(f"Error during vector ingestion: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector Ingestion Pipeline")
    parser.add_argument('--input_jsonl', type=str, default='data/processed/chunks.jsonl')
    parser.add_argument('--persist_dir', type=str, default='chroma_db')
    
    args = parser.parse_args()
    ingest_vectors(args.input_jsonl, args.persist_dir)
