import os
import json
import argparse
import logging
import torch

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# --- THE FIX: Import from langchain_classic ---
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_advanced_retriever(jsonl_path: str, persist_dir: str):
    logging.info("1. Building Sparse Retriever (BM25 for exact keywords)...")
    documents = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            documents.append(Document(page_content=record['text'], metadata=record['metadata']))
            
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10 
    
    logging.info("2. Loading Dense Retriever (ChromaDB for semantic search)...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embedding_model, collection_name="enterprise_rag_docs")
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10}) 
    
    logging.info("3. Combining into Hybrid Ensemble Retriever...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6] 
    )
    
    logging.info("4. Initializing Cross-Encoder Re-ranker...")
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
    
    advanced_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    
    return advanced_retriever

def ask_advanced_rag(question: str, jsonl_path: str, persist_dir: str):
    retriever = build_advanced_retriever(jsonl_path, persist_dir)
    
    logging.info("Loading Open-Source LLM (Qwen2.5-0.5B-Instruct)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        device_map="auto" if device == "cuda" else None
    )
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, temperature=0.1)
    llm = HuggingFacePipeline(pipeline=pipe)
    
    prompt = PromptTemplate.from_template("""<|im_start|>system
Answer using ONLY the provided context.<|im_end|>
<|im_start|>user
Context:\n{context}\n\nQuestion: {question}<|im_end|>
<|im_start|>assistant\n""")
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    logging.info(f"Executing Advanced Query: {question}")
    response = rag_chain.invoke(question)
    
    print("\n" + "="*50)
    print("QUESTION:", question)
    print("="*50)
    answer = response.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    print("GENERATED ANSWER:\n", answer)
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', type=str, default="What is the exact formula for Scaled Dot-Product Attention?")
    parser.add_argument('--jsonl_path', type=str, default='data/processed/chunks.jsonl')
    parser.add_argument('--persist_dir', type=str, default='chroma_db')
    args = parser.parse_args()
    
    ask_advanced_rag(args.question, args.jsonl_path, args.persist_dir)
