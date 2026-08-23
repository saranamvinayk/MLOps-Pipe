import argparse
import logging
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def format_docs(docs):
    """Joins retrieved document chunks into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(persist_dir: str):
    logging.info("Loading ChromaDB Vector Store...")
    
    # 1. Load the same embedding model used in Day 7
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Reconnect to the persistent vector store
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_name="enterprise_rag_docs"
    )
    
    # Top-k retrieval: fetch top 3 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    logging.info("Loading Open-Source LLM (Qwen2.5-0.5B-Instruct)...")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        device_map="auto" if device == "cuda" else None
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=False,
        repetition_penalty=1.1
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # 3. Prompt Template (Instructing the LLM to ground answers in retrieved context)
    prompt_template = """<|im_start|>system
You are a helpful technical assistant. Answer the question using ONLY the provided context. If the answer cannot be found in the context, say "I cannot answer based on the provided document."
<|im_end|>
<|im_start|>user
Context:
{context}

Question: {question}
<|im_end|>
<|im_start|>assistant
"""
    prompt = PromptTemplate.from_template(prompt_template)
    
    # 4. Modern LCEL Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def ask_rag(question: str, persist_dir: str):
    rag_chain = build_rag_chain(persist_dir)
    logging.info(f"Querying: {question}")
    
    response = rag_chain.invoke(question)
    print("\n" + "="*50)
    print("QUESTION:", question)
    print("="*50)
    # Strip prompt formatting to isolate response
    answer = response.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    print("GENERATED ANSWER:\n", answer)
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End RAG Query Engine")
    parser.add_argument('--question', type=str, default="What is Multi-Head Attention and why is it useful?")
    parser.add_argument('--persist_dir', type=str, default='chroma_db')
    
    args = parser.parse_args()
    ask_rag(args.question, args.persist_dir)
