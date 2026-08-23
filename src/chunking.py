import os
import json
import argparse
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def chunk_document(input_pdf: str, output_jsonl: str):
    logging.info(f"Loading document: {input_pdf}")
    
    try:
        # 1. Extract Text from PDF
        loader = PyPDFLoader(input_pdf)
        documents = loader.load()
        logging.info(f"Extracted {len(documents)} pages.")
        
        # 2. Configure the Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # 3. Apply the Splitter
        logging.info("Chunking text...")
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Document split into {len(chunks)} contextual chunks.")
        
        # 4. Save to JSONL
        os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
        
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                record = {
                    "text": chunk.page_content,
                    "metadata": chunk.metadata
                }
                f.write(json.dumps(record) + '\n')
                
        logging.info(f"Chunks successfully saved to {output_jsonl}")
        
    except Exception as e:
        logging.error(f"Error during chunking: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Extraction and Chunking")
    parser.add_argument('--input_pdf', type=str, default='data/raw/sample_document.pdf')
    parser.add_argument('--output_jsonl', type=str, default='data/processed/chunks.jsonl')
    
    args = parser.parse_args()
    chunk_document(args.input_pdf, args.output_jsonl)
