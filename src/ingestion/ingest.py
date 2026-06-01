"""
PDF ingestion pipeline for EvalForge.
Loads PDFs, splits into chunks, generates embeddings, stores in Chroma.
"""

import os
import time
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

def load_pdfs(pdf_dir: str = "data/pdfs") -> List[Document]:
    pdf_path = Path(pdf_dir)
    documents = []

    for pdf_file in pdf_path.glob("*.pdf"):
        print(f"Loading {pdf_file.name}...")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        for doc in docs:
            doc.metadata["source_file"] = pdf_file.name
        documents.extend(docs)
    
    print(f"Loaded {len(documents)} pages from {pdf_path}")
    return documents

def chunk_documents(
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
) -> List[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators=["\n\n", "\n",". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(
    chunks: List[Document],
    persist_dir: str = "./chroma_db",
    collection_name: str = "evalforge"
) -> Chroma:
        """Generate embeddings and store in Chroma."""
        embeddings = GoogleGenerativeAIEmbeddings(
             model="models/gemini-embedding-2"
        )

        vector_store = Chroma.from_documents(
             documents=chunks,
             embedding=embeddings,
             collection_name=collection_name,
             persist_directory=persist_dir
        )

        print(f"Vector store created at {persist_dir}")
        print(f"Collection: {collection_name}, count: {vector_store._collection.count()}")
        return vector_store

def main():
     docs = load_pdfs()
     chunks = chunk_documents(docs)
     create_vector_store(chunks)

if __name__ == "__main__":
    main()

