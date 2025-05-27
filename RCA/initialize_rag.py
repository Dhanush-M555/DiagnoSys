#!/usr/bin/env python3
import os
import sys
from langchain_community.document_loaders import PyPDFLoader, TextLoader  # Added TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def initialize_rag():
    """
    Initialize the RAG system by:
    1. Loading the support documentation (PDF or TXT)
    2. Splitting it into chunks
    3. Creating embeddings for each chunk
    4. Storing the embeddings in a ChromaDB vector store
    5. Saving the vector store to disk
    """
    print("Initializing RAG system...")

    # Define paths
    SUPPORT_DOC_FILENAME = "support_documentation.txt"  # Or change to .pdf
    VECTOR_STORE_PATH = "vector_store"  # Will be created if it doesn't exist

    base_dir = os.path.dirname(os.path.abspath(__file__))
    support_doc_path = os.path.join(base_dir, SUPPORT_DOC_FILENAME)
    vector_store_path = os.path.join(base_dir, VECTOR_STORE_PATH)

    print(f"Using Support Document Path: {support_doc_path}")
    print(f"Using Vector Store Path: {vector_store_path}")

    # Check if support document exists
    if not os.path.exists(support_doc_path):
        print(f"Error: Support document not found at {support_doc_path}")
        sys.exit(1)

    # Check if the vector store already exists
    if os.path.exists(vector_store_path):
        overwrite = input(f"Vector store already exists at {vector_store_path}. Overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            print("Initialization cancelled.")
            return
        else:
            # Need to remove the old directory if overwriting
            import shutil
            print(f"Removing existing vector store at {vector_store_path}...")
            shutil.rmtree(vector_store_path)
            
    # Create vector store directory if it doesn't exist (shouldn't if removed)
    os.makedirs(vector_store_path, exist_ok=True)

    # Step 1: Load the document
    try:
        if support_doc_path.endswith('.pdf'):
            print(f"Loading PDF document: {support_doc_path}")
            loader = PyPDFLoader(support_doc_path)
            documents = loader.load()
        elif support_doc_path.endswith('.txt'):
            print(f"Loading text document: {support_doc_path}")
            loader = TextLoader(support_doc_path) # Use TextLoader for .txt
            documents = loader.load()
        else:
            print(f"Error: Unsupported document format: {support_doc_path}")
            sys.exit(1)

    except Exception as e:
        print(f"Error loading document: {e}")
        sys.exit(1)

    print(f"Loaded document with {len(documents)} page(s)/section(s)")

    # Step 2: Split the document into chunks
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} chunks")

    # Step 3: Initialize the embedding model
    print("Initializing embedding model...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        print("You may need to install the required packages:")
        print("pip install sentence-transformers")
        sys.exit(1)

    # Step 4: Create and Persist the vector store
    print("Creating and saving vector store...")
    try:
        # When persist_directory is specified, Chroma handles persistence automatically.
        # The object returned here might not have a .persist() method.
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=vector_store_path  # Specify the path here
        )
        print(f"Vector store created and persisted to {vector_store_path}")
    except Exception as e:
        print(f"Error creating or persisting vector store: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        sys.exit(1)

    # Step 5: Verify the vector store by reloading it
    print("Verifying vector store...")
    try:
        loaded_store = Chroma(
            persist_directory=vector_store_path,
            embedding_function=embeddings
        )
        collection_count = loaded_store._collection.count()
        print(f"Vector store verification successful. Contains {collection_count} embeddings.")
    except Exception as e:
        print(f"Error verifying vector store: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nRAG initialization complete!")
    print(f"Vector store is ready at: {vector_store_path}")
    print("You can now run the AI agent.")
    print("If you were in the RCA directory, run: python run_app.py")
    print("If you were in the root directory, run: python RCA/run_app.py")

if __name__ == "__main__":
    initialize_rag() 