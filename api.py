import os
import logging
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
import chromadb
from chromadb.config import Settings
import requests
from pydantic import BaseModel


# -----------------------------------------
# Setup Logging
# -----------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# -----------------------------------------
# Constants: Configure the RAG here
# -----------------------------------------
# Directory to store uploaded PDFs
UPLOAD_DIR = "./uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Directory where ChromaDB will persist its data
CHROMA_DB_DIR = "./chromadb"  # Path to ChromaDB storage

# Name of the ChromaDB collection to create or retrieve
COLLECTION_NAME = "pdf_documents"

# Maximum size of text chunks when splitting the PDF content
CHUNK_SIZE = 500  # Number of characters per text chunk

# Configurable model selection: "chatgpt" or "ollama"
MODEL_TYPE = "ollama"  # Default is Ollama; change to "chatgpt" for OpenAI's GPT

# Ollama-specific configurations
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"  # Local Ollama server endpoint
DEFAULT_OLLAMA_MODEL = "phi3.5:latest"  # Default Ollama model

# OpenAI-specific configurations
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your_openai_api_key")  # Replace with your OpenAI API key
CHATGPT_MODEL_NAME = "gpt-3.5-turbo"

# -----------------------------------------
# FastAPI Setup
# -----------------------------------------
app = FastAPI()


# -----------------------------------------
# Utility Functions
# -----------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        logger.info("Text extracted from PDF.")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract text from PDF.")


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Splits a large text into smaller chunks for embedding."""
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    logger.info(f"Text split into {len(chunks)} chunks.")
    return chunks


def create_chroma_client_and_collection(collection_name: str) -> chromadb.api.Collection:
    """Creates a ChromaDB client and retrieves a collection."""
    chroma_client = chromadb.Client(Settings(persist_directory=CHROMA_DB_DIR))
    collection = chroma_client.get_or_create_collection(name=collection_name)
    logger.info(f"ChromaDB collection '{collection_name}' created or retrieved.")
    return collection


def ingest_pdf_to_chromadb(pdf_path: str, collection: chromadb.api.Collection) -> None:
    """Ingests text from a PDF into a ChromaDB collection."""
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text, CHUNK_SIZE)

    for idx, chunk in enumerate(chunks):
        if chunk.strip():  # Skip empty chunks
            collection.upsert(
                documents=[chunk],
                ids=[f"doc_{idx}"]
            )
    logger.info(f"Ingested {len(chunks)} chunks into ChromaDB.")


def query_vector_db(collection: chromadb.api.Collection, query: str, max_results: int = 3) -> List[str]:
    """Queries the vector database for the most similar chunks."""
    results = collection.query(
        query_texts=[query],
        n_results=max_results
    )
    documents = results["documents"][0]
    logger.info(f"Retrieved {len(documents)} matching chunks from the vector DB.")
    return documents


def query_llm(
    query: str,
    context: List[str],
    model_type: str = MODEL_TYPE,
    ollama_model: str = DEFAULT_OLLAMA_MODEL
) -> str:
    """
    Queries a language model (LLM) with additional context.
    """
    if model_type == "chatgpt":
        # Construct the messages with context
        messages = [
            {"role": "system", "content": "You are an assistant that answers questions about PDF documents."},
            {"role": "user", "content": f"Context: {' '.join(context)}\n\nQuestion: {query}"}
        ]

        # Simulate response (Replace with OpenAI API call if needed)
        return "ChatGPT response simulation"

    elif model_type == "ollama":
        # Combine context into a single string
        combined_context = " ".join(context)
        prompt = f"Context: {combined_context}\n\nQuestion: {query}"

        # Specify the model and send the query to the Ollama server
        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "stream": False  # Disable streaming for simplicity
        }
        try:
            response = requests.post(OLLAMA_ENDPOINT, json=payload)
            response.raise_for_status()
            return response.json().get("response", "No response received from Ollama.")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTPError: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to query LLM: {e.response.text}")
        except Exception as e:
            logger.error(f"Unexpected error querying LLM: {e}")
            raise HTTPException(status_code=500, detail="Unexpected error querying LLM.")
    else:
        raise HTTPException(status_code=400, detail="Invalid model type. Use 'chatgpt' or 'ollama'.")


# -----------------------------------------
# FastAPI Endpoints
# -----------------------------------------
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile):
    """
    Endpoint to upload a PDF and ingest it into ChromaDB.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    logger.info(f"PDF uploaded to {file_path}")

    collection = create_chroma_client_and_collection(COLLECTION_NAME)
    ingest_pdf_to_chromadb(file_path, collection)

    return {"message": f"PDF {file.filename} uploaded and ingested successfully."}


@app.post("/query/")
async def query_pdf(query: str = Form(...), model_type: str = MODEL_TYPE):
    """
    Endpoint to query the ingested PDF content via an LLM.
    """
    collection = create_chroma_client_and_collection(COLLECTION_NAME)

    # Query the vector database
    try:
        top_matches = query_vector_db(collection, query)
    except Exception as e:
        logger.error(f"Error querying vector DB: {e}")
        raise HTTPException(status_code=500, detail="Failed to query vector database.")

    # Query the LLM
    try:
        response = query_llm(query, top_matches, model_type=model_type)
        return JSONResponse(content={"response": response})
    except Exception as e:
        logger.error(f"Error querying LLM: {e}")
        raise HTTPException(status_code=500, detail="Failed to query LLM.")


@app.get("/")
async def root():
    """Basic root endpoint to check API status."""
    return {"message": "PDF Query API is up and running."}
