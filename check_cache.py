from sentence_transformers import SentenceTransformer

from embedding_cache import load_or_create_embeddings
from pdf_ingestion import load_pdf_chunks

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name, device="cpu")
documents = load_pdf_chunks()

# Change only the in-memory copy, not the PDF.
documents[0]["text"] += "\nTemporary cache verification text."

load_or_create_embeddings(model, documents, model_name)