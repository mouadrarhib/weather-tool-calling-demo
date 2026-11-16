from sentence_transformers import SentenceTransformer

from embedding_cache import load_or_create_embeddings
from pdf_ingestion import load_pdf_chunks


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

documents = load_pdf_chunks()

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu",
)

document_embeddings = load_or_create_embeddings(
    model,
    documents,
    MODEL_NAME,
)


def retrieve_passage(question, location="Casablanca", min_similarity=0.3,):
    # Keep the positions of documents that apply.
    eligible_indices = [
        index
        for index, document in enumerate(documents)
        if document["location"] == location
        and document["status"] == "active"
    ]

    if not eligible_indices:
        return None

    # Only the question gets a new embedding for each search.
    question_embedding = model.encode(question)

    scores = model.similarity(
        question_embedding,
        document_embeddings[eligible_indices],
    )[0]

    best_position = int(scores.argmax())
    document_index = eligible_indices[best_position]
    best_document = documents[document_index]
    best_score = float(scores[best_position])
    if best_score < min_similarity:
        return None

    return {
        "id": best_document["id"],
        "text": best_document["text"],
        "source": best_document["source"],
        "page": best_document["page"],
        "similarity": best_score,
    }


if __name__ == "__main__":
    questions = [
        "How can I cancel my booking?",
        "What is the cancellation fee?",
    ]

    for question in questions:
        passage = retrieve_passage(question)
        print("\nQuestion:", question)
        print("Retrieved:", passage)