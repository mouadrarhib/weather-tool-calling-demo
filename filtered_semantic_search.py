from sentence_transformers import SentenceTransformer

documents = [
    {
        "id": "policy-01",
        "text": "You can cancel your reservation from your account.",
        "source": "casablanca-active.pdf",
        "location": "Casablanca",
        "status": "active",
    },
    {
        "id": "policy-02",
        "text": "Cancel your booking by contacting the old support number.",
        "source": "casablanca-expired.pdf",
        "location": "Casablanca",
        "status": "expired",
    },
    {
        "id": "policy-03",
        "text": "Cancel your booking at our Marrakech office.",
        "source": "marrakech-active.pdf",
        "location": "Marrakech",
        "status": "active",
    },
    {
        "id": "policy-04",
        "text": "Our vehicles receive regular maintenance.",
        "source": "casablanca-maintenance.pdf",
        "location": "Casablanca",
        "status": "active",
    },
]

# Prepare the model and document vectors once when this module loads.
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
)

document_embeddings = model.encode([
    document["text"] for document in documents
])


def retrieve_passage(question, location="Casablanca"):
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

    return {
        "id": best_document["id"],
        "text": best_document["text"],
        "source": best_document["source"],
        "similarity": float(scores[best_position]),
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