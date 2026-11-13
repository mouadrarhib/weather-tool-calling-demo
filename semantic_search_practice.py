import math

documents = [
    {
        "id": "policy-01",
        "text": "Cancellation is free until 24 hours before pickup.",
        "embedding": [2, 4],
        "source": "casablanca-active.pdf",
        "location": "Casablanca",
        "status": "active",
    },
    {
        "id": "policy-02",
        "text": "A cancellation fee applies to every booking.",
        "embedding": [2, 3],
        "source": "casablanca-expired.pdf",
        "location": "Casablanca",
        "status": "expired",
    },
    {
        "id": "policy-03",
        "text": "Cancellation is free until 48 hours before pickup.",
        "embedding": [2, 3],
        "source": "marrakech-active.pdf",
        "location": "Marrakech",
        "status": "active",
    },
    {
        "id": "policy-04",
        "text": "Vehicles receive regular maintenance.",
        "embedding": [8, 1],
        "source": "casablanca-maintenance.pdf",
        "location": "Casablanca",
        "status": "active",
    },
]

question_vector = [2, 3]

applicable_documents = []

for document in documents:
    if (
        document["location"] == "Casablanca"
        and document["status"] == "active"
    ):
        applicable_documents.append(document)

for document in applicable_documents:
    print(document["id"])


def euclidean_distance(vector_a, vector_b):
    squared_total = 0

    for a, b in zip(vector_a, vector_b):
        squared_total += (a - b) ** 2

    return math.sqrt(squared_total)


for document in applicable_documents:
    distance = euclidean_distance(
        question_vector,
        document["embedding"],
    )
    print(document["id"], "distance:", round(distance, 2))


ranked_results = []

for document in applicable_documents:
    distance = euclidean_distance(
        question_vector,
        document["embedding"],
    )

    ranked_results.append({
        "document": document,
        "distance": distance,
    })

ranked_results.sort(key=lambda item: item["distance"])

for item in ranked_results:
    document = item["document"]
    print(document["id"], "distance:", round(item["distance"], 2))

if ranked_results:
    best_document = ranked_results[0]["document"]

    print("\nRetrieved passage:", best_document["text"])
    print("Source:", best_document["source"])
else:
    print("No applicable documents found.")