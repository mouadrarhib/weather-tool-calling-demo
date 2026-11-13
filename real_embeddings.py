from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
)

text = "How can I cancel my booking?"
embedding = model.encode(text)

print("Text:", text)
print("Number of dimensions:", len(embedding))
print("First 5 values:", embedding[:5])


passages = [
    "You can cancel your reservation from your account.",
    "Our vehicles receive regular maintenance.",
]

passage_embeddings = model.encode(passages)

scores = model.similarity(
    embedding,
    passage_embeddings,
)[0]

for passage, score in zip(passages, scores):
    print(f"\nSimilarity: {float(score):.3f}")
    print("Passage:", passage)


ranked_passages = sorted(
    zip(passages, scores),
    key=lambda item: float(item[1]),
    reverse=True,
)

print("\nRanked passages:")

for passage, score in ranked_passages:
    print(f"{float(score):.3f} — {passage}")