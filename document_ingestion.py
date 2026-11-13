from pathlib import Path


def load_policy_chunks(location="Casablanca", status="active"):
    document_path = (
        Path(__file__).resolve().parent
        / "data"
        / "rental_policy.txt"
    )

    text = document_path.read_text(encoding="utf-8")
    sections = text.strip().split("\n\n")

    chunks = []

    for index, section in enumerate(sections, start=1):
        if not section.strip():
            continue

        chunks.append({
            "id": f"rental-policy-{index:02d}",
            "text": section.strip(),
            "source": document_path.name,
            "location": location,
            "status": status,
        })

    return chunks


if __name__ == "__main__":
    chunks = load_policy_chunks()

    print("Chunks created:", len(chunks))

    for chunk in chunks:
        print("\nID:", chunk["id"])
        print("Source:", chunk["source"])
        print(chunk["text"])