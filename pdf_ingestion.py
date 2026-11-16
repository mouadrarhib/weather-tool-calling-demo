from pathlib import Path
import re

from pypdf import PdfReader


def load_pdf_chunks(location="Casablanca", status="active"):
    pdf_path = (
        Path(__file__).resolve().parent
        / "data"
        / "rental_policy.pdf"
    )

    reader = PdfReader(pdf_path)
    chunks = []

    repeated_lines = {
        "Rental Policy",
        "Fictional policy for an AI engineering learning exercise.",
        "Practice document only. These are not real company policies.",
    }

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        kept_lines = []

        for line in raw_text.splitlines():
            line = line.strip()

            if not line or line in repeated_lines:
                continue

            if re.fullmatch(r"Page \d+ of \d+", line):
                continue

            kept_lines.append(line)

        text = "\n".join(kept_lines)

        if not text:
            raise ValueError(
                f"No usable text extracted from page {page_number}. "
                "Inspect the PDF; it may require OCR."
            )

        chunks.append({
            "id": f"rental-pdf-{page_number:02d}",
            "text": text,
            "source": pdf_path.name,
            "page": page_number,
            "location": location,
            "status": status,
        })

    return chunks


if __name__ == "__main__":
    chunks = load_pdf_chunks()

    print("Chunks created:", len(chunks))

    for chunk in chunks:
        print(f"\n{chunk['id']} — page {chunk['page']}")
        print(chunk["text"])