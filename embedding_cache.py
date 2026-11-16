import hashlib
import json
from pathlib import Path

import numpy as np


CACHE_PATH = (
    Path(__file__).resolve().parent
    / ".cache"
    / "document_embeddings.npz"
)


def load_or_create_embeddings(model, documents, model_name):
    # Include text, source, page and metadata in the fingerprint.
    # Bump cache_version if embedding settings change.
    configuration = {
        "cache_version": 1,
        "model_name": model_name,
        "normalize_embeddings": False,
        "documents": documents,
    }

    serialized = json.dumps(
        configuration,
        sort_keys=True,
        ensure_ascii=False,
    )

    fingerprint = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()

    if CACHE_PATH.exists():
        try:
            with np.load(CACHE_PATH, allow_pickle=False) as cached:
                saved_fingerprint = cached["fingerprint"].item()
                embeddings = cached["embeddings"]

                if (
                    saved_fingerprint == fingerprint
                    and embeddings.ndim == 2
                    and embeddings.shape[0] == len(documents)
                    and embeddings.shape[1]
                    == model.get_embedding_dimension()
                    and np.isfinite(embeddings).all()
                ):
                    print("Document embeddings: loaded from cache")
                    return embeddings
        except (OSError, ValueError, KeyError, EOFError):
            print("Cache could not be read; rebuilding.")

    embeddings = model.encode(
        [document["text"] for document in documents],
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = CACHE_PATH.with_suffix(".tmp")

    with temporary_path.open("wb") as file:
        np.savez_compressed(
            file,
            fingerprint=np.array(fingerprint),
            embeddings=embeddings,
        )

    temporary_path.replace(CACHE_PATH)

    print("Document embeddings: created and saved")
    return embeddings