## Semantic Search and RAG Prototype

This project also includes a small retrieval-augmented generation (RAG)
prototype over sample rental-policy passages.

### How it works

1. Load a local embedding model.
2. Generate document embeddings once per running process.
3. Filter documents by location and active status.
4. Embed the user's question.
5. Rank eligible passages by cosine similarity.
6. Send the highest-ranked passage and its source to an LLM.
7. Generate an evidence-supported answer or acknowledge missing information.

Higher cosine similarity ranks first. A high score indicates relevance,
not a guarantee that the passage answers the question.

### Files

| File | Purpose |
|---|---|
| `semantic_search_practice.py` | Learn filtering and Euclidean distance using invented vectors |
| `real_embeddings.py` | Generate real embeddings and compare passage similarity |
| `filtered_semantic_search.py` | Filter sample documents and retrieve the highest-ranked passage |
| `rag_demo.py` | Generate an answer from retrieved evidence through OpenRouter |

### Additional Setup

With the virtual environment activated, install CPU-only PyTorch and
Sentence Transformers:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install sentence-transformers
```

The embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

It generates 384-dimensional embeddings locally on the CPU.
The first run downloads the model from Hugging Face.

Local embedding generation does not require an API key.
Answer generation uses the OpenRouter key in `.env`.

### Run Semantic Search

```bash
python filtered_semantic_search.py
```

This runs two example questions without calling OpenRouter.

### Run the RAG Demo

```bash
python rag_demo.py
```

Change `question` in `rag_demo.py` to try another question.
Each successful answer-generation attempt makes one model request.

### Demonstrated Scenarios

**Answerable question**

- Question: “How can I cancel my booking?”
- Evidence: “You can cancel your reservation from your account.”
- Observed result: an answer supported by the passage, with its source and ID.

**Missing information**

- Question: “What is the cancellation fee?”
- Evidence: the same cancellation passage, which contains no fee.
- Observed result: the model explained that the fee information was missing.

These manual examples demonstrate the workflow; they are not a full
evaluation of answer reliability.

### RAG Limitations

- Documents are hard-coded sample passages.
- PDF filenames are source labels; actual PDFs are not loaded or parsed.
- Retrieval selects one passage even when none adequately answers the question.
- The embedding model is intended primarily for English text.
- Embeddings are held in memory, not persisted in a database.
- Restarting the program reloads the model and recomputes document embeddings.
- Document changes during execution require rebuilding the prepared embeddings.
- Citations and factual support are prompted, not automatically verified.
- RAG-specific automated tests and provider-error handling remain pending.