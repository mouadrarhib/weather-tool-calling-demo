The result includes the chunk ID, text, source filename, page number, and similarity score.

Embeddings stay in memory. They are not persisted to a vector database.
Restarting the program reloads the document and recomputes its embeddings.

Editing the file while the process is running does not automatically refresh
the prepared chunks or vectors. Reload them after document changes.
Changing the embedding model also requires rebuilding the corresponding vectors.

## RAG Prototype

RAG means retrieval-augmented generation: retrieve relevant evidence,
then provide it to a model when generating an answer.

### Workflow

```text
Local policy PDF
→ text extraction and header/footer cleanup
→ page-aware chunks
→ chunk embeddings prepared in memory

User question
→ filter applicable chunks
→ compare embeddings
→ select relevant evidence
→ send question and evidence to the LLM
→ answer with a source or acknowledge missing information
```

### Run

```bash
python rag_demo.py
```

Change `question` in `rag_demo.py` to try another question.

When evidence is selected, the script makes one OpenRouter model request.
When retrieval returns `None`, it skips answer generation.

### Demonstrated Supported Answer

Question:

> What is the cancellation fee?

Retrieved evidence includes both the free-cancellation period and fee.

Observed answer:

- Free if cancelled at least 24 hours before pickup.
- 100 MAD if cancelled less than 24 hours before pickup.

The PDF-based answer cited `rental_policy.pdf`, page 1, chunk `rental-pdf-01`.

The earlier cancellation-method example also produced a supported answer
explaining that cancellation is available from the customer's account.

### Demonstrated Missing Information

Question:

> Is the cancellation fee refundable?

The cancellation chunk is relevant and specifies the fee, but says nothing
about refundability.

Observed with the earlier text-file pipeline: the model explained that
refundability was unspecified, without inventing a refund rule, and cited
the retrieved chunk. The PDF contains the same cleaned cancellation text;
a PDF-based refundability generation check has not yet been demonstrated.

Relevance and answerability are separate.
A related chunk does not necessarily support the requested answer.

### No-Evidence Behaviour

For unrelated questions, retrieval can reject low-similarity candidates.
When retrieval returns `None`, the script displays:

```text
No sufficiently relevant evidence was found in the applicable documents.
```

No OpenRouter request is made on this path.

This behaviour was demonstrated with an unrelated cooking question before
the move to file-based chunks. The updated local retrieval evaluation also
checks unrelated cooking and geography questions.

## Retrieval Evaluation

Run:

```bash
python evaluate_retrieval.py
```

This evaluation runs locally without calling OpenRouter.

### Current Dataset

Eleven manually labelled cases cover:

- Cancellation questions with different wording
- Vehicle maintenance questions
- Pickup identification and driving-licence requirements
- A location with no applicable chunks
- The cancellation fee, now supported by the document
- Fee refundability, which remains unspecified
- Unrelated geography and cooking questions

Each case records an expected chunk ID and expected page number, or `None`
for both when no result is expected.
The `answer_supported` field records a human-assigned expectation about
whether the applicable document contains enough information to answer.

### Observed Results

All eleven checks passed against the PDF chunks, verifying both the retrieved
chunk ID and page number. The existing threshold remained unchanged.

Example observed similarities:

| Question | Selected chunk | Similarity |
|---|---|---:|
| How can I cancel my booking? | `rental-pdf-01` | 0.727 |
| What is the cancellation fee? | `rental-pdf-01` | 0.696 |
| Are the cars maintained regularly? | `rental-pdf-02` | 0.451 |
| Is the cancellation fee refundable? | `rental-pdf-01` | 0.666 |
| What documents should I bring when picking up the car? | `rental-pdf-03` | 0.559 |

The refundability question has a relatively high relevance score,
but the chunk does not contain its answer.

PASS requires both the retrieved ID and page to match their expected values,
including `None` for no-result cases.
It does not mean an LLM answer was automatically checked for correctness.

The earlier 9/9 result evaluated the previous hard-coded sample passages.
The intermediate 10/10 result evaluated text-file chunks.
The current 11/11 result evaluates PDF chunk IDs and page metadata.
These are different dataset versions and are not directly comparable
as a measure of improvement.

The evaluation is small and development-oriented.
It does not establish general RAG reliability.

### Similarity Threshold

`retrieve_passage` defaults to:

```python
min_similarity=0.3
```

It returns `None` when:

- No chunks match the metadata filters, or
- The highest similarity is below the threshold.

The threshold was originally selected using the earlier sample passages.
It remained unchanged for the text-file and PDF evaluations.

It is not a universal cutoff and requires broader evaluation.
Similarity does not establish answer completeness or correctness.

## Limitations and Pending Work

- Weather information is fixed sample data, not a live forecast.
- The rental policy is fictional; current retrieval uses one small text-based PDF.
- PDF cleanup is specific to known headers and footers in this practice file.
- PDF chunking uses one page per chunk; the earlier text loader splits on blank lines.
- Neither splitter enforces token limits or adds overlap.
- OCR, encrypted-document handling, and complex layout/table reconstruction are not implemented.
- Location and status are assigned metadata, not extracted facts.
- Chunk IDs depend on page order for PDFs and section order for the text loader.
- Page metadata checks verify retrieval output, not every generated citation.
- Retrieval returns at most one chunk.
- The provisional similarity threshold needs broader evaluation.
- Embeddings are held in memory rather than persisted in a database.
- The embedding model is primarily intended for English.
- Citations and factual support are prompted, not automatically verified.
- Retrieval checks do not automatically evaluate generated-answer quality.
- Automated ingestion tests and end-to-end RAG tests remain pending.
- RAG provider-error handling remains pending.
- Weather model-provider error handlers still need automated mock tests.
- Free-provider availability, rate limits, and model selection can vary.
- The weather implementation supports one tool-execution round followed
  by a final model response, not an unrestricted agent loop.
- Runtime dependencies and the curriculum revision are not pinned.