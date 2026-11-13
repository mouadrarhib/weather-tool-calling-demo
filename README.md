# Weather Tool Calling and RAG Learning Project

A Python learning project covering two AI application workflows:

1. A real LLM requesting a validated weather tool through OpenRouter.
2. A RAG prototype that reads a local policy file, retrieves relevant chunks,
   and generates answers supported by that evidence.

Weather data is fixed sample data. The rental policy is fictional.
Model API calls and locally generated embeddings are real.

## Learning Source

Built while studying
[AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch).

The weather function is loaded from the curriculum repository.
The practice code adds validation, dispatching, tests, live model integration,
document ingestion, semantic retrieval, and a small retrieval evaluation.

## Requirements

- Python 3.10 or newer, compatible with the installed dependencies
- Git
- Internet access for initial downloads and model API requests
- An OpenRouter API key for answer generation

Local retrieval and automated tests do not require a model API key.
Embedding model inference runs on the CPU.

## Project Files

| File | Purpose |
|---|---|
| `lesson.py` | Weather validation, JSON parsing, and tool dispatching |
| `test_lesson.py` | Seven automated tests for tool handling |
| `live_weather.py` | Real model integration with the weather tool |
| `semantic_search_practice.py` | Filtering and distance using invented vectors |
| `real_embeddings.py` | Real embeddings and cosine-similarity examples |
| `data/rental_policy.txt` | Fictional rental policy used as the document source |
| `document_ingestion.py` | Reads the policy file and creates section-based chunks |
| `filtered_semantic_search.py` | Metadata filtering and semantic retrieval over chunks |
| `evaluate_retrieval.py` | Ten manually labelled retrieval checks |
| `rag_demo.py` | Evidence-based answer generation through OpenRouter |
| `.env.example` | API-key configuration template |
| `.gitignore` | Excludes credentials, environments, and local artifacts |

## Setup

### 1. Clone This Project

```bash
git clone https://github.com/mouadrarhib/weather-tool-calling-demo.git
cd weather-tool-calling-demo
```

### 2. Clone the Curriculum

```bash
git clone https://github.com/rohitg00/ai-engineering-from-scratch.git
```

The curriculum clone is required by `lesson.py`, which loads its sample
weather function. The document-retrieval scripts read the local policy file.

Expected structure:

```text
weather-tool-calling-demo/
├── README.md
├── lesson.py
├── test_lesson.py
├── live_weather.py
├── document_ingestion.py
├── filtered_semantic_search.py
├── evaluate_retrieval.py
├── rag_demo.py
├── data/
│   └── rental_policy.txt
├── .env.example
└── ai-engineering-from-scratch/
```

### 3. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

For the weather tool and OpenRouter integration:

```bash
python -m pip install openai python-dotenv
```

For local CPU embeddings and semantic search:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install sentence-transformers
```

Text-file ingestion uses Python's standard library.

### 5. Configure OpenRouter

Copy the configuration template:

```bash
cp .env.example .env
chmod 600 .env
```

Edit `.env` so it contains:

```dotenv
OPENROUTER_API_KEY=your_actual_key_here
```

Replace the placeholder with your key. Never commit or share `.env`.

The live scripts load `.env` from their own directory.

## Weather Tool-Calling Demo

### Workflow

1. Send the user question and tool definition to the model.
2. Receive a tool name and JSON arguments.
3. Parse and validate the request.
4. Clean the city name.
5. Execute the sample weather lookup.
6. Append the tool result with its matching `tool_call_id`.
7. Ask the model to explain the result.

The application maintains one shared conversation list.

### Run Local Examples

```bash
python lesson.py
```

These examples do not call OpenRouter.

### Run the Live Model Integration

```bash
python live_weather.py
```

Change the user question in the shared `messages` list to try another city.

Demonstrated scenarios include:

- Tokyo: sample weather returned and explained.
- London: sample weather returned and explained.
- Atlantis: `CITY_NOT_FOUND` returned and the limitation explained.

The model is instructed to label weather information as sample data.

### Tool Error Codes

| Code | Meaning |
|---|---|
| `INVALID_JSON` | Request or argument text cannot be parsed as JSON |
| `INVALID_TOOL_CALL` | Parsed tool request is not a dictionary |
| `UNKNOWN_TOOL` | Unsupported or missing tool name in the current dispatcher |
| `INVALID_ARGUMENTS` | Arguments are missing or are not a dictionary |
| `INVALID_CITY` | City is missing, blank, or is not a string |
| `CITY_NOT_FOUND` | The sample database has no matching city |
| `WEATHER_UNAVAILABLE` | The weather lookup raised a handled `ConnectionError` |

Error codes are defined by this application.

### Model Provider Failures

The live weather integration handles:

- Rate limits
- Request timeouts
- Connection failures

If the first model request fails, execution stops with a clear message.
If the final request fails after tool execution, the script reports that
the tool completed but the final explanation could not be generated.

A timeout or connection error does not prove that the provider's server is down.

### Automated Tests

```bash
python -m unittest -v test_lesson
```

Seven tests cover:

1. Missing arguments
2. Unknown tool names
3. Blank cities without executing the lookup
4. City cleaning and exactly one lookup call
5. Simulated weather connection failure
6. Malformed argument JSON
7. Arguments that parse as a list instead of a dictionary

Tests use `unittest` and `unittest.mock` and make no model API requests.

All seven tests passed in the demonstrated local run.
Model-provider error handlers have been syntax-checked but are not yet
covered by automated mock tests.

## Document Ingestion and Chunking

### Source Document

The RAG prototype reads:

```text
data/rental_policy.txt
```

This is a fictional policy created for the exercise. It contains three sections:

- Cancellation
- Vehicle maintenance
- Pickup requirements

### Run the Loader

```bash
python document_ingestion.py
```

The reusable `load_policy_chunks()` function:

1. Reads the file as UTF-8 text.
2. Splits sections at blank lines.
3. Keeps each heading with its section.
4. Assigns a chunk ID, source filename, location, and status.
5. Returns a list of chunk dictionaries.

Example chunk:

```python
{
    "id": "rental-policy-01",
    "text": "CANCELLATION\n...",
    "source": "rental_policy.txt",
    "location": "Casablanca",
    "status": "active",
}
```

The current file produces:

| Chunk ID | Section |
|---|---|
| `rental-policy-01` | Cancellation |
| `rental-policy-02` | Vehicle maintenance |
| `rental-policy-03` | Pickup requirements |

Location and status are assigned by the loader's parameters.
They are not extracted from the document.

The cancellation section keeps the fee and its timing conditions together:

- Cancellation is free at least 24 hours before pickup.
- Cancellation less than 24 hours before pickup costs 100 MAD.

### Current Splitter Scope

The splitter uses blank lines and suits this small, structured sample.
It does not yet enforce token limits, add overlap, or handle large sections,
complex tables, or multiple document formats.

Chunk IDs are based on section order and may change if sections are reordered.

## Embeddings and Semantic Search

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model generates 384-dimensional text embeddings locally on the CPU.
The first run downloads the model from Hugging Face.
Later runs normally reuse the downloaded model files.

This model is intended primarily for English text.

### Learning Examples

Invented vectors and Euclidean distance:

```bash
python semantic_search_practice.py
```

Real embeddings and cosine similarity:

```bash
python real_embeddings.py
```

Euclidean distance ranks smaller values first.
Cosine similarity ranks larger values first.

A similarity score is not an accuracy percentage or a probability
that a passage contains the answer.

### Filtered Retrieval

```bash
python filtered_semantic_search.py
```

Retrieval:

1. Loads chunks from `data/rental_policy.txt`.
2. Loads the embedding model.
3. Prepares chunk embeddings once per running process.
4. Filters candidates by location and active status.
5. Generates an embedding for the question.
6. Compares it with eligible chunk vectors.
7. Returns the highest-ranked chunk if it meets the similarity threshold.

The result includes the chunk ID, text, source filename, and similarity score.

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
Local policy file
→ section-based chunks
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

The answer cited `rental_policy.txt`, chunk `rental-policy-01`.

The earlier cancellation-method example also produced a supported answer
explaining that cancellation is available from the customer's account.

### Demonstrated Missing Information

Question:

> Is the cancellation fee refundable?

The cancellation chunk is relevant and specifies the fee, but says nothing
about refundability.

Observed result: the model explained that refundability was unspecified,
without inventing a refund rule, and cited the retrieved chunk.

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

Ten manually labelled cases cover:

- Cancellation questions with different wording
- Vehicle maintenance questions
- A location with no applicable chunks
- The cancellation fee, now supported by the document
- Fee refundability, which remains unspecified
- Unrelated geography and cooking questions

Each case records an expected chunk ID or `None`.
The `answer_supported` field records a human-assigned expectation about
whether the applicable document contains enough information to answer.

### Observed Results

All ten chunk-selection checks passed against the text-file chunks,
with the existing threshold unchanged.

Example observed similarities:

| Question | Selected chunk | Similarity |
|---|---|---:|
| How can I cancel my booking? | `rental-policy-01` | 0.727 |
| What is the cancellation fee? | `rental-policy-01` | 0.696 |
| Are the cars maintained regularly? | `rental-policy-02` | 0.451 |
| Is the cancellation fee refundable? | `rental-policy-01` | 0.666 |

The refundability question has a relatively high relevance score,
but the chunk does not contain its answer.

PASS means the retrieved ID matches the expected ID, or both are `None`.
It does not mean an LLM answer was automatically checked for correctness.

The earlier 9/9 result evaluated the previous hard-coded sample passages.
The current 10/10 result evaluates the file-based chunks.
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
It remained unchanged for the file-based evaluation.

It is not a universal cutoff and requires broader evaluation.
Similarity does not establish answer completeness or correctness.

## Limitations and Pending Work

- Weather information is fixed sample data, not a live forecast.
- The rental policy is fictional and stored in one local text file.
- PDF extraction has not been implemented.
- Chunking relies on blank lines and has no token-size limit or overlap.
- Location and status are assigned metadata, not extracted facts.
- Chunk IDs depend on section order.
- The current retrieval evaluation does not directly test pickup requirements.
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

## Next Steps

- Add PDF text extraction while preserving source and page metadata.
- Evaluate retrieval for pickup requirements and additional question types.
- Add focused ingestion and answer-generation tests.
- Improve chunking and evaluate its effect on retrieval.
- Add persistent vector storage when the document collection grows.