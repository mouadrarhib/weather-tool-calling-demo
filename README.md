# Weather Tool Calling and RAG Learning Project

A Python learning project covering two AI application workflows:

1. A real LLM requesting a validated weather tool through OpenRouter.
2. A RAG prototype that retrieves sample policy passages and generates
   answers supported by that evidence.

Weather data and document passages are sample data.
Model API calls and locally generated embeddings are real.

## Learning Source

Built while studying
[AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch).

The weather function is loaded from the curriculum repository.
The practice code adds validation, dispatching, tests, live model integration,
semantic retrieval, and a small retrieval evaluation.

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
| `lesson.py` | Weather input validation, JSON parsing, and tool dispatching |
| `test_lesson.py` | Seven automated tests for tool handling |
| `live_weather.py` | Real model integration with the weather tool |
| `semantic_search_practice.py` | Filtering and distance using invented vectors |
| `real_embeddings.py` | Real embeddings and cosine-similarity examples |
| `filtered_semantic_search.py` | Metadata filtering and semantic retrieval |
| `evaluate_retrieval.py` | Nine manually labelled retrieval checks |
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
weather function.

Expected structure:

```text
weather-tool-calling-demo/
├── README.md
├── lesson.py
├── test_lesson.py
├── live_weather.py
├── filtered_semantic_search.py
├── evaluate_retrieval.py
├── rag_demo.py
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

1. Loads the embedding model.
2. Prepares document embeddings once per running process.
3. Filters document candidates by location and active status.
4. Generates an embedding for the question.
5. Compares it with eligible document vectors.
6. Returns the highest-ranked passage if it meets the similarity threshold.

The result includes the passage ID, text, source label, and similarity score.

Document embeddings stay in memory. They are not persisted to a vector database.
Restarting the program loads the model and recomputes document embeddings.

Document or embedding-model changes require rebuilding the corresponding vectors.

## RAG Prototype

RAG means retrieval-augmented generation: retrieve relevant evidence,
then provide it to a model when generating an answer.

### Workflow

```text
Question
→ filter applicable documents
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

### Demonstrated Answerable Question

Question:

> How can I cancel my booking?

Evidence:

> You can cancel your reservation from your account.

Observed result: an answer supported by that passage, citing
`casablanca-active.pdf` and passage ID `policy-01`.

### Demonstrated Missing Information

Question:

> What is the cancellation fee?

The same passage is relevant to cancellation but contains no fee amount.

Observed result: the model acknowledged that the supplied evidence
does not specify the fee.

Relevance and answerability are separate.
A related passage does not necessarily support an answer.

### No-Evidence Behaviour

For an unrelated question, such as:

> How do I bake a chocolate cake?

Retrieval rejects the low-similarity result and the script displays:

```text
No sufficiently relevant evidence was found in the applicable documents.
```

No OpenRouter request is made on this path.

## Retrieval Evaluation

Run:

```bash
python evaluate_retrieval.py
```

This evaluation runs locally without calling OpenRouter.

### Dataset

Nine manually labelled cases cover:

- Cancellation questions with different wording
- Vehicle maintenance questions
- A location with no applicable documents
- A cancellation-fee question whose answer is missing
- Unrelated geography and cooking questions

Each case records an expected passage ID or `None`.
The `answer_supported` field records a human-assigned expectation about
whether the applicable sample documents contain the answer.

### Observed Results

All nine passage-selection checks passed in the demonstrated run.

Three additional questions were tested without changing the threshold.
This remains a tiny development evaluation, not evidence of general reliability.

PASS means the retrieved ID matches the expected ID, or both are `None`.
It does not mean an LLM answer was checked for correctness.

### Similarity Threshold

`retrieve_passage` defaults to:

```python
min_similarity=0.3
```

It returns `None` when:

- No documents match the metadata filters, or
- The highest similarity is below the threshold.

The threshold was selected using a small development sample.
It is not a universal cutoff and requires broader evaluation.

The cancellation-fee question can remain above the threshold while
still lacking enough evidence for an answer.

## Limitations and Pending Work

- Weather information is fixed sample data, not a live forecast.
- Documents are hard-coded sample passages.
- PDF filenames are source labels; actual PDFs are not loaded or parsed.
- Retrieval returns at most one passage.
- The provisional similarity threshold needs broader evaluation.
- Embeddings are stored in memory rather than persisted in a database.
- The sample embedding model is primarily intended for English.
- Citations and factual support are prompted, not automatically verified.
- The nine retrieval checks do not evaluate generated-answer quality.
- End-to-end RAG automated tests remain pending.
- RAG provider-error handling remains pending.
- Weather model-provider error handlers still need automated mock tests.
- Free-provider availability, rate limits, and model selection can vary.
- The weather implementation supports one tool-execution round followed
  by a final model response, not an unrestricted agent loop.
- Runtime dependencies and the curriculum revision are not pinned.