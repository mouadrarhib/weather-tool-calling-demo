# Weather Tool-Calling Demo

A Python learning project that connects a real LLM through OpenRouter
to a validated weather tool.

The LLM calls are real. Weather information comes from the fixed sample
database in the AI Engineering from Scratch curriculum.

## How it works

1. Send the user question and tool definition to the model.
2. Receive a tool request containing a name and JSON arguments.
3. Parse and validate the request.
4. Clean the city name and execute the weather lookup.
5. Send the tool result back with its matching tool_call_id.
6. Ask the model to explain the result.

## Files

- `lesson.py`: input validation, JSON parsing, and tool dispatching.
- `test_lesson.py`: seven automated tests.
- `live_weather.py`: OpenRouter integration and conversation handling.
- `.env.example`: API-key configuration template.

## Requirements

- Python 3.10 or newer
- Git
- An OpenRouter API key
- Internet access for model requests

## Setup

Clone this project and enter its directory. Then clone the curriculum
into the project folder:

```bash
git clone https://github.com/rohitg00/ai-engineering-from-scratch.git
```

The expected structure is:

```text
project/
├── lesson.py
├── test_lesson.py
├── live_weather.py
└── ai-engineering-from-scratch/
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install openai python-dotenv
```

Create a `.env` file beside `live_weather.py`:

```dotenv
OPENROUTER_API_KEY=your_key_here
```

Replace the placeholder locally. Never commit the real key.

## Run

Run the local examples without model API calls:

```bash
python lesson.py
```

Run the real model integration:

```bash
python live_weather.py
```

Change the user question in the shared `messages` list to try another city.

## Tests

```bash
python -m unittest -v test_lesson
```

Seven tests cover:

- Missing arguments
- Unknown tools
- Blank cities without executing the lookup
- City cleaning and exactly one lookup call
- Simulated weather connection failure
- Malformed argument JSON
- Arguments that parse as a list instead of a dictionary

Tests use mocks and do not call the model API.

## Error handling

The dispatcher returns structured errors with stable codes.
The live integration handles model rate limits, connection failures,
and timeouts around both model calls.

Provider-error handlers have been syntax-checked but are not yet
covered by automated mock tests.

## Limitations

- Weather data is sample data, not a live forecast.
- Free-model availability and rate limits may interrupt requests.
- The free router may select different models between runs.
- This implementation supports one tool-execution round followed by
  a final model response.
- Prompt instructions do not guarantee correct model behaviour.
- Runtime dependencies and the curriculum revision are not pinned yet.

## Learning source

Built while studying
[AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch).

The weather implementation is loaded from that repository.
The surrounding practice code adds validation, dispatching, tests,
and a live model integration.