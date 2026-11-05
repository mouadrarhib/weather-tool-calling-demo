import os
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    OpenAI,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)
import json
from lesson import handle_tool_request

SYSTEM_INSTRUCTION = (
    "Weather tools return fixed demonstration data, not live weather. "
    "Clearly label weather answers as sample data. "
    "For weather questions, call get_weather before answering. "
    "Do not infer whether a city is supported from its name. "
    "After receiving a tool result, answer using that result. "
    "If the tool returns CITY_NOT_FOUND, explain that the sample "
    "weather database has no data for that city."
)

# Load .env from the same folder as this script.
env_path = Path(__file__).resolve().parent / ".env"

if not env_path.is_file():
    raise RuntimeError(f"Cannot find .env file at: {env_path}")

# Prefer the saved .env value over an old terminal value.
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

if not api_key:
    raise RuntimeError(
        "The .env file must contain: "
        "OPENROUTER_API_KEY=your_actual_key_here"
    )



client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    max_retries=0,
    timeout=60.0,
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get sample weather data for a supported city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, such as Tokyo.",
                    }
                },
                "required": ["city"],
            },
        },
    }
]

messages = [
    {
        "role": "system",
        "content": SYSTEM_INSTRUCTION,
    },
    {
        "role": "user",
        "content": "What is the weather in Atlantis?",
    },
]

try:
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
        tools=tools,
        max_tokens=500,
    )
except RateLimitError:
    raise SystemExit("Model rate limit reached. Please try later.")
except APITimeoutError:
    raise SystemExit("The model request timed out. Please try later.")
except APIConnectionError:
    raise SystemExit("Could not connect to the model provider.")

message = response.choices[0].message

print("Model:", response.model)

messages.append(message.model_dump(exclude_none=True))

if message.tool_calls:
    for call in message.tool_calls:

        result = handle_tool_request(
            call.function.name,
            call.function.arguments,
        )
        print("Tool result:", result)

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result),
        })

    print("Conversation roles:", [item["role"] for item in messages])

    try:
        final_response = client.chat.completions.create(
            model=response.model,
            messages=messages,
            max_tokens=500,
        )
    except RateLimitError:
        print("The tool completed, but the model is rate-limited.")
    except APITimeoutError:
        print("The tool completed, but the final model request timed out.")
    except APIConnectionError:
        print("The tool completed, but the model provider could not be reached.")
    else:
        print("Final answer:", final_response.choices[0].message.content)
else:
    print("Answer:", message.content)
