from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

from filtered_semantic_search import retrieve_passage

from fastapi import HTTPException
from openai import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
)

from rag_demo import answer_question

app = FastAPI(title="Rental Policy RAG API")


class SearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)
    location: str = Field(
        default="Casablanca",
        min_length=1,
        max_length=100,
    )


class Passage(BaseModel):
    id: str
    text: str
    source: str
    page: int
    similarity: float


class SearchResponse(BaseModel):
    found: bool
    passage: Passage | None


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    passage = retrieve_passage(
        question=request.question,
        location=request.location,
    )

    return {
        "found": passage is not None,
        "passage": passage,
    }


class AskResponse(BaseModel):
    found: bool
    answer: str
    passage: Passage | None


@app.post("/ask", response_model=AskResponse)
def ask(request: SearchRequest):
    try:
        return answer_question(
            question=request.question,
            location=request.location,
        )
    except RateLimitError:
        raise HTTPException(
            status_code=503,
            detail="The model provider is rate-limited. Please try later.",
        ) from None
    except APITimeoutError:
        raise HTTPException(
            status_code=504,
            detail="The model provider did not respond in time.",
        ) from None
    except APIConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to the model provider.",
        ) from None
    except APIStatusError:
        raise HTTPException(
            status_code=502,
            detail="The model provider returned an unsuccessful response.",
        ) from None
    except RuntimeError:
        raise HTTPException(
            status_code=500,
            detail="Answer generation is unavailable. Check server configuration and logs.",
        ) from None