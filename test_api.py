import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app

import httpx
from openai import APITimeoutError
from openai import APIConnectionError
from openai import APIConnectionError, APITimeoutError, RateLimitError

import rag_demo
from rag_demo import answer_question




class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_blank_question_is_rejected(self):
        with patch("api.retrieve_passage") as mock_retrieve:
            response = self.client.post(
                "/search",
                json={"question": "   ", "location": "Casablanca"},
            )

        self.assertEqual(response.status_code, 422)
        mock_retrieve.assert_not_called()

    def test_ask_returns_answer_and_evidence(self):
        expected_result = {
            "found": True,
            "answer": "Cancellation costs 100 MAD within 24 hours of pickup.",
            "passage": {
                "id": "rental-pdf-01",
                "text": "Cancellations less than 24 hours before pickup cost 100 MAD.",
                "source": "rental_policy.pdf",
                "page": 1,
                "similarity": 0.7,
            },
        }

        with patch(
            "api.answer_question",
            return_value=expected_result,
        ) as mock_answer:
            response = self.client.post(
                "/ask",
                json={
                    "question": "What is the cancellation fee?",
                    "location": "Casablanca",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected_result)
        mock_answer.assert_called_once_with(
            question="What is the cancellation fee?",
            location="Casablanca",
    )

    def test_ask_provider_timeout_returns_504(self):
        timeout = APITimeoutError(
            request=httpx.Request(
                "POST",
                "https://openrouter.ai/api/v1/chat/completions",
            )
        )

        with patch("api.answer_question", side_effect=timeout):
            response = self.client.post(
                "/ask",
                json={
                    "question": "What is the cancellation fee?",
                    "location": "Casablanca",
                },
            )

        self.assertEqual(response.status_code, 504)
        self.assertEqual(
            response.json()["detail"],
            "The model provider did not respond in time.",
        )

    def test_ask_connection_failure_returns_503(self):
        error = APIConnectionError(
            request=httpx.Request(
                "POST",
                "https://openrouter.ai/api/v1/chat/completions",
            )
        )

        with patch("api.answer_question", side_effect=error):
            response = self.client.post(
                "/ask",
                json={
                    "question": "What is the cancellation fee?",
                    "location": "Casablanca",
                },
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "Could not connect to the model provider.",
        )


    def test_ask_rate_limit_returns_503(self):
        provider_response = httpx.Response(
            status_code=429,
            request=httpx.Request(
                "POST",
                "https://openrouter.ai/api/v1/chat/completions",
            ),
        )

        error = RateLimitError(
            "Rate limit reached",
            response=provider_response,
            body={"error": {"message": "Rate limit reached"}},
        )

        with patch("api.answer_question", side_effect=error):
            response = self.client.post(
                "/ask",
                json={
                    "question": "What is the cancellation fee?",
                    "location": "Casablanca",
                },
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "The model provider is rate-limited. Please try later.",
        )

    def test_no_evidence_skips_model_client(self):
        with (
            patch.object(rag_demo, "retrieve_passage", return_value=None),
            patch.object(rag_demo, "OpenAI") as mock_client,
        ):
            result = answer_question("How do I bake a chocolate cake?")

        self.assertFalse(result["found"])
        self.assertIsNone(result["passage"])
        mock_client.assert_not_called()