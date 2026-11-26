
"""
Groq LLM judge integration for DeepEval.

Generator: Ollama
Evaluation judge: Groq
Framework: DeepEval 4.3.6
"""

import os
from typing import Optional

from dotenv import load_dotenv
from groq import Groq, AsyncGroq
from pydantic import BaseModel
from deepeval.models import DeepEvalBaseLLM


# Load environment variables from project .env
load_dotenv()


class GroqJudge(DeepEvalBaseLLM):

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        api_key: Optional[str] = None,
        temperature: float = 0
    ):
        self.model_name = model_name
        self.temperature = temperature

        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found. "
                "Please configure it in your project's .env file."
            )

        self.client = Groq(api_key=self.api_key)
        self.async_client = AsyncGroq(api_key=self.api_key)

    def load_model(self):
        """Return the initialized Groq client."""
        return self.client

    def get_model_name(self):
        """Return the configured judge model name."""
        return self.model_name

    def generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None
    ) -> str:
        """Generate an evaluation response synchronously."""

        request = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise and objective LLM evaluation judge. "
                        "Follow the evaluation instructions carefully. "
                        "Return valid JSON when JSON output is requested."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature
        }

        if schema is not None:
            request["response_format"] = {
                "type": "json_object"
            }

        response = self.client.chat.completions.create(**request)

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError("Groq returned an empty response.")

        return content

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None
    ) -> str:
        """Generate an evaluation response asynchronously."""

        request = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise and objective LLM evaluation judge. "
                        "Follow the evaluation instructions carefully. "
                        "Return valid JSON when JSON output is requested."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature
        }

        if schema is not None:
            request["response_format"] = {
                "type": "json_object"
            }

        response = await self.async_client.chat.completions.create(
            **request
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError("Groq returned an empty response.")

        return content

    def get_model(self):
        """Return the underlying Groq client."""
        return self.client
