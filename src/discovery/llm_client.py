"""
Thin wrapper around the Anthropic Messages API.

Reads ANTHROPIC_API_KEY from the environment (or a .env file in the project
root).  Retries on rate-limit and server-overload errors with exponential
back-off before re-raising.
"""

import os
import time
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

# Load .env once at import time (safe to call multiple times)
_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_env_path)


class LLMError(Exception):
    pass


class LLMClient:
    """
    Parameters
    ----------
    model   : Anthropic model ID, e.g. ``"claude-sonnet-4-6"``.
    api_key : Override the ANTHROPIC_API_KEY environment variable.
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> str:
        """
        Call the Messages API and return the assistant's text response.

        Retries up to 3 times on RateLimitError or 5xx server errors with
        exponential back-off (5 s, 10 s, 20 s).
        """
        last_exc: Optional[Exception] = None
        for attempt in range(4):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    temperature=temperature,
                )
                return response.content[0].text
            except anthropic.RateLimitError as exc:
                last_exc = exc
                time.sleep(5 * (2 ** attempt))
            except anthropic.APIStatusError as exc:
                if exc.status_code in (500, 503, 529):
                    last_exc = exc
                    time.sleep(2 ** attempt)
                else:
                    raise LLMError(f"Anthropic API error {exc.status_code}: {exc}") from exc
        raise LLMError(f"API call failed after retries: {last_exc}") from last_exc
