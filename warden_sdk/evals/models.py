"""The LLM behind judges and simulated users.

Anything with a `name` and a `json()` method works, so tests can use a fake
and other providers can be plugged in. ClaudeModel is the default.
"""

import json
from typing import Any, Protocol

DEFAULT_MODEL = "claude-opus-5-5"


class ModelError(Exception):
    """The model gave no usable answer (refused, truncated, or not valid JSON)."""


class Model(Protocol):
    # Recorded with every score, so a change of model shows up as a new scorer version.
    name: str

    def json(self, system: str, messages: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
        """Return the model's reply as a dict matching `schema` (a JSON Schema object)."""
        ...


class ClaudeModel:
    """Claude through the Anthropic SDK, with replies constrained to a JSON schema.

    Credentials come from the environment (ANTHROPIC_API_KEY, or an `ant auth
    login` profile). Refusal fallbacks are deliberately left off: a fallback
    would grade some rows with a different model, and a comparison between runs
    only holds if the same judge graded both.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "low",
        max_tokens: int = 16000,
        client: Any = None,
        lazy: bool = False,
    ):
        """lazy=True defers creating the SDK client (and so needing credentials) until the first call."""
        self.name = f"{model}@{effort}"
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self._client = client if client is not None or lazy else self._new_client()

    @staticmethod
    def _new_client() -> Any:
        import anthropic  # imported here so the SDK is only needed when a Claude model is used

        return anthropic.Anthropic()

    def json(self, system: str, messages: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
        import anthropic

        if self._client is None:
            self._client = self._new_client()
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=messages,
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
            )
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
            # The SDK already retried these; let the runner decide what an infra failure means.
            raise ModelError(f"{type(e).__name__}: {e}") from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise ModelError(f"API error {e.status_code}") from e
            raise  # 4xx other than 429 is a bug in the request, not a flaky judge
        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            raise ModelError(f"model refused ({category or 'no category'})")
        if response.stop_reason == "max_tokens":
            raise ModelError("reply was cut off at max_tokens")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise ModelError("reply had no text")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ModelError(f"reply was not valid JSON: {e}") from e


_default: Model | None = None


def default_model() -> Model:
    """A shared ClaudeModel, created on first use."""
    global _default
    if _default is None:
        _default = ClaudeModel()
    return _default
