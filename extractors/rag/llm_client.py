"""
LLMClient interface (LLD section 15 — schema-constrained structured output).

Same pattern as every other external-service boundary in this project
(GrobidClient in Step 3, EmbeddingModel/NerModel in Steps 4/6): an
interface, a real implementation, and an offline-testable fake. The
difference here is `api.anthropic.com` IS reachable from this sandbox's
network allow-list — but there's no API key configured in this
environment, so `AnthropicLLMClient` still can't be exercised by the test
suite. It's written to the real SDK regardless, ready to run wherever a key
is available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMOutputInvalid(RuntimeError):
    """The model responded, but its output didn't validate against the
    requested schema. Retryable — a reworded prompt or another sample often
    succeeds (LLD section 26: retry transient errors)."""


class LLMTransportError(RuntimeError):
    """Network/timeout/5xx talking to the LLM provider. Also retryable."""


class LLMExtractionError(RuntimeError):
    """Raised by call_with_retry once retries are exhausted. Permanent for
    this call — the caller should skip this field, not retry indefinitely
    (LLD section 26: don't retry permanent errors forever)."""


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, response_model: Type[T]) -> T:
        """Return an instance of `response_model` built from the LLM's
        response. Implementations must raise LLMOutputInvalid if the
        response doesn't validate, and LLMTransportError for network-level
        failures — call_with_retry depends on that distinction."""
        ...


def call_with_retry(
    client: LLMClient,
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T],
    max_retries: int = 2,
) -> T:
    last_exc: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            return client.complete(system_prompt, user_prompt, response_model)
        except (LLMOutputInvalid, LLMTransportError) as exc:
            last_exc = exc
            continue
    raise LLMExtractionError(
        f"LLM extraction failed after {max_retries + 1} attempt(s) for {response_model.__name__}"
    ) from last_exc


class AnthropicLLMClient(LLMClient):
    """
    Production client. Uses tool-use to force schema-constrained output:
    the response schema is passed as a single tool's input_schema, and
    `tool_choice` forces the model to call it, so the response is
    structurally guaranteed to match `response_model`'s JSON schema shape
    (values can still be semantically wrong, which is what evaluation in
    Step 11 is for).

    Not exercised by the test suite — there's no API key configured in this
    environment. Swap this in for FakeLLMClient wherever one is available;
    both implement `LLMClient`.
    """

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None):
        import anthropic  # local import: optional heavy dep

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.name = model

    def complete(self, system_prompt: str, user_prompt: str, response_model: Type[T]) -> T:
        import anthropic

        tool_name = "submit_extraction"
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                tools=[
                    {
                        "name": tool_name,
                        "description": f"Submit the extracted {response_model.__name__} data.",
                        "input_schema": response_model.model_json_schema(),
                    }
                ],
                tool_choice={"type": "tool", "name": tool_name},
            )
        except anthropic.APIError as exc:
            raise LLMTransportError(str(exc)) from exc

        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use is None:
            raise LLMOutputInvalid("model did not call the extraction tool")

        try:
            return response_model.model_validate(tool_use.input)
        except ValidationError as exc:
            raise LLMOutputInvalid(str(exc)) from exc


class FakeLLMClient(LLMClient):
    """
    Offline, deterministic, scriptable. Each call to `complete()` pops the
    next item off a queue:
      - a dict            -> validated against `response_model`
                              (raises LLMOutputInvalid on failure, exactly
                              like the real client would)
      - an Exception      -> raised as-is
      - a BaseModel        -> returned directly

    Records every call in `.calls` so tests can assert on what was actually
    asked (which query, which chunks made it into the prompt).
    """

    def __init__(self, scripted_responses: list):
        self._queue = list(scripted_responses)
        self.calls: list[tuple[str, str, Type[BaseModel]]] = []

    def complete(self, system_prompt: str, user_prompt: str, response_model: Type[T]) -> T:
        self.calls.append((system_prompt, user_prompt, response_model))
        if not self._queue:
            raise LLMTransportError("FakeLLMClient: scripted response queue exhausted")

        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, response_model):
            return item
        try:
            return response_model.model_validate(item)
        except ValidationError as exc:
            raise LLMOutputInvalid(str(exc)) from exc
