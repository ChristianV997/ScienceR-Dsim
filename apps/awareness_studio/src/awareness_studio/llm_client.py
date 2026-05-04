from __future__ import annotations

from typing import Protocol

from awareness_studio import config


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicClient:
    def __init__(
        self,
        model: str = config.LLM_MODEL,
        api_key: str = config.LLM_API_KEY,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package not found. Install with: pip install anthropic"
            ) from exc
        self._client = __import__("anthropic").Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=config.LLM_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


def get_llm_client() -> LLMClient:
    provider = config.LLM_PROVIDER
    if provider == "anthropic":
        return AnthropicClient()
    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r}. Supported values: 'anthropic'."
    )
