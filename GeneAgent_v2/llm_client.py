"""OpenAI Responses API helpers used by GeneAgent."""

from __future__ import annotations

import copy
from typing import Any

from config import require_openai_settings


def get_openai_client() -> Any:
    """Create a client lazily so imports never require credentials."""

    settings = require_openai_settings()
    if settings.provider == "azure":
        from openai import AzureOpenAI

        return AzureOpenAI(
            api_key=settings.api_key,
            azure_endpoint=settings.endpoint,
            api_version=settings.api_version,
        )

    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": settings.api_key}
    if settings.endpoint:
        kwargs["base_url"] = settings.endpoint
    return OpenAI(**kwargs)


def response_tool_schema(function_doc: dict[str, Any]) -> dict[str, Any]:
    """Convert the old Chat Completions function schema to Responses tools."""

    if function_doc.get("type") == "function":
        return function_doc

    parameters = copy.deepcopy(
        function_doc.get("parameters", {"type": "object", "properties": {}})
    )
    return {
        "type": "function",
        "name": function_doc["name"],
        "description": function_doc.get("description", ""),
        "parameters": parameters,
        "strict": False,
    }


def item_get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def response_text(response: Any) -> str:
    """Extract text from a Responses API result with a small compatibility shim."""

    text = getattr(response, "output_text", None)
    if text:
        return text

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if item_get(item, "type") != "message":
            continue
        for content in item_get(item, "content", []) or []:
            if item_get(content, "type") in {"output_text", "text"}:
                value = item_get(content, "text", "")
                if value:
                    parts.append(value)
    return "\n".join(parts)


def function_calls(response: Any) -> list[Any]:
    return [
        item
        for item in (getattr(response, "output", []) or [])
        if item_get(item, "type") == "function_call"
    ]


def replayable_output_items(response: Any) -> list[Any]:
    """Return output items that can be safely replayed as next-turn input."""

    from config import get_openai_settings

    settings = get_openai_settings()
    replay_items: list[Any] = []
    for item in getattr(response, "output", []) or []:
        if (
            item_get(item, "type") == "reasoning"
            and not settings.store_responses
            and not item_get(item, "encrypted_content")
        ):
            continue
        replay_items.append(item)
    return replay_items


def build_response_kwargs(
    *,
    instructions: str,
    input_items: Any,
    model: str,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from config import get_openai_settings

    settings = get_openai_settings()
    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_items,
        "store": settings.store_responses,
    }
    if settings.reasoning_effort:
        kwargs["reasoning"] = {"effort": settings.reasoning_effort}
        if tools and not settings.store_responses:
            # GPT-5 reasoning items must be replayed with function outputs.
            # With store=False, encrypted_content makes that stateless replay valid.
            kwargs["include"] = ["reasoning.encrypted_content"]
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return kwargs


def generate_text(
    *,
    instructions: str,
    input_items: str | list[dict[str, str]],
    model: str | None = None,
    client: Any | None = None,
) -> str:
    """Generate text through the Responses API."""

    from config import get_openai_settings

    settings = get_openai_settings()
    active_client = client or get_openai_client()
    response = active_client.responses.create(
        **build_response_kwargs(
            instructions=instructions,
            input_items=input_items,
            model=model or settings.model,
        )
    )
    return response_text(response)
