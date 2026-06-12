"""Optional gene-set meaningfulness detector."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from config import get_openai_settings
from llm_client import (
    build_response_kwargs,
    function_calls,
    get_openai_client,
    item_get,
    replayable_output_items,
    response_text,
    response_tool_schema,
)

from apis.get_enrichment_for_gene_set import (
    get_enrichment_for_gene_set,
    get_enrichment_for_gene_set_doc,
)
from apis.get_pathway_for_gene_set import (
    get_pathway_for_gene_set,
    get_pathway_for_gene_set_doc,
)


FunctionInfo = tuple[Callable[..., Any], dict[str, Any]]

func2info: dict[str, FunctionInfo] = {
    "get_enrichment_for_gene_set": (
        get_enrichment_for_gene_set,
        get_enrichment_for_gene_set_doc,
    ),
    "get_pathway_for_gene_set": (
        get_pathway_for_gene_set,
        get_pathway_for_gene_set_doc,
    ),
}


class AgentDetect:
    """Detect whether a gene set has coherent enrichment/pathway signal."""

    def __init__(
        self,
        function_names: list[str],
        *,
        model: str | None = None,
        max_turns: int = 5,
    ) -> None:
        unknown = sorted(set(function_names) - set(func2info))
        if unknown:
            raise ValueError(f"Unknown detector tool names: {', '.join(unknown)}")

        self.model = model
        self.max_turns = max_turns
        self.name2function = {
            function_name: func2info[function_name][0]
            for function_name in function_names
        }
        self.function_docs = [
            response_tool_schema(func2info[function_name][1])
            for function_name in function_names
        ]
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = get_openai_client()
        return self._client

    def inference(self, genes: str) -> str:
        settings = get_openai_settings()
        model = self.model or settings.detection_model
        instructions = """
You are a gene-set coherence detector.
Your job is to decide whether an input gene set shows a coherent biological signal using enrichment and pathway evidence.

Decision standard:
Meaningful means the tools return one or more significant terms with coherent biological themes and reasonable overlap with the input genes.
Possibly meaningful means there is some coherent evidence but it is weak, broad, sparse, or driven by a small subset.
Unrelated means there is no significant/coherent enrichment or pathway signal, or the returned evidence is too generic to support a shared function.
Too long to process means the returned evidence cannot be summarized safely within the response budget.

Final output contract:
Start the final response with exactly "Decision:".
After "Decision:", write one compact paragraph in this shape:
Label=<Meaningful|Possibly meaningful|Unrelated|Too long to process>. Evidence=<specific enrichment/pathway terms, overlapping genes, databases, and p-values when available>. Rationale=<why this supports or fails to support gene-set coherence>.
Use plain text only. Do not use markdown, bullets, numbering, or tables.
"""
        input_items: list[Any] = [
            {
                "role": "user",
                    "content": f"""
Gene set to evaluate:
{genes}

Detection task:
Use the available enrichment and pathway tools to evaluate biological coherence.
Consider term specificity, p-value significance, database/source, overlap size, and whether overlapping genes represent a substantial portion of the input.
Do not judge coherence from gene symbols alone when tool evidence is available.
If the tools return errors or empty results, report limited evidence rather than inventing a process.
""",
            }
        ]

        for _ in range(self.max_turns):
            time.sleep(1)
            response = self.client.responses.create(
                **build_response_kwargs(
                    instructions=instructions,
                    input_items=input_items,
                    model=model,
                    tools=self.function_docs,
                )
            )

            calls = function_calls(response)
            if calls:
                input_items.extend(replayable_output_items(response))
                for tool_call in calls:
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": item_get(tool_call, "call_id"),
                            "output": self._run_tool_call(tool_call),
                        }
                    )
                continue

            text = response_text(response).strip()
            if "Decision:" in text:
                return text.split("Decision:", 1)[1].strip()

            input_items.append(
                {
                    "role": "user",
                    "content": 'Return the final detection now using the required format that starts with "Decision:".',
                }
            )

        return "Failed."

    def _run_tool_call(self, tool_call: Any) -> str:
        function_name = item_get(tool_call, "name")
        raw_arguments = item_get(tool_call, "arguments", "{}")
        try:
            function_params = json.loads(raw_arguments or "{}")
            function_to_call = self.name2function[function_name]
            function_response = function_to_call(**function_params)
            return json.dumps(
                {
                    "status": "ok",
                    "function": function_name,
                    "params": function_params,
                    "result": function_response,
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            return json.dumps(
                {
                    "status": "error",
                    "function": function_name,
                    "arguments": raw_arguments,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
