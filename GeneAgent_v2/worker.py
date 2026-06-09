"""Tool-using claim verification agent."""

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

from apis.get_complex_for_gene_set import (
    get_complex_for_gene_set,
    get_complex_for_gene_set_doc,
)
from apis.get_complex_names_for_complex_ids import (
    get_complex_names_for_complex_ids,
    get_complex_names_for_complex_ids_doc,
)
from apis.get_disease_for_single_gene import (
    get_disease_for_single_gene,
    get_disease_for_single_gene_doc,
)
from apis.get_disease_name_for_disease_id import (
    get_disease_name_for_disease_id,
    get_disease_name_for_disease_id_doc,
)
from apis.get_domain_for_single_gene import (
    get_domain_for_single_gene,
    get_domain_for_single_gene_doc,
)
from apis.get_domain_name_for_domain_id import (
    get_domain_name_for_domain_id,
    get_domain_name_for_domain_id_doc,
)
from apis.get_enrichment_for_gene_set import (
    get_enrichment_for_gene_set,
    get_enrichment_for_gene_set_doc,
)
from apis.get_gene_id_for_single_gene import (
    get_gene_id_for_single_gene,
    get_gene_id_for_single_gene_doc,
)
from apis.get_gene_summary_for_single_gene import (
    get_gene_summary_for_single_gene,
    get_gene_summary_for_single_gene_doc,
)
from apis.get_interactions_for_gene_set import (
    get_interactions_for_gene_set,
    get_interactions_for_gene_set_doc,
)
from apis.get_pathway_for_gene_set import (
    get_pathway_for_gene_set,
    get_pathway_for_gene_set_doc,
)
from apis.get_pathway_id_for_single_gene import (
    get_pathway_id_for_single_gene,
    get_pathway_id_for_single_gene_doc,
)
from apis.get_pathway_name_for_pathway_id import (
    get_pathway_name_for_pathway_id,
    get_pathway_name_for_pathway_id_doc,
)
from apis.get_pubmed_articles import get_pubmed_articles, get_pubmed_articles_doc
from apis.get_rifsinfo_for_single_gene import (
    get_rifsinfo_for_single_gene,
    get_rifsinfo_for_single_gene_doc,
)


FunctionInfo = tuple[Callable[..., Any], dict[str, Any]]

func2info: dict[str, FunctionInfo] = {
    "get_complex_for_gene_set": (
        get_complex_for_gene_set,
        get_complex_for_gene_set_doc,
    ),
    "get_complex_names_for_complex_ids": (
        get_complex_names_for_complex_ids,
        get_complex_names_for_complex_ids_doc,
    ),
    "get_disease_for_single_gene": (
        get_disease_for_single_gene,
        get_disease_for_single_gene_doc,
    ),
    "get_disease_name_for_disease_id": (
        get_disease_name_for_disease_id,
        get_disease_name_for_disease_id_doc,
    ),
    "get_domain_for_single_gene": (
        get_domain_for_single_gene,
        get_domain_for_single_gene_doc,
    ),
    "get_domain_name_for_domain_id": (
        get_domain_name_for_domain_id,
        get_domain_name_for_domain_id_doc,
    ),
    "get_enrichment_for_gene_set": (
        get_enrichment_for_gene_set,
        get_enrichment_for_gene_set_doc,
    ),
    "get_gene_id_for_single_gene": (
        get_gene_id_for_single_gene,
        get_gene_id_for_single_gene_doc,
    ),
    "get_gene_summary_for_single_gene": (
        get_gene_summary_for_single_gene,
        get_gene_summary_for_single_gene_doc,
    ),
    "get_interactions_for_gene_set": (
        get_interactions_for_gene_set,
        get_interactions_for_gene_set_doc,
    ),
    "get_pathway_for_gene_set": (
        get_pathway_for_gene_set,
        get_pathway_for_gene_set_doc,
    ),
    "get_pathway_id_for_single_gene": (
        get_pathway_id_for_single_gene,
        get_pathway_id_for_single_gene_doc,
    ),
    "get_pathway_name_for_pathway_id": (
        get_pathway_name_for_pathway_id,
        get_pathway_name_for_pathway_id_doc,
    ),
    "get_pubmed_articles": (get_pubmed_articles, get_pubmed_articles_doc),
    "get_rifsinfo_for_single_gene": (
        get_rifsinfo_for_single_gene,
        get_rifsinfo_for_single_gene_doc,
    ),
}


class AgentPhD:
    """Fact-check a biological claim by letting GPT-5 call curated tools."""

    def __init__(
        self,
        function_names: list[str],
        *,
        model: str | None = None,
        max_turns: int = 20,
    ) -> None:
        unknown = sorted(set(function_names) - set(func2info))
        if unknown:
            raise ValueError(f"Unknown tool names: {', '.join(unknown)}")

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

    def inference(self, claim: str) -> str:
        settings = get_openai_settings()
        model = self.model or settings.model
        instructions = """
            You are an objective biomedical claim verifier for gene-set analysis.
            Your job is to verify one claim using only the provided tools and their returned evidence.

            Evidence standard:
            Strong support means the tool evidence directly supports the main biological relationship in the claim.
            Partial support means the evidence supports only part of the claim, a subset of genes, or a broader/narrower related process.
            Weak support means the evidence is biologically related but indirect, ambiguous, or too general.
            Refuted means the evidence contradicts the claim.
            Insufficient evidence means the available tools do not return enough usable evidence.

            Tool-use guidance:
            Choose the fewest tools needed for a reliable decision.
            Prefer enrichment and pathway tools for gene-set process claims.
            Use gene summaries, domains, diseases, complexes, interactions, and PubMed when the claim mentions specific genes, mechanisms, complexes, domains, interactions, disease links, or literature support.
            Do not infer beyond tool evidence.

            Final output contract:
            Start the final response with exactly "Report:".
            After "Report:", write one compact paragraph in this shape:
            Decision=<Strong support|Partial support|Weak support|Refuted|Insufficient evidence>. Evidence=<specific tool-backed evidence with gene names, term names, database/source names, and p-values or IDs when available>. Limitation=<short caveat if support is partial, weak, refuted, or insufficient>.
            Use plain text only. Do not use markdown, bullets, numbering, tables, or citations outside the tool evidence.
            """
        input_items: list[Any] = [
            {
                "role": "user",
                    "content": f"""
                        Claim to verify:
                        {claim}

                        Verification task:
                        Identify the central biological assertion in the claim.
                        Call relevant tools to gather direct evidence.
                        Compare the claim against the returned evidence using the evidence standard in the instructions.
                        Report only evidence that came from tool results.
                        If a tool returns an error or empty result, treat that as missing evidence rather than negative evidence unless another tool gives contradictory evidence.
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
                    output = self._run_tool_call(tool_call)
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": item_get(tool_call, "call_id"),
                            "output": output,
                        }
                    )
                continue

            text = response_text(response).strip()
            if "Report:" in text:
                return text.split("Report:", 1)[1].strip()

            input_items.append(
                {
                    "role": "user",
                    "content": 'Return the final verification now using the required format that starts with "Report:".',
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
