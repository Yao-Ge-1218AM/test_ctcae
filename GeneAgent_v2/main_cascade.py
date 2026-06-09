"""GeneAgent cascade for gene-set analysis and tool-backed verification."""

from __future__ import annotations

import argparse
import ast
import pandas as pd
from email import header
import json
import re
import traceback
from pathlib import Path
from typing import Iterable

from config import PROJECT_ROOT, get_openai_settings
from detector import AgentDetect
from llm_client import generate_text
from worker import AgentPhD


SUMMARY_OUTPUT = PROJECT_ROOT / "result/geneagent.initial_summary.txt"
FINAL_OUTPUT = PROJECT_ROOT / "result/geneagent.final_result.txt"
CLAIM_LOG = PROJECT_ROOT / "result/geneagent.claim_verification.txt"

SYSTEM = (
    "You are a precise biological-analysis assistant for molecular biologists. "
    "Write compact, evidence-aware gene-set analyses using standard biological terminology. "
    "Prefer specific mechanisms, pathways, complexes, molecular functions, and cellular contexts over broad generic wording."
)
SYSTEM_VERIFY = (
    "You are an objective biomedical claim verifier. "
    "Convert gene-set summaries into checkable claims without adding new biology, assumptions, or interpretation."
)

DETECTION_TOOLS = [
    "get_pathway_for_gene_set",
    "get_enrichment_for_gene_set",
]

VERIFICATION_TOOLS = [
    "get_complex_for_gene_set",
    "get_complex_names_for_complex_ids",
    "get_disease_for_single_gene",
    "get_disease_name_for_disease_id",
    "get_domain_for_single_gene",
    "get_domain_name_for_domain_id",
    "get_enrichment_for_gene_set",
    "get_gene_id_for_single_gene",
    "get_pathway_for_gene_set",
    "get_pathway_id_for_single_gene",
    "get_pathway_name_for_pathway_id",
    "get_interactions_for_gene_set",
    "get_gene_summary_for_single_gene",
    "get_pubmed_articles",
    "get_rifsinfo_for_single_gene",
]

agentdetect = AgentDetect(function_names=DETECTION_TOOLS)
agentphd = AgentPhD(function_names=VERIFICATION_TOOLS)


class CascadeError(RuntimeError):
    """Raised when one cascade step returns an unusable result."""


def baseline_prompt(genes: str) -> str:
    return f"""
        Purpose:
        Create the initial biological interpretation of the input gene set. This is a concise working hypothesis that will be verified in later steps.

        Input gene set:
        {genes}

        Task:
        Infer the most prominent shared biological process, molecular function, pathway, complex, or cellular context represented by the gene set.
        Name the process using a brief, specific biological phrase.
        Explain the analysis by grouping genes that support the same function or mechanism.
        For each functional statement, include the relevant gene names in the same sentence.
        Use cautious factual language when the relationship is plausible but not directly established from the gene names alone.

        Output contract:
        Line 1 must be exactly: Process: <specific process name>
        Line 2 must begin exactly: Analysis: <concise biological analysis>
        Use plain text only.
        Do not use markdown, bullets, numbering, tables, quotation marks around the process name, or citation placeholders.
        Do not mention that this is a hypothesis or that later verification will occur.

        Quality requirements:
        Use precise terms such as signaling, transcriptional regulation, cell-cycle control, chromatin remodeling, immune response, metabolism, transport, complex assembly, or DNA repair when supported.
        Avoid vague phrases such as "various cellular processes", "multiple pathways", "biological functions", or "important roles" unless immediately specified.
        Do not invent experimental conditions, disease contexts, tissue contexts, species, or literature citations.
        """


def topic_prompt(process: str) -> str:
    return f"""
        Purpose:
        Generate process-level claims that can verify whether the proposed process name is appropriate for the entire gene set.

        Proposed process name:
        {process}

        Task:
        Write affirmative, decontextualized claims that test the biological meaning of the proposed process name.
        Each claim must be about the entire gene set, not a single gene, subset of genes, or the wording quality of the previous summary.
        Claims must be independently understandable without seeing the original prompt.
        Claims should be specific enough that enrichment, pathway, interaction, complex, domain, gene-summary, or PubMed tools can verify them.

        Output contract:
        Return only a valid JSON array of strings.
        Return 1 to 4 claims.
        Do not wrap the JSON in markdown fences.
        Do not include comments, labels, or explanatory text outside the JSON array.
        Use this exact shape:
        ["claim_1", "claim_2", "claim_3"]
        """


def analysis_prompt(summary: str) -> str:
    return f"""
        Purpose:
        Generate analysis-level claims that can verify the gene-specific biological statements in the revised summary.

        Revised summary:
        {summary}

        Task:
        Extract only factual biological assertions that connect named genes to functions, pathways, complexes, domains, or processes.
        Each claim must include the exact gene names involved and the function attributed to them.
        If several genes are grouped together in one analytical sentence, keep those genes together in one claim.
        Do not create claims about style, summary quality, uncertainty, verification status, or the process of reasoning.
        Do not add new biological facts that are absent from the revised summary.

        Output contract:
        Return only a valid JSON array of strings.
        Return 1 to 8 claims, prioritizing claims most important to the process name.
        Do not wrap the JSON in markdown fences.
        Do not include comments, labels, or explanatory text outside the JSON array.
        Use this exact shape:
        ["claim_1", "claim_2", "claim_3"]
        """


def modification_prompt(verification_topic: str) -> str:
    return f"""
        Purpose:
        Revise the process name and summary after process-level verification.

        Verification report for the process name:
        {verification_topic}

        Decision rules:
        Use only claims with successful evidence from the verification report.
        If the process-level claims are strongly supported, keep the original process name unless a more precise equivalent term is directly supported.
        If the evidence partially supports the process name, narrow the process name to the supported biological scope and remove unsupported wording.
        If the process-level claims are weakly supported, refuted, or unsupported, replace the process name with the strongest specific enrichment, pathway, complex, or gene-summary function supported by the report.
        If no coherent shared process is supported, use a conservative process name that reflects the supported evidence without exaggeration.

        Revision requirements:
        Revise only what is necessary to align the process name and analysis with verified evidence.
        Retain gene-function statements that are not contradicted by the report.
        Remove or soften unsupported claims without discussing their failure.
        Do not write phrases such as "no direct evidence", "not confirmed", "needs further investigation", or "verification report".

        Output contract:
        Line 1 must be exactly: Process: <updated specific process name>
        Line 2 must begin exactly: Analysis: <revised concise biological analysis>
        Use plain text only.
        Do not use markdown, bullets, numbering, tables, citations, or explanatory metadata.
        """


def summarization_prompt(verification_analysis: str) -> str:
    return f"""
        Purpose:
        Produce the final verified gene-set analysis after checking the gene-specific analytical claims.

        Verification report for analytical claims:
        {verification_analysis}

        Task:
        Revise the current summary so that each retained gene-function statement is supported by the verification report.
        Keep supported statements and make them more precise when the report provides standard names or clearer evidence.
        Remove, narrow, or soften unsupported statements without describing the verification failure.
        If the verified analytical evidence no longer supports the current process name, replace it with the most specific process name supported by the retained analysis.
        Group genes that contribute to the same biological function in the same sentence.

        Output contract:
        Line 1 must be exactly: Process: <final verified process name>
        Line 2 must begin exactly: Analysis: <final verified concise biological analysis>
        Use plain text only.
        Do not use markdown, bullets, numbering, tables, citations, or explanatory metadata.
        Do not mention unsupported evidence, verification status, tool results, or future investigation.
        """


def normalize_genes(genes: str) -> str:
    return ",".join(
        gene
        for gene in re.sub(r"[, ;|\s]+", ",", str(genes)).split(",")
        if gene
    )


def append_record(path: Path, *parts: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for part in parts:
            handle.write(str(part))
            if not str(part).endswith("\n"):
                handle.write("\n")


def call_llm(step: str, instructions: str, input_items, *, model: str | None = None) -> str:
    try:
        text = generate_text(
            instructions=instructions,
            input_items=input_items,
            model=model,
        )
    except Exception as exc:
        raise CascadeError(f"{step} LLM call failed: {exc}") from exc
    if not text or not text.strip():
        raise CascadeError(f"{step} returned an empty response")
    return text.strip()


def extract_process_name(summary: str) -> str:
    match = re.search(r"^Process:\s*(.+?)\s*$", summary, flags=re.MULTILINE)
    if not match:
        raise CascadeError('Initial summary did not include a "Process: <name>" line')
    return match.group(1).strip()


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json|python|list)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_json_list(text: str, step: str) -> list[str]:
    cleaned = strip_code_fence(text)
    candidates = [cleaned]

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end > start:
        candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]

    raise CascadeError(f"{step} did not return a valid JSON list: {text[:300]}")


def verify_claims(
    claims: Iterable[str],
    *,
    gene_context: str | None = None,
    log_id: str,
) -> str:
    report_parts = []
    for claim in claims:
        claim_for_verification = claim.strip()
        if gene_context:
            claim_for_verification = (
                f"{claim_for_verification}\n"
                f"Here is the entire gene set used for verification:\n##{gene_context}##"
            )

        try:
            claim_result = agentphd.inference(claim_for_verification)
        except Exception as exc:
            claim_result = f"Failed to verify claim: {exc}"

        append_record(CLAIM_LOG, log_id, claim_for_verification, claim_result, "&&")
        report_parts.append(
            f"Original_claim: {claim_for_verification}\n"
            f"Verified_claim: {claim_result}\n"
        )
    return "\n".join(report_parts)


def _gene_agent(id_value: str, genes: str) -> str:
    settings = get_openai_settings()
    genes = normalize_genes(genes)
    print(id_value)
    print(genes)
    print("RUNNING GENEAGENT")

    messages = [
        {"role": "user", "content": baseline_prompt(genes)},
    ]
    summary = call_llm("Initial summary", SYSTEM, messages, model=settings.model)
    messages.append({"role": "assistant", "content": summary})
    append_record(SUMMARY_OUTPUT, summary, id_value, "//")
    print("=====Summary=====")
    print(summary)

    process = extract_process_name(summary)
    raw_topic_claims = call_llm(
        "Topic claim generation",
        SYSTEM_VERIFY,
        topic_prompt(process),
        model=settings.model,
    )
    claims_topic = parse_json_list(raw_topic_claims, "Topic claim generation")
    append_record(CLAIM_LOG, str(id_value))
    print("=====Topic Claim=====")
    print(claims_topic)

    verification_topic = verify_claims(
        claims_topic,
        gene_context=genes,
        log_id=str(id_value),
    )

    messages.append({"role": "user", "content": modification_prompt(verification_topic)})
    updated_topic = call_llm(
        "Process-name modification",
        SYSTEM,
        messages,
        model=settings.model,
    )
    messages.append({"role": "assistant", "content": updated_topic})
    print("=====Updated Topic=====")
    print(updated_topic)

    raw_analysis_claims = call_llm(
        "Analysis claim generation",
        SYSTEM_VERIFY,
        analysis_prompt(updated_topic),
        model=settings.model,
    )
    claims_analysis = parse_json_list(raw_analysis_claims, "Analysis claim generation")
    print("=====Analysis Claim=====")
    print(claims_analysis)

    verification_analysis = verify_claims(
        claims_analysis,
        log_id=str(id_value),
    )

    messages.append({"role": "user", "content": summarization_prompt(verification_analysis)})
    update = call_llm("Final summarization", SYSTEM, messages, model=settings.model)
    append_record(FINAL_OUTPUT, update, id_value, "//")
    append_record(CLAIM_LOG, "////////")
    print("====Final Update====")
    return update


def GeneAgent(ID, genes):
    try:
        return _gene_agent(str(ID), str(genes))
    except Exception as exc:
        failure = (
            "Process: Failed\n"
            f"The gene set could not be processed. Reason: {exc}"
        )
        append_record(FINAL_OUTPUT, failure, str(ID), "//")
        append_record(CLAIM_LOG, str(ID), traceback.format_exc(), "////////")
        return failure


# def iter_cases(path: Path):
#     if not path.exists():
#         raise FileNotFoundError(f"Input TSV not found: {path}")

#     with path.open("r", encoding="utf-8", newline="") as handle:
#         reader = csv.DictReader(handle, delimiter="\t")
#         required = {"num_gene", "gene_list"}
#         missing = sorted(required - set(reader.fieldnames or []))
#         if missing:
#             raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
#         for row in reader:
#             yield row["num_gene"], row["gene_list"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GeneAgent cascade on a TSV file.")
    parser.add_argument(
        "input_tsv",
        nargs="?",
        default="data/gene.cluster.tsv",
        help="TSV file with Gene List and Gene Count columns.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_tsv)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    data = pd.read_table(input_path, header=0, index_col=None, sep="\t")
    for case_id, genes in zip(data["Cluster ID"], data["Gene List"]):
        id_value = f"{case_id}"
        response = GeneAgent(id_value, genes)
        print(response)

    print("===Finished!===")


if __name__ == "__main__":
    main()
