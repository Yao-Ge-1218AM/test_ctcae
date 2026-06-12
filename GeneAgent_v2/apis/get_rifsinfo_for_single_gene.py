from apis.utils import dumps, request_json


def get_rifsinfo_for_single_gene(gene_name, alpha=0.1):
    data, error = request_json(
        "GET",
        "https://www.ncbi.nlm.nih.gov/research/pubtator-api/agentapi/",
        params={"name": gene_name, "retmode": "json"},
    )
    if error:
        return error
    if not isinstance(data, dict):
        return "Error: unexpected PubTator gene response format"

    results = data.get("results", [])
    if not isinstance(results, list) or not results:
        return dumps([])

    first = results[0]
    if isinstance(first, dict):
        generifs = first.get("generifs", [])
        if isinstance(generifs, list):
            keep = max(1, int(len(generifs) * float(alpha))) if generifs else 0
            first["generifs"] = generifs[:keep]
    return dumps(results)


get_rifsinfo_for_single_gene_doc = {
    "name": "get_rifsinfo_for_single_gene",
    "description": "Given a single gene name, return GeneRIF information and PubMed IDs.",
    "parameters": {
        "type": "object",
        "properties": {
            "gene_name": {
                "type": "string",
                "description": "A single gene name to search.",
            },
            "alpha": {
                "type": "number",
                "description": "Fraction of GeneRIF entries to return.",
            },
        },
        "required": ["gene_name"],
    },
}
