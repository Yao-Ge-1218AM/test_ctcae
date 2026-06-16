from apis.utils import dumps, request_json


def get_pathway_id_for_single_gene(gene_name):
    data, error = request_json(
        "GET",
        "https://www.ncbi.nlm.nih.gov/research/pubtator-api/agentapi/",
        params={"name": gene_name, "table": "pathway", "retmode": "json"},
    )
    if error:
        return error
    if not isinstance(data, dict):
        return "Error: unexpected PubTator pathway response format"
    results = data.get("results", [])
    if not isinstance(results, list):
        return "Error: unexpected PubTator pathway results format"
    return dumps(results[:10])


get_pathway_id_for_single_gene_doc = {
    "name": "get_pathway_id_for_single_gene",
    "description": "Given a single gene name, return corresponding pathway IDs.",
    "parameters": {
        "type": "object",
        "properties": {
            "gene_name": {
                "type": "string",
                "description": "A single gene name to search.",
            }
        },
        "required": ["gene_name"],
    },
}
