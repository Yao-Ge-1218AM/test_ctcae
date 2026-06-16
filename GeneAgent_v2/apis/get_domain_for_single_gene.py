from apis.utils import dumps, request_json


def get_domain_for_single_gene(gene_name):
    data, error = request_json(
        "GET",
        "https://www.ncbi.nlm.nih.gov/research/pubtator-api/agentapi/cdd/",
        params={"name": gene_name, "retmode": "json", "limit": 10},
    )
    if error:
        return error
    if not isinstance(data, dict):
        return "Error: unexpected PubTator domain response format"
    return dumps(data.get("results", []))


get_domain_for_single_gene_doc = {
    "name": "get_domain_for_single_gene",
    "description": "Given a gene name, return related biological domains containing domain IDs and names.",
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
