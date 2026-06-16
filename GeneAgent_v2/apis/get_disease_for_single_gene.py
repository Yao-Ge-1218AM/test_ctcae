from apis.utils import dumps, request_json


def get_disease_for_single_gene(gene_name):
    data, error = request_json(
        "GET",
        "https://www.ncbi.nlm.nih.gov/research/pubtator-api/agentapi/disease/",
        params={"name": gene_name, "retmode": "json", "limit": 100},
    )
    if error:
        return error
    if not isinstance(data, dict):
        return "Error: unexpected PubTator disease response format"
    return dumps(data.get("results", []))


get_disease_for_single_gene_doc = {
    "name": "get_disease_for_single_gene",
    "description": "Given a gene name, return related diseases containing disease IDs and names.",
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
