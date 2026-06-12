from apis.utils import dumps, request_json


def get_interactions_for_gene_set(gene_set):
    gene_set = ",".join(gene.strip() for gene in gene_set.split(",") if gene.strip())
    if not gene_set:
        return dumps([])

    data, error = request_json(
        "GET",
        "https://www.ncbi.nlm.nih.gov/research/pubtator-api/agentapi/ppi/",
        params={"name": gene_set, "retmode": "json", "limit": 50},
    )
    if error:
        return error
    if not isinstance(data, dict):
        return "Error: unexpected PubTator PPI response format"
    return dumps(data.get("results", []))


get_interactions_for_gene_set_doc = {
    "name": "get_interactions_for_gene_set",
    "description": "Given a comma-delimited gene set, return information on interacting genes.",
    "parameters": {
        "type": "object",
        "properties": {
            "gene_set": {
                "type": "string",
                "description": 'A gene set separated by commas, for example "x,y,z".',
            }
        },
        "required": ["gene_set"],
    },
}
