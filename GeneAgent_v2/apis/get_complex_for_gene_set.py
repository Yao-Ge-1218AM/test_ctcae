from apis.utils import dumps, request_json


def get_complex_for_gene_set(gene_set):
    gene_set = ",".join(gene.strip() for gene in gene_set.split(",") if gene.strip())
    if not gene_set:
        return dumps([])

    data, error = request_json(
        "GET",
        "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/agentapi/complex/",
        params={"name": gene_set, "retmode": "json", "limit": 10},
    )
    if error:
        return error
    if not isinstance(data, dict):
        return "Error: unexpected PubTator complex response format"
    return dumps(data.get("results", []))


get_complex_for_gene_set_doc = {
    "name": "get_complex_for_gene_set",
    "description": "Given a comma-delimited gene set, return possible protein complex IDs and corresponding complex names.",
    "parameters": {
        "type": "object",
        "properties": {
            "gene_set": {
                "type": "string",
                "description": 'A gene set separated only by commas, for example "x,y,z".',
            }
        },
        "required": ["gene_set"],
    },
}
