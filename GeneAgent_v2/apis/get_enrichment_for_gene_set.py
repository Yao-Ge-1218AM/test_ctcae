from apis.utils import dumps, request_json


def get_enrichment_for_gene_set(gene_set, organism="hsapiens"):
    gene_list = [gene.strip() for gene in gene_set.split(",") if gene.strip()]
    if not gene_list:
        return dumps([])

    payload = {
        "organism": organism,
        "query": gene_list,
        "sources": [],
        "all_results": False,
        "user_threshold": 0.05,
    }

    data, error = request_json(
        "POST",
        "https://biit.cs.ut.ee/gprofiler/api/gost/profile/",
        headers={"Content-Type": "application/json"},
        json=payload,
    )
    if error:
        return error
    if not isinstance(data, dict) or not isinstance(data.get("result"), list):
        return "Error: unexpected g:Profiler response format"

    answer = []
    for item in data["result"][:5]:
        if not isinstance(item, dict):
            continue
        answer.append(
            {
                "description": item.get("description"),
                "enrichment name": item.get("name"),
                "p-value": item.get("p_value"),
                "intersection_size": item.get("intersection_size"),
            }
        )
    return dumps(answer)


get_enrichment_for_gene_set_doc = {
    "name": "get_enrichment_for_gene_set",
    "description": "Given a comma-delimited gene set, return its top-5 enrichment biological function names. Use this for claims that mention functional triggers such as regulation, signaling, response, and related biological processes.",
    "parameters": {
        "type": "object",
        "properties": {
            "gene_set": {
                "type": "string",
                "description": 'A gene set separated only by commas, for example "x,y,z".',
            },
            "organism": {
                "type": "string",
                "description": "Species for the gene set.",
                "enum": ["hsapiens", "mmusculus"],
            },
        },
        "required": ["gene_set", "organism"],
    },
}
