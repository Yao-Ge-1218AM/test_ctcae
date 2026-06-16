from apis.utils import dumps, request_json


BACKGROUND_TYPES = [
    "KEGG_2021_Human",
    "Reactome_2022",
    "BioPlanet_2019",
    "WikiPathways_2016",
    "GO_Cellular_Component_2021",
    "GO_Molecular_Function_2021",
    "GO_Biological_Process_2021",
]


def get_pathway_for_gene_set(gene_set):
    """Return top pathway/enrichment terms from Enrichr."""

    gene_list = [gene.strip() for gene in gene_set.split(",") if gene.strip()]
    if not gene_list:
        return dumps([])

    payload = {
        "list": (None, "\n".join(gene_list)),
        "description": (None, "GeneAgent gene set"),
    }
    data, error = request_json(
        "POST",
        "https://maayanlab.cloud/Enrichr/addList",
        files=payload,
    )
    if error:
        return error
    if not isinstance(data, dict) or "userListId" not in data:
        return "Error: unexpected Enrichr addList response format"

    list_id = data["userListId"]
    terms = {}
    for background_type in BACKGROUND_TYPES:
        results, result_error = request_json(
            "GET",
            "https://maayanlab.cloud/Enrichr/enrich",
            params={"userListId": list_id, "backgroundType": background_type},
        )
        if result_error or not isinstance(results, dict):
            continue

        pathway_data = results.get(background_type, [])
        if not isinstance(pathway_data, list):
            continue

        for value in pathway_data[:3]:
            if not isinstance(value, list) or len(value) < 6:
                continue
            term_name = value[1]
            p_value = value[2]
            overlapping = value[5] if isinstance(value[5], list) else []
            terms[term_name] = [p_value, ",".join(overlapping), background_type]

    pathway_analysis = []
    for term, value in sorted(terms.items(), key=lambda item: item[1][0])[:5]:
        pathway_analysis.append(
            {
                "term": term,
                "p-value": value[0],
                "overlapping genes": value[1],
                "database": value[2],
            }
        )
    return dumps(pathway_analysis)


get_pathway_for_gene_set_doc = {
    "name": "get_pathway_for_gene_set",
    "description": "Given a comma-delimited gene set, return its top-5 biological pathway names and overlapping genes.",
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
