from apis.utils import dumps, request_json


def get_gene_id_for_single_gene(gene, specie):
    data, error = request_json(
        "GET",
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={
            "db": "gene",
            "term": f"{gene} AND {specie}",
            "retmode": "json",
            "retmax": 5,
            "sort": "relevance",
        },
    )
    if error:
        return error
    if not isinstance(data, dict):
        return "Error: unexpected NCBI gene ID response format"
    gene_ids = data.get("esearchresult", {}).get("idlist", [])
    return dumps(gene_ids)


get_gene_id_for_single_gene_doc = {
    "name": "get_gene_id_for_single_gene",
    "description": "Given a single gene name, return related NCBI Gene IDs in the selected species.",
    "parameters": {
        "type": "object",
        "properties": {
            "gene": {
                "type": "string",
                "description": "A single gene name to search.",
            },
            "specie": {
                "type": "string",
                "description": "Species name. Use Homo for human and Mus for mouse.",
                "enum": ["Homo", "Mus"],
            },
        },
        "required": ["gene", "specie"],
    },
}
