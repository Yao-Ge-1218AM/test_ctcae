from apis.utils import request_json


def get_gene_summary_for_single_gene(gene_name, specie):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    term = f"{gene_name} AND {specie}"
    search_data, search_error = request_json(
        "GET",
        f"{base_url}/esearch.fcgi",
        params={
            "db": "gene",
            "term": term,
            "retmode": "json",
            "sort": "relevance",
        },
    )
    if search_error:
        return search_error
    if not isinstance(search_data, dict):
        return "Error: unexpected NCBI esearch response format"

    gene_ids = search_data.get("esearchresult", {}).get("idlist", [])
    if not gene_ids:
        return "Error: unable to fetch data"

    summary_data, summary_error = request_json(
        "GET",
        f"{base_url}/esummary.fcgi",
        params={
            "db": "gene",
            "id": gene_ids[0],
            "retmode": "json",
            "sort": "relevance",
        },
    )
    if summary_error:
        return summary_error
    if not isinstance(summary_data, dict):
        return "Error: unexpected NCBI esummary response format"

    result = summary_data.get("result", {})
    gene_summary = result.get(gene_ids[0])
    if not isinstance(gene_summary, dict):
        return "Error: gene summary not found"
    gene_summary.pop("locationhist", None)
    return gene_summary


get_gene_summary_for_single_gene_doc = {
    "name": "get_gene_summary_for_single_gene",
    "description": "Given a single gene name, return summary information on function and related metadata.",
    "parameters": {
        "type": "object",
        "properties": {
            "gene_name": {
                "type": "string",
                "description": "A single gene name to search.",
            },
            "specie": {
                "type": "string",
                "description": "Species name. Use Homo for human and Mus for mouse.",
                "enum": ["Homo", "Mus"],
            },
        },
        "required": ["gene_name", "specie"],
    },
}
