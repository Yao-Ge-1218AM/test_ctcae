from apis.utils import dumps, read_tsv_records


def get_pathway_name_for_pathway_id(pathway_id, dbpath="backend_pathways.tsv"):
    records = read_tsv_records(
        dbpath,
        "backend_pathways.tsv",
        ["PathwayID", "PathwayName"],
    )
    pathway_dict = {}
    for record in records:
        full_id = record["PathwayID"]
        pathway_dict[full_id] = record["PathwayName"]
        if ":" in full_id:
            pathway_dict[full_id.split(":", 1)[1]] = record["PathwayName"]

    pathway_name = pathway_dict.get(pathway_id)
    return dumps(pathway_name) if pathway_name else "Error: pathway ID not found"


get_pathway_name_for_pathway_id_doc = {
    "name": "get_pathway_name_for_pathway_id",
    "description": "Given a pathway ID, return the corresponding pathway name from the local backend_pathways.tsv database.",
    "parameters": {
        "type": "object",
        "properties": {
            "pathway_id": {
                "type": "string",
                "description": "A pathway ID to search, with or without database prefix.",
            },
            "dbpath": {
                "type": "string",
                "description": "Local pathway database path. Relative paths resolve inside the apis directory.",
                "enum": ["backend_pathways.tsv"],
            },
        },
        "required": ["pathway_id"],
    },
}
