from apis.utils import dumps, read_tsv_records


def get_disease_name_for_disease_id(disease_id, dbpath="backend_diseases.tsv"):
    records = read_tsv_records(
        dbpath,
        "backend_diseases.tsv",
        ["DiseaseID", "DiseaseName"],
    )
    disease_dict = {
        record["DiseaseID"]: record["DiseaseName"]
        for record in records
    }
    disease_name = disease_dict.get(disease_id)
    return dumps(disease_name) if disease_name else "Error: disease ID not found"


get_disease_name_for_disease_id_doc = {
    "name": "get_disease_name_for_disease_id",
    "description": "Given a disease ID, return the corresponding disease name from the local backend_diseases.tsv database.",
    "parameters": {
        "type": "object",
        "properties": {
            "disease_id": {
                "type": "string",
                "description": "A disease ID to search.",
            },
            "dbpath": {
                "type": "string",
                "description": "Local disease database path. Relative paths resolve inside the apis directory.",
                "enum": ["backend_diseases.tsv"],
            },
        },
        "required": ["disease_id"],
    },
}
