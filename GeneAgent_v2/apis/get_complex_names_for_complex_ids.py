from apis.utils import dumps, read_tsv_records


def get_complex_names_for_complex_ids(complex_acs, dbpath="backend_complex.tsv"):
    records = read_tsv_records(
        dbpath,
        "backend_complex.tsv",
        ["Complex ac", "Recommended name"],
    )
    complex_dict = {
        record["Complex ac"]: record["Recommended name"]
        for record in records
    }
    complex_names = [
        complex_dict[complex_ac.strip()]
        for complex_ac in complex_acs.split(",")
        if complex_ac.strip() in complex_dict
    ]
    return dumps(complex_names)


get_complex_names_for_complex_ids_doc = {
    "name": "get_complex_names_for_complex_ids",
    "description": "Given complex IDs, return representative complex names from the local backend_complex.tsv database.",
    "parameters": {
        "type": "object",
        "properties": {
            "complex_acs": {
                "type": "string",
                "description": 'Complex IDs separated by commas, for example "CPX-6,CPX-594".',
            },
            "dbpath": {
                "type": "string",
                "description": "Local complex database path. Relative paths resolve inside the apis directory.",
                "enum": ["backend_complex.tsv"],
            },
        },
        "required": ["complex_acs"],
    },
}
