from apis.utils import dumps, read_tsv_records


def get_domain_name_for_domain_id(domain_id, dbpath="backend_domains.tsv"):
    records = read_tsv_records(
        dbpath,
        "backend_domains.tsv",
        ["domainID", "domainName"],
    )
    domain_dict = {
        record["domainID"]: record["domainName"]
        for record in records
    }
    domain_name = domain_dict.get(domain_id)
    return dumps(domain_name) if domain_name else "Error: domain ID not found"


get_domain_name_for_domain_id_doc = {
    "name": "get_domain_name_for_domain_id",
    "description": "Given a domain ID, return the corresponding domain name from the local backend_domains.tsv database.",
    "parameters": {
        "type": "object",
        "properties": {
            "domain_id": {
                "type": "string",
                "description": "A domain ID to search.",
            },
            "dbpath": {
                "type": "string",
                "description": "Local domain database path. Relative paths resolve inside the apis directory.",
                "enum": ["backend_domains.tsv"],
            },
        },
        "required": ["domain_id"],
    },
}
