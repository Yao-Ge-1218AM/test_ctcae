"""Document the active GeneAgent agent/tool layout.

The older version of this file instantiated a missing GPT class and referenced
tools that are no longer present. The executable cascade now lives in
main_cascade.py, worker.py, and detector.py. This module is kept as a readable
registry for people inspecting the project.
"""

AGENTS = {
    "detector": {
        "class": "detector.AgentDetect",
        "purpose": "Optional pre-filter for meaningful gene-set signal.",
        "tools": [
            "get_pathway_for_gene_set",
            "get_enrichment_for_gene_set",
        ],
    },
    "verifier": {
        "class": "worker.AgentPhD",
        "purpose": "Tool-backed fact-checker for process and analysis claims.",
        "tools": [
            "get_complex_for_gene_set",
            "get_complex_names_for_complex_ids",
            "get_disease_for_single_gene",
            "get_disease_name_for_disease_id",
            "get_domain_for_single_gene",
            "get_domain_name_for_domain_id",
            "get_enrichment_for_gene_set",
            "get_gene_id_for_single_gene",
            "get_pathway_for_gene_set",
            "get_pathway_id_for_single_gene",
            "get_pathway_name_for_pathway_id",
            "get_interactions_for_gene_set",
            "get_gene_summary_for_single_gene",
            "get_pubmed_articles",
            "get_rifsinfo_for_single_gene",
        ],
    },
}

# Backward-compatible lowercase name for notebooks/scripts that import agents.
agents = AGENTS
