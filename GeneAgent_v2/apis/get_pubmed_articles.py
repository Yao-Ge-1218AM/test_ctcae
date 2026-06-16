from xml.etree import ElementTree

from apis.utils import request_content


def get_pubmed_articles(term):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    search_content, search_error = request_content(
        "GET",
        f"{base_url}/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": term,
            "retmode": "xml",
            "retmax": "5",
            "sort": "relevance",
        },
    )
    if search_error:
        return search_error

    try:
        search_results = ElementTree.fromstring(search_content or b"")
    except ElementTree.ParseError as exc:
        return f"Error: parsing PubMed search results failed: {exc}"

    id_list = [id_tag.text for id_tag in search_results.findall(".//Id") if id_tag.text]
    if not id_list:
        return "No articles found for the query."

    fetch_content, fetch_error = request_content(
        "GET",
        f"{base_url}/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(id_list), "retmode": "xml"},
    )
    if fetch_error:
        return fetch_error

    try:
        articles = ElementTree.fromstring(fetch_content or b"")
    except ElementTree.ParseError as exc:
        return f"Error: parsing PubMed fetch results failed: {exc}"

    results = []
    for article in articles.findall(".//PubmedArticle"):
        pmid_elem = article.find(".//PMID")
        title_elem = article.find(".//ArticleTitle")
        abstract_elem = article.find(".//Abstract/AbstractText")
        pmid = pmid_elem.text if pmid_elem is not None else "No PMID available"
        title = title_elem.text if title_elem is not None else "No title available"
        abstract_text = (
            abstract_elem.text if abstract_elem is not None else "No abstract available"
        )
        results.append(f"PMID: {pmid}\nTitle: {title}\nAbstract: {abstract_text}\n")
    return "".join(results)


get_pubmed_articles_doc = {
    "name": "get_pubmed_articles",
    "description": "Given a gene-related query, return related PubMed articles containing titles and abstracts.",
    "parameters": {
        "type": "object",
        "properties": {
            "term": {
                "type": "string",
                "description": "A gene-related PubMed query to search.",
            },
        },
        "required": ["term"],
    },
}
