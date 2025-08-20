import os
import requests
from dotenv import load_dotenv

# Load your API key from .env
load_dotenv()
WOS_API_KEY = os.getenv("WOS_API_KEY")

def test_researcherid_lookup(author_query="AU=(Furze, Morgan)"):
    """Trial function to check if ResearcherID appears in Starter API responses."""
    if not WOS_API_KEY:
        raise RuntimeError("WOS_API_KEY not set")

    base_url = "https://api.clarivate.com/apis/wos-starter/v1/documents"
    headers = {"X-ApiKey": WOS_API_KEY, "Accept": "application/json"}
    params = {"q": author_query, "db": "WOS", "limit": 5, "page": 1}

    resp = requests.get(base_url, headers=headers, params=params, timeout=30)
    if resp.status_code != 200:
        print(f"❌ API error {resp.status_code}: {resp.text}")
        return

    data = resp.json()
    hits = data.get("hits", [])

    print(f"Found {len(hits)} documents for query {author_query!r}\n")

    for i, hit in enumerate(hits, 1):
        authors = hit.get("names", {}).get("authors", [])
        print(f"--- Document {i} ---")
        for auth in authors:
            name = auth.get("wosStandard") or auth.get("displayName")
            rid = auth.get("researcherId")
            orcid = auth.get("orcidId")
            print(f"  Author: {name}")
            print(f"    ResearcherID: {rid or 'N/A'}")
            print(f"    ORCID: {orcid or 'N/A'}")
        print()

if __name__ == "__main__":
    test_researcherid_lookup()
