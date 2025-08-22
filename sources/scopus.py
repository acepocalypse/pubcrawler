from __future__ import annotations
def normalize_scopus_ids(scopus_input):
    """Normalize Scopus ID input to a list format."""
    if isinstance(scopus_input, str):
        ids = [id_.strip() for id_ in scopus_input.split(',') if id_.strip()]
        return ids
    elif isinstance(scopus_input, list):
        ids = []
        for item in scopus_input:
            if isinstance(item, str):
                ids.extend([id_.strip() for id_ in item.split(',') if id_.strip()])
        return ids
    else:
        raise TypeError("scopus_id must be a string or list of strings")
"""pubcrawler.sources.scopus
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Scopus harvesting module for the **pubcrawler** pipeline.

▸ Public entry‑point
    fetch(scopus_id, api_key, **options) -> list[Publication]
    --------------------------------------------------------
    • Wraps the Scopus Search API.
    • Fetches all works associated with the provided Scopus ID.
    • Converts the raw JSON response into the pipeline's *canonical* schema
      (see models.Publication).
    • Returns `List[Publication]` ready for aggregation.
"""

import re
import time
from typing import Any, Dict, List
from urllib.parse import quote

import pandas as pd
import requests

# ----- pubcrawler core ------------------------------------------------------
try:
    from ..models import Publication  # dataclass defined in pubcrawler/models.py
except ImportError:
    # Fallback for when running as a standalone script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models import Publication

__all__ = ["fetch"]


# ---------------------------------------------------------------------------
# 1)  Core Scopus API Client
# ---------------------------------------------------------------------------

def _fetch_scopus_works(
    scopus_id: str, api_key: str, batch_size: int = 25
) -> list[dict]:
    """Fetch all publications for a single Scopus Author ID."""
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    works, start_index = [], 0
    max_api_calls = 50  # Safety break to prevent infinite loops
    call_count = 0

    while call_count < max_api_calls:
        params = {
            "query": f"AU-ID({scopus_id})",
            "view": "COMPLETE",
            "start": start_index,
            "count": batch_size,
        }
        try:
            resp = requests.get(
                "https://api.elsevier.com/content/search/scopus",
                headers=headers,
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching works for Scopus ID {scopus_id} (start={start_index}): {e}")
            break

        batch_entries = data.get("search-results", {}).get("entry", [])
        if not batch_entries:
            break  # No more results

        works.extend(batch_entries)
        start_index += batch_size
        call_count += 1
        time.sleep(0.1)  # Be polite to the API

    if call_count >= max_api_calls:
        print(f"Warning: Reached API call limit ({max_api_calls}) for Scopus ID {scopus_id}.")

    return works


def _query_scopus_by_id(
    scopus_id: str, api_key: str, page_batch: int = 25
) -> pd.DataFrame:
    """Fetches works for a given Scopus ID, returning a raw DataFrame."""
    print(f"Fetching works for Scopus ID: {scopus_id}")

    raw_works = _fetch_scopus_works(scopus_id, api_key, batch_size=page_batch)
    
    if not raw_works:
        print(f"No works found for Scopus ID: {scopus_id}.")
        return pd.DataFrame()

    return pd.json_normalize(raw_works)


# ---------------------------------------------------------------------------
# 2)  Scopus-raw ➜ canonical DataFrame
# ---------------------------------------------------------------------------

_CANON_COLS: list[str] = [
    "title", "authors", "journal", "year",
    "doi", "issn", "source", "citations", "url",
]

_EXTRA_COLS: list[str] = [
    "volume", "issue", "page_start", "page_end", "page_count",
    "publisher", "links", "scopus_pub_id", "work_type",
]


def _format_issn(code: Any) -> str | None:
    """Cleans and formats an ISSN code to the standard ####-#### format."""
    if isinstance(code, str):
        cleaned = re.sub(r"[^0-9X]", "", code.upper())
        if len(cleaned) == 8:
            return f"{cleaned[:4]}-{cleaned[4:]}"
    return None


def _parse_scopus_authors(author_list: Any) -> list[str]:
    """Extracts a list of author names from the Scopus 'author' field."""
    if not isinstance(author_list, list):
        return []
    names = []
    for a in author_list:
        if isinstance(a, dict) and (name := a.get("ce:indexed-name")):
            names.append(name)
    return names


def _scopus_to_canonical(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Coerces the raw DataFrame from the Scopus API into the canonical schema."""
    if raw_df.empty:
        return pd.DataFrame(columns=_CANON_COLS + _EXTRA_COLS)

    df = raw_df.copy()

    colmap = {
        "dc:title": "title",
        "author": "authors",
        "prism:publicationName": "journal",
        "citedby-count": "citations",
        "prism:coverDate": "year",
        "prism:url": "url",
        "prism:volume": "volume",
        "prism:issueIdentifier": "issue",
        "prism:doi": "doi",
        "eid": "scopus_pub_id",
        "subtypeDescription": "work_type",
        "prism:issn": "issn_raw",
        "prism:eIssn": "eissn_raw",
        "prism:pageRange": "page_range",
        "prism:publisher": "publisher",
    }
    df = df.rename(columns=colmap)

    # --- Clean and transform columns ---
    df["title"] = df["title"].astype(str).str.strip()  # Preserve original casing
    df["authors"] = df["authors"].apply(_parse_scopus_authors)
    df["year"] = pd.to_datetime(df["year"], errors="coerce").dt.year.astype("Int64")
    df["citations"] = pd.to_numeric(df["citations"], errors="coerce").fillna(0).astype(int)
    df["doi"] = df["doi"].astype(str).str.strip().str.lower().replace("", None)
    df["source"] = "Scopus"

    # --- Extract primary ISSN ---
    df["issn"] = df.get("issn_raw", pd.Series(dtype=str)).apply(_format_issn)
    df["eissn"] = df.get("eissn_raw", pd.Series(dtype=str)).apply(_format_issn)
    df["issn"] = df["issn"].fillna(df["eissn"])

    # --- Parse page information ---
    def _parse_pages(page_range):
        """Parse page range into start, end, and count"""
        if not page_range or pd.isna(page_range):
            return None, None, None
        
        page_str = str(page_range).strip()
        if "-" in page_str:
            parts = page_str.split("-")
            if len(parts) == 2:
                try:
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                    count = end - start + 1
                    return start, end, count
                except ValueError:
                    pass
        return None, None, None

    if "page_range" in df.columns:
        page_info = df["page_range"].apply(_parse_pages)
        df["page_start"] = page_info.apply(lambda x: x[0] if x else None)
        df["page_end"] = page_info.apply(lambda x: x[1] if x else None)
        df["page_count"] = page_info.apply(lambda x: x[2] if x else None)
    else:
        df["page_start"] = None
        df["page_end"] = None
        df["page_count"] = None

    # --- Add missing columns ---
    df["links"] = df.apply(lambda row: [row["url"]] if pd.notna(row.get("url")) else [], axis=1)

    # --- Deduplicate ---
    initial_count = len(df)
    df = df.sort_values("citations", ascending=False)
    df = df.drop_duplicates(subset=["doi"], keep="first")
    df = df.drop_duplicates(subset=["title", "year"], keep="first")
    if len(df) < initial_count:
        print(f"  Deduplicated {initial_count - len(df)} records. Remaining: {len(df)}")

    # Ensure all canonical/extra columns exist
    for col in _CANON_COLS + _EXTRA_COLS:
        if col not in df.columns:
            df[col] = None

    return df[_CANON_COLS + _EXTRA_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3)  Public API – the only function the rest of pubcrawler calls
# ---------------------------------------------------------------------------


def fetch(
    scopus_id: str = None,
    api_key: str = None,
    *,
    orcid_id: str = None,
    first_name: str = None,
    last_name: str = None,
    affiliation: str = None,
    page_batch: int = 25,
) -> List[Publication]:
    """
    Harvest publications for an author from Scopus using their Scopus ID, ORCID ID, or name/affiliation and return them as a list
    of :class:`pubcrawler.models.Publication`.

    Parameters
    ----------
    scopus_id : str, optional
        The Scopus Author ID (numeric string).
    api_key : str
        Your Elsevier/Scopus API key.
    orcid_id : str, optional
        ORCID ID to search for in Scopus.
    first_name, last_name, affiliation : str, optional
        Author details for name-based search.
    page_batch : int, default 25
        Number of results to fetch per API call.
    """
    # Determine search strategy
    all_dfs = []
    # Handle multiple Scopus IDs
    if scopus_id and api_key:
        normalized_ids = normalize_scopus_ids(scopus_id)
        print(f"[DEBUG] Normalized Scopus IDs: {normalized_ids}")
        for sid in normalized_ids:
            print(f"[DEBUG] Using Scopus ID for API call: '{sid}'")
            raw_df = _query_scopus_by_id(sid, api_key, page_batch)
            if not raw_df.empty:
                raw_df['scopus_id'] = sid
                all_dfs.append(raw_df)
    elif orcid_id and api_key:
        raw_df = _query_scopus_by_orcid(orcid_id, api_key, page_batch)
        if not raw_df.empty:
            all_dfs.append(raw_df)
    # elif first_name and last_name and api_key:
    #     raw_df = _query_scopus_by_name(first_name, last_name, affiliation, api_key, page_batch)
    else:
        print("Error: Insufficient parameters for Scopus search")
        return []

    if not all_dfs:
        print("No works found for provided Scopus IDs.")
        return []

    combined_df = pd.concat(all_dfs, ignore_index=True)
    canon_df = _scopus_to_canonical(combined_df)

    pubs: List[Publication] = []
    for _, row in canon_df.iterrows():
        scopus_eid = getattr(row, 'scopus_pub_id', None) or row.get('scopus_pub_id', None)
        public_url = None
        if scopus_eid:
            public_url = f"https://www.scopus.com/record/display.uri?eid={scopus_eid}&origin=resultslist"
        else:
            public_url = row.url
        pubs.append(
            Publication(
                title=row.title,
                authors=row.authors,
                journal=row.journal or None,
                year=int(row.year) if pd.notna(row.year) else None,
                doi=row.doi,
                issn=row.issn,
                source=row.source,  # "Scopus"
                citations=int(row.citations),
                url=public_url,
            )
        )

    return sorted(pubs, key=lambda p: (p.year or 0, p.citations or 0), reverse=True)


def _query_scopus_by_orcid(
    orcid_id: str, api_key: str, page_batch: int = 25
) -> pd.DataFrame:
    """Fetches works for a given ORCID ID from Scopus, returning a raw DataFrame."""
    print(f"Fetching works for ORCID ID: {orcid_id}")

    raw_works = _fetch_scopus_works_by_orcid(orcid_id, api_key, batch_size=page_batch)
    
    if not raw_works:
        print(f"No works found for ORCID ID: {orcid_id}")
        return pd.DataFrame()

    return pd.json_normalize(raw_works)


def _fetch_scopus_works_by_orcid(
    orcid_id: str, api_key: str, batch_size: int = 25
) -> list[dict]:
    """Fetch all publications for an ORCID ID from Scopus."""
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    works, start_index = [], 0
    max_api_calls = 50
    call_count = 0

    # Format ORCID for Scopus search
    orcid_formatted = orcid_id.replace('https://orcid.org/', '')
    
    while call_count < max_api_calls:
        params = {
            "query": f"ORCID({orcid_formatted})",
            "view": "COMPLETE",
            "start": start_index,
            "count": batch_size,
        }
        try:
            resp = requests.get(
                "https://api.elsevier.com/content/search/scopus",
                headers=headers,
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching works for ORCID ID {orcid_id} (start={start_index}): {e}")
            break

        batch_entries = data.get("search-results", {}).get("entry", [])
        if not batch_entries:
            break

        works.extend(batch_entries)
        start_index += batch_size
        call_count += 1
        time.sleep(0.1)

    if call_count >= max_api_calls:
        print(f"Warning: Reached API call limit ({max_api_calls}) for ORCID ID {orcid_id}.")

    return works


# ---------------------------------------------------------------------------
# 4)  CLI for quick manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    # --- Configuration for manual test ---
    # IMPORTANT: Set your Scopus API key as an environment variable
    # for this test to run.
    # Load from .env file
    load_dotenv()
    SCOPUS_API_KEY = os.environ.get("SCOPUS_API_KEY")
    SCOPUS_ID = "56518239200"  # Example Scopus ID - replace with actual ID

    if not SCOPUS_API_KEY:
        print("Error: SCOPUS_API_KEY environment variable not set.")
        print("Please set it to your Elsevier API key to run this test.")
    else:
        print(f"Fetching Scopus data for Scopus ID: {SCOPUS_ID}...")
        publications = fetch(
            scopus_id=SCOPUS_ID,
            api_key=SCOPUS_API_KEY,
        )

        print(f"\n✅ Fetched {len(publications)} publications from Scopus.")
        if publications:
            print("--- First 5 publications ---")
            for p in publications[:5]:
                print(f"  · {p.year} [{p.citations} citations] {p.title[:80]}")
                print(f"    DOI: {p.doi or 'N/A'}, ISSN: {p.issn or 'N/A'}")