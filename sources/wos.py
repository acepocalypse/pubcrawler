"""pubcrawler.sources.wos
~~~~~~~~~~~~~~~~~~~~~~~~
Web of Science harvesting module for the **pubcrawler** pipeline.

▸ Public entry‑point
    fetch(first_name, last_name, affiliation, api_key, **options) -> list[Publication]
    ----------------------------------------------------------------------------------
    • Wraps the Web of Science Starter API (or Expanded API if available).
    • Prioritizes Web of Science Author ID searches for accurate, targeted results.
    • Falls back to name and affiliation searches when Author ID is not available.
    • Converts the raw JSON response into the pipeline's *canonical* schema
      (see models.Publication).
    • Returns `List[Publication]` ready for aggregation.

Note: This implementation uses the Web of Science Starter API which has different
endpoints and data structures compared to the Expanded API. Adjust accordingly
based on your API access level.

For best results, provide Web of Science Author IDs when available.
"""

from __future__ import annotations

import ast
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
# 1) Helpers
# ---------------------------------------------------------------------------

def _create_retry_session(max_retries: int = 3, backoff_factor: float = 1.5) -> requests.Session:
    """Create a requests session with enhanced retry logic for transient failures."""
    session = requests.Session()
    
    # Define which HTTP status codes should trigger a retry
    status_forcelist = [429, 500, 502, 503, 504]  # Include 504 Gateway Timeout
    
    retry_strategy = Retry(
        total=max_retries,
        read=max_retries,
        connect=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET"],  # Only retry GET requests
        respect_retry_after_header=True,  # Honor server retry-after headers
        raise_on_status=False,  # Don't raise HTTPError, let us handle status codes
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=5, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def normalize_wos_ids(wos_input):
    """Normalize WoS ID input to a list format."""
    if isinstance(wos_input, str):
        return [wos_input]
    elif isinstance(wos_input, list):
        return wos_input
    else:
        raise TypeError("WOS_AUTHOR_ID must be a string or list of strings")

def _format_issn(code: str | None) -> str | None:
    """Format ISSN to standard ####-#### format."""
    if isinstance(code, str) and len(code) == 8 and code.isdigit():
        return f"{code[:4]}-{code[4:]}"
    return None

def _norm_issn(code: str | None) -> str | None:
    """Normalize ISSN code."""
    if not code or pd.isna(code):
        return None
    code_str = str(code).strip()
    if len(code_str) == 9 and code_str[4] == "-":
        return code_str
    if len(code_str) == 8 and code_str.isdigit():
        return f"{code_str[:4]}-{code_str[4:]}"
    return None

def _parse_authors(author_list) -> list[str]:
    """Parse WoS author list into standardized format."""
    if not isinstance(author_list, list):
        return []
    
    authors = []
    for a in author_list:
        if isinstance(a, dict):
            name = a.get("wosStandard", a.get("displayName", "")).strip()
            if name:
                authors.append(name)
    return authors

def _parse_cell_to_list(x) -> list[str]:
    """Parse cell content to list format."""
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return [x]
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def deduplicate_by_doi(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate DataFrame by DOI, keeping the most complete record."""
    if df.empty or "doi" not in df.columns:
        return df

    # Separate records with and without DOIs
    has_doi = df["doi"].notna() & (df["doi"].str.strip() != "")
    df_with_doi = df[has_doi].copy()
    df_without_doi = df[~has_doi].copy()
    
    # If no records have DOIs, return original
    if df_with_doi.empty:
        return df.reset_index(drop=True)
    
    # Deduplicate only records with DOIs
    df_with_doi["completeness_score"] = df_with_doi.notna().sum(axis=1)
    df_with_doi["citation_score"] = pd.to_numeric(df_with_doi["citations"], errors="coerce").fillna(0)
    df_with_doi["_rank"] = (
        df_with_doi.groupby("doi")[["completeness_score", "citation_score"]]
        .transform(lambda x: -x.rank(method="first"))
        .sum(axis=1)
    )
    df_with_doi = df_with_doi.sort_values("_rank").drop_duplicates("doi", keep="first")
    df_with_doi.drop(columns=["completeness_score", "citation_score", "_rank"], inplace=True)
    
    # Combine deduplicated DOI records with non-DOI records
    result_df = pd.concat([df_with_doi, df_without_doi], ignore_index=True)
    return result_df.reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2) Enhanced API Fetcher
# ---------------------------------------------------------------------------

def _collect_wos_documents(
    author_id: str,
    api_key: str,
    limit_per_page: int = 50,
) -> list[dict]:
    """Collect WoS documents using author ID with enhanced API handling."""
    headers = {
        "X-ApiKey": api_key,
        "Accept": "application/json",
        "User-Agent": "pubcrawler/1.0"
    }
    base_url = "https://api.clarivate.com/apis/wos-starter/v1/documents"
    
    # Try different query formats for better compatibility
    # Some WoS Author IDs work better with different query structures
    query_formats = [
        f'AI=("{author_id}")',      # Standard format with quotes
        f'AI={author_id}',          # Without quotes
        f'AI="{author_id}"',        # Alternative format
    ]
    
    page = 1
    all_hits: list[dict] = []
    session = _create_retry_session(max_retries=3, backoff_factor=1.5)
    
    # Try each query format until we get results or exhaust all options
    for query_idx, query in enumerate(query_formats):
        page = 1
        hits_found = False
        
        if query_idx > 0:
            print(f"🔄 Trying alternative query format {query_idx + 1} for WoS Author ID: {author_id}")
        
        while True:
            params = {"q": query, "db": "WOS", "limit": limit_per_page, "page": page}
            
            try:
                # Use more conservative timeout settings
                resp = session.get(base_url, headers=headers, params=params, timeout=(5, 30))
                
                if resp.status_code != 200:
                    error_message = f"WoS API error: {resp.status_code} - {resp.text}"
                    if resp.status_code == 403:
                        error_message += "\n💡 This usually means your API key doesn't have access to the Web of Science service."
                    elif resp.status_code == 401:
                        error_message += "\n💡 Authentication failed. Please check your API key."
                    elif resp.status_code == 429:
                        error_message += "\n💡 Rate limit exceeded. Please wait before making more requests."
                        # Add exponential backoff for rate limiting
                        time.sleep(min(2 ** (query_idx + 1), 30))  # Reduced max wait
                    elif resp.status_code == 504:
                        error_message += "\n💡 Gateway timeout occurred. Trying different query format if available."
                    
                    # If it's a 504 or rate limiting, try next query format
                    if resp.status_code in [504, 429]:
                        print(f"⚠️ {error_message}")
                        break  # Try next query format
                    else:
                        raise RuntimeError(error_message)
                
                data = resp.json()
                hits = data.get("hits", [])
                
                if not hits:
                    if page == 1:
                        # No results with this query format, try next one
                        break
                    else:
                        # End of results for this query format
                        break
                
                hits_found = True
                
                # Enhance records with identifier information
                for rec in hits:
                    ids = rec.get("identifiers", {})
                    rec["doi"] = ids.get("doi", "")
                    rec["issn"] = ids.get("issn", "")
                    rec["eissn"] = ids.get("eissn", "")
                
                all_hits.extend(hits)
                
                if len(hits) < limit_per_page:
                    break  # Last page for this query format
                    
                page += 1
                time.sleep(0.3)  # Slightly longer delay between requests
                
            except requests.exceptions.Timeout:
                print(f"⚠️ Request timeout for WoS Author ID {author_id} (query format {query_idx + 1}, page {page}). Trying next format.")
                break
            except requests.exceptions.ConnectionError as e:
                print(f"⚠️ Connection error for WoS Author ID {author_id} (query format {query_idx + 1}, page {page}): {e}")
                break
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 504:
                    print(f"⚠️ Gateway timeout for WoS Author ID {author_id} (query format {query_idx + 1}, page {page}). Trying next format.")
                    break
                else:
                    print(f"⚠️ HTTP error for WoS Author ID {author_id} (query format {query_idx + 1}, page {page}): {e}")
                    break
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Error querying Web of Science for Author ID {author_id} (query format {query_idx + 1}, page {page}): {e}")
                break
            except RuntimeError as e:
                print(f"⚠️ {e}")
                break
        
        # If we found results with this query format, we're done
        if hits_found:
            if query_idx > 0:
                print(f"✅ Found results using query format {query_idx + 1} for WoS Author ID: {author_id}")
            break
            
        # Add delay before trying next query format
        if query_idx < len(query_formats) - 1:
            time.sleep(1.0)
    
    session.close()
    return all_hits

def _search_wos_by_name_affiliation(
    first: str, last: str, affil: str, api_key: str, max_records: int = 100
) -> list[dict]:
    """Fallback search by name and affiliation when author ID is not available."""
    
    headers = {"X-ApiKey": api_key}
    base_url = "https://api.clarivate.com/apis/wos-starter/v1/documents"
    session = _create_retry_session(max_retries=2, backoff_factor=1.0)
    
    # Try multiple query variations
    query_variations = [
        # Original format
        f'AU=("{last}, {first}") AND AD=("{affil}")' if affil else f'AU=("{last}, {first}")',
        # Without quotes around author
        f'AU={last}, {first} AND AD="{affil}"' if affil else f'AU={last}, {first}',
        # Just author without affiliation
        f'AU="{last}, {first}"',
        # First name first
        f'AU="{first} {last}"',
        # Just last name
        f'AU={last}',
        # Alternative affiliation format
        f'AU={last} AND OG={affil}' if affil else f'AU={last}',
    ]
    
    for query_idx, query in enumerate(query_variations):
        all_hits = []
        page = 1
        limit_per_page = min(50, max_records)
        
        try:
            while len(all_hits) < max_records:
                remaining = max_records - len(all_hits)
                current_limit = min(limit_per_page, remaining)
                
                params = {
                    "q": query,
                    "db": "WOS",
                    "limit": current_limit,
                    "page": page
                }
                
                try:
                    # Use conservative timeout settings
                    resp = session.get(base_url, headers=headers, params=params, timeout=(5, 25))
                    
                    if resp.status_code != 200:
                        error_message = f"WoS API error: {resp.status_code} - {resp.text}"
                        if resp.status_code == 403:
                            error_message += "\n💡 This usually means your API key doesn't have access to the Web of Science service."
                        elif resp.status_code == 504:
                            error_message += "\n💡 Gateway timeout occurred. Trying next query variation if available."
                        print(f"⚠️ {error_message}")
                        break
                    
                    data = resp.json()
                    hits = data.get("hits", [])
                    
                    if not hits:
                        break
                    
                    # Enhance records with identifier information
                    for rec in hits:
                        ids = rec.get("identifiers", {})
                        rec["doi"] = ids.get("doi", "")
                        rec["issn"] = ids.get("issn", "")
                        rec["eissn"] = ids.get("eissn", "")
                    
                    all_hits.extend(hits)
                    
                    if len(hits) < current_limit:
                        break
                        
                    page += 1
                    time.sleep(0.4)  # More conservative delay
                    
                except requests.exceptions.Timeout:
                    print(f"⚠️ Request timeout for query variation {query_idx + 1} (page {page}). Trying next variation.")
                    break
                except requests.exceptions.ConnectionError as e:
                    print(f"⚠️ Connection error for query variation {query_idx + 1} (page {page}): {e}")
                    break
                except requests.exceptions.HTTPError as e:
                    if e.response and e.response.status_code == 504:
                        print(f"⚠️ Gateway timeout for query variation {query_idx + 1} (page {page}). Retries exhausted, trying next variation.")
                    else:
                        print(f"⚠️ HTTP error for query variation {query_idx + 1} (page {page}): {e}")
                    break
                except requests.exceptions.RequestException as e:
                    print(f"⚠️ Error querying Web of Science for query variation {query_idx + 1} (page {page}): {e}")
                    break
                    
        except Exception as e:
            print(f"⚠️ Unexpected error with query variation {query_idx + 1}: {e}")
            continue
        
        # If we found results, return them
        if all_hits:
            session.close()
            return all_hits
        
        # Add delay before trying next query variation
        if query_idx < len(query_variations) - 1:
            time.sleep(0.5)
    
    session.close()
    return []

# ---------------------------------------------------------------------------
# 3) Enhanced Data Processing Pipeline
# ---------------------------------------------------------------------------

def _wos_pipeline_single(
    author_id: str,
    api_key: str,
    limit_per_page: int = 50,
) -> pd.DataFrame:
    """Process WoS data for a single author ID."""
    raw_docs = _collect_wos_documents(author_id, api_key, limit_per_page)
    if not raw_docs:
        return pd.DataFrame()

    df = pd.json_normalize(raw_docs)

    # Extract and clean data fields
    df["title"] = df.get("title", "").astype(str).str.strip()  # Preserve original casing
    df["authors"] = df.get("names.authors", []).apply(_parse_authors)
    df["journal"] = df.get("source.sourceTitle", "")
    df["year"] = pd.to_numeric(df.get("source.publishYear", ""), errors="coerce")
    df["volume"] = df.get("source.volume", "")
    df["issue"] = df.get("source.issue", "")
    df["article_number"] = df.get("source.articleNumber", "")
    df["page_count"] = pd.to_numeric(df.get("source.pages.count", ""), errors="coerce")
    
    # Extract citations
    df["citations"] = df.get("citations", []).apply(
        lambda lst: next((c.get("count", 0) for c in lst if c.get("db") == "WOS"), 0)
        if isinstance(lst, list) else 0
    )

    def clean_doi(x):
        if isinstance(x, str):
            s = x.strip().lower()
            return s if s else None
        return None

    df["doi"] = df.get("doi", "").apply(clean_doi)

    # Handle ISSN data
    df["issn_list"] = df.apply(
        lambda r: list(filter(None, [_norm_issn(r.get("issn")), _norm_issn(r.get("eissn"))])),
        axis=1
    )
    
    # Select primary ISSN
    df["issn"] = df["issn_list"].apply(lambda x: x[0] if x else None)

    df["work_type"] = df.get("types", []).apply(
        lambda x: x[0].lower() if isinstance(x, list) and x else None
    )
    df["url"] = df.get("links.record", "")
    df["wos_id"] = df.get("uid", "")

    # Select and reorder columns
    keep_cols = [
        "title", "authors", "journal", "citations", "year", "url",
        "volume", "issue", "article_number", "page_count",
        "doi", "issn", "wos_id", "work_type", "issn_list",
    ]
    
    result_df = df.reindex(columns=keep_cols).fillna("")
    result_df = result_df.replace({None: "", "None": ""})
    
    return result_df.sort_values("year", ascending=False, na_position="last").reset_index(drop=True)

def _wos_pipeline_name_based(
    first: str, last: str, affil: str, api_key: str, max_records: int = 100
) -> pd.DataFrame:
    """Process WoS data using name and affiliation search."""
    raw_docs = _search_wos_by_name_affiliation(first, last, affil, api_key, max_records)
    if not raw_docs:
        return pd.DataFrame()

    df = pd.json_normalize(raw_docs)

    # Process similar to single author ID pipeline
    df["title"] = df.get("title", "").astype(str).str.strip()  # Preserve original casing
    df["authors"] = df.get("names.authors", []).apply(_parse_authors)
    df["journal"] = df.get("source.sourceTitle", "")
    df["year"] = pd.to_numeric(df.get("source.publishYear", ""), errors="coerce")
    df["volume"] = df.get("source.volume", "")
    df["issue"] = df.get("source.issue", "")
    df["page_count"] = pd.to_numeric(df.get("source.pages.count", ""), errors="coerce")
    
    df["citations"] = df.get("citations", []).apply(
        lambda lst: next((c.get("count", 0) for c in lst if c.get("db") == "WOS"), 0)
        if isinstance(lst, list) else 0
    )

    def clean_doi(x):
        if isinstance(x, str):
            s = x.strip().lower()
            return s if s else None
        return None

    df["doi"] = df.get("doi", "").apply(clean_doi)
    df["issn"] = df.get("issn", "").apply(_norm_issn)
    df["work_type"] = df.get("types", []).apply(
        lambda x: x[0].lower() if isinstance(x, list) and x else None
    )
    df["url"] = df.get("links.record", "")
    df["wos_id"] = df.get("uid", "")

    # Select columns for canonical format
    keep_cols = [
        "title", "authors", "journal", "year", "doi", "issn", 
        "citations", "url", "volume", "issue", "page_count", 
        "wos_id", "work_type"
    ]
    
    result_df = df.reindex(columns=keep_cols).fillna("")
    result_df = result_df.replace({None: "", "None": ""})
    
    return result_df.sort_values("year", ascending=False, na_position="last").reset_index(drop=True)

# ---------------------------------------------------------------------------
# 4) Canonical Conversion
# ---------------------------------------------------------------------------

_CANON_COLS: list[str] = [
    "title", "authors", "journal", "year",
    "doi", "issn", "source", "citations", "url",
]

_EXTRA_COLS: list[str] = [
    "volume", "issue", "page_start", "page_end", "page_count",
    "publisher", "links", "wos_id", "work_type"
]

def _to_canonical_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert processed WoS DataFrame to canonical format."""
    if df.empty:
        return pd.DataFrame(columns=_CANON_COLS + _EXTRA_COLS)
    
    canonical_df = df.copy()
    
    # Add missing canonical columns
    canonical_df["source"] = "WoS"
    canonical_df["publisher"] = ""
    canonical_df["links"] = canonical_df.get("url", "").apply(lambda x: [x] if x else [])
    canonical_df["page_start"] = None
    canonical_df["page_end"] = None
    
    # Ensure all required columns exist
    for col in _CANON_COLS + _EXTRA_COLS:
        if col not in canonical_df.columns:
            canonical_df[col] = None
    
    # Clean and deduplicate
    initial_count = len(canonical_df)
    canonical_df = canonical_df.sort_values("citations", ascending=False, na_position="last")
    canonical_df = deduplicate_by_doi(canonical_df)
    
    if len(canonical_df) < initial_count:
        print(f"  Deduplicated {initial_count - len(canonical_df)} WoS records. Remaining: {len(canonical_df)}")
    
    # Always include public-facing WoS URL if wos_id (UT) is present
    def wos_links(row):
        links = []
        # Add public WoS URL if wos_id exists
        if row.get("wos_id"):
            links.append(f"https://www.webofscience.com/wos/woscc/full-record/{row['wos_id']}")
        # Add API-provided record link if present and not duplicate
        url = row.get("url", "")
        if url and url not in links:
            links.append(url)
        return links
    canonical_df["links"] = canonical_df.apply(wos_links, axis=1)
    
    return canonical_df[_CANON_COLS + _EXTRA_COLS].reset_index(drop=True)

# ---------------------------------------------------------------------------
# 5) Public API – the only function the rest of pubcrawler calls
# ---------------------------------------------------------------------------

def fetch(
    first_name: str,
    last_name: str,
    affiliation: str,
    api_key: str,
    *,
    author_ids: Optional[List[str]] = None,
    max_records: int = 100,
) -> List[Publication]:
    """
    Harvest publications for an author from Web of Science and return them as a list
    of :class:`pubcrawler.models.Publication`.

    Parameters
    ----------
    first_name, last_name, affiliation : str
        Author details used to search for their publications.
    api_key : str
        Your Web of Science API key.
    author_ids : List[str], optional
        WoS Author IDs to search by. If provided, these take precedence over name search.
        This is now the preferred method for searching.
    max_records : int, default 100
        Maximum number of records to fetch.
    """
    
    all_dfs = []
    has_timeout_errors = False
    
    # Prioritize author ID search if provided
    if author_ids:
        author_ids = normalize_wos_ids(author_ids)
        print(f"🔍 Searching Web of Science using Author IDs: {', '.join(author_ids)}")
        for aid in author_ids:
            try:
                df = _wos_pipeline_single(aid, api_key, limit_per_page=50)
                if not df.empty:
                    df["wos_author_id"] = aid
                    all_dfs.append(df)
                    print(f"✅ Found {len(df)} publications for WoS Author ID: {aid}")
            except Exception as e:
                error_msg = str(e)
                if "504" in error_msg or "gateway timeout" in error_msg.lower():
                    has_timeout_errors = True
                    print(f"⚠️ Gateway timeout for Author ID={aid}. This is usually temporary - try again later.")
                else:
                    print(f"⚠️ Failed to fetch for Author ID={aid}: {e}")
    
    # Fallback to name/affiliation search if no author ID results and affiliation is available
    if not all_dfs and affiliation:
        print(f"🔍 No results from Author ID search, falling back to name/affiliation search for {first_name} {last_name} at {affiliation}")
        try:
            df = _wos_pipeline_name_based(first_name, last_name, affiliation, api_key, max_records)
            if not df.empty:
                all_dfs.append(df)
                print(f"✅ Found {len(df)} publications using name/affiliation search")
        except Exception as e:
            error_msg = str(e)
            if "504" in error_msg or "gateway timeout" in error_msg.lower():
                has_timeout_errors = True
                print(f"⚠️ Gateway timeout during name/affiliation search. This is usually temporary - try again later.")
            else:
                print(f"⚠️ Failed to fetch by name/affiliation: {e}")
    
    # Handle case where no search methods are available
    if not all_dfs and not author_ids and not affiliation:
        print("⚠️ Web of Science search requires either Author ID(s) or institutional affiliation")
        return []
    
    # Combine results
    if not all_dfs:
        if has_timeout_errors:
            print("ℹ️ No results due to gateway timeouts. The Web of Science servers may be experiencing high load.")
            print("ℹ️ Please try again in a few minutes. The request will automatically retry transient errors.")
            print("ℹ️ Consider using different WoS Author ID formats if available (e.g., ResearcherID vs. Author ID).")
        else:
            print("ℹ️ No publications found in Web of Science with the provided criteria")
        return []
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Apply deduplication
    combined_df = deduplicate_by_doi(combined_df)
    
    # Convert to canonical format
    canon_df = _to_canonical_format(combined_df)

    # --- map into dataclass ----------------------------------------------
    pubs: List[Publication] = []
    for _, row in canon_df.iterrows():
        pubs.append(
            Publication(
                title=row.title,
                authors=row.authors,
                journal=row.journal or None,
                year=int(row.year) if pd.notna(row.year) else None,
                doi=row.doi or None,
                issn=row.issn or None,
                source=row.source,  # "WoS"
                citations=int(row.citations) if pd.notna(row.citations) else 0,
                url=row.url or None,
                links=row.links if hasattr(row, 'links') else [],
            )
        )

    result = sorted(pubs, key=lambda p: (p.year or 0, p.citations or 0), reverse=True)
    
    if has_timeout_errors and result:
        print(f"ℹ️ Retrieved {len(result)} publications despite some gateway timeouts. Some data may be incomplete.")
    
    return result


# ---------------------------------------------------------------------------
# 6) CLI for quick manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    # --- Configuration for manual test ---
    load_dotenv()
    WOS_API_KEY = os.environ.get("WOS_API_KEY")
    WOS_AUTHOR_ID = "B-1234-5678"  # Example WoS Author ID (preferred)
    FIRST_NAME = "Morgan"
    LAST_NAME = "Furze"
    AFFILIATION = "Purdue University"

    if not WOS_API_KEY:
        print("Error: WOS_API_KEY environment variable not set.")
        print("Please set it to your Web of Science API key to run this test.")
    else:
        # Test with Author ID (preferred method)
        if WOS_AUTHOR_ID:
            print(f"Fetching Web of Science data using Author ID: {WOS_AUTHOR_ID}")
            publications = fetch(
                first_name=FIRST_NAME,
                last_name=LAST_NAME,
                affiliation=AFFILIATION,
                api_key=WOS_API_KEY,
                author_ids=[WOS_AUTHOR_ID]
            )
        else:
            # Fallback to name/affiliation search
            print(f"Fetching Web of Science data for {FIRST_NAME} {LAST_NAME} @ {AFFILIATION}...")
            publications = fetch(
                first_name=FIRST_NAME,
                last_name=LAST_NAME,
                affiliation=AFFILIATION,
                api_key=WOS_API_KEY,
            )

        print(f"\n✅ Fetched {len(publications)} publications from Web of Science.")
        if publications:
            print("--- First 5 publications ---")
            for p in publications[:5]:
                print(f"  · {p.year} [{p.citations} citations] {p.title[:80]}")
                print(f"    DOI: {p.doi or 'N/A'}, ISSN: {p.issn or 'N/A'}")