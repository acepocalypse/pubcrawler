"""pubcrawler.sources.orcid
~~~~~~~~~~~~~~~~~~~~~~~~~
ORCID harvesting module for the **pubcrawler** pipeline.

▸ Public entry‑point
    fetch(orcid_id, client_id, client_secret, **options) -> list[Publication]
    ------------------------------------------------------------------------
    • Wraps the ORCID Public API v3.0.
    • Fetches all works associated with the provided ORCID ID.
    • Converts the raw JSON response into the pipeline's *canonical* schema
      (see models.Publication).
    • Returns `List[Publication]` ready for aggregation.
"""

from __future__ import annotations

import re
import time
import json
from typing import Any, Dict, List, Optional
from urllib.parse import quote
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
# 1) ORCID API Client
# ---------------------------------------------------------------------------

class ORCIDClient:
    """ORCID API client with OAuth2 authentication."""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://pub.orcid.org/v3.0"
        self.access_token = None
        
    def _get_access_token(self) -> str:
        """Get OAuth2 access token for ORCID API."""
        if self.access_token:
            return self.access_token
            
        token_url = "https://orcid.org/oauth/token"
        
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'scope': '/read-public',
            'grant_type': 'client_credentials'
        }
        
        try:
            response = requests.post(token_url, headers=headers, data=data, timeout=30)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data.get('access_token')
            
            if not self.access_token:
                raise RuntimeError("Failed to obtain access token from ORCID")
                
            return self.access_token
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"ORCID authentication failed: {e}")
    
    def _make_request(self, endpoint: str) -> Dict[str, Any]:
        """Make authenticated request to ORCID API."""
        access_token = self._get_access_token()
        
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 404:
                raise RuntimeError(f"ORCID ID not found or profile is private")
            elif response.status_code == 401:
                raise RuntimeError(f"ORCID API authentication failed. Check your client credentials.")
            elif response.status_code != 200:
                raise RuntimeError(f"ORCID API error: {response.status_code} - {response.text}")
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"ORCID API request failed: {e}")
    
    def get_works(self, orcid_id: str) -> List[Dict[str, Any]]:
        """Get all works for an ORCID ID."""
        # Normalize ORCID ID format
        orcid_id = self._normalize_orcid_id(orcid_id)
        
        # First, get the list of work summaries
        works_endpoint = f"{orcid_id}/works"
        works_summary = self._make_request(works_endpoint)
        
        work_details = []
        
        # Get detailed information for each work
        if 'group' in works_summary:
            for group in works_summary['group']:
                for work_summary in group.get('work-summary', []):
                    put_code = work_summary.get('put-code')
                    if put_code:
                        try:
                            # Get detailed work information
                            work_detail_endpoint = f"{orcid_id}/work/{put_code}"
                            work_detail = self._make_request(work_detail_endpoint)
                            # Only append if work_detail is not None
                            if work_detail:
                                work_details.append(work_detail)
                            
                            # Be respectful to the API
                            time.sleep(0.1)
                            
                        except Exception as e:
                            print(f"Warning: Failed to fetch work detail for put-code {put_code}: {e}")
                            continue
        
        return work_details
    
    def _normalize_orcid_id(self, orcid_id: str) -> str:
        """Normalize ORCID ID to standard format."""
        # Remove any URLs or prefixes
        orcid_id = orcid_id.replace('https://orcid.org/', '')
        orcid_id = orcid_id.replace('http://orcid.org/', '')
        orcid_id = orcid_id.replace('orcid.org/', '')
        
        # Ensure proper format: ####-####-####-####
        orcid_id = re.sub(r'[^\d\-X]', '', orcid_id.upper())
        
        # Add hyphens if missing
        if len(orcid_id) == 16 and '-' not in orcid_id:
            orcid_id = f"{orcid_id[:4]}-{orcid_id[4:8]}-{orcid_id[8:12]}-{orcid_id[12:16]}"
        
        # Validate format
        if not re.match(r'^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$', orcid_id):
            raise ValueError(f"Invalid ORCID ID format: {orcid_id}")
        
        return orcid_id


# ---------------------------------------------------------------------------
# 2) Data Processing Functions
# ---------------------------------------------------------------------------

def _extract_publication_data(work: Dict[str, Any]) -> Dict[str, Any]:
    """Extract publication data from ORCID work object."""
    # Check if work is None or empty
    if not work:
        print("Warning: Encountered None or empty work object")
        return None
    
    # Debug: Print work structure for troubleshooting
    if not isinstance(work, dict):
        print(f"Warning: Work object is not a dict, type: {type(work)}")
        return None
    
    pub_data = {
        'title': None,
        'authors': [],
        'journal': None,
        'year': None,
        'doi': None,
        'issn': None,
        'url': None,
        'citations': 0,
        'work_type': None,
        'publisher': None
    }
    
    try:
        # Extract title
        title_info = work.get('title')
        if title_info and isinstance(title_info, dict) and 'title' in title_info:
            title_obj = title_info['title']
            if title_obj and isinstance(title_obj, dict):
                pub_data['title'] = title_obj.get('value', '').strip()
        
        # Extract publication year
        pub_date = work.get('publication-date')
        if pub_date and isinstance(pub_date, dict) and 'year' in pub_date:
            year_obj = pub_date['year']
            if year_obj and isinstance(year_obj, dict):
                try:
                    pub_data['year'] = int(year_obj.get('value', 0))
                except (ValueError, TypeError):
                    pass
        
        # Extract journal name
        journal_title = work.get('journal-title')
        if journal_title and isinstance(journal_title, dict):
            pub_data['journal'] = journal_title.get('value', '').strip()
        
        # Extract work type
        work_type = work.get('type')
        if work_type:
            pub_data['work_type'] = str(work_type).lower()
        
        # Extract external identifiers (DOI, etc.)
        external_ids = work.get('external-ids')
        if external_ids and isinstance(external_ids, dict):
            ext_id_list = external_ids.get('external-id', [])
            if ext_id_list and isinstance(ext_id_list, list):
                for ext_id in ext_id_list:
                    if not ext_id or not isinstance(ext_id, dict):
                        continue
                        
                    id_type = ext_id.get('external-id-type', '').lower()
                    id_value = ext_id.get('external-id-value', '').strip()
                    
                    if id_type == 'doi' and id_value:
                        # Clean DOI
                        doi = id_value.lower()
                        if doi.startswith('http'):
                            doi = doi.split('doi.org/')[-1]
                        pub_data['doi'] = doi
                    elif id_type == 'issn' and id_value:
                        pub_data['issn'] = id_value
                    elif id_type in ['uri', 'url'] and id_value:
                        if not pub_data['url']:
                            pub_data['url'] = id_value
                
                # Extract URL from external-id-url if no URL found yet
                if not pub_data['url']:
                    for ext_id in ext_id_list:
                        if not ext_id or not isinstance(ext_id, dict):
                            continue
                        ext_id_url = ext_id.get('external-id-url')
                        if ext_id_url and isinstance(ext_id_url, dict):
                            url = ext_id_url.get('value')
                            if url:
                                pub_data['url'] = url
                                break
        
        # Extract contributors (authors)
        contributors = work.get('contributors')
        if contributors and isinstance(contributors, dict):
            contributor_list = contributors.get('contributor', [])
            if contributor_list and isinstance(contributor_list, list):
                authors = []
                for contrib in contributor_list:
                    if not contrib or not isinstance(contrib, dict):
                        continue
                    credit_name = contrib.get('credit-name')
                    if credit_name and isinstance(credit_name, dict) and 'value' in credit_name:
                        author_name = credit_name['value']
                        if author_name:
                            authors.append(str(author_name).strip())
                pub_data['authors'] = authors
        
    except Exception as e:
        print(f"Warning: Error extracting data from work object: {e}")
        print(f"Work object keys: {list(work.keys()) if isinstance(work, dict) else 'Not a dict'}")
        return None
    
    return pub_data


def _orcid_to_canonical(works: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert ORCID works to canonical format."""
    canonical_pubs = []
    
    for work in works:
        # Skip None or empty works
        if not work:
            continue
            
        pub_data = _extract_publication_data(work)
        
        # Skip if extraction failed or no title
        if not pub_data or not pub_data['title']:
            continue
        
        canonical_pubs.append({
            'title': pub_data['title'],
            'authors': pub_data['authors'],
            'journal': pub_data['journal'],
            'year': pub_data['year'],
            'doi': pub_data['doi'],
            'issn': pub_data['issn'],
            'source': 'ORCID',
            'citations': pub_data['citations'],
            'url': pub_data['url'],
            'work_type': pub_data['work_type'],
            'publisher': pub_data['publisher']
        })
    
    return canonical_pubs


# ---------------------------------------------------------------------------
# 3) Public API
# ---------------------------------------------------------------------------

def fetch(
    orcid_id: str,
    client_id: str,
    client_secret: str,
    *,
    max_records: int = 1000,
) -> List[Publication]:
    """
    Harvest publications for an author from ORCID and return them as a list
    of :class:`pubcrawler.models.Publication`.

    Parameters
    ----------
    orcid_id : str
        The ORCID identifier (e.g., "0000-0000-0000-0000").
    client_id : str
        ORCID API client ID.
    client_secret : str
        ORCID API client secret.
    max_records : int, default 1000
        Maximum number of records to fetch.

    Returns
    -------
    List[Publication]
        List of Publication objects.
    """
    print(f"Fetching ORCID data for ID: {orcid_id}")
    
    # Initialize ORCID client
    client = ORCIDClient(client_id, client_secret)
    
    try:
        # Get works from ORCID
        works = client.get_works(orcid_id)
        
        if not works:
            print(f"No works found for ORCID ID: {orcid_id}")
            return []
        
        print(f"Found {len(works)} works from ORCID")
        
        # Limit results if specified
        if max_records and len(works) > max_records:
            works = works[:max_records]
            print(f"Limited to {max_records} works")
        
        # Convert to canonical format
        canonical_works = _orcid_to_canonical(works)
        
        # Convert to Publication objects
        publications = []
        for work_data in canonical_works:
            publications.append(
                Publication(
                    title=work_data['title'],
                    authors=work_data['authors'],
                    journal=work_data['journal'],
                    year=work_data['year'],
                    doi=work_data['doi'],
                    issn=work_data['issn'],
                    source=work_data['source'],
                    citations=work_data['citations'],
                    url=work_data['url']
                )
            )
        
        # Sort by year (descending) then by title
        publications.sort(key=lambda p: (-(p.year or 0), p.title or ''))
        
        print(f"✅ Successfully processed {len(publications)} publications from ORCID")
        return publications
        
    except Exception as e:
        print(f"❌ Error fetching from ORCID: {e}")
        raise


# ---------------------------------------------------------------------------
# 4) CLI for quick manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()
    
    ORCID_CLIENT_ID = os.environ.get("ORCID_CLIENT_ID")
    ORCID_CLIENT_SECRET = os.environ.get("ORCID_CLIENT_SECRET")
    ORCID_ID = "0000-0001-9690-6218"  # Replace with actual ORCID ID for testing

    if not ORCID_CLIENT_ID or not ORCID_CLIENT_SECRET:
        print("Error: ORCID_CLIENT_ID and ORCID_CLIENT_SECRET environment variables not set.")
        print("Please set them to your ORCID API credentials to run this test.")
    else:
        print(f"Fetching ORCID data for ID: {ORCID_ID}...")
        try:
            publications = fetch(
                orcid_id=ORCID_ID,
                client_id=ORCID_CLIENT_ID,
                client_secret=ORCID_CLIENT_SECRET,
            )

            print(f"\n✅ Fetched {len(publications)} publications from ORCID.")
            if publications:
                print("--- First 5 publications ---")
                for i, p in enumerate(publications[:5]):
                    print(f"  {i+1}. ({p.year or 'N/A'}) {p.title[:80]}...")
                    print(f"     Journal: {p.journal or 'N/A'}")
                    if p.doi:
                        print(f"     DOI: {p.doi}")
        except Exception as e:
            print(f"❌ Error: {e}")
