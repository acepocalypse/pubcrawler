"""pubcrawler.aggregate
~~~~~~~~~~~~~~~~~~~~~~
Core orchestration for the pubcrawler pipeline.

It aggregates data from configured sources (Google Scholar, Scopus, Web of Science),
runs harvesting in parallel (when available), and merges results into a single,
deduplicated list of publications.

Resilience improvements:
- Lazy, per-source imports so missing optional dependencies don't crash the whole app.
- Deduplication gracefully degrades when pandas/rapidfuzz aren't installed.
- A small CLI (and a package entry via ``python -m pubcrawler``) for easier running.

Public entry-point:
    aggregate_publications(author, api_keys, **options) -> List[Publication]
"""
from __future__ import annotations

import concurrent.futures
import sys
from typing import List, Dict, Optional, Tuple

# Avoid hard dependencies at import time; import where needed.
try:
    from .models import Author, Publication
    from .coverage import analyze_publication_coverage, print_coverage_report
except ImportError:
    # Allow running this file as a script for debugging (python aggregate.py)
    from models import Author, Publication  # type: ignore
    from coverage import analyze_publication_coverage, print_coverage_report  # type: ignore


def _deduplicate_publications(publications: List[Publication]) -> List[Publication]:
    """
    Deduplicate a list of publications using sophisticated matching algorithms.

    The strategy is as follows:
    1. Group publications by a sanitized DOI. Publications with a valid DOI are
       considered unique. The one with the highest citation count is kept.
    2. For publications without a DOI, group them by a combination of year and a
       fuzzy-matched title.
    3. The representative from each group is chosen based on citation count.
    
    Parameters
    ----------
    publications : List[Publication]
        List of publications to deduplicate.
        
    Returns
    -------
    List[Publication]
        Deduplicated list of publications, sorted by year and citations.
    """
    if not publications:
        return []

    # Clean all publications first to handle any NaN values
    cleaned_pubs = [_clean_publication_data(pub) for pub in publications]

    # Try to use pandas/rapidfuzz for strong dedup; otherwise do a minimal fallback.
    try:
        import pandas as pd  # type: ignore
        from rapidfuzz import fuzz, process  # type: ignore
        return _advanced_deduplication(cleaned_pubs, pd, fuzz, process)
    except ImportError:
        print("⚠️ Advanced deduplication unavailable (missing pandas/rapidfuzz). Using basic deduplication.")
        return _basic_deduplication(cleaned_pubs)


def _basic_deduplication(publications: List[Publication]) -> List[Publication]:
    """
    Basic deduplication using exact matches only.
    
    Fallback method when pandas/rapidfuzz are not available.
    """
    by_doi: Dict[str, Publication] = {}
    remaining: List[Publication] = []
    
    for pub in publications:
        # Clean the publication data first
        cleaned_pub = _clean_publication_data(pub)
        
        if cleaned_pub.doi:
            existing = by_doi.get(cleaned_pub.doi)
            if existing is None or (cleaned_pub.citations or 0) > (existing.citations or 0):
                by_doi[cleaned_pub.doi] = cleaned_pub
        else:
            remaining.append(cleaned_pub)
    
    # Naive dedup by (title, year)
    seen = set()
    result = list(by_doi.values())
    for pub in sorted(remaining, key=lambda x: (x.citations or 0), reverse=True):
        key = (pub.title, pub.year)
        if key in seen:
            continue
        seen.add(key)
        result.append(pub)
    
    # Sort by year (descending) then citations (descending)
    return sorted(result, key=lambda p: (p.year or 0, p.citations or 0), reverse=True)


def _clean_publication_data(pub: Publication) -> Publication:
    """Clean a Publication object to ensure proper data types and handle NaN values."""
    # Handle each field carefully
    title = pub.title if pub.title and not _is_empty_value(pub.title) else ""
    
    authors = pub.authors if isinstance(pub.authors, list) else []
    # Filter out empty authors
    authors = [author for author in authors if author and not _is_empty_value(author)]
    
    journal = pub.journal if pub.journal and not _is_empty_value(pub.journal) else None
    
    year = None
    if pub.year and not _is_empty_value(pub.year):
        try:
            year = int(float(pub.year))  # Handle cases where year might be float
        except (ValueError, TypeError):
            year = None
    
    doi = pub.doi if pub.doi and not _is_empty_value(pub.doi) else None
    issn = pub.issn if pub.issn and not _is_empty_value(pub.issn) else None
    source = pub.source if pub.source and not _is_empty_value(pub.source) else "Unknown"
    
    citations = 0
    if pub.citations and not _is_empty_value(pub.citations):
        try:
            citations = int(float(pub.citations))  # Handle float citations
        except (ValueError, TypeError):
            citations = 0
    
    url = pub.url if pub.url and not _is_empty_value(pub.url) else None
    
    # Create a new publication with cleaned data
    return Publication(
        title=title,
        authors=authors,
        journal=journal,
        year=year,
        doi=doi,
        issn=issn,
        source=source,
        citations=citations,
        url=url
    )


def _advanced_deduplication(publications: List[Publication], pd, fuzz, process) -> List[Publication]:
    """
    Advanced deduplication using pandas and fuzzy string matching.
    Enhanced to handle cross-database matching more effectively, including
    fuzzy matching between publications with and without DOIs.
    """
    df = pd.DataFrame([p.__dict__ for p in publications])
    df['doi'] = df['doi'].str.lower().str.strip().replace('', None)
    df['citations'] = pd.to_numeric(df['citations'], errors='coerce').fillna(0)
    df = df.sort_values('citations', ascending=False).reset_index(drop=True)

    # --- Step 1: Comprehensive deduplication using both DOI and fuzzy matching ---
    unique_pubs = []
    processed_indices = set()

    for idx, pub_row in df.iterrows():
        if idx in processed_indices:
            continue
        
        # Find all potential matches (both DOI and fuzzy)
        potential_matches = []
        
        for other_idx, other_row in df.iterrows():
            if other_idx <= idx or other_idx in processed_indices:
                continue
            
            # Check for DOI match first (highest priority)
            if (pub_row['doi'] and other_row['doi'] and 
                pub_row['doi'] == other_row['doi']):
                potential_matches.append(other_idx)
            # Then check for fuzzy match (including DOI vs no-DOI cases)
            elif _is_fuzzy_match(pub_row, other_row, fuzz):
                potential_matches.append(other_idx)
        
        # Create merged publication with source information
        merged_pub = _merge_duplicate_sources(pub_row, df, potential_matches)
        unique_pubs.append(merged_pub)
        
        # Mark all matches as processed
        processed_indices.add(idx)
        for match_idx in potential_matches:
            processed_indices.add(match_idx)

    # Convert back to dataframe and sort
    final_df = pd.DataFrame(unique_pubs)
    if not final_df.empty:
        # Clean up NaN values before converting to Publication objects
        final_df = _clean_dataframe_for_json(final_df, pd)
        final_df = final_df.sort_values(['year', 'citations'], ascending=[False, False])
        
        # Convert back to Publication objects, excluding internal tracking fields
        valid_fields = {
            'title', 'authors', 'journal', 'year', 'doi', 'issn', 
            'source', 'citations', 'url'
        }
        
        final_pubs = []
        for idx, row in final_df.iterrows():
            # Convert row to dict
            row_dict = row.to_dict()
            
            # Only include fields that are valid for Publication constructor
            pub_data = {k: v for k, v in row_dict.items() if k in valid_fields}
            
            # Handle NaN values that might cause issues
            for key, value in pub_data.items():
                # Handle different data types appropriately
                if key == 'authors':
                    # Authors should be a list
                    if value is None or (hasattr(value, '__len__') and len(value) == 0):
                        pub_data[key] = []
                    elif isinstance(value, str):
                        pub_data[key] = [value]
                    elif isinstance(value, list):
                        pub_data[key] = [str(item) for item in value if item is not None]
                    else:
                        pub_data[key] = []
                elif isinstance(value, (list, tuple, set)):
                    # Handle unexpected arrays/lists for non-author fields
                    if len(value) > 0:
                        pub_data[key] = value[0]  # Take first element
                    else:
                        pub_data[key] = None
                else:
                    # Handle scalar values
                    try:
                        is_na = pd.isna(value)
                    except (ValueError, TypeError):
                        # If pd.isna() fails, treat as not-NA
                        is_na = value is None
                    
                    if is_na:
                        if key == 'citations':
                            pub_data[key] = 0
                        elif key in ['year']:
                            pub_data[key] = None
                        elif key in ['title', 'source']:
                            pub_data[key] = ""
                        else:
                            pub_data[key] = None
            
            final_pubs.append(Publication(**pub_data))
        
    else:
        final_pubs = []
    
    return final_pubs


def _clean_dataframe_for_json(df, pd):
    """Clean DataFrame by replacing NaN/None values appropriately for JSON serialization."""
    import numpy as np
    
    # Replace NaN with None for optional string fields
    string_fields = ['doi', 'issn', 'journal', 'url']
    for field in string_fields:
        if field in df.columns:
            # Use a safer method to replace NaN values
            df[field] = df[field].apply(lambda x: None if _is_empty_value(x) else x)
    
    # Handle year field specially
    if 'year' in df.columns:
        df['year'] = df['year'].apply(lambda x: None if _is_empty_value(x) else (int(x) if x is not None else None))
    
    # Handle citations field
    if 'citations' in df.columns:
        df['citations'] = df['citations'].apply(lambda x: 0 if _is_empty_value(x) else int(x))
    
    # Ensure authors is always a list
    if 'authors' in df.columns:
        df['authors'] = df['authors'].apply(lambda x: x if isinstance(x, list) and len(x) > 0 else [])
    
    # Handle title field
    if 'title' in df.columns:
        df['title'] = df['title'].apply(lambda x: str(x) if not _is_empty_value(x) else "")
    
    # Handle source field
    if 'source' in df.columns:
        df['source'] = df['source'].apply(lambda x: str(x) if not _is_empty_value(x) else "Unknown")
    
    return df


def _is_fuzzy_match(pub1_row, pub2_row, fuzz) -> bool:
    """
    Determine if two publication rows are the same using comprehensive fuzzy matching.
    This matches the logic used in app.py for consistency.
    """
    # Enhanced fuzzy matching
    title_match = False
    year_match = False
    author_match = False
    journal_match = False
    
    # Title matching with multiple strategies
    if pub1_row['title'] and pub2_row['title']:
        norm_title1 = _normalize_title_for_matching(pub1_row['title'])
        norm_title2 = _normalize_title_for_matching(pub2_row['title'])
        
        title_ratio = fuzz.ratio(norm_title1, norm_title2)
        token_ratio = fuzz.token_set_ratio(norm_title1, norm_title2)
        partial_ratio = fuzz.partial_ratio(norm_title1, norm_title2)
        
        best_title_similarity = max(title_ratio, token_ratio, partial_ratio)
        title_match = best_title_similarity >= 80
        
        # For very high similarity, be more confident
        if best_title_similarity >= 95:
            title_match = True
    
    # Year matching (allow 1 year difference for cross-database variations)
    if pub1_row['year'] and pub2_row['year']:
        year_match = abs(pub1_row['year'] - pub2_row['year']) <= 1
    
    # Author matching
    if pub1_row['authors'] and pub2_row['authors']:
        author_similarity = _calculate_author_similarity_for_dedup(pub1_row['authors'], pub2_row['authors'], fuzz)
        author_match = author_similarity >= 0.3
    
    # Journal matching with enhanced normalization
    if pub1_row['journal'] and pub2_row['journal']:
        norm_journal1 = _normalize_title_for_matching(pub1_row['journal'])
        norm_journal2 = _normalize_title_for_matching(pub2_row['journal'])
        
        # Handle common journal name variations
        norm_journal1 = _normalize_journal_name(norm_journal1)
        norm_journal2 = _normalize_journal_name(norm_journal2)
        
        journal_similarity = max(
            fuzz.ratio(norm_journal1, norm_journal2),
            fuzz.partial_ratio(norm_journal1, norm_journal2),
            fuzz.token_set_ratio(norm_journal1, norm_journal2)
        )
        journal_match = journal_similarity >= 70
    
    # Confidence scoring with enhanced logic
    confidence_score = 0
    
    # Title is the most important indicator
    if title_match:
        if best_title_similarity >= 95:
            confidence_score += 4.0
        elif best_title_similarity >= 90:
            confidence_score += 3.5
        elif best_title_similarity >= 85:
            confidence_score += 3.0
        else:
            confidence_score += 2.5
    
    # Year matching adds confidence
    if year_match:
        confidence_score += 2.0
    
    # Author matching is important
    if author_match:
        confidence_score += 2.0
    
    # Journal matching provides additional confirmation
    if journal_match:
        confidence_score += 1.5
    
    # Decision logic - more flexible for cross-database variations
    # High confidence match
    if confidence_score >= 4.0:
        return True
    # Strong title + year combination (common for same paper)
    elif title_match and year_match and confidence_score >= 3.5:
        return True
    # Very strong title match alone (for cases with missing data)
    elif title_match and best_title_similarity >= 95 and confidence_score >= 3.0:
        return True
    # Title + authors match (good indicator)
    elif title_match and author_match and confidence_score >= 3.0:
        return True
    # Title + journal match (also good indicator)
    elif title_match and journal_match and confidence_score >= 3.0:
        return True
    
    return False


def _normalize_journal_name(journal_name: str) -> str:
    """Normalize journal names to handle common variations."""
    if not journal_name:
        return ""
    
    import re
    
    # Common journal name normalizations
    normalized = journal_name.lower().strip()
    
    # Remove common suffixes that might differ
    normalized = re.sub(r',?\s*\d{4}.*$', '', normalized)  # Remove year and everything after
    normalized = re.sub(r'\s*\([^)]*\)$', '', normalized)  # Remove parentheses at end
    
    # Standardize common abbreviations
    abbreviations = {
        'oecologia': 'oecologia',
        'ieee trans': 'ieee transactions',
        'ieee transactions': 'ieee transactions',
        'acm trans': 'acm transactions',
        'acm transactions': 'acm transactions',
        'proc natl acad sci': 'proceedings national academy sciences',
        'pnas': 'proceedings national academy sciences',
        'j exp biol': 'journal experimental biology',
        'j comp physiol': 'journal comparative physiology',
        'nature': 'nature',
        'science': 'science',
        'cell': 'cell',
        'plos one': 'plos one',
    }
    
    for abbrev, full in abbreviations.items():
        if abbrev in normalized:
            normalized = normalized.replace(abbrev, full)
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


def _normalize_title_for_matching(text: str) -> str:
    """Normalize text for better fuzzy matching during deduplication."""
    if not text:
        return ""
    
    import re
    
    # Convert to lowercase and strip
    normalized = text.lower().strip()
    
    # Remove extra whitespace and normalize spacing
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove common punctuation that might vary between sources
    normalized = re.sub(r'[.,;:!?"\'\(\)\[\]{}]', '', normalized)
    
    # Remove common prefixes/suffixes that might differ
    normalized = re.sub(r'\b(the|a|an)\b', '', normalized)
    
    # Handle common abbreviations and variations
    normalized = re.sub(r'\bvol\.?\s*', '', normalized)
    normalized = re.sub(r'\bno\.?\s*', '', normalized)
    normalized = re.sub(r'\bpp\.?\s*', '', normalized)
    normalized = re.sub(r'\bissue\s*', '', normalized)
    normalized = re.sub(r'\bvolume\s*', '', normalized)
    normalized = re.sub(r'\bnumber\s*', '', normalized)
    
    # Remove page numbers and volume/issue info that might differ
    normalized = re.sub(r'\b\d+[-–—]\d+\b', '', normalized)  # page ranges
    normalized = re.sub(r'\b\d+\s*\(\d+\)\b', '', normalized)  # vol(issue)
    
    # Remove years in parentheses or standalone
    normalized = re.sub(r'\b(19|20)\d{2}\b', '', normalized)
    
    # Clean up multiple spaces created by removals
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized.strip()


def _calculate_author_similarity_for_dedup(authors1, authors2, fuzz) -> float:
    """Calculate similarity between two author lists during deduplication."""
    import re
    
    if not authors1 or not authors2:
        return 0.0
    
    # Handle case where authors might be stored as string vs list
    if isinstance(authors1, str):
        authors1 = [authors1]
    if isinstance(authors2, str):
        authors2 = [authors2]
    
    # Normalize author names
    def normalize_author(author):
        if not author:
            return ""
        
        normalized = author.lower().strip()
        # Remove titles and suffixes
        normalized = re.sub(r'\b(dr|prof|professor|phd|md|jr|sr|ii|iii)\b\.?', '', normalized)
        # Remove extra spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    norm_authors1 = [normalize_author(author) for author in authors1 if author]
    norm_authors2 = [normalize_author(author) for author in authors2 if author]
    
    # Remove empty authors
    norm_authors1 = [a for a in norm_authors1 if a]
    norm_authors2 = [a for a in norm_authors2 if a]
    
    if not norm_authors1 or not norm_authors2:
        return 0.0
    
    # Count matches using fuzzy matching
    matches = 0
    
    for author1 in norm_authors1:
        best_match = 0
        for author2 in norm_authors2:
            similarity = fuzz.ratio(author1, author2)
            best_match = max(best_match, similarity)
        
        if best_match >= 75:
            matches += 1
    
    # Calculate similarity based on shorter list (handle "et al." cases)
    min_authors = min(len(norm_authors1), len(norm_authors2))
    base_similarity = matches / min_authors if min_authors > 0 else 0.0
    
    return base_similarity



def _merge_duplicate_sources(base_row, df, match_indices):
    """Merge a publication with its fuzzy matches, combining source information and preserving citation data."""
    import pandas as pd
    import numpy as np
    
    merged_pub = base_row.to_dict()
    
    if match_indices:
        # Collect all sources with their citation counts
        source_citations = []
        source_citations.append({
            'source': base_row['source'],
            'citations': base_row.get('citations', 0)
        })
        
        for idx in match_indices:
            match_row = df.loc[idx]
            source_citations.append({
                'source': match_row['source'],
                'citations': match_row.get('citations', 0)
            })
        
        # Create detailed source information
        unique_sources = []
        citation_details = {}
        max_citations = 0
        
        for item in source_citations:
            source = item['source']
            citations = item['citations'] if not _is_empty_value(item['citations']) else 0
            
            if source not in [s['source'] for s in unique_sources]:
                unique_sources.append({'source': source, 'citations': citations})
                citation_details[source] = citations
                max_citations = max(max_citations, citations)
        
        # Format source information with citation details
        if len(unique_sources) > 1:
            source_parts = []
            for item in unique_sources:
                source_parts.append(f"{item['source']} ({item['citations']} cites)")
            merged_pub['source'] = f"Multiple sources: {', '.join(source_parts)}"
            
            # Store citation details for later use
            merged_pub['_citation_details'] = citation_details
        
        # Use the highest citation count as the main citation count
        merged_pub['citations'] = max_citations
        
        # Use the best available data from all matches
        for idx in match_indices:
            match_row = df.loc[idx]
            
            # Fill in missing DOI if available
            current_doi = merged_pub.get('doi')
            match_doi = match_row.get('doi')
            if (_is_empty_value(current_doi)) and not _is_empty_value(match_doi):
                merged_pub['doi'] = match_doi
            
            # Fill in missing URL if available (prefer DOI source)
            current_url = merged_pub.get('url')
            match_url = match_row.get('url')
            if _is_empty_value(current_url) and not _is_empty_value(match_url):
                merged_pub['url'] = match_url
            elif (not _is_empty_value(match_doi) and not _is_empty_value(match_url) and 
                  _is_empty_value(merged_pub.get('doi'))):
                # Prefer URL from source that has DOI
                merged_pub['url'] = match_url
            
            # Use more complete title if one is significantly longer
            match_title = match_row.get('title', '')
            current_title = merged_pub.get('title', '')
            if not _is_empty_value(match_title) and len(str(match_title)) > len(str(current_title)):
                merged_pub['title'] = match_title
            
            # Fill in missing year if available
            current_year = merged_pub.get('year')
            match_year = match_row.get('year')
            if _is_empty_value(current_year) and not _is_empty_value(match_year):
                merged_pub['year'] = match_year
            
            # Use more complete journal name if available
            match_journal = match_row.get('journal', '')
            current_journal = merged_pub.get('journal', '')
            if (not _is_empty_value(match_journal) and 
                (_is_empty_value(current_journal) or len(str(match_journal)) > len(str(current_journal)))):
                merged_pub['journal'] = match_journal
    
    # Clean up any problematic values
    for key, value in merged_pub.items():
        if key != '_citation_details' and _is_empty_value(value):
            if key in ['citations']:
                merged_pub[key] = 0
            elif key in ['authors']:
                merged_pub[key] = []
            else:
                merged_pub[key] = None
    
    return merged_pub


def _is_empty_value(value):
    """Check if a value is empty, None, NaN, or otherwise should be considered missing."""
    import pandas as pd
    import numpy as np
    
    if value is None:
        return True
    
    # Handle scalar values
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        # If pd.isna fails (e.g., on lists), check other conditions
        pass
    
    # Handle string representations of missing values
    if isinstance(value, str):
        return value.strip() == '' or value.lower() in ['nan', 'none', 'null']
    
    # Handle lists/arrays
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    
    # Handle numpy arrays
    if hasattr(value, '__len__') and hasattr(value, 'dtype'):
        try:
            return np.isnan(value).all() if value.dtype.kind in 'fc' else len(value) == 0
        except (TypeError, ValueError):
            return len(value) == 0
    
    return False


def aggregate_publications(
    author: Author,
    api_keys: Dict[str, str],
    *,
    max_pubs_g_scholar: int = 100,
    headless_g_scholar: bool = False,
    analyze_coverage: bool = True,
) -> List[Publication]:
    """
    Fetches publications from all sources in parallel, then merges and deduplicates.

    Parameters
    ----------
    author : Author
        The author to search for.
    api_keys : dict
        A dictionary containing the required API keys, e.g.,
        {'scopus_api_key': '...', 'wos_api_key': '...'}.
    max_pubs_g_scholar : int, default 100
        Max publications to fetch from Google Scholar.
    headless_g_scholar : bool, default False
        Whether to run the Google Scholar scraper in headless mode.
    analyze_coverage : bool, default True
        Whether to perform index coverage analysis.

    Returns
    -------
    List[Publication]
        A single, deduplicated list of publications.
    """
    all_publications: List[Publication] = []
    available_sources: List[str] = []

    # Lazy import sources so missing optional deps don't crash the whole run
    gs_mod = None
    scopus_mod = None
    wos_mod = None
    try:
        try:
            from .sources import google_scholar as gs_mod  # type: ignore
        except ImportError:
            from sources import google_scholar as gs_mod  # type: ignore
    except Exception as e:
        print(f"⚠️ Google Scholar disabled: {e}")
    try:
        try:
            from .sources import scopus as scopus_mod  # type: ignore
        except ImportError:
            from sources import scopus as scopus_mod  # type: ignore
    except Exception as e:
        print(f"⚠️ Scopus disabled: {e}")
    try:
        try:
            from .sources import wos as wos_mod  # type: ignore
        except ImportError:
            from sources import wos as wos_mod  # type: ignore
    except Exception as e:
        print(f"⚠️ Web of Science disabled: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_source: Dict[concurrent.futures.Future, str] = {}

        # --- Submit Google Scholar task ---
        if author.gs_id and gs_mod is not None:
            future_gs = executor.submit(
                gs_mod.fetch,
                gs_id=author.gs_id,
                first_name=author.first_name,
                last_name=author.last_name,
                max_publications_detail=max_pubs_g_scholar,
                headless=headless_g_scholar,
            )
            future_to_source[future_gs] = "Google Scholar"
            available_sources.append("Google Scholar")

        # --- Submit Scopus task ---
        if api_keys.get("scopus_api_key") and (author.scopus_id or author.affiliation) and scopus_mod is not None:
            if author.scopus_id:
                # Use Scopus ID for direct lookup
                future_scopus = executor.submit(
                    scopus_mod.fetch,
                    scopus_id=author.scopus_id,
                    api_key=api_keys["scopus_api_key"],
                )
            else:
                # Fallback to name-based search (if the old method is still available)
                future_scopus = executor.submit(
                    scopus_mod.fetch,
                    first_name=author.first_name,
                    last_name=author.last_name,
                    affiliation=author.affiliation,
                    api_key=api_keys["scopus_api_key"],
                )
            future_to_source[future_scopus] = "Scopus"
            available_sources.append("Scopus")

        # --- Submit Web of Science task ---
        if api_keys.get("wos_api_key") and author.affiliation and wos_mod is not None:
            future_wos = executor.submit(
                wos_mod.fetch,
                first_name=author.first_name,
                last_name=author.last_name,
                affiliation=author.affiliation,
                api_key=api_keys["wos_api_key"],
            )
            future_to_source[future_wos] = "Web of Science"
            available_sources.append("Web of Science")

        # --- Collect results ---
        for future in concurrent.futures.as_completed(future_to_source):
            source_name = future_to_source[future]
            try:
                pubs = future.result()
                print(f"✅ Fetched {len(pubs)} publications from {source_name}.")
                all_publications.extend(pubs)
            except Exception as exc:
                print(f"❌ {source_name} generated an exception: {exc}")

    print(f"\nTotal publications fetched before deduplication: {len(all_publications)}")
    
    # --- Deduplicate ---
    deduplicated_pubs = _deduplicate_publications(all_publications)
    
    print(f"Total publications after deduplication: {len(deduplicated_pubs)}")

    # --- Coverage Analysis ---
    if analyze_coverage and available_sources:
        print(f"\n🔍 Analyzing index coverage across {len(available_sources)} sources...")
        try:
            print_coverage_report(deduplicated_pubs, available_sources)
        except Exception as e:
            print(f"⚠️ Coverage analysis failed: {e}")

    return deduplicated_pubs


def main() -> None:
    """Command-line interface for the pubcrawler aggregate module."""
    import os
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate publications for a researcher from multiple sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --first John --last Doe --affiliation "MIT" --gs-id ABC123
  %(prog)s --first Jane --last Smith --scopus-key YOUR_KEY --max-gs 50
        """
    )
    
    # Author information
    parser.add_argument("--first", dest="first_name", required=True,
                       help="Author's first name")
    parser.add_argument("--last", dest="last_name", required=True,
                       help="Author's last name")
    parser.add_argument("--affiliation", 
                       help="Author's institutional affiliation")
    parser.add_argument("--gs-id", dest="gs_id", 
                       help="Google Scholar author ID")
    parser.add_argument("--scopus-id", dest="scopus_id",
                       help="Scopus author ID")
    
    # API keys
    parser.add_argument("--scopus-key", dest="scopus_api_key", 
                       default=os.environ.get("SCOPUS_API_KEY"),
                       help="Scopus API key (or set SCOPUS_API_KEY env var)")
    parser.add_argument("--wos-key", dest="wos_api_key", 
                       default=os.environ.get("WOS_API_KEY"),
                       help="Web of Science API key (or set WOS_API_KEY env var)")
    
    # Options
    parser.add_argument("--max-gs", dest="max_pubs_g_scholar", type=int, default=100,
                       help="Maximum publications to fetch from Google Scholar (default: 100)")
    parser.add_argument("--headless", action="store_true", 
                       help="Run Google Scholar scraping in headless mode")
    parser.add_argument("--no-coverage", action="store_true",
                       help="Skip index coverage analysis")
    
    args = parser.parse_args()

    # Validate required information
    if not (args.gs_id or args.scopus_id or args.affiliation):
        parser.error("At least one of --gs-id, --scopus-id, or --affiliation is required")

    # Prepare API keys
    api_keys = {
        "scopus_api_key": args.scopus_api_key,
        "wos_api_key": args.wos_api_key,
    }

    # Create author object
    author = Author(
        first_name=args.first_name,
        last_name=args.last_name,
        affiliation=args.affiliation,
        gs_id=args.gs_id,
        scopus_id=args.scopus_id,
    )

    # Show configuration
    print(f"🔍 Searching for publications by {author.full_name}")
    if author.affiliation:
        print(f"   📍 Affiliation: {author.affiliation}")
    if author.gs_id:
        print(f"   🎓 Google Scholar ID: {author.gs_id}")
    if author.scopus_id:
        print(f"   🔬 Scopus ID: {author.scopus_id}")
    
    available_sources = []
    if author.gs_id:
        available_sources.append("Google Scholar")
    if api_keys.get("scopus_api_key") and (author.scopus_id or author.affiliation):
        available_sources.append("Scopus")
    if api_keys.get("wos_api_key") and author.affiliation:
        available_sources.append("Web of Science")
    
    if not available_sources:
        print("❌ No data sources available. Please provide:")
        print("   • Google Scholar ID (--gs-id)")
        print("   • Scopus ID (--scopus-id)")
        print("   • Scopus API key + affiliation (--scopus-key, --affiliation)")
        print("   • Web of Science API key + affiliation (--wos-key, --affiliation)")
        sys.exit(1)
    
    print(f"   📚 Sources: {', '.join(available_sources)}")
    print()

    # Run aggregation
    try:
        final_publications = aggregate_publications(
            author=author,
            api_keys=api_keys,
            max_pubs_g_scholar=args.max_pubs_g_scholar,
            headless_g_scholar=args.headless,
            analyze_coverage=not args.no_coverage,
        )

        if final_publications:
            print(f"\n✅ Aggregated {len(final_publications)} unique publications.")
            
            # Display results
            _display_results(final_publications)
        else:
            print("\n❌ No publications found.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during aggregation: {e}")
        sys.exit(1)


def _display_results(publications: List[Publication]) -> None:
    """Display publication results in a formatted manner."""
    try:
        import pandas as pd  # type: ignore
        
        results_df = pd.DataFrame([p.__dict__ for p in publications])
        
        print("\n--- Top 10 Publications (by Year, then Citations) ---")
        display_cols = [c for c in ["year", "citations", "title", "doi", "source"] 
                       if c in results_df.columns]
        
        top_10 = results_df.head(10)[display_cols]
        # Truncate title for display
        if "title" in top_10.columns:
            top_10["title"] = top_10["title"].str[:80] + "..."
        
        print(top_10.to_string(index=False))
        
        print("\n--- Source Distribution ---")
        source_counts = results_df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count} publications")
            
        print(f"\n--- Summary Statistics ---")
        total_citations = results_df['citations'].sum()
        avg_citations = results_df['citations'].mean()
        year_range = f"{results_df['year'].min()}-{results_df['year'].max()}"
        with_doi = results_df['doi'].notna().sum()
        
        print(f"  Total citations: {total_citations:,}")
        print(f"  Average citations per paper: {avg_citations:.1f}")
        print(f"  Publication years: {year_range}")
        print(f"  Publications with DOI: {with_doi}/{len(results_df)} ({with_doi/len(results_df)*100:.1f}%)")
        
    except ImportError:
        # Fallback display without pandas
        print("\n--- Top 10 Publications ---")
        for i, pub in enumerate(publications[:10], 1):
            print(f"{i:2d}. ({pub.year or 'N/A'}) [{pub.citations or 0} cites] {pub.title[:80]}...")
            print(f"     Source: {pub.source}")
            if pub.doi:
                print(f"     DOI: {pub.doi}")
        
        # Simple source count
        sources = {}
        for pub in publications:
            sources[pub.source] = sources.get(pub.source, 0) + 1
        
        print("\n--- Source Distribution ---")
        for source, count in sources.items():
            print(f"  {source}: {count} publications")


if __name__ == '__main__':
    main()
