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

import argparse
import concurrent.futures
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Optional

# Avoid hard dependencies at import time; import where needed.
try:
    from .coverage import analyze_publication_coverage, print_coverage_report
    from .models import Author, Publication
except ImportError:
    # Allow running this file as a script for debugging (python aggregate.py)
    from coverage import analyze_publication_coverage, print_coverage_report  # type: ignore
    from models import Author, Publication  # type: ignore

# --- Pre-compiled Regular Expressions for Performance ---
_DOI_VALIDATION_PATTERN = re.compile(r"^10\.\d{4,9}/[^\s]+$", re.IGNORECASE)
_CORRECTION_TITLE_PATTERNS = [
    re.compile(r"^\s*(erratum|corrigendum|correction|retraction|expression of concern)\b", re.IGNORECASE),
    re.compile(r"\[\s*(erratum|corrigendum|correction|retraction)\s*\]", re.IGNORECASE),
    re.compile(r"^\s*(editorial|commentary|reply|response|letter( to the editor)?)\b", re.IGNORECASE),
]


def _normalize_doi(raw: Optional[str]) -> Optional[str]:
    """Normalize DOI strings and validate. Supports DOI URLs and 'doi:' prefixes."""
    if not raw:
        return None

    doi = str(raw).strip().lower()

    # Strip URL and common prefixes
    doi = re.sub(r"^\s*(https?://(dx\.)?doi\.org/)", "", doi)
    doi = re.sub(r"^\s*doi:\s*", "", doi)

    # Trim wrapping punctuation/spaces/brackets
    doi = doi.strip().strip(" .;,:<>[]()")

    # Drop URL query/fragment and any trailing whitespace after the DOI
    doi = re.split(r"[?#\s]", doi)[0]

    # Remove zero-width/invisible spaces and other whitespace
    doi = re.sub(r"[\u200B-\u200D\u2060]", "", doi)
    doi = re.sub(r"\s*/\s*", "/", doi)
    doi = doi.replace(" ", "")

    # Basic DOI validation
    if not _DOI_VALIDATION_PATTERN.match(doi):
        return None
    return doi


def _is_empty_value(value) -> bool:
    """Check if a value is empty, None, NaN, or otherwise should be considered missing."""
    if value is None:
        return True

    # Handle pandas NaN for optional dependency
    try:
        import pandas as pd
        if pd.isna(value):
            return True
    except (TypeError, ValueError, ImportError):
        pass  # Fallback to other checks

    # Handle string representations of missing values
    if isinstance(value, str):
        return value.strip() == "" or value.lower() in ["nan", "none", "null"]

    # Handle empty collections
    if isinstance(value, (list, tuple)):
        return not value

    return False


def _clean_publication_data(pub: Publication) -> Publication:
    """Clean a Publication object to ensure proper data types and handle missing values."""
    title = "" if _is_empty_value(pub.title) else pub.title
    authors = [author for author in (pub.authors or []) if not _is_empty_value(author)]
    journal = None if _is_empty_value(pub.journal) else pub.journal
    issn = None if _is_empty_value(pub.issn) else pub.issn
    source = "Unknown" if _is_empty_value(pub.source) else pub.source
    url = None if _is_empty_value(pub.url) else pub.url

    year = None
    if not _is_empty_value(pub.year):
        try:
            year = int(float(pub.year))  # Handle cases like "2020.0"
        except (ValueError, TypeError):
            year = None

    citations = 0
    if not _is_empty_value(pub.citations):
        try:
            citations = int(float(pub.citations))
        except (ValueError, TypeError):
            citations = 0

    # Normalize DOI from either the 'doi' field or a doi.org URL
    norm_doi = _normalize_doi(pub.doi)
    if not norm_doi and isinstance(url, str) and "doi.org" in url.lower():
        norm_doi = _normalize_doi(url)

    return Publication(
        title=title,
        authors=authors,
        journal=journal,
        year=year,
        doi=norm_doi,
        issn=issn,
        source=source,
        citations=citations,
        url=url,
    )


def _normalize_text_for_matching(text: str) -> str:
    """Enhanced text normalization for better fuzzy matching during deduplication."""
    if not text:
        return ""

    # 1. Unicode normalize and strip diacritics (e.g., "café" -> "cafe")
    def _latinize(s: str) -> str:
        nkfd = unicodedata.normalize("NFKD", s)
        return "".join(c for c in nkfd if not unicodedata.combining(c))

    # 2. Unify common punctuation variants
    text = text.replace("–", "-").replace("—", "-").replace("’", "'").replace("“", '"').replace("”", '"')

    # 3. Lowercase, trim, and latinize
    normalized = _latinize(text.lower().strip())

    # 4. Remove common punctuation that might vary between sources
    normalized = re.sub(r"[.,;:!?\"'()\[\]{}]", "", normalized)

    # 5. Remove common prefixes/suffixes that might differ (articles, metadata)
    normalized = re.sub(r"\b(the|a|an|abstract|full text|pdf|html|preprint|arxiv)\b", "", normalized)

    # 6. Remove common abbreviations and metadata terms
    normalized = re.sub(r"\b(vol|no|pp|issue|volume|number)\.?\s*", "", normalized)

    # 7. Remove page numbers, volume/issue info, and years
    normalized = re.sub(r"\b\d+[-–—]\d+\b", "", normalized)  # page ranges
    normalized = re.sub(r"\b\d+\s*\(\d+\)\b", "", normalized)  # vol(issue)
    normalized = re.sub(r"\b(19|20)\d{2}\b", "", normalized)  # years

    # 8. Collapse all whitespace to single spaces and trim
    return re.sub(r"\s+", " ", normalized).strip()


def _calculate_author_similarity(authors1: List[str], authors2: List[str]) -> float:
    """Calculate an order-insensitive author similarity score, lenient for truncated lists."""
    if not authors1 or not authors2:
        return 0.0

    def normalize_author(name: str) -> str:
        if not name:
            return ""
        name = name.lower().strip()
        name = re.sub(r"\b(dr|prof|professor|phd|md|jr|sr|ii|iii|mr|ms|mrs)\b\.?", "", name)
        name = re.sub(r"[.,]", " ", name)  # Keep token boundaries
        return re.sub(r"\s+", " ", name).strip()

    norm_authors1 = [s for s in (normalize_author(a) for a in authors1) if s]
    norm_authors2 = [s for s in (normalize_author(b) for b in authors2) if s]

    if not norm_authors1 or not norm_authors2:
        return 0.0

    # Detect if either list is likely truncated
    is_truncated1 = any("et al" in str(a).lower() or "..." in str(a) for a in authors1)
    is_truncated2 = any("et al" in str(b).lower() or "..." in str(b) for b in authors2)
    len_ratio = min(len(norm_authors1), len(norm_authors2)) / max(len(norm_authors1), len(norm_authors2))
    is_likely_truncated = is_truncated1 or is_truncated2 or len_ratio < 0.5

    try:
        from rapidfuzz import fuzz
    except ImportError:
        # Fallback logic without rapidfuzz
        if is_likely_truncated:
            return 1.0 if any(a == b for a in norm_authors1 for b in norm_authors2) else 0.3
        matches = sum(1 for a in norm_authors1 for b in norm_authors2 if a == b or set(a.split()) == set(b.split()))
        return min(1.0, matches / min(len(norm_authors1), len(norm_authors2)))

    # Enhanced logic with rapidfuzz
    if is_likely_truncated:
        # For truncated lists, be very generous. A single good match is significant.
        best_match_score = 0
        for a in norm_authors1:
            for b in norm_authors2:
                similarity = max(
                    fuzz.ratio(a, b),
                    fuzz.token_sort_ratio(a, b),
                    fuzz.token_set_ratio(a, b),
                )
                best_match_score = max(best_match_score, similarity)
        return min(1.0, best_match_score / 100 + 0.2) if best_match_score >= 60 else 0.2

    # Standard logic for complete author lists
    first_author_similarity = max(
        fuzz.ratio(norm_authors1[0], norm_authors2[0]),
        fuzz.token_sort_ratio(norm_authors1[0], norm_authors2[0]),
    )
    first_author_match = first_author_similarity >= 85

    matches = 0
    for a in norm_authors1:
        best_score = 0
        for b in norm_authors2:
            best_score = max(best_score, fuzz.token_set_ratio(a, b))
        if best_score >= 85:
            matches += 1

    shorter_list_len = min(len(norm_authors1), len(norm_authors2))
    base_score = matches / shorter_list_len
    if first_author_match:
        base_score = min(1.0, base_score + 0.15)  # Bonus for matching first author
    return base_score


def _last_name_overlap(authors1: List[str], authors2: List[str]) -> float:
    """Calculate a conservative last-name overlap score (0-1), lenient for truncated lists."""
    if not authors1 or not authors2:
        return 0.0

    def extract_last_name(raw: str) -> Optional[str]:
        if not raw or "et al" in str(raw).lower():
            return None
        s = str(raw).strip()
        # Prefer surname before comma: 'Fanton, AC' -> 'Fanton'
        surname_part = s.split(",", 1)[0] if "," in s else s
        # Strip titles/suffixes
        surname_part = re.sub(r"\b(dr|prof|phd|md|jr|sr|ii|iii)\b\.?", "", surname_part, flags=re.IGNORECASE)
        # Normalize and keep only letters/hyphens
        surname_part = "".join(c for c in unicodedata.normalize("NFKD", surname_part) if not unicodedata.combining(c))
        surname_part = re.sub(r"[^A-Za-z\- ]", " ", surname_part).lower().strip()
        if not surname_part:
            return None
        # Pick rightmost non-initial token; fallback to the longest
        parts = [p for p in surname_part.split() if p]
        non_initials = [p for p in parts if len(p) > 1]
        return non_initials[-1] if non_initials else max(parts, key=len, default=None)

    last_names1 = {ln for ln in (extract_last_name(a) for a in authors1) if ln}
    last_names2 = {ln for ln in (extract_last_name(a) for a in authors2) if ln}
    if not last_names1 or not last_names2:
        return 0.0

    intersection_len = len(last_names1 & last_names2)
    # Be more generous for truncated lists where any overlap is significant
    is_truncated = any("et al" in str(a).lower() for a in authors1 + authors2)
    len_ratio = min(len(authors1), len(authors2)) / max(len(authors1), len(authors2))
    if is_truncated or len_ratio < 0.5:
        return min(1.0, intersection_len / min(len(last_names1), len(last_names2)) * 2) if intersection_len > 0 else 0.0

    return intersection_len / min(len(last_names1), len(last_names2))


def _is_correction_or_note(title: str) -> bool:
    """Detect if a title represents an erratum, correction, retraction, or editorial note."""
    if not title:
        return False
    t_lower = str(title).strip().lower()
    return any(pattern.search(t_lower) for pattern in _CORRECTION_TITLE_PATTERNS)


def _publications_match(pub1: Publication, pub2: Publication) -> bool:
    """Determine if two publications are likely duplicates using fuzzy matching."""
    # --- Highest Priority: DOI Match ---
    doi1 = _normalize_doi(pub1.doi)
    doi2 = _normalize_doi(pub2.doi)
    if doi1 and doi2:
        return doi1 == doi2

    # Avoid self-comparison within the same source harvest
    if pub1.source.lower() == pub2.source.lower():
        return False

    # --- Pre-flight Checks ---
    if not (pub1.title and pub2.title):
        return False
    if _is_correction_or_note(pub1.title) != _is_correction_or_note(pub2.title):
        return False

    # --- Fuzzy Matching Logic (requires rapidfuzz) ---
    try:
        from rapidfuzz import fuzz
    except ImportError:
        # Fallback to basic matching if rapidfuzz is not installed
        return pub1.title.lower().strip() == pub2.title.lower().strip() and pub1.year == pub2.year

    norm_title1 = _normalize_text_for_matching(pub1.title)
    norm_title2 = _normalize_text_for_matching(pub2.title)
    if not norm_title1 or not norm_title2:
        return False

    # Calculate title similarity
    best_title_similarity = max(
        fuzz.ratio(norm_title1, norm_title2),
        fuzz.token_set_ratio(norm_title1, norm_title2),
        fuzz.partial_ratio(norm_title1, norm_title2),
        fuzz.WRatio(norm_title1, norm_title2),
    )

    # --- Decision Thresholds ---
    TITLE_STRONG_MATCH = 98
    TITLE_HIGH_MATCH = 95
    TITLE_BASE_MATCH = 91

    if best_title_similarity < TITLE_BASE_MATCH:
        return False

    # Check year alignment (looser for stronger title matches)
    year_match = True
    if pub1.year and pub2.year:
        year_diff = abs(pub1.year - pub2.year)
        year_match = year_diff <= 1 if best_title_similarity < TITLE_STRONG_MATCH else year_diff <= 2

    # Check author alignment
    author_match = True
    if pub1.authors and pub2.authors:
        author_similarity = _calculate_author_similarity(pub1.authors, pub2.authors)
        last_name_similarity = _last_name_overlap(pub1.authors, pub2.authors)
        # Use more lenient thresholds for truncated lists
        is_truncated = any("et al" in str(a).lower() for a in pub1.authors + pub2.authors)
        author_match = (author_similarity >= (0.2 if is_truncated else 0.4)) or (
            last_name_similarity >= (0.3 if is_truncated else 0.6)
        )

    # Final decision logic
    if best_title_similarity >= TITLE_STRONG_MATCH:
        return year_match or author_match
    if best_title_similarity >= TITLE_HIGH_MATCH:
        return year_match and author_match
    # Base match requires both year and author alignment
    return year_match and author_match


def _clean_title_metadata(title: str) -> str:
    """Conservatively clean titles of trailing metadata like (vol 1, pp 2-3, 2020)."""
    if not title:
        return ""
    cleaned = title.strip()
    # Patterns for trailing metadata in parentheses
    patterns_to_remove = [
        r"\s*\(\s*vol\.?\s+\d+.*,\s+\d{4}\s*\)$",  # (vol 15, ..., 2015)
        r"\s*\(\s*pp?\.?\s+\d+[-–—]\d+\s*\)$",  # (pp. 123-145)
        r'[,\.]\s*\(\s*\d{4}\s*\)$',  # , (2020)
        r"\s*\(\s*doi:\s*[^\)]+\s*\)$",  # (doi: ...)
    ]
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s*[,;:\.\s]+$", "", cleaned).strip()

    # Only accept cleaning if it doesn't drastically shorten the title
    return cleaned if len(cleaned) >= len(title) * 0.7 else title.strip()


def _select_best_title(titles: List[str]) -> str:
    """Select the best-formatted title from a list of candidates."""
    if not titles:
        return ""
    cleaned_titles = [t for t in (_clean_title_metadata(t) for t in titles) if t]
    if not cleaned_titles:
        return ""

    def title_quality_score(title: str) -> float:
        score = len(title) / 100.0  # Prefer longer, more complete titles
        if not title.islower() and not title.isupper():
            score += 2.0  # Bonus for proper sentence casing
        if title.endswith((".", "?", "!")):
            score += 0.5
        if title.isupper():
            score -= 1.0  # Penalty for ALL CAPS
        if re.search(r"\b(vol|pp|issue)\b", title, re.IGNORECASE):
            score -= 1.0  # Penalty for remaining metadata
        return score

    return max(cleaned_titles, key=title_quality_score)


def _normalize_source_label(s: str) -> Optional[str]:
    """Normalize source names to a canonical form."""
    s = (s or "").strip().lower()
    if "google scholar" in s or s == "gs":
        return "Google Scholar"
    if "scopus" in s:
        return "Scopus"
    if "web of science" in s or s == "wos":
        return "Web of Science"
    if "orcid" in s:
        return "ORCID"
    return None


def _extract_source_map(pub: Publication) -> Dict[str, int]:
    """Extract a map of {NormalizedSource: citation_count} from a publication's source field."""
    src_text = (getattr(pub, "source", "") or "").strip()
    citations = int(getattr(pub, "citations", 0) or 0)
    result: Dict[str, int] = {}

    if src_text.lower().startswith("multiple sources:"):
        # Parse merged source strings like "Google Scholar (360 cites), Scopus (307 cites)"
        pattern = re.compile(
            r"(google scholar|scopus|web of science|wos|orcid)(?:\s*\((\d+)\s*cites?\))?",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(src_text):
            norm_label = _normalize_source_label(match.group(1))
            if norm_label:
                cites_str = match.group(2)
                # ORCID provides presence, not citations
                cites_int = int(cites_str) if cites_str is not None else (0 if norm_label == "ORCID" else citations)
                result[norm_label] = max(result.get(norm_label, 0), cites_int)
    else:
        # Handle single-source records
        norm_label = _normalize_source_label(src_text)
        if norm_label:
            result[norm_label] = 0 if norm_label == "ORCID" else citations

    return result


def _merge_publications(publications: List[Publication]) -> Publication:
    """Merge a list of duplicate publications into a single, comprehensive record."""
    if not publications:
        raise ValueError("Cannot merge an empty list of publications.")
    if len(publications) == 1:
        return publications[0]

    # Sort publications by quality to select the best base record
    def quality_score(p: Publication) -> int:
        score = 0
        if p.doi: score += 5
        if p.citations: score += 3
        if p.journal: score += 2
        if p.authors and len(p.authors) > 2: score += 1
        return score

    sorted_pubs = sorted(publications, key=quality_score, reverse=True)
    primary = sorted_pubs[0]

    # Aggregate all unique sources and their citation counts
    aggregated_sources: Dict[str, int] = {}
    for p in sorted_pubs:
        for source, cites in _extract_source_map(p).items():
            aggregated_sources[source] = max(aggregated_sources.get(source, 0), cites)

    # Build a deterministic, human-readable source string
    source_order = ["Google Scholar", "Scopus", "Web of Science", "ORCID"]
    source_parts = []
    for name in source_order:
        if name in aggregated_sources:
            source_parts.append(f"{name} ({aggregated_sources[name]} cites)" if name != "ORCID" else "ORCID")
    merged_source = f"Multiple sources: {', '.join(source_parts)}" if len(source_parts) > 1 else (source_parts[0] if source_parts else primary.source)

    # Select the best value for each field
    merged_title = _select_best_title([p.title for p in sorted_pubs if p.title])
    merged_authors = max((p.authors for p in sorted_pubs), key=lambda x: len(x or []), default=[])
    merged_journal = next((p.journal for p in sorted_pubs if p.journal), None)
    merged_year = next((p.year for p in sorted_pubs if p.year), None)
    merged_url = next((p.url for p in sorted_pubs if p.url), None)

    # Select the most common, non-null DOI
    dois = [_normalize_doi(p.doi) for p in sorted_pubs if _normalize_doi(p.doi)]
    merged_doi = Counter(dois).most_common(1)[0][0] if dois else None

    # Global max citations (excluding ORCID)
    max_citations = max((c for s, c in aggregated_sources.items() if s != "ORCID"), default=0)

    return Publication(
        title=merged_title,
        authors=merged_authors,
        journal=merged_journal,
        year=merged_year,
        doi=merged_doi,
        issn=primary.issn,
        source=merged_source,
        citations=max_citations,
        url=merged_url,
    )


def _deduplicate_publications(publications: List[Publication]) -> List[Publication]:
    """Robustly deduplicate publications using a multi-step process."""
    if not publications:
        return []

    print(f"🔄 Starting deduplication of {len(publications)} publications...")

    # 1. Clean all incoming data and normalize DOIs
    cleaned = [_clean_publication_data(p) for p in publications]

    # 2. Group and merge publications by their normalized DOI
    doi_buckets: Dict[str, List[Publication]] = defaultdict(list)
    no_doi_pubs: List[Publication] = []
    for p in cleaned:
        doi = _normalize_doi(p.doi) or (_normalize_doi(p.url) if p.url and "doi.org" in p.url else None)
        if doi:
            doi_buckets[doi].append(p)
        else:
            no_doi_pubs.append(p)

    candidates = [_merge_publications(bucket) for bucket in doi_buckets.values()]
    candidates.extend(no_doi_pubs)
    print(f"📦 Candidates after DOI collapse: {len(candidates)}")

    # 3. Iteratively merge remaining candidates using fuzzy matching.
    # This allows DOI-less records to merge with DOI-bearing records.
    i = 0
    while i < len(candidates):
        j = i + 1
        merged_into_i = False
        while j < len(candidates):
            if _publications_match(candidates[i], candidates[j]):
                # Merge j into i, then remove j
                candidates[i] = _merge_publications([candidates[i], candidates[j]])
                candidates.pop(j)
                merged_into_i = True
            else:
                j += 1
        # If no merges occurred for item i, move to the next item
        if not merged_into_i:
            i += 1

    # 4. Final sort by year and citations
    candidates.sort(key=lambda p: (-(p.year or 0), -(p.citations or 0)))

    print(f"✅ Deduplication complete: {len(publications)} → {len(candidates)} publications")
    return candidates


def aggregate_publications(
    author: Author,
    api_keys: Dict[str, str],
    *,
    max_pubs_g_scholar: int = 100,
    headless_g_scholar: bool = True,
    analyze_coverage: bool = True,
) -> List[Publication]:
    """
    Fetch publications from all sources in parallel, then merge and deduplicate.

    Parameters
    ----------
    author : Author
        The author to search for.
    api_keys : dict
        A dictionary containing the required API keys.
    max_pubs_g_scholar : int, default 100
        Max publications to fetch from Google Scholar.
    headless_g_scholar : bool, default True
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

    # Lazy import sources so missing optional dependencies don't crash the run
    def get_source_module(name: str):
        try:
            # Try relative import first
            from importlib import import_module
            return import_module(f".sources.{name}", package="pubcrawler")
        except (ImportError, ModuleNotFoundError):
            try:
                # Fallback for script execution
                return import_module(f"sources.{name}")
            except Exception as e:
                print(f"⚠️ {name.title()} source disabled: {e}")
                return None

    gs_mod = get_source_module("google_scholar")
    scopus_mod = get_source_module("scopus")
    wos_mod = get_source_module("wos")
    orcid_mod = get_source_module("orcid")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_source: Dict[concurrent.futures.Future, str] = {}

        # Submit Google Scholar task
        if author.gs_id and gs_mod:
            future = executor.submit(gs_mod.fetch, gs_id=author.gs_id, max_publications_detail=max_pubs_g_scholar, headless=headless_g_scholar)
            future_to_source[future] = "Google Scholar"

        # Submit Scopus task
        if api_keys.get("scopus_api_key") and scopus_mod:
            scopus_args = {}
            if author.scopus_id: scopus_args["scopus_id"] = author.scopus_id
            elif getattr(author, 'orcid_id', None): scopus_args["orcid_id"] = author.orcid_id
            elif author.affiliation: scopus_args.update({"first_name": author.first_name, "last_name": author.last_name, "affiliation": author.affiliation})
            if scopus_args:
                future = executor.submit(scopus_mod.fetch, api_key=api_keys["scopus_api_key"], **scopus_args)
                future_to_source[future] = "Scopus"

        # Submit Web of Science task
        if api_keys.get("wos_api_key") and (author.wos_id or author.affiliation) and wos_mod:
            future = executor.submit(wos_mod.fetch, first_name=author.first_name, last_name=author.last_name, affiliation=author.affiliation or "", api_key=api_keys["wos_api_key"], author_ids=[author.wos_id] if author.wos_id else None)
            future_to_source[future] = "Web of Science"

        # Submit ORCID task
        if api_keys.get("orcid_client_id") and api_keys.get("orcid_client_secret") and getattr(author, 'orcid_id', None) and orcid_mod:
            future = executor.submit(orcid_mod.fetch, orcid_id=author.orcid_id, client_id=api_keys["orcid_client_id"], client_secret=api_keys["orcid_client_secret"])
            future_to_source[future] = "ORCID"

        available_sources = list(future_to_source.values())
        print(f"🚀 Fetching from sources: {', '.join(available_sources) or 'None'}")

        # Collect results
        for future in concurrent.futures.as_completed(future_to_source):
            source_name = future_to_source[future]
            try:
                pubs = future.result()
                print(f"✅ Fetched {len(pubs)} publications from {source_name}.")
                all_publications.extend(pubs)
            except Exception as exc:
                print(f"❌ {source_name} generated an exception: {exc}")

    print(f"\nTotal publications fetched before deduplication: {len(all_publications)}")

    deduplicated_pubs = _deduplicate_publications(all_publications)

    if analyze_coverage and available_sources:
        print(f"\n🔍 Analyzing index coverage across {len(available_sources)} sources...")
        try:
            print_coverage_report(deduplicated_pubs, available_sources)
        except Exception as e:
            print(f"⚠️ Coverage analysis failed: {e}")

    return deduplicated_pubs


def _display_results(publications: List[Publication]) -> None:
    """Display publication results in a formatted manner, using pandas if available."""
    if not publications:
        print("\n❌ No publications found.")
        return

    print(f"\n✅ Aggregated {len(publications)} unique publications.")

    try:
        import pandas as pd
        pd.set_option('display.max_colwidth', 80)
        pd.set_option('display.width', 120)

        df = pd.DataFrame([p.to_dict() for p in publications])
        display_cols = ["year", "citations", "title", "doi", "source"]
        df_display = df[[c for c in display_cols if c in df.columns]]

        print("\n--- Top 10 Publications (by Year, then Citations) ---")
        print(df_display.head(10).to_string(index=False))

        print("\n--- Source Distribution ---")
        source_counts = df['source'].apply(lambda x: 'Multiple sources' if 'Multiple' in x else x).value_counts()
        print(source_counts.to_string())

        print("\n--- Summary Statistics ---")
        print(f"  Total citations: {int(df['citations'].sum()):,}")
        print(f"  Average citations per paper: {df['citations'].mean():.1f}")
        print(f"  Publication years: {int(df['year'].min())}-{int(df['year'].max())}")
        print(f"  Publications with DOI: {df['doi'].notna().sum()}/{len(df)} ({df['doi'].notna().mean():.1%})")

    except ImportError:
        # Fallback display without pandas
        print("\n--- Top 10 Publications ---")
        for i, pub in enumerate(publications[:10], 1):
            print(f"{i:2d}. ({pub.year or 'N/A'}) [{pub.citations or 0} cites] {pub.title[:80]}...")
            print(f"     Source: {pub.source}")
            if pub.doi:
                print(f"     DOI: {pub.doi}")


def main() -> None:
    """Command-line interface for the pubcrawler aggregate module."""
    parser = argparse.ArgumentParser(
        description="Aggregate publications for a researcher from multiple sources.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python -m pubcrawler.aggregate --first John --last Doe --affiliation "MIT" --gs-id "scholar_id"
  python -m pubcrawler.aggregate --first Jane --last Smith --scopus-key "key" --max-gs 50
""",
    )
    # Author and API key arguments...
    parser.add_argument("--first", required=True, help="Author's first name")
    parser.add_argument("--last", required=True, help="Author's last name")
    parser.add_argument("--affiliation", help="Author's institutional affiliation")
    parser.add_argument("--gs-id", help="Google Scholar author ID")
    parser.add_argument("--scopus-id", help="Scopus author ID")
    parser.add_argument("--wos-id", help="Web of Science ResearcherID")
    parser.add_argument("--orcid-id", help="ORCID iD")
    parser.add_argument("--scopus-key", default=os.environ.get("SCOPUS_API_KEY"), help="Scopus API key (or SCOPUS_API_KEY env var)")
    parser.add_argument("--wos-key", default=os.environ.get("WOS_API_KEY"), help="Web of Science API key (or WOS_API_KEY env var)")
    parser.add_argument("--orcid-client-id", default=os.environ.get("ORCID_CLIENT_ID"), help="ORCID client ID (or ORCID_CLIENT_ID env var)")
    parser.add_argument("--orcid-client-secret", default=os.environ.get("ORCID_CLIENT_SECRET"), help="ORCID client secret (or ORCID_CLIENT_SECRET env var)")
    parser.add_argument("--max-gs", type=int, default=100, help="Max publications from Google Scholar (default: 100)")
    parser.add_argument("--no-headless", action="store_false", dest="headless", help="Run Google Scholar scraping with a visible browser")
    parser.add_argument("--no-coverage", action="store_false", dest="analyze_coverage", help="Skip index coverage analysis")
    args = parser.parse_args()

    if not any([args.gs_id, args.scopus_id, args.wos_id, args.orcid_id, args.affiliation]):
        parser.error("At least one identifier (--gs-id, --scopus-id, etc.) or --affiliation is required.")

    author = Author(
        first_name=args.first, last_name=args.last, affiliation=args.affiliation,
        gs_id=args.gs_id, scopus_id=args.scopus_id, wos_id=args.wos_id, orcid_id=args.orcid_id
    )
    api_keys = {
        "scopus_api_key": args.scopus_key, "wos_api_key": args.wos_key,
        "orcid_client_id": args.orcid_client_id, "orcid_client_secret": args.orcid_client_secret,
    }

    print(f"🔍 Searching for publications by {author.full_name}")
    if author.affiliation: print(f"   📍 Affiliation: {author.affiliation}")
    if author.gs_id: print(f"   🎓 Google Scholar ID: {author.gs_id}")
    if author.scopus_id: print(f"   🔬 Scopus ID: {author.scopus_id}")
    if author.wos_id: print(f"   🔬 Web of Science ID: {author.wos_id}")
    if author.orcid_id: print(f"   🆔 ORCID iD: {author.orcid_id}")
    print()

    try:
        final_publications = aggregate_publications(
            author=author, api_keys=api_keys,
            max_pubs_g_scholar=args.max_gs,
            headless_g_scholar=args.headless,
            analyze_coverage=args.analyze_coverage,
        )
        _display_results(final_publications)
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during aggregation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()