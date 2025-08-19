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
from typing import List, Dict, Optional

# Avoid hard dependencies at import time; import where needed.
try:
    from .models import Author, Publication
    from .coverage import analyze_publication_coverage, print_coverage_report
except ImportError:
    # Allow running this file as a script for debugging (python aggregate.py)
    from models import Author, Publication  # type: ignore
    from coverage import analyze_publication_coverage, print_coverage_report  # type: ignore


def _normalize_doi(raw: Optional[str]) -> Optional[str]:
    """Normalize DOI strings and validate. Supports DOI URLs and 'doi:' prefixes."""
    if not raw:
        return None
    import re
    doi = str(raw).strip().lower()

    # Strip URL and common prefixes
    doi = re.sub(r'^\s*(https?://(dx\.)?doi\.org/)', '', doi)
    doi = re.sub(r'^\s*doi:\s*', '', doi)

    # Trim wrapping punctuation/spaces/brackets
    doi = doi.strip().strip(' .;,:<>[]()')

    # Drop URL query/fragment and any trailing whitespace after the DOI
    doi = re.split(r'[?#\s]', doi)[0]

    # Remove zero-width/invisible spaces
    doi = re.sub(r'[\u200B-\u200D\u2060]', '', doi)

    # Normalize spaces around slash and remove remaining spaces
    doi = re.sub(r'\s*/\s*', '/', doi)
    doi = doi.replace(' ', '')

    # Basic DOI validation (case-insensitive allowed chars in suffix)
    # Ref: Crossref-style validation (simplified)
    pattern = re.compile(r'^10\.\d{4,9}/[^\s]+$', re.IGNORECASE)
    if not pattern.match(doi):
        return None
    return doi


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

    # Normalize/derive DOI (handles doi URLs and 'doi:' prefixes)
    norm_doi = _normalize_doi(doi)
    if not norm_doi and url and isinstance(url, str) and 'doi.org' in url.lower():
        norm_doi = _normalize_doi(url)

    # Create a new publication with cleaned data
    return Publication(
        title=title,
        authors=authors,
        journal=journal,
        year=year,
        doi=norm_doi,
        issn=issn,
        source=source,
        citations=citations,
        url=url
    )


def _is_empty_value(value):
    """Check if a value is empty, None, NaN, or otherwise should be considered missing."""
    if value is None:
        return True

    # Handle scalar values
    try:
        import pandas as pd
        if pd.isna(value):
            return True
    except (TypeError, ValueError, ImportError):
        # If pd.isna fails or pandas not available, check other conditions
        pass

    # Handle string representations of missing values
    if isinstance(value, str):
        return value.strip() == '' or value.lower() in ['nan', 'none', 'null']

    # Handle lists/arrays
    if isinstance(value, (list, tuple)):
        return len(value) == 0

    return False


def _normalize_text_for_matching(text: str) -> str:
    """Enhanced text normalization for better fuzzy matching during deduplication."""
    if not text:
        return ""

    import re
    import unicodedata

    # Unicode normalize and strip diacritics
    def _latinize(s: str) -> str:
        nkfd = unicodedata.normalize('NFKD', s)
        return "".join(c for c in nkfd if not unicodedata.combining(c))

    # Unify common punctuation variants first
    text = (text or "").replace('–', '-').replace('—', '-').replace('’', "'").replace('“', '"').replace('”', '"')

    # Lowercase, trim, and latinize (remove accents)
    normalized = _latinize(text.lower().strip())

    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)

    # Remove common punctuation that might vary between sources
    normalized = re.sub(r'[.,;:!?"\'\(\)\[\]{}]', '', normalized)

    # Remove common prefixes/suffixes that might differ
    normalized = re.sub(r'\b(the|a|an)\b', '', normalized)

    # Handle common abbreviations and variations - more comprehensive
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

    # Remove common suffixes that might differ - enhanced
    normalized = re.sub(r'\b(abstract|full text|pdf|html|preprint|arxiv)\b', '', normalized)

    # Clean up multiple spaces created by removals
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized


def _calculate_author_similarity(authors1: List[str], authors2: List[str]) -> float:
    """Enhanced author similarity calculation with better handling of partial lists."""
    import re

    if not authors1 or not authors2:
        return 0.0

    # Handle case where authors might be stored as string vs list
    if isinstance(authors1, str):
        authors1 = [authors1]
    if isinstance(authors2, str):
        authors2 = [authors2]

    # Normalize author names - enhanced
    def normalize_author(author):
        if not author:
            return ""

        normalized = author.lower().strip()
        # Remove titles and suffixes - more comprehensive
        normalized = re.sub(r'\b(dr|prof|professor|phd|md|jr|sr|ii|iii|mr|ms|mrs)\b\.?', '', normalized)
        # Remove extra spaces and punctuation
        normalized = re.sub(r'[,.]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    norm_authors1 = [normalize_author(author) for author in authors1 if author]
    norm_authors2 = [normalize_author(author) for author in authors2 if author]

    # Remove empty authors
    norm_authors1 = [a for a in norm_authors1 if a]
    norm_authors2 = [a for a in norm_authors2 if a]

    if not norm_authors1 or not norm_authors2:
        return 0.0

    # Import fuzzy matching
    try:
        from rapidfuzz import fuzz
    except ImportError:
        # Fallback to basic string matching
        matches = sum(1 for a1 in norm_authors1 for a2 in norm_authors2 if a1 == a2)
        return matches / min(len(norm_authors1), len(norm_authors2)) if min(len(norm_authors1), len(norm_authors2)) > 0 else 0.0

    # Strategy 1: First author matching (often most important) - stricter
    first_author_match = False
    if norm_authors1 and norm_authors2:
        first_similarity = fuzz.ratio(norm_authors1[0], norm_authors2[0])
        first_author_match = first_similarity >= 85  # Much stricter - raised from 70

    # Strategy 2: Count matches using fuzzy matching - much stricter thresholds
    matches = 0
    for author1 in norm_authors1:
        best_match = max(fuzz.ratio(author1, author2) for author2 in norm_authors2)
        if best_match >= 85:  # Much stricter - raised from 70
            matches += 1

    # Strategy 3: Handle "et al." cases more conservatively
    shorter_len = min(len(norm_authors1), len(norm_authors2))
    longer_len = max(len(norm_authors1), len(norm_authors2))

    if longer_len > shorter_len * 2:  # More conservative - raised from 1.5
        shorter_authors = norm_authors1 if len(norm_authors1) < len(norm_authors2) else norm_authors2
        longer_authors = norm_authors2 if len(norm_authors1) < len(norm_authors2) else norm_authors1

        matched_short = sum(
            1 for short_author in shorter_authors
            if any(fuzz.ratio(short_author, long_author) >= 85 for long_author in longer_authors)  # Stricter threshold
        )

        if len(shorter_authors) > 0 and matched_short / len(shorter_authors) >= 0.8:  # Much stricter - raised from 0.6
            return 0.75

    # Calculate final similarity - more conservative boost
    base_similarity = matches / shorter_len if shorter_len > 0 else 0.0

    # Small boost if first author matches well - reduced boost
    if first_author_match:
        base_similarity = min(1.0, base_similarity + 0.15)  # Reduced from 0.25

    return base_similarity


def _last_name_overlap(authors1: List[str], authors2: List[str]) -> float:
    """Compute conservative overlap of last names between two author lists (0..1)."""
    if not authors1 or not authors2:
        return 0.0

    def to_last(name: str) -> Optional[str]:
        if not name:
            return None
        import re, unicodedata
        s = str(name).strip().lower()
        # Remove titles/suffixes
        s = re.sub(r'\b(dr|prof|professor|phd|md|jr|sr|ii|iii|mr|ms|mrs)\b\.?', '', s)
        # Unicode normalize and strip accents
        s = "".join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
        s = re.sub(r'[.,]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        if not s:
            return None
        parts = s.split(' ')
        # Heuristic: last token as last name
        return parts[-1] if parts else None

    set1 = {ln for ln in (to_last(a) for a in authors1) if ln}
    set2 = {ln for ln in (to_last(a) for a in authors2) if ln}
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    denom = min(len(set1), len(set2))
    return inter / denom if denom else 0.0


def _is_correction_or_note(title: str) -> bool:
    """Detect if a title represents an erratum/correction/retraction/editorial note."""
    if not title:
        return False
    import re
    t = str(title).strip().lower()
    patterns = [
        r'^\s*(erratum|corrigendum|correction|retraction|expression of concern)\b',
        r'\[\s*(erratum|corrigendum|correction|retraction)\s*\]',
        r'^\s*(editorial|commentary|reply|response|letter( to the editor)?)\b',
    ]
    return any(re.search(p, t, re.IGNORECASE) for p in patterns)


def _publications_match(pub1: Publication, pub2: Publication) -> bool:
    """Simplified publication matching with conservative thresholds."""
    # DOI match (highest priority) with normalization
    doi1 = _normalize_doi(getattr(pub1, 'doi', None))
    doi2 = _normalize_doi(getattr(pub2, 'doi', None))
    if doi1 and doi2:
        return doi1 == doi2

    # Skip if from the same source (shouldn't be duplicates within same source)
    if pub1.source.lower() == pub2.source.lower():
        return False

    # Both must have titles
    if not (pub1.title and pub2.title):
        return False

    # Never match a correction-like item with a non-correction item
    if _is_correction_or_note(pub1.title) != _is_correction_or_note(pub2.title):
        return False

    # Import fuzzy matching
    try:
        from rapidfuzz import fuzz
    except ImportError:
        # Fallback to basic matching
        return (pub1.title.lower().strip() == pub2.title.lower().strip() and
                pub1.year == pub2.year)

    # Normalize titles for comparison
    norm_title1 = _normalize_text_for_matching(pub1.title)
    norm_title2 = _normalize_text_for_matching(pub2.title)

    if not norm_title1 or not norm_title2:
        return False

    # Cheap token overlap pre-filter to avoid spurious fuzzy matches
    t1 = {t for t in norm_title1.split() if len(t) >= 3}
    t2 = {t for t in norm_title2.split() if len(t) >= 3}
    if t1 and t2 and len(t1.intersection(t2)) == 0:
        return False

    # Calculate title similarity using multiple methods
    title_ratio = fuzz.ratio(norm_title1, norm_title2)
    token_ratio = fuzz.token_set_ratio(norm_title1, norm_title2)
    partial_ratio = fuzz.partial_ratio(norm_title1, norm_title2)
    try:
        token_sort = fuzz.token_sort_ratio(norm_title1, norm_title2)
        wratio = fuzz.WRatio(norm_title1, norm_title2)
    except Exception:
        token_sort = 0
        wratio = 0

    best_title_similarity = max(title_ratio, token_ratio, partial_ratio, token_sort, wratio)

    # If titles do not match strongly enough, stop early
    if best_title_similarity < 85:
        return False

    # Year check (looser when title is very strong)
    year_match = True  # Default to True if years missing
    if pub1.year and pub2.year:
        year_diff = abs(pub1.year - pub2.year)
        year_match = year_diff <= 1 if best_title_similarity < 95 else year_diff <= 2

    # Author similarity
    author_match = True  # Default to True if authors missing
    last_overlap = 0.0
    if pub1.authors and pub2.authors:
        author_similarity = _calculate_author_similarity(pub1.authors, pub2.authors)
        last_overlap = _last_name_overlap(pub1.authors, pub2.authors)
        # Combine conservatively: require either solid fuzzy or strong last-name overlap
        author_match = (author_similarity >= 0.4) or (last_overlap >= 0.6)

    # Decision thresholds:
    # - Very high title similarity (>=95): accept if either year or authors align
    if best_title_similarity >= 95:
        return year_match or author_match

    # - High title similarity (90-95): require year alignment AND at least weak author alignment
    if 90 <= best_title_similarity < 95:
        return year_match and (author_match or last_overlap >= 0.5)

    # - Moderate-high (85-90): require both tight year and stronger author confirmation
    return year_match and (last_overlap >= 0.6 or (pub1.authors and pub2.authors and _calculate_author_similarity(pub1.authors, pub2.authors) >= 0.6))


def _clean_title_metadata(title: str) -> str:
    """
    Clean titles of metadata like volume/issue info, page numbers, and year suffixes.
    Very conservative to avoid removing actual title content.
    """
    if not title:
        return ""

    import re

    cleaned = title.strip()

    # Pattern 1: Remove volume/issue/page info in parentheses at the end
    # Examples: "(vol 15, 60, 2015)", "(Volume 10, Issue 3, 2020)", "(pp. 123-145)"
    patterns_to_remove = [
        # Volume/issue/year patterns - must be at end and in parentheses
        r'\s*\(\s*vol\.?\s+\d+[,\s]+[\d\-]+[,\s]+\d{4}\s*\)$',  # (vol 15, 60, 2015)
        r'\s*\(\s*volume\s+\d+[,\s]+issue\s+\d+[,\s]+\d{4}\s*\)$',  # (volume 10, issue 3, 2020)
        r'\s*\(\s*v\.?\s*\d+[,\s]+n\.?\s*\d+[,\s]+\d{4}\s*\)$',  # (v. 10, n. 3, 2020)

        # Page number patterns - must be at end and in parentheses
        r'\s*\(\s*pp?\.?\s+\d+[-–—]\d+\s*\)$',  # (pp. 123-145), (p. 123-145)
        r'\s*\(\s*pages?\s+\d+[-–—]\d+\s*\)$',  # (pages 123-145)

        # Year in parentheses at the end (only if preceded by comma or period)
        r'[,\.]\s*\(\s*\d{4}\s*\)$',  # , (2020) or . (2020)

        # Volume/issue without year but with clear indicators
        r'\s*\(\s*vol\.?\s+\d+[,\s]+no\.?\s+\d+\s*\)$',  # (vol 15, no 3)
        r'\s*\(\s*volume\s+\d+[,\s]+number\s+\d+\s*\)$',  # (volume 15, number 3)

        # Simple volume/page patterns
        r'\s*\(\s*vol\.?\s*\d+[,\s]+\d+\s*\)$',  # (vol. 15, 123)

        # DOI or article identifiers in parentheses at end
        r'\s*\(\s*doi:\s*[^\)]+\s*\)$',  # (doi: 10.1234/example)
        r'\s*\(\s*article\s+\w+\s*\)$',  # (article 123456)
    ]

    # Apply patterns case-insensitively
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # Clean up any trailing punctuation that might be left
    cleaned = re.sub(r'\s*[,;:\.\s]+$', '', cleaned)

    # Only return the cleaned version if it's substantially the same length
    # This prevents over-aggressive cleaning that removes actual content
    if len(cleaned) >= len(title) * 0.7:  # At least 70% of original length
        return cleaned.strip()
    else:
        # If we removed too much, return the original
        return title.strip()


def _select_best_title(titles: List[str]) -> str:
    """Select the best-formatted title from a list of title options."""
    if not titles:
        return ""

    if len(titles) == 1:
        return _clean_title_metadata(titles[0])

    # Clean all titles first
    cleaned_titles = [_clean_title_metadata(title) for title in titles if title]
    cleaned_titles = [title for title in cleaned_titles if title]  # Remove empty titles

    if not cleaned_titles:
        return ""

    if len(cleaned_titles) == 1:
        return cleaned_titles[0]

    def title_quality_score(title: str) -> float:
        """Score a title based on formatting quality."""
        if not title:
            return 0.0

        score = 0.0

        # Prefer titles with proper capitalization (not all lowercase or all uppercase)
        if not title.islower() and not title.isupper():
            score += 2.0

        # Prefer titles with mixed case
        if any(c.isupper() for c in title) and any(c.islower() for c in title):
            score += 1.0

        # Prefer longer titles (more complete) - but not if they're just metadata
        score += len(title) / 1000.0  # Small bonus for length

        # Prefer titles with proper punctuation
        if title.endswith('.') or title.endswith('?') or title.endswith('!'):
            score += 0.5

        # Penalize titles that are all caps
        if title.isupper():
            score -= 1.0

        # Penalize titles that are all lowercase
        if title.islower():
            score -= 0.5

        # Penalize titles that still contain obvious metadata markers
        import re
        if re.search(r'\b(vol|volume|pp|pages?|issue|no)\b', title, re.IGNORECASE):
            score -= 1.0

        return score

    # Find the title with the best quality score
    best_title = max(cleaned_titles, key=title_quality_score)
    return best_title


def _merge_publications(publications: List[Publication]) -> Publication:
    """Merge duplicate publications, selecting the best data from each source."""
    if not publications:
        raise ValueError("Cannot merge empty list of publications")

    if len(publications) == 1:
        return publications[0]

    # Sort by data quality: prefer sources with more complete information
    def publication_quality_score(pub):
        score = 0
        score += 3 if pub.doi else 0
        score += 2 if pub.citations and pub.citations > 0 else 0
        score += 1 if pub.journal else 0
        score += 1 if pub.authors and len(pub.authors) > 1 else 0
        score += 1 if pub.url else 0
        return score

    sorted_pubs = sorted(publications, key=publication_quality_score, reverse=True)
    primary = sorted_pubs[0]

    # Collect source information
    source_info = []
    max_citations = 0

    for pub in sorted_pubs:
        source_name = pub.source
        citations = pub.citations or 0

        # Normalize source names
        if 'google scholar' in source_name.lower():
            source_name = 'Google Scholar'
        elif 'scopus' in source_name.lower():
            source_name = 'Scopus'
        elif 'web of science' in source_name.lower() or 'wos' in source_name.lower():
            source_name = 'Web of Science'
        elif 'orcid' in source_name.lower():
            source_name = 'ORCID'

        source_info.append(f"{source_name} ({citations} cites)")
        max_citations = max(max_citations, citations)

    # Create merged publication with best available data
    merged_title = _select_best_title([pub.title for pub in sorted_pubs if pub.title])
    merged_authors = primary.authors
    merged_journal = primary.journal
    merged_year = primary.year

    # Select the most common normalized DOI across all candidates, tie-breaking by quality order
    from collections import Counter
    norm_dois = []
    for pub in sorted_pubs:
        nd = _normalize_doi(getattr(pub, 'doi', None))
        if nd:
            norm_dois.append(nd)
    merged_doi = None
    if norm_dois:
        counts = Counter(norm_dois)
        top_count = max(counts.values())
        top_dois = {d for d, c in counts.items() if c == top_count}
        for pub in sorted_pubs:
            nd = _normalize_doi(getattr(pub, 'doi', None))
            if nd in top_dois:
                merged_doi = nd
                break

    merged_url = primary.url

    # Fill missing fields from other sources
    for pub in sorted_pubs[1:]:
        if not merged_authors and pub.authors:
            merged_authors = pub.authors
        elif pub.authors and len(pub.authors) > len(merged_authors or []):
            merged_authors = pub.authors

        if not merged_journal and pub.journal:
            merged_journal = pub.journal

        if not merged_year and pub.year:
            merged_year = pub.year

        # Keep existing fallback unchanged (will only run if no DOI chosen above) — but normalize it
        if not merged_doi and pub.doi:
            nd = _normalize_doi(pub.doi)
            if nd:
                merged_doi = nd

        if not merged_url and pub.url:
            merged_url = pub.url

    # Create source string
    unique_sources = list(dict.fromkeys(source_info))  # Preserve order, remove duplicates
    merged_source = f"Multiple sources: {', '.join(unique_sources)}"

    return Publication(
        title=merged_title,
        authors=merged_authors,
        journal=merged_journal,
        year=merged_year,
        doi=merged_doi,
        issn=primary.issn,
        source=merged_source,
        citations=max_citations,
        url=merged_url
    )


def _deduplicate_publications(publications: List[Publication]) -> List[Publication]:
    """
    Deduplicate publications using a simple greedy approach to ensure each publication appears exactly once.
    Ensures one-to-one relationships between sources (no multiple Google Scholar items matching same WoS/Scopus item).
    """
    if not publications:
        return []

    print(f"🔄 Starting deduplication of {len(publications)} publications...")

    # Clean all publications first
    cleaned_pubs = [_clean_publication_data(pub) for pub in publications]

    # Group publications by source for analysis
    by_source = {}
    for pub in cleaned_pubs:
        source = pub.source.lower()
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(pub)

    print(f"📊 Publications by source: {', '.join(f'{source}: {len(pubs)}' for source, pubs in by_source.items())}")

    # Sort publications by citation count (highest first) for greedy processing
    # This ensures higher-cited publications become the primary publications in merges
    indexed_pubs = [(i, pub) for i, pub in enumerate(cleaned_pubs)]
    indexed_pubs.sort(key=lambda x: -(x[1].citations or 0))  # Sort by citations descending

    print(f"🎯 Processing publications in citation order (highest first)...")

    # Simple greedy deduplication with one-to-one relationship enforcement
    unique_publications = []
    used_indices = set()

    for orig_idx, pub1 in indexed_pubs:
        if orig_idx in used_indices:
            continue

        # Find all publications that match this one (excluding already used ones)
        matches = [orig_idx]  # Start with the publication itself

        for other_idx, pub2 in indexed_pubs:
            if other_idx in used_indices or other_idx == orig_idx:
                continue

            if _publications_match(pub1, pub2):
                matches.append(other_idx)
                print(f"🔗 Found duplicate: '{pub1.title[:50]}...' <-> '{pub2.title[:50]}...'")

        # Mark ALL matched publications as used to prevent them from being matched again
        for match_idx in matches:
            used_indices.add(match_idx)

        # Merge all matches into one publication
        if len(matches) == 1:
            unique_publications.append(cleaned_pubs[orig_idx])
        else:
            pubs_to_merge = [cleaned_pubs[idx] for idx in matches]
            # Sort by citation count to prioritize highest-cited as primary
            pubs_to_merge.sort(key=lambda p: -(p.citations or 0))
            merged_pub = _merge_publications(pubs_to_merge)
            unique_publications.append(merged_pub)

            sources = list(set(pub.source for pub in pubs_to_merge))
            print(f"🔗 Merged {len(pubs_to_merge)} publications from {', '.join(sources)}: '{merged_pub.title[:60]}...'")

    # Sort by year (descending) then citations (descending)
    sorted_pubs = sorted(unique_publications, key=lambda p: (-(p.year or 0), -(p.citations or 0)))

    print(f"✅ Deduplication complete: {len(publications)} → {len(sorted_pubs)} publications")
    return sorted_pubs


def aggregate_publications(
    author: Author,
    api_keys: Dict[str, str],
    *,
    max_pubs_g_scholar: int = 100,
    headless_g_scholar: bool = True,
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
        {'scopus_api_key': '...', 'wos_api_key': '...', 'orcid_client_id': '...', 'orcid_client_secret': '...'}.
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

    # Lazy import sources so missing optional deps don't crash the whole run
    gs_mod = None
    scopus_mod = None
    wos_mod = None
    orcid_mod = None
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
    try:
        try:
            from .sources import orcid as orcid_mod  # type: ignore
        except ImportError:
            from sources import orcid as orcid_mod  # type: ignore
    except Exception as e:
        print(f"⚠️ ORCID disabled: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_source: Dict[concurrent.futures.Future, str] = {}

        # --- Submit Google Scholar task ---
        if author.gs_id and gs_mod is not None:
            future_gs = executor.submit(
                gs_mod.fetch,
                gs_id=author.gs_id,
                first_name=author.first_name,
                last_name=author.last_name,
                orcid_id=getattr(author, 'orcid_id', None),
                max_publications_detail=max_pubs_g_scholar,
                headless=headless_g_scholar,
            )
            future_to_source[future_gs] = "Google Scholar"
            available_sources.append("Google Scholar")

        # --- Submit Scopus task ---
        if api_keys.get("scopus_api_key") and (author.scopus_id or getattr(author, 'orcid_id', None) or author.affiliation) and scopus_mod is not None:
            if author.scopus_id:
                # Use Scopus ID for direct lookup
                future_scopus = executor.submit(
                    scopus_mod.fetch,
                    scopus_id=author.scopus_id,
                    api_key=api_keys["scopus_api_key"],
                )
            elif getattr(author, 'orcid_id', None):
                # Use ORCID ID for Scopus search
                future_scopus = executor.submit(
                    scopus_mod.fetch,
                    orcid_id=author.orcid_id,
                    api_key=api_keys["scopus_api_key"],
                )
            else:
                # Fallback to name-based search
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
        if api_keys.get("wos_api_key") and (author.wos_id or author.affiliation) and wos_mod is not None:
            # Prepare author_ids if WoS ID is available
            wos_author_ids = [author.wos_id] if author.wos_id else None

            future_wos = executor.submit(
                wos_mod.fetch,
                first_name=author.first_name,
                last_name=author.last_name,
                affiliation=author.affiliation or "",
                api_key=api_keys["wos_api_key"],
                author_ids=wos_author_ids,
            )
            future_to_source[future_wos] = "Web of Science"
            available_sources.append("Web of Science")

        # --- Submit ORCID task ---
        if (api_keys.get("orcid_client_id") and api_keys.get("orcid_client_secret") and
            getattr(author, 'orcid_id', None) and orcid_mod is not None):
            future_orcid = executor.submit(
                orcid_mod.fetch,
                orcid_id=author.orcid_id,
                client_id=api_keys["orcid_client_id"],
                client_secret=api_keys["orcid_client_secret"],
            )
            future_to_source[future_orcid] = "ORCID"
            available_sources.append("ORCID")

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
    parser.add_argument("--orcid-client-id", dest="orcid_client_id",
                       default=os.environ.get("ORCID_CLIENT_ID"),
                       help="ORCID client ID (or set ORCID_CLIENT_ID env var)")
    parser.add_argument("--orcid-client-secret", dest="orcid_client_secret",
                       default=os.environ.get("ORCID_CLIENT_SECRET"),
                       help="ORCID client secret (or set ORCID_CLIENT_SECRET env var)")

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
        "orcid_client_id": args.orcid_client_id,
        "orcid_client_secret": args.orcid_client_secret,
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
    if api_keys.get("orcid_client_id") and api_keys.get("orcid_client_secret") and getattr(author, 'orcid_id', None):
        available_sources.append("ORCID")

    if not available_sources:
        print("❌ No data sources available. Please provide:")
        print("   • Google Scholar ID (--gs-id)")
        print("   • Scopus ID (--scopus-id)")
        print("   • Scopus API key + affiliation (--scopus-key, --affiliation)")
        print("   • Web of Science API key + affiliation (--wos-key, --affiliation)")
        print("   • ORCID client ID/secret + ORCID ID (--orcid-client-id, --orcid-client-secret, --scopus-id)")
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
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _display_results(publications: List[Publication]) -> None:
    """Display publication results in a formatted manner."""
    try:
        import pandas as pd  # type: ignore

        results_df = pd.DataFrame([p.to_dict() for p in publications])

        print("\n--- Top 10 Publications (by Year, then Citations) ---")
        display_cols = [c for c in ["year", "citations", "title", "doi", "source"]
                       if c in results_df.columns]

        top_10 = results_df.head(10)[display_cols]
        # Truncate title for display
        if "title" in top_10.columns:
            top_10["title"] = top_10["title"].str.slice(0, 80).str.ljust(83, '.').str.slice(0, 83)

        print(top_10.to_string(index=False))

        print("\n--- Source Distribution ---")
        # Handle merged sources
        source_counts = results_df['source'].apply(lambda x: 'Multiple sources' if 'Multiple' in x else x).value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count} publications")

        print(f"\n--- Summary Statistics ---")
        total_citations = results_df['citations'].sum()
        avg_citations = results_df['citations'].mean()
        year_range = f"{int(results_df['year'].min())}-{int(results_df['year'].max())}" if not results_df['year'].empty else "N/A"
        with_doi = results_df['doi'].notna().sum()

        print(f"  Total citations: {int(total_citations):,}")
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
            source_key = 'Multiple sources' if 'Multiple' in pub.source else pub.source
            sources[source_key] = sources.get(source_key, 0) + 1

        print("\n--- Source Distribution ---")
        for source, count in sources.items():
            print(f"  {source}: {count} publications")


if __name__ == '__main__':
    main()