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

    # Try to use pandas/rapidfuzz for strong dedup; otherwise do a minimal fallback.
    try:
        import pandas as pd  # type: ignore
        from rapidfuzz import fuzz, process  # type: ignore
        return _advanced_deduplication(publications, pd, fuzz, process)
    except ImportError:
        print("⚠️ Advanced deduplication unavailable (missing pandas/rapidfuzz). Using basic deduplication.")
        return _basic_deduplication(publications)


def _basic_deduplication(publications: List[Publication]) -> List[Publication]:
    """
    Basic deduplication using exact matches only.
    
    Fallback method when pandas/rapidfuzz are not available.
    """
    by_doi: Dict[str, Publication] = {}
    remaining: List[Publication] = []
    
    for pub in publications:
        if pub.doi:
            existing = by_doi.get(pub.doi)
            if existing is None or (pub.citations or 0) > (existing.citations or 0):
                by_doi[pub.doi] = pub
        else:
            remaining.append(pub)
    
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


def _advanced_deduplication(publications: List[Publication], pd, fuzz, process) -> List[Publication]:
    """
    Advanced deduplication using pandas and fuzzy string matching.
    """
    df = pd.DataFrame([p.__dict__ for p in publications])
    df['doi'] = df['doi'].str.lower().str.strip().replace('', None)
    df['citations'] = pd.to_numeric(df['citations'], errors='coerce').fillna(0)

    # --- Step 1: Deduplicate by DOI ---
    doi_pubs = df.dropna(subset=['doi'])
    no_doi_pubs = df[df['doi'].isna()]

    if not doi_pubs.empty:
        doi_pubs = doi_pubs.sort_values('citations', ascending=False)
        deduped_by_doi = doi_pubs.drop_duplicates(subset=['doi'], keep='first')
    else:
        deduped_by_doi = pd.DataFrame(columns=df.columns)

    # --- Step 2: Deduplicate by fuzzy title matching for the rest ---
    if not no_doi_pubs.empty:
        no_doi_pubs = no_doi_pubs.sort_values('citations', ascending=False).reset_index(drop=True)

        unique_titles = []
        processed_indices = set()

        for year, group in no_doi_pubs.groupby('year'):
            indices = list(group.index)
            while indices:
                idx = indices.pop(0)
                if idx in processed_indices:
                    continue
                unique_titles.append(group.loc[idx].to_dict())
                processed_indices.add(idx)

                if indices:
                    other_titles = [group.loc[i, 'title'] for i in indices]
                    matches = process.extract(
                        group.loc[idx, 'title'],
                        other_titles,
                        scorer=fuzz.WRatio,
                        limit=None,
                        score_cutoff=90,
                    )
                    for _match_title, _score, match_idx_in_others in matches:
                        original_idx = indices[match_idx_in_others]
                        processed_indices.add(original_idx)

        deduped_by_title = pd.DataFrame(unique_titles)
    else:
        deduped_by_title = pd.DataFrame(columns=df.columns)

    final_df = pd.concat([deduped_by_doi, deduped_by_title], ignore_index=True)
    final_df = final_df.sort_values(['year', 'citations'], ascending=[False, False])

    final_pubs = [Publication(**row) for _, row in final_df.iterrows()]
    return final_pubs


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
