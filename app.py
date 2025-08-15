"""PubCrawler Web Application
~~~~~~~~~~~~~~~~~~~~~~~~~
Flask web application providing a user-friendly interface for the PubCrawler
publication coverage analysis tool.

Features:
- Search interface for researchers
- Real-time publication aggregation
- Coverage analysis visualization
- Export functionality
- Responsive design
"""

import os
import json
import traceback
from datetime import datetime
from typing import Dict, List, Optional

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.exceptions import BadRequest

# Import PubCrawler modules
import sys
import os
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from aggregate import aggregate_publications
from models import Author, Publication
from coverage import analyze_publication_coverage
from config_keys import get_api_keys
from rapidfuzz import fuzz


app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'pubcrawler-dev-key-change-in-production')


def _normalize_text(text: str) -> str:
    """Normalize text for better fuzzy matching."""
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


def _calculate_author_similarity(authors1: List[str], authors2: List[str]) -> float:
    """Calculate similarity between two author lists with improved matching."""
    import re
    
    if not authors1 or not authors2:
        return 0.0
    
    # Normalize author names - handle different formats
    def normalize_author(author):
        if not author:
            return ""
        
        # Remove common suffixes and prefixes
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
    
    # Strategy 1: Try to match first author (often most important)
    first_author_match = False
    if norm_authors1 and norm_authors2:
        first_similarity = fuzz.ratio(norm_authors1[0], norm_authors2[0])
        first_author_match = first_similarity >= 75
    
    # Strategy 2: Count overall matches using fuzzy matching
    matches = 0
    total_comparisons = 0
    
    for author1 in norm_authors1:
        best_match = 0
        for author2 in norm_authors2:
            similarity = fuzz.ratio(author1, author2)
            best_match = max(best_match, similarity)
            total_comparisons += 1
        
        # Consider it a match if similarity is high enough
        if best_match >= 75:  # Lowered threshold
            matches += 1
    
    # Strategy 3: Handle "et al." cases - if one list is much shorter, be more lenient
    shorter_list_len = min(len(norm_authors1), len(norm_authors2))
    longer_list_len = max(len(norm_authors1), len(norm_authors2))
    
    # If one list has significantly fewer authors (suggesting "et al."), 
    # check if the shorter list authors are in the longer list
    if longer_list_len > shorter_list_len * 2:
        # Focus on matching the shorter list
        shorter_authors = norm_authors1 if len(norm_authors1) < len(norm_authors2) else norm_authors2
        longer_authors = norm_authors2 if len(norm_authors1) < len(norm_authors2) else norm_authors1
        
        matched_short = 0
        for short_author in shorter_authors:
            for long_author in longer_authors:
                if fuzz.ratio(short_author, long_author) >= 75:
                    matched_short += 1
                    break
        
        short_list_similarity = matched_short / len(shorter_authors) if shorter_authors else 0
        
        # If most of the shorter list matches, consider it a good match
        if short_list_similarity >= 0.7:  # 70% of short list matches
            return 0.8
    
    # Calculate final similarity
    min_authors = min(len(norm_authors1), len(norm_authors2))
    base_similarity = matches / min_authors if min_authors > 0 else 0.0
    
    # Boost score if first author matches
    if first_author_match:
        base_similarity = min(1.0, base_similarity + 0.2)
    
    return base_similarity


@app.route('/')
def index():
    """Main page with search interface."""
    return render_template('index.html')


@app.route('/test-matching')
def test_matching_page():
    """Test page for fuzzy matching (debug only)."""
    if not app.debug:
        return "Test page only available in debug mode", 403
    
    # Serve the test HTML file
    with open('test_matching.html', 'r', encoding='utf-8') as f:
        return f.read()


@app.route('/api/search', methods=['POST'])
def api_search():
    """
    API endpoint to search for publications.
    
    Expected JSON payload:
    {
        "researcher_name": "John Doe",
        "researcher_id": "optional-orcid-or-gs-id",
        "affiliation": "University Name",
        "filters": {
            "year_range": "2020-2023",
            "database": "all",
            "sort_by": "newest"
        },
        "api_keys": {
            "scopus_api_key": "optional",
            "wos_api_key": "optional"
        }
    }
    """
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("No JSON data provided")
        
        # Extract researcher information
        researcher_name = data.get('researcher_name', '').strip()
        if not researcher_name:
            return jsonify({'error': 'Researcher name is required'}), 400
        
        # Parse name (simple heuristic)
        name_parts = researcher_name.split()
        if len(name_parts) < 2:
            return jsonify({'error': 'Please provide both first and last name'}), 400
        
        first_name = name_parts[0]
        last_name = ' '.join(name_parts[1:])
        
        # Extract other parameters
        google_scholar_id = data.get('google_scholar_id', '').strip()
        scopus_id = data.get('scopus_id', '').strip()
        affiliation = data.get('affiliation', '').strip()
        api_keys = data.get('api_keys', {})
        filters = data.get('filters', {})
        
        # Create author object
        author = Author(
            first_name=first_name,
            last_name=last_name,
            affiliation=affiliation,
            gs_id=google_scholar_id,
            scopus_id=scopus_id
        )
        
        # Validate that we have enough information to search
        if not google_scholar_id and not scopus_id and not affiliation:
            return jsonify({
                'error': 'At least one of the following is required: Google Scholar ID, Scopus ID, or institutional affiliation'
            }), 400
        
        # Clean API keys - merge user-provided with defaults
        default_keys = get_api_keys()
        clean_api_keys = {
            'scopus_api_key': api_keys.get('scopus_api_key', '').strip() or default_keys.get('scopus_api_key'),
            'wos_api_key': api_keys.get('wos_api_key', '').strip() or default_keys.get('wos_api_key')
        }
        
        # Perform the search
        print(f"🔍 Starting search for {author.full_name}")
        publications = aggregate_publications(
            author=author,
            api_keys=clean_api_keys,
            max_pubs_g_scholar=100,
            headless_g_scholar=True,  # Always headless for web interface
            analyze_coverage=False  # We'll do this separately
        )
        
        # Determine which sources were actually used
        available_sources = []
        if google_scholar_id:
            available_sources.append("Google Scholar")
        if clean_api_keys.get('scopus_api_key') and scopus_id:
            available_sources.append("Scopus")
        elif clean_api_keys.get('scopus_api_key') and affiliation:
            available_sources.append("Scopus")
        if clean_api_keys.get('wos_api_key') and affiliation:
            available_sources.append("Web of Science")
        
        if not available_sources:
            return jsonify({
                'error': 'No data sources available. Please provide either:\n' +
                        '• Google Scholar ID, or\n' +
                        '• Scopus ID with API key, or\n' +
                        '• Institutional affiliation (for Scopus/Web of Science access)',
                'suggestions': [
                    'Add a Google Scholar ID for basic search',
                    'Add a Scopus ID for targeted Scopus search',
                    'Add institutional affiliation for comprehensive coverage',
                    'Verify API keys are valid and have proper permissions'
                ]
            }), 400
        
        # Perform coverage analysis
        coverage_report = analyze_publication_coverage(publications, available_sources)
        
        # Apply filters
        filtered_publications = _apply_filters(publications, filters)
        
        # Debug information (only in development)
        debug_info = {}
        if app.debug:
            debug_info = debug_publication_matching(publications, verbose=False)
            print(f"📊 Debug info: {debug_info['unique_publications']}/{debug_info['total_publications']} unique publications found")
        
        # Format response
        response_data = {
            'success': True,
            'researcher': {
                'name': author.full_name,
                'affiliation': affiliation,
                'gs_id': google_scholar_id,
                'scopus_id': scopus_id,
                'search_timestamp': datetime.now().isoformat()
            },
            'summary': {
                'total_publications': len(filtered_publications),
                'total_before_filters': len(publications),
                'unique_publications': debug_info.get('unique_publications', len(filtered_publications)),
                'sources_used': available_sources,
                'coverage_report': coverage_report.get('summary', {})
            },
            'publications': [_format_publication(pub, publications) for pub in filtered_publications],
            'coverage_analysis': coverage_report,
            'filters_applied': filters
        }
        
        # Add debug info in development mode
        if app.debug and debug_info:
            response_data['debug'] = debug_info
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': f'Search failed: {str(e)}',
            'success': False
        }), 500


@app.route('/api/coverage/<publication_title>')
def api_publication_coverage(publication_title):
    """Get detailed coverage information for a specific publication."""
    # This would be implemented to provide detailed coverage info
    # For now, return a placeholder
    return jsonify({
        'title': publication_title,
        'coverage': {
            'google_scholar': True,
            'scopus': False,
            'wos': True
        },
        'recommendations': [
            'Consider submitting to Scopus for better visibility'
        ]
    })


@app.route('/api/test-matching', methods=['POST'])
def api_test_matching():
    """Test endpoint to verify fuzzy matching logic."""
    if not app.debug:
        return jsonify({'error': 'Test endpoint only available in debug mode'}), 403
    
    try:
        data = request.get_json()
        pub1_data = data.get('publication1', {})
        pub2_data = data.get('publication2', {})
        
        # Create Publication objects for testing
        pub1 = Publication(
            title=pub1_data.get('title', ''),
            authors=pub1_data.get('authors', []),
            journal=pub1_data.get('journal', ''),
            year=pub1_data.get('year'),
            doi=pub1_data.get('doi'),
            citations=pub1_data.get('citations', 0),
            source=pub1_data.get('source', 'Test1')
        )
        
        pub2 = Publication(
            title=pub2_data.get('title', ''),
            authors=pub2_data.get('authors', []),
            journal=pub2_data.get('journal', ''),
            year=pub2_data.get('year'),
            doi=pub2_data.get('doi'),
            citations=pub2_data.get('citations', 0),
            source=pub2_data.get('source', 'Test2')
        )
        
        # Test the matching
        is_match = _publications_match(pub1, pub2)
        
        # Get detailed scores
        title_similarity = 0
        if pub1.title and pub2.title:
            norm1 = _normalize_text(pub1.title)
            norm2 = _normalize_text(pub2.title)
            title_similarity = max(
                fuzz.ratio(norm1, norm2),
                fuzz.token_set_ratio(norm1, norm2),
                fuzz.partial_ratio(norm1, norm2)
            )
        
        author_similarity = 0
        if pub1.authors and pub2.authors:
            author_similarity = _calculate_author_similarity(pub1.authors, pub2.authors)
        
        journal_similarity = 0
        if pub1.journal and pub2.journal:
            norm1 = _normalize_text(pub1.journal)
            norm2 = _normalize_text(pub2.journal)
            journal_similarity = max(
                fuzz.ratio(norm1, norm2),
                fuzz.partial_ratio(norm1, norm2),
                fuzz.token_set_ratio(norm1, norm2)
            )
        
        return jsonify({
            'match': is_match,
            'details': {
                'title_similarity': title_similarity,
                'author_similarity': author_similarity,
                'journal_similarity': journal_similarity,
                'year_match': abs(pub1.year - pub2.year) <= 1 if pub1.year and pub2.year else False,
                'doi_match': pub1.doi and pub2.doi and pub1.doi.lower().strip() == pub2.doi.lower().strip()
            },
            'normalized': {
                'title1': _normalize_text(pub1.title) if pub1.title else '',
                'title2': _normalize_text(pub2.title) if pub2.title else '',
                'journal1': _normalize_text(pub1.journal) if pub1.journal else '',
                'journal2': _normalize_text(pub2.journal) if pub2.journal else ''
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Test failed: {str(e)}'}), 500


@app.route('/api/export', methods=['POST'])
def api_export():
    """Export publications to CSV format."""
    try:
        data = request.get_json()
        publications_data = data.get('publications', [])
        export_format = data.get('format', 'csv')
        
        if export_format == 'csv':
            return _export_to_csv(publications_data)
        else:
            return jsonify({'error': 'Unsupported export format'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Export failed: {str(e)}'}), 500


def _apply_filters(publications: List[Publication], filters: Dict) -> List[Publication]:
    """Apply user-specified filters to publications."""
    filtered_pubs = publications.copy()
    
    # Year range filter
    year_range = filters.get('year_range')
    if year_range and year_range != 'All Years':
        if year_range == '2020-2023':
            filtered_pubs = [p for p in filtered_pubs if p.year and 2020 <= p.year <= 2023]
        elif year_range == '2015-2019':
            filtered_pubs = [p for p in filtered_pubs if p.year and 2015 <= p.year <= 2019]
        elif year_range == 'Before 2015':
            filtered_pubs = [p for p in filtered_pubs if p.year and p.year < 2015]
    
    # Database filter
    database_filter = filters.get('database')
    if database_filter and database_filter != 'All Databases':
        if database_filter == 'Google Scholar Only':
            filtered_pubs = [p for p in filtered_pubs if p.source == 'Google Scholar']
        elif database_filter == 'Scopus Only':
            filtered_pubs = [p for p in filtered_pubs if p.source == 'Scopus']
        elif database_filter == 'Web of Science Only':
            filtered_pubs = [p for p in filtered_pubs if p.source == 'Web of Science']
    
    # Sort
    sort_by = filters.get('sort_by', 'newest')
    if sort_by == 'newest':
        filtered_pubs.sort(key=lambda p: p.year or 0, reverse=True)
    elif sort_by == 'oldest':
        filtered_pubs.sort(key=lambda p: p.year or 0)
    elif sort_by == 'most_cited':
        filtered_pubs.sort(key=lambda p: p.citations or 0, reverse=True)
    
    return filtered_pubs


def _format_publication(pub: Publication, all_publications: List[Publication] = None) -> Dict:
    """Format a publication for JSON response with coverage analysis using fuzzy matching."""
    # Analyze coverage by looking for the same publication across sources
    coverage = {
        'google_scholar': False,
        'scopus': False,
        'wos': False
    }
    
    # If we have all publications, check for duplicates across sources using fuzzy matching
    if all_publications:
        # Find publications that match this one using multiple criteria
        matching_pubs = []
        
        for other_pub in all_publications:
            # Skip if comparing with itself
            if pub is other_pub:
                matching_pubs.append(other_pub)
                continue
            
            # Method 1: Exact DOI match (highest priority)
            if pub.doi and other_pub.doi and pub.doi.lower().strip() == other_pub.doi.lower().strip():
                matching_pubs.append(other_pub)
                continue
            
            # Method 2: Fuzzy matching on title, authors, year, and journal
            title_match = False
            author_match = False
            year_match = False
            journal_match = False
            
            # Title fuzzy matching with multiple strategies
            title_match = False
            title_similarity = 0
            if pub.title and other_pub.title:
                norm_title1 = _normalize_text(pub.title)
                norm_title2 = _normalize_text(other_pub.title)
                
                # Strategy 1: Standard fuzzy ratio
                title_similarity = fuzz.ratio(norm_title1, norm_title2)
                
                # Strategy 2: Token set ratio (handles word order differences)
                token_similarity = fuzz.token_set_ratio(norm_title1, norm_title2)
                
                # Strategy 3: Partial ratio (handles truncated titles)
                partial_similarity = fuzz.partial_ratio(norm_title1, norm_title2)
                
                # Use the best similarity score
                best_title_similarity = max(title_similarity, token_similarity, partial_similarity)
                title_match = best_title_similarity >= 80  # Lowered threshold
                
                # Additional check for very similar titles
                if best_title_similarity >= 95:
                    title_match = True
            
            # Year exact match (allow 1-year difference for edge cases)
            year_match = False
            if pub.year and other_pub.year:
                year_match = abs(pub.year - other_pub.year) <= 1  # Allow 1 year difference
            
            # Author similarity using the improved helper function
            author_match = False
            author_similarity = 0
            if pub.authors and other_pub.authors:
                author_similarity = _calculate_author_similarity(pub.authors, other_pub.authors)
                author_match = author_similarity >= 0.3  # More lenient threshold
            
            # Journal fuzzy matching with better normalization
            journal_match = False
            journal_similarity = 0
            if pub.journal and other_pub.journal:
                norm_journal1 = _normalize_text(pub.journal)
                norm_journal2 = _normalize_text(other_pub.journal)
                
                # Try multiple fuzzy matching strategies for journals
                journal_ratio = fuzz.ratio(norm_journal1, norm_journal2)
                journal_partial = fuzz.partial_ratio(norm_journal1, norm_journal2)
                journal_token = fuzz.token_set_ratio(norm_journal1, norm_journal2)
                
                journal_similarity = max(journal_ratio, journal_partial, journal_token)
                journal_match = journal_similarity >= 70  # More lenient for journals
            
            # Enhanced decision logic with more nuanced scoring
            confidence_score = 0
            
            # Title scoring with gradual points
            if title_match:
                if best_title_similarity >= 95:
                    confidence_score += 4  # Very strong title match
                elif best_title_similarity >= 90:
                    confidence_score += 3.5
                elif best_title_similarity >= 85:
                    confidence_score += 3
                else:
                    confidence_score += 2.5
            
            # Year scoring
            if year_match:
                confidence_score += 2
            
            # Author scoring with gradual points
            if author_match:
                if author_similarity >= 0.7:
                    confidence_score += 2.5  # Strong author match
                elif author_similarity >= 0.5:
                    confidence_score += 2
                else:
                    confidence_score += 1.5
            
            # Journal scoring
            if journal_match:
                if journal_similarity >= 90:
                    confidence_score += 1.5
                else:
                    confidence_score += 1
            
            # More flexible matching criteria
            is_match = False
            
            # High confidence match
            if confidence_score >= 4.5:
                is_match = True
            # Strong title + year combination
            elif title_match and year_match and confidence_score >= 4:
                is_match = True
            # Very strong title match alone (for cases with missing data)
            elif title_match and best_title_similarity >= 95 and confidence_score >= 3:
                is_match = True
            # Title + either authors or journal
            elif title_match and (author_match or journal_match) and confidence_score >= 3.5:
                is_match = True
            
            if is_match:
                matching_pubs.append(other_pub)
        
        # Check which sources have this publication
        for match_pub in matching_pubs:
            source_lower = match_pub.source.lower()
            if 'google scholar' in source_lower:
                coverage['google_scholar'] = True
            elif 'scopus' in source_lower:
                coverage['scopus'] = True
            elif 'web of science' in source_lower or 'wos' in source_lower:
                coverage['wos'] = True
    else:
        # Fallback: just mark the source of this publication
        source_lower = pub.source.lower()
        if 'google scholar' in source_lower:
            coverage['google_scholar'] = True
        elif 'scopus' in source_lower:
            coverage['scopus'] = True
        elif 'web of science' in source_lower or 'wos' in source_lower:
            coverage['wos'] = True
    
    return {
        'title': pub.title,
        'authors': pub.authors,
        'journal': pub.journal,
        'year': pub.year,
        'doi': pub.doi,
        'citations': pub.citations or 0,
        'source': pub.source,
        'url': pub.url,
        'coverage': coverage
    }


def _export_to_csv(publications_data: List[Dict]) -> str:
    """Export publications to CSV format."""
    import io
    import csv
    from flask import make_response
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        'Title', 'Authors', 'Journal', 'Year', 'DOI', 'Citations', 
        'Source', 'Google Scholar', 'Scopus', 'Web of Science'
    ])
    
    # Data rows
    for pub in publications_data:
        coverage = pub.get('coverage', {})
        writer.writerow([
            pub.get('title', ''),
            '; '.join(pub.get('authors', [])),
            pub.get('journal', ''),
            pub.get('year', ''),
            pub.get('doi', ''),
            pub.get('citations', 0),
            pub.get('source', ''),
            'Yes' if coverage.get('google_scholar') else 'No',
            'Yes' if coverage.get('scopus') else 'No',
            'Yes' if coverage.get('wos') else 'No'
        ])
    
    output.seek(0)
    
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename=publications_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    response.headers["Content-type"] = "text/csv"
    
    return response


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('500.html'), 500


def debug_publication_matching(publications: List[Publication], verbose: bool = False) -> Dict:
    """
    Debug utility to analyze publication matching across databases.
    Returns statistics about potential duplicates and matching quality.
    """
    from collections import defaultdict
    
    stats = {
        'total_publications': len(publications),
        'by_source': defaultdict(int),
        'potential_duplicates': [],
        'unique_publications': 0,
        'coverage_summary': defaultdict(int)
    }
    
    # Count by source
    for pub in publications:
        stats['by_source'][pub.source] += 1
    
    # Find potential duplicates using the same fuzzy matching logic
    seen_publications = []
    
    for pub in publications:
        # Format with coverage analysis
        formatted_pub = _format_publication(pub, publications)
        
        # Count coverage patterns
        coverage = formatted_pub['coverage']
        coverage_count = sum(coverage.values())
        stats['coverage_summary'][coverage_count] += 1
        
        # Check if this is truly unique
        is_unique = True
        for seen_pub in seen_publications:
            # Use the same matching logic as _format_publication
            if _publications_match(pub, seen_pub):
                is_unique = False
                if verbose:
                    stats['potential_duplicates'].append({
                        'pub1': {'title': pub.title, 'source': pub.source, 'year': pub.year},
                        'pub2': {'title': seen_pub.title, 'source': seen_pub.source, 'year': seen_pub.year}
                    })
                break
        
        if is_unique:
            seen_publications.append(pub)
    
    stats['unique_publications'] = len(seen_publications)
    
    return dict(stats)


def _publications_match(pub1: Publication, pub2: Publication) -> bool:
    """Check if two publications are the same using improved fuzzy matching."""
    # DOI match
    if pub1.doi and pub2.doi and pub1.doi.lower().strip() == pub2.doi.lower().strip():
        return True
    
    # Enhanced fuzzy matching
    title_match = False
    year_match = False
    author_match = False
    journal_match = False
    
    # Title matching with multiple strategies
    if pub1.title and pub2.title:
        norm_title1 = _normalize_text(pub1.title)
        norm_title2 = _normalize_text(pub2.title)
        
        title_ratio = fuzz.ratio(norm_title1, norm_title2)
        token_ratio = fuzz.token_set_ratio(norm_title1, norm_title2)
        partial_ratio = fuzz.partial_ratio(norm_title1, norm_title2)
        
        best_title_similarity = max(title_ratio, token_ratio, partial_ratio)
        title_match = best_title_similarity >= 80
    
    # Year matching (allow 1 year difference)
    if pub1.year and pub2.year:
        year_match = abs(pub1.year - pub2.year) <= 1
    
    # Author matching
    if pub1.authors and pub2.authors:
        author_similarity = _calculate_author_similarity(pub1.authors, pub2.authors)
        author_match = author_similarity >= 0.3
    
    # Journal matching
    if pub1.journal and pub2.journal:
        norm_journal1 = _normalize_text(pub1.journal)
        norm_journal2 = _normalize_text(pub2.journal)
        journal_similarity = max(
            fuzz.ratio(norm_journal1, norm_journal2),
            fuzz.partial_ratio(norm_journal1, norm_journal2),
            fuzz.token_set_ratio(norm_journal1, norm_journal2)
        )
        journal_match = journal_similarity >= 70
    
    # Use same confidence scoring as main function
    confidence_score = 0
    if title_match:
        confidence_score += 3
    if year_match:
        confidence_score += 2
    if author_match:
        confidence_score += 2
    if journal_match:
        confidence_score += 1
    
    # Same matching criteria as main function
    return (confidence_score >= 4.5 or 
            (title_match and year_match and confidence_score >= 4) or
            (title_match and (author_match or journal_match) and confidence_score >= 3.5))


if __name__ == '__main__':
    # Development server
    app.run(debug=True, host='0.0.0.0', port=5000)
