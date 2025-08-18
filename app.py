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

from aggregate import aggregate_publications, _publications_match, _normalize_text_for_matching, _calculate_author_similarity
from models import Author, Publication
from coverage import analyze_publication_coverage
from config_keys import get_api_keys
from rapidfuzz import fuzz


app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'pubcrawler-dev-key-change-in-production')


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
        orcid_id = data.get('orcid_id', '').strip()
        affiliation = data.get('affiliation', '').strip()
        api_keys = data.get('api_keys', {})
        
        # Create author object
        author = Author(
            first_name=first_name,
            last_name=last_name,
            affiliation=affiliation,
            gs_id=google_scholar_id,
            scopus_id=scopus_id,
            orcid_id=orcid_id
        )
        
        # Validate that we have enough information to search
        if not google_scholar_id and not scopus_id and not orcid_id and not affiliation:
            return jsonify({
                'error': 'At least one of the following is required: Google Scholar ID, Scopus ID, ORCID ID, or institutional affiliation'
            }), 400
        
        # Clean API keys - merge user-provided with defaults
        default_keys = get_api_keys()
        clean_api_keys = {
            'scopus_api_key': api_keys.get('scopus_api_key', '').strip() or default_keys.get('scopus_api_key'),
            'wos_api_key': api_keys.get('wos_api_key', '').strip() or default_keys.get('wos_api_key'),
            'orcid_client_id': api_keys.get('orcid_client_id', '').strip() or default_keys.get('orcid_client_id'),
            'orcid_client_secret': api_keys.get('orcid_client_secret', '').strip() or default_keys.get('orcid_client_secret')
        }
        
        # Perform the search
        print(f"🔍 Starting search for {author.full_name}")
        publications = aggregate_publications(
            author=author,
            api_keys=clean_api_keys,
            max_pubs_g_scholar=1000,
            headless_g_scholar=False,  # Always headless for web interface
            analyze_coverage=False  # We'll do this separately
        )
        
        # Determine which sources were actually used
        available_sources = []
        if google_scholar_id:
            available_sources.append("Google Scholar")
        if clean_api_keys.get('scopus_api_key') and (scopus_id or orcid_id):
            available_sources.append("Scopus")
        elif clean_api_keys.get('scopus_api_key') and affiliation:
            available_sources.append("Scopus")
        if clean_api_keys.get('wos_api_key') and (affiliation or orcid_id):
            available_sources.append("Web of Science")
        if (clean_api_keys.get('orcid_client_id') and clean_api_keys.get('orcid_client_secret') and orcid_id):
            available_sources.append("ORCID")
        
        if not available_sources:
            return jsonify({
                'error': 'No data sources available. Please provide either:\n' +
                        '• Google Scholar ID, or\n' +
                        '• Scopus ID with API key, or\n' +
                        '• ORCID ID with client credentials, or\n' +
                        '• Institutional affiliation (for Scopus/Web of Science access)',
                'suggestions': [
                    'Add a Google Scholar ID for basic search',
                    'Add a Scopus ID for targeted Scopus search',
                    'Add an ORCID ID for comprehensive coverage',
                    'Add institutional affiliation for broader coverage',
                    'Verify API keys are valid and have proper permissions'
                ]
            }), 400
        
        # Perform coverage analysis
        coverage_report = analyze_publication_coverage(publications, available_sources)
        
        # Debug information (only in development)
        debug_info = {}
        if app.debug:
            debug_info = debug_publication_matching(publications, verbose=False)
            print(f"📊 Debug info: {debug_info['unique_publications']}/{debug_info['total_publications']} unique publications found")
        
        # Format response (no server-side filtering - done on client)
        response_data = {
            'success': True,
            'researcher': {
                'name': author.full_name,
                'affiliation': affiliation,
                'gs_id': google_scholar_id,
                'scopus_id': scopus_id,
                'orcid_id': orcid_id,  # Keep as orcid_id for frontend consistency
                'search_timestamp': datetime.now().isoformat()
            },
            'summary': {
                'total_publications': len(publications),
                'total_before_filters': len(publications),
                'unique_publications': debug_info.get('unique_publications', len(publications)),
                'sources_used': available_sources,
                'coverage_report': coverage_report.get('summary', {})
            },
            'publications': [_format_publication(pub, publications) for pub in publications],
            'coverage_analysis': coverage_report
        }
        
        # Clean the response data to ensure JSON serialization
        response_data = _clean_response_for_json(response_data)
        
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
            norm1 = _normalize_text_for_matching(pub1.title)
            norm2 = _normalize_text_for_matching(pub2.title)
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
            norm1 = _normalize_text_for_matching(pub1.journal)
            norm2 = _normalize_text_for_matching(pub2.journal)
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
                'title1': _normalize_text_for_matching(pub1.title) if pub1.title else '',
                'title2': _normalize_text_for_matching(pub2.title) if pub2.title else '',
                'journal1': _normalize_text_for_matching(pub1.journal) if pub1.journal else '',
                'journal2': _normalize_text_for_matching(pub2.journal) if pub2.journal else ''
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
        researcher_info = data.get('researcher', {})
        export_format = data.get('format', 'csv')
        
        if export_format == 'csv':
            return _export_to_csv(publications_data, researcher_info)
        else:
            return jsonify({'error': 'Unsupported export format'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Export failed: {str(e)}'}), 500


def _clean_response_for_json(data):
    """Recursively clean response data to ensure JSON serialization."""
    import math
    
    if isinstance(data, dict):
        return {key: _clean_response_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_clean_response_for_json(item) for item in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    elif data is None:
        return None
    elif isinstance(data, str) and data.lower() in ['nan', 'none', 'null']:
        return None
    else:
        return data


def _format_publication(pub: Publication, all_publications: List[Publication] = None) -> Dict:
    """Format a publication for JSON response with coverage analysis using fuzzy matching."""
    import math
    
    # Helper function to clean values for JSON serialization
    def clean_value(value):
        if value is None:
            return None
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if isinstance(value, str) and value.lower() in ['nan', 'none', 'null', '']:
            return None
        return value
    
    # Analyze coverage by looking for the same publication across sources
    coverage = {
        'google_scholar': False,
        'scopus': False,
        'wos': False,
        'orcid': False
    }
    
    # Track citations from each source
    source_citations = {}
    
    # Check if this publication has already been merged from multiple sources
    source_lower = pub.source.lower() if pub.source else ""
    if 'multiple sources:' in source_lower:
        # Parse the merged sources with citation information
        # Format: "Multiple sources: Google Scholar (45 cites), Scopus (52 cites), ORCID (0 cites)"
        import re
        
        if 'google scholar' in source_lower:
            coverage['google_scholar'] = True
            # Extract citation count for Google Scholar
            match = re.search(r'google scholar \((\d+) cites?\)', source_lower)
            if match:
                source_citations['google_scholar'] = int(match.group(1))
        
        if 'scopus' in source_lower:
            coverage['scopus'] = True
            # Extract citation count for Scopus
            match = re.search(r'scopus \((\d+) cites?\)', source_lower)
            if match:
                source_citations['scopus'] = int(match.group(1))
        
        if 'web of science' in source_lower or 'wos' in source_lower:
            coverage['wos'] = True
            # Extract citation count for Web of Science
            match = re.search(r'(?:web of science|wos) \((\d+) cites?\)', source_lower)
            if match:
                source_citations['wos'] = int(match.group(1))
        
        if 'orcid' in source_lower:
            coverage['orcid'] = True
            # Extract citation count for ORCID
            match = re.search(r'orcid \((\d+) cites?\)', source_lower)
            if match:
                source_citations['orcid'] = int(match.group(1))
    else:
        # Single source publication - mark its source and look for fuzzy matches
        if 'google scholar' in source_lower:
            coverage['google_scholar'] = True
            source_citations['google_scholar'] = clean_value(pub.citations) or 0
        elif 'scopus' in source_lower:
            coverage['scopus'] = True
            source_citations['scopus'] = clean_value(pub.citations) or 0
        elif 'web of science' in source_lower or 'wos' in source_lower:
            coverage['wos'] = True
            source_citations['wos'] = clean_value(pub.citations) or 0
        elif 'orcid' in source_lower:
            coverage['orcid'] = True
            source_citations['orcid'] = clean_value(pub.citations) or 0
        
        # If we have all publications, check for duplicates across sources using fuzzy matching
        if all_publications:
            # Find publications that match this one using multiple criteria
            matching_pubs = []
            
            for other_pub in all_publications:
                # Skip if comparing with itself
                if pub is other_pub:
                    continue
                
                # Method 1: Exact DOI match (highest priority)
                if (pub.doi and other_pub.doi and 
                    clean_value(pub.doi) and clean_value(other_pub.doi) and
                    pub.doi.lower().strip() == other_pub.doi.lower().strip()):
                    matching_pubs.append(other_pub)
                    continue
                
                # Method 2: Fuzzy matching on title, authors, year, and journal
                if _publications_match(pub, other_pub):
                    matching_pubs.append(other_pub)
            
            # Check which sources have this publication and their citation counts
            for match_pub in matching_pubs:
                other_source_lower = match_pub.source.lower() if match_pub.source else ""
                match_citations = clean_value(match_pub.citations) or 0
                
                # Handle both single sources AND already-merged sources
                if 'multiple sources:' in other_source_lower:
                    # Parse merged sources to extract individual source information
                    import re
                    
                    if 'google scholar' in other_source_lower and not coverage['google_scholar']:
                        coverage['google_scholar'] = True
                        # Extract citation count for Google Scholar
                        match = re.search(r'google scholar \((\d+) cites?\)', other_source_lower)
                        if match:
                            source_citations['google_scholar'] = int(match.group(1))
                    
                    if 'scopus' in other_source_lower and not coverage['scopus']:
                        coverage['scopus'] = True
                        # Extract citation count for Scopus
                        match = re.search(r'scopus \((\d+) cites?\)', other_source_lower)
                        if match:
                            source_citations['scopus'] = int(match.group(1))
                    
                    if ('web of science' in other_source_lower or 'wos' in other_source_lower) and not coverage['wos']:
                        coverage['wos'] = True
                        # Extract citation count for Web of Science
                        match = re.search(r'(?:web of science|wos) \((\d+) cites?\)', other_source_lower)
                        if match:
                            source_citations['wos'] = int(match.group(1))
                    
                    if 'orcid' in other_source_lower and not coverage['orcid']:
                        coverage['orcid'] = True
                        # Extract citation count for ORCID
                        match = re.search(r'orcid \((\d+) cites?\)', other_source_lower)
                        if match:
                            source_citations['orcid'] = int(match.group(1))
                else:
                    # Handle single sources
                    if 'google scholar' in other_source_lower and not coverage['google_scholar']:
                        coverage['google_scholar'] = True
                        source_citations['google_scholar'] = match_citations
                    elif 'scopus' in other_source_lower and not coverage['scopus']:
                        coverage['scopus'] = True
                        source_citations['scopus'] = match_citations
                    elif ('web of science' in other_source_lower or 'wos' in other_source_lower) and not coverage['wos']:
                        coverage['wos'] = True
                        source_citations['wos'] = match_citations
                    elif 'orcid' in other_source_lower and not coverage['orcid']:
                        coverage['orcid'] = True
                        source_citations['orcid'] = match_citations
    
    # Clean and format all values for JSON
    formatted_pub = {
        'title': clean_value(pub.title) or "",
        'authors': pub.authors if isinstance(pub.authors, list) else [],
        'journal': clean_value(pub.journal),
        'year': clean_value(pub.year),
        'doi': clean_value(pub.doi),
        'citations': int(clean_value(pub.citations)) if clean_value(pub.citations) is not None else 0,
        'source': clean_value(pub.source) or "Unknown",
        'url': clean_value(pub.url),
        'coverage': coverage
    }
    
    # Add detailed citation information if available
    if source_citations:
        formatted_pub['source_citations'] = source_citations
        
        # Calculate total unique coverage
        total_coverage = sum(1 for covered in coverage.values() if covered)
        formatted_pub['coverage_count'] = total_coverage
    
    return formatted_pub


def _export_to_csv(publications_data: List[Dict], researcher_info: Dict = None) -> str:
    """Export publications to CSV format."""
    import io
    import csv
    import re
    from flask import make_response
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header with detailed citation information and ORCID
    writer.writerow([
        'Title', 'Authors', 'Journal', 'Year', 'DOI', 'Max Citations', 
        'Source', 'Google Scholar', 'Scopus', 'Web of Science', 'ORCID',
        'GS Citations', 'Scopus Citations', 'WoS Citations', 'Coverage Count'
    ])
    
    # Data rows
    for pub in publications_data:
        coverage = pub.get('coverage', {})
        source_citations = pub.get('source_citations', {})
        
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
            'Yes' if coverage.get('wos') else 'No',
            'Yes' if coverage.get('orcid') else 'No',
            source_citations.get('google_scholar', '') if source_citations.get('google_scholar') is not None else '',
            source_citations.get('scopus', '') if source_citations.get('scopus') is not None else '',
            source_citations.get('wos', '') if source_citations.get('wos') is not None else '',
            pub.get('coverage_count', sum(1 for v in coverage.values() if v))
        ])
    
    output.seek(0)
    
    # Generate filename with researcher name
    if researcher_info:
        researcher_name = researcher_info.get('name', 'Unknown_Researcher')
        # Clean the name for filename (remove special characters)
        clean_name = re.sub(r'[^\w\s-]', '', researcher_name).strip()
        clean_name = re.sub(r'[-\s]+', '_', clean_name)
        orcid_id = researcher_info.get('orcid_id', '')
        orcid_suffix = f"_{orcid_id}" if orcid_id else ""
        filename = f"publications_{clean_name}{orcid_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    else:
        filename = f"publications_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
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
        # Handle both single sources and merged sources
        if 'multiple sources:' in pub.source.lower():
            # Extract individual sources from merged string
            source_part = pub.source.split(':', 1)[1].strip()
            individual_sources = [s.strip() for s in source_part.split(',')]
            for source in individual_sources:
                stats['by_source'][source] += 1
        else:
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


if __name__ == '__main__':
    # Development server
    app.run(debug=True, host='0.0.0.0', port=5000)
