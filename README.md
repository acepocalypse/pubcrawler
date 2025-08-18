# PubCrawler 📚🔍

A modular pipeline for harvesting academic publications from multiple sources with **index coverage analysis**.

PubCrawler aggregates publications from Google Scholar, Scopus, Web of Science, and ORCID, deduplicates them, and provides detailed analysis of which databases index each publication - helping identify coverage gaps and improve research visibility.

## ✨ Key Features

- **Multi-source aggregation**: Google Scholar, Scopus, Web of Science, ORCID
- **🆕 Direct ID Support**: Use Google Scholar IDs, Scopus IDs, Web of Science IDs, ORCID IDs, or institutional affiliations
- **🔍 Index Coverage Analysis**: Identify where publications are indexed and where they're missing
- **Smart deduplication**: Uses DOI matching and fuzzy title matching
- **Parallel processing**: Fast concurrent data retrieval
- **Robust error handling**: Graceful degradation when sources are unavailable
- **Coverage reports**: Actionable insights for improving publication visibility
- **Web Interface**: User-friendly web application with real-time search
- **CLI and API**: Use from command line or integrate into your code

## 🚀 Recent Improvements

- ✅ **ORCID Support**: Direct lookup using ORCID IDs for authoritative publication data
- ✅ **Scopus ID Support**: Direct lookup using Scopus Author IDs for faster, more accurate results
- ✅ **Web of Science ID Support**: Direct lookup using WoS Author IDs for targeted, accurate results
- ✅ **Enhanced Web Interface**: Improved frontend with separate input fields for all ID types
- ✅ **Better Error Handling**: More informative error messages and suggestions
- ✅ **Optimized Backend**: Updated aggregation logic to handle multiple ID types efficiently
- ✅ **All Scrapers Working**: Verified and optimized all four data sources

## 🔍 Index Coverage Analysis

PubCrawler's unique **coverage analysis** feature helps you understand:

- ✅ **Complete Coverage**: Publications indexed in all available databases
- 🟡 **Partial Coverage**: Publications missing from some databases  
- 🔴 **Coverage Gaps**: Publications indexed in only one source
- ⭐ **High-impact Gaps**: Highly-cited papers with limited visibility
- 📊 **Source Statistics**: Coverage rates across databases
- 💡 **Recommendations**: Actionable steps to improve visibility

### Coverage Categories

| Category | Description | Action Needed |
|----------|-------------|---------------|
| ✅ Complete Coverage | Indexed in all queried sources | Monitor for new publications |
| 🟢 Good Coverage | Indexed in 75%+ of sources | Consider submitting to missing databases |
| 🟡 Partial Coverage | Indexed in 50-75% of sources | Priority for submission to missing sources |
| 🟠 Limited Coverage | Indexed in 25-50% of sources | High priority for broader indexing |
| 🔴 No Coverage | Not found in any source | Investigate publication status |

### Sample Coverage Report

```
📊 PUBLICATION INDEX COVERAGE ANALYSIS
================================================================================

📈 Overview:
  • Total Publications: 87
  • Average Coverage: 67.8%
  • Complete Coverage: 23 publications
  • No Coverage: 5 publications

🎯 Coverage Distribution:
  • Complete Coverage: 23 (26.4%)
  • Good Coverage: 31 (35.6%)
  • Partial Coverage: 18 (20.7%)
  • Limited Coverage: 10 (11.5%)

🗃️ Source Statistics:
  • Google Scholar: 82/87 (94.3%) indexed
  • Scopus: 65/87 (74.7%) indexed  
  • Web of Science: 58/87 (66.7%) indexed
  • ORCID: 50/87 (57.5%) indexed

💡 Recommendations:
  1. 🔍 Consider submitting 22 publications missing from Scopus
  2. 📈 Priority: Submit 29 publications missing from Web of Science
  3. ⭐ 12 high-impact publications have coverage gaps
```

## Features

- **Multi-source aggregation**: Fetch publications from Google Scholar, Scopus, Web of Science, and ORCID
- **Intelligent deduplication**: Advanced algorithms using DOI matching and fuzzy title matching
- **Standardized output**: All sources normalized to consistent data schema
- **Parallel processing**: Concurrent fetching from multiple sources for speed
- **Resilient design**: Graceful handling of missing dependencies and API failures
- **Command-line interface**: Easy-to-use CLI for quick searches
- **Flexible API**: Programmatic access for integration into larger workflows

## Installation

```bash
# Clone the repository
git clone https://github.com/acepocalypse/pubcrawler.git
cd pubcrawler

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

## Quick Start

### Web Interface

1. **Start the web application**:
   ```bash
   python run_web.py
   ```

2. **Open your browser** to `http://localhost:5000`

3. **Search for publications**:
   - Enter a researcher's name
   - Add Google Scholar ID, Scopus ID, Web of Science ID, ORCID ID, or institutional affiliation
   - Optionally provide API keys for enhanced access
   - View results with coverage analysis

### Command Line Usage

```bash
# Basic usage with Google Scholar ID
python -m pubcrawler --first John --last Doe --gs-id ABC123DEF

# Using Scopus ID for direct lookup
python -m pubcrawler --first John --last Doe --scopus-id 56518239200

# Using ORCID ID for direct lookup
python -m pubcrawler --first John --last Doe --orcid-id 0000-0002-1825-0097

# Include all sources (requires API keys)
python -m pubcrawler \
    --first John --last Doe \
    --affiliation "MIT" \
    --gs-id ABC123DEF \
    --scopus-id 56518239200 \
    --orcid-id 0000-0002-1825-0097 \
    --scopus-key YOUR_SCOPUS_KEY \
    --wos-key YOUR_WOS_KEY \
    --max-gs 100 \
    --headless
```

### Python API Usage

```python
from pubcrawler import aggregate_publications, analyze_publication_coverage
from pubcrawler.models import Author

# Create author object with multiple IDs for better coverage
author = Author(
    first_name="John",
    last_name="Doe",
    affiliation="Massachusetts Institute of Technology",
    gs_id="ABC123DEF",           # Google Scholar ID
    scopus_id="56518239200",      # Scopus Author ID
    wos_id="B-1234-5678",        # Web of Science Author ID (preferred for WoS)
    orcid_id="0000-0002-1825-0097" # ORCID ID
)

# Set up API keys (optional for web scraping, required for API access)
api_keys = {
    "scopus_api_key": "your_scopus_key",
    "wos_api_key": "your_wos_key"
}

# Fetch and aggregate publications with coverage analysis
publications = aggregate_publications(
    author=author,
    api_keys=api_keys,
    max_pubs_g_scholar=100,
    headless_g_scholar=True,
    analyze_coverage=True  # Enable coverage analysis
)

print(f"Found {len(publications)} unique publications")

# Get detailed coverage report
coverage_report = analyze_publication_coverage(
    publications, 
    available_sources=["Google Scholar", "Scopus", "Web of Science", "ORCID"]
)

# Show coverage summary
print(f"Average coverage: {coverage_report['summary']['average_coverage']:.1%}")
print(f"Complete coverage: {coverage_report['summary']['complete_coverage_count']} publications")
```

### Coverage Analysis Only

```python
from pubcrawler.coverage import analyze_publication_coverage, print_coverage_report

# Analyze existing publications list
coverage_report = analyze_publication_coverage(publications)

# Print detailed coverage report
print_coverage_report(publications, ["Google Scholar", "Scopus", "Web of Science", "ORCID"])

# Access specific gap information
gaps = coverage_report["gap_analysis"]
high_impact_gaps = gaps["high_impact_missing"]

print(f"High-impact publications with gaps: {len(high_impact_gaps)}")
for pub in high_impact_gaps[:3]:  # Top 3
    print(f"- {pub['title']} ({pub['year']}) - {pub['citations']} citations")
    print(f"  Missing from: {pub['indexed_in']}")
```

## Coverage Analysis Features

The coverage analysis system helps identify indexing gaps across academic databases:

- **Complete Coverage**: Publications indexed in all available databases
- **Partial Coverage**: Publications missing from one or more databases
- **High Impact Gaps**: Well-cited publications with indexing gaps
- **Coverage Recommendations**: Actionable insights for improving visibility

### Coverage Report Example

```
Publication Coverage Analysis
============================

Summary:
  Total Publications: 42
  Average Coverage: 78.6%
  Complete Coverage: 28 publications (66.7%)

Gap Analysis:
  Publications with gaps: 14
  High-impact papers missing: 3
  Most common gap: Web of Science (8 publications)

Recommendations:
  1. Submit 3 high-impact papers to Web of Science
  2. Update Scopus profiles for 2 publications
  3. Verify Google Scholar indexing for recent work
```

For more examples, see `example.py` in the repository.

## Individual Source Usage

```python
# Google Scholar only
from pubcrawler.sources import google_scholar
pubs = google_scholar.fetch("scholar_user_id", "John", "Doe")

# Scopus with direct ID lookup (NEW - Faster and more accurate)
from pubcrawler.sources import scopus
pubs = scopus.fetch(scopus_id="56518239200", api_key="your_scopus_key")

# Scopus with name/affiliation search (legacy method)
from pubcrawler.sources import scopus
pubs = scopus.fetch("John", "Doe", "MIT", "your_scopus_key")

# Web of Science only
from pubcrawler.sources import wos
pubs = wos.fetch("John", "Doe", "MIT", "your_wos_key")

# ORCID only
from pubcrawler.sources import orcid
pubs = orcid.fetch("0000-0002-1825-0097")
```

## Finding Your IDs

### Google Scholar ID
1. Go to your Google Scholar profile
2. The ID is in the URL: `https://scholar.google.com/citations?user=YOUR_ID_HERE`

### Scopus ID
1. Search for the author on Scopus
2. Click on the author's name in search results
3. The ID is in the URL: `https://www.scopus.com/authid/detail.uri?authorId=YOUR_ID_HERE`
4. Or use the Author Search API to find the ID programmatically

### Web of Science Author ID
1. Log into Web of Science and search for the author
2. Click on the author's name in search results
3. Look for the Author Identifier (e.g., `B-1234-5678`) on the author profile page
4. This can also be found in the ResearcherID system
5. Example formats: `B-1234-5678`, `A-1234-2022`, etc.

### ORCID ID
1. Go to the ORCID website
2. Search for the author or use the direct URL if known: `https://orcid.org/0000-0002-1825-0097`
3. The ID is the 16-digit number after `/orcid.org/` in the URL
4. Example: For `https://orcid.org/0000-0002-1825-0097`, the ORCID ID is `0000-0002-1825-0097`

## Configuration

### API Keys

PubCrawler requires API keys for Scopus and Web of Science. Set them as environment variables:

```bash
export SCOPUS_API_KEY="your_scopus_api_key"
export WOS_API_KEY="your_wos_api_key"
```

Or pass them directly to the functions.

### Google Scholar Setup

Google Scholar scraping uses Selenium with undetected Chrome driver. The system will automatically:
- Download the appropriate ChromeDriver
- Configure anti-detection settings
- Handle rate limiting and CAPTCHA avoidance

## Data Schema

All sources are normalized to a common `Publication` schema:

```python
@dataclass
class Publication:
    title: str                    # Publication title (normalized)
    authors: List[str]           # List of author names
    journal: Optional[str]       # Journal/venue name
    year: Optional[int]          # Publication year
    doi: Optional[str]           # Digital Object Identifier
    issn: Optional[str]          # International Standard Serial Number
    source: str                  # Source database ("Google Scholar", "Scopus", "WoS", "ORCID")
    citations: Optional[int]     # Citation count
    url: Optional[str]           # Direct URL to publication
```

## Deduplication Strategy

1. **DOI-based**: Publications with the same DOI are considered duplicates (highest citation count kept)
2. **Fuzzy title matching**: For publications without DOIs, uses fuzzy string matching on titles within the same publication year
3. **Citation preference**: When duplicates are found, the version with the highest citation count is retained

## Dependencies

Core dependencies:
- `pandas` - Data manipulation and analysis
- `requests` - HTTP requests for API calls
- `selenium` - Web scraping for Google Scholar
- `undetected-chromedriver` - Anti-detection Chrome driver
- `beautifulsoup4` - HTML parsing
- `rapidfuzz` - Fast fuzzy string matching (optional, for advanced deduplication)

See `requirements.txt` for the complete list.

## Error Handling

- **Graceful degradation**: Missing optional dependencies don't crash the pipeline
- **Source isolation**: Failures in one source don't affect others
- **Rate limiting**: Automatic handling of API rate limits and CAPTCHA detection
- **Retry logic**: Automatic retries for transient failures

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Scholar for providing open access to academic publications
- Elsevier (Scopus) and Clarivate (Web of Science) for their APIs
- The open-source community for the excellent tools that make this project possible
