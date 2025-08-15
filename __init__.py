"""PubCrawler: A modular pipeline for harvesting academic publications.

PubCrawler aggregates publications from Google Scholar, Scopus, and Web of Science,
deduplicates them, and returns a clean dataset ready for analysis or front-end display.

Quick Start
-----------

```python
from pubcrawler import aggregate_publications
from pubcrawler.models import Author

# Basic usage
author = Author(
    first_name="John",
    last_name="Doe", 
    affiliation="University of Example",
    gs_id="ABC123DEF"  # Optional
)

api_keys = {
    "scopus_api_key": "your_scopus_key",  # Optional
    "wos_api_key": "your_wos_key",  # Optional
}

publications = aggregate_publications(
    author=author,
    api_keys=api_keys,
    max_pubs_g_scholar=100,
    headless_g_scholar=True
)

print(f"Found {len(publications)} unique publications")
```

Individual Source Usage
-----------------------

```python
# Google Scholar only
from pubcrawler.sources import google_scholar
pubs = google_scholar.fetch("scholar_user_id", "John", "Doe")

# Scopus only  
from pubcrawler.sources import scopus
pubs = scopus.fetch("John", "Doe", "University", "api_key")

# Web of Science only
from pubcrawler.sources import wos
pubs = wos.fetch("John", "Doe", "University", "api_key")
```
"""

from .models import Publication, Author
from .aggregate import aggregate_publications
from .config import get_config, PubCrawlerConfig
from .utils import setup_logging
from .coverage import analyze_publication_coverage, print_coverage_report

# Source modules are available for direct import if needed
from . import sources

__version__ = "0.1.0"
__author__ = "PubCrawler Team"

__all__ = [
    # Core models
    "Publication",
    "Author",
    
    # Main aggregation function
    "aggregate_publications",
    
    # Coverage analysis
    "analyze_publication_coverage",
    "print_coverage_report",
    
    # Configuration
    "get_config",
    "PubCrawlerConfig",
    
    # Utilities
    "setup_logging",
    
    # Source modules
    "sources",
]
