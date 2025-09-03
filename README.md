# PubCrawler

A modular pipeline for harvesting academic publications from multiple sources with **index coverage analysis**.  

PubCrawler aggregates publications from Google Scholar, Scopus, Web of Science, and ORCID, deduplicates results, and analyzes database coverage to help identify indexing gaps and improve research visibility.

---

## Key Features

- **Multi-source aggregation**: Google Scholar, Scopus, Web of Science, ORCID  
- **Direct ID support**: Query by Google Scholar, Scopus, Web of Science, ORCID IDs, or institutional affiliation  
- **Coverage analysis**: Identify where publications are indexed or missing  
- **Smart deduplication**: DOI-based and fuzzy title matching  
- **Parallel processing**: Concurrent retrieval for speed  
- **Resilient design**: Error handling and graceful degradation when sources are unavailable  
- **Coverage reports**: Summaries and recommendations to improve visibility  
- **Interfaces**: Web UI, CLI, and Python API  

---

## Recent Improvements

- ORCID, Scopus, and Web of Science ID lookup for faster, authoritative results  
- Enhanced web interface with dedicated ID input fields  
- Improved error handling with more informative messages  
- Optimized backend aggregation for multiple ID types  
- Verified and updated scrapers across all supported sources  

---

## Coverage Analysis

PubCrawler provides a detailed view of publication indexing across sources. It categorizes each publication as:  

- **Complete coverage**: Indexed in all queried sources  
- **Good coverage**: Indexed in 75% or more of sources  
- **Partial coverage**: Indexed in 50–75% of sources  
- **Limited coverage**: Indexed in 25–50% of sources  
- **No coverage**: Not indexed in any source  

Example statistics include coverage distribution, source-by-source indexing rates, and actionable recommendations (e.g., submitting missing publications to databases).  

---

## Installation

```bash
git clone https://github.com/acepocalypse/pubcrawler.git
cd pubcrawler
pip install -r requirements.txt
# Optional: development mode
pip install -e .
```

---

## Usage

### Web Interface

1. Start the web application:
   ```bash
   python run_web.py
   ```
2. Open `http://localhost:5000` in a browser.  
3. Search using a researcher’s name, IDs, or affiliation.  

### Desktop App (One-Click)

A native desktop launcher is available for single-user, local use. It embeds the existing web UI in a window using a lightweight webview and runs the Flask server in the background. The original network-accessible entry (`run_web.py`) and scripts (`start_network.sh` / `.bat`) remain unchanged and continue to work as before.

- Run from source (requires `pywebview`):
  ```bash
  pip install pywebview
  python desktop_app.py
  ```
  The app starts locally on a random free port and opens in a desktop window.

- Build a standalone executable with PyInstaller:
  - macOS/Linux:
    ```bash
    ./build_desktop.sh
    # Launch the built app
    ./dist/PubCrawlerDesktop/PubCrawlerDesktop
    ```
  - Windows:
    ```bat
    build_desktop.bat
    rem Launch the built app
    dist\PubCrawlerDesktop\PubCrawlerDesktop.exe
    ```

These scripts bundle templates and static assets, producing a single-folder app under `dist/`. If you prefer a single-file binary, add `--onefile` to the PyInstaller command in the scripts.

### Command Line

```bash
# Basic example with Google Scholar ID
python -m pubcrawler --first John --last Doe --gs-id ABC123DEF

# Direct lookup using Scopus ID
python -m pubcrawler --first John --last Doe --scopus-id 56518239200

# Include multiple IDs and sources
python -m pubcrawler   --first John --last Doe   --affiliation "MIT"   --gs-id ABC123DEF   --scopus-id 56518239200   --orcid-id 0000-0002-1825-0097   --scopus-key YOUR_SCOPUS_KEY   --wos-key YOUR_WOS_KEY   --max-gs 100   --headless
```

### Python API

```python
from pubcrawler import aggregate_publications, analyze_publication_coverage
from pubcrawler.models import Author

author = Author(
    first_name="John",
    last_name="Doe",
    affiliation="MIT",
    gs_id="ABC123DEF",
    scopus_id="56518239200",
    wos_id="B-1234-5678",
    orcid_id="0000-0002-1825-0097"
)

api_keys = {
    "scopus_api_key": "your_scopus_key",
    "wos_api_key": "your_wos_key"
}

publications = aggregate_publications(author, api_keys, analyze_coverage=True)
coverage_report = analyze_publication_coverage(publications)
```

---

## Configuration

- **API Keys**: Required for Scopus and Web of Science. Set via environment variables:  
  ```bash
  export SCOPUS_API_KEY="your_key"
  export WOS_API_KEY="your_key"
  ```
- **Google Scholar**: Uses Selenium with undetected ChromeDriver for scraping, including automatic driver setup and CAPTCHA handling.  

---

## Data Model

All publications are normalized to a standard schema:

```python
@dataclass
class Publication:
    title: str
    authors: List[str]
    journal: Optional[str]
    year: Optional[int]
    doi: Optional[str]
    issn: Optional[str]
    source: str
    citations: Optional[int]
    url: Optional[str]
```

---

## Deduplication

1. **DOI-based**: Treats identical DOIs as duplicates (retaining the highest citation count).  
2. **Title-based**: Fuzzy title matching within the same publication year.  
3. **Citation preference**: Keeps the version with the highest citation count.  

---

## Dependencies

- `pandas` – Data processing  
- `requests` – API interaction  
- `selenium`, `undetected-chromedriver` – Google Scholar scraping  
- `beautifulsoup4` – HTML parsing  
- `rapidfuzz` – Fuzzy matching  

See `requirements.txt` for the full list.

Desktop build requires in addition:
- `pywebview` – Native window wrapper for the web UI
- `pyinstaller` – To package as a standalone executable

---

## Contributing

1. Fork the repository  
2. Create a feature branch  
3. Commit changes  
4. Push to your fork  
5. Open a Pull Request  

---

## License

MIT License. See the [LICENSE](LICENSE) file for details.  
