"""
PubCrawler Configuration
~~~~~~~~~~~~~~~~~~~~~~~
Configuration settings for PubCrawler including API keys and default settings.
"""

import os
from typing import Dict, Optional

# API Keys - Set these as environment variables for security
DEFAULT_API_KEYS = {
    "scopus_api_key": None,  # Set SCOPUS_API_KEY environment variable
    "wos_api_key": None,     # Set WOS_API_KEY environment variable
    "orcid_client_id": None, # Set ORCID_CLIENT_ID environment variable
    "orcid_client_secret": None # Set ORCID_CLIENT_SECRET environment variable
}

def get_api_keys() -> Dict[str, Optional[str]]:
    """
    Get API keys from environment variables or defaults.
    
    Returns
    -------
    Dict[str, Optional[str]]
        Dictionary containing API keys
    """
    return {
        "scopus_api_key": os.environ.get("SCOPUS_API_KEY", DEFAULT_API_KEYS["scopus_api_key"]),
        "wos_api_key": os.environ.get("WOS_API_KEY", DEFAULT_API_KEYS["wos_api_key"]),
        "orcid_client_id": os.environ.get("ORCID_CLIENT_ID", DEFAULT_API_KEYS["orcid_client_id"]),
        "orcid_client_secret": os.environ.get("ORCID_CLIENT_SECRET", DEFAULT_API_KEYS["orcid_client_secret"])
    }

# Default search settings
DEFAULT_SEARCH_SETTINGS = {
    "max_pubs_g_scholar": 2000,
    "headless_g_scholar": True,
    "analyze_coverage": True
}

# Web interface settings
WEB_SETTINGS = {
    "default_host": "127.0.0.1",
    "default_port": 5000,
    "debug_mode": False
}

# Error handling settings
ERROR_HANDLING = {
    "retry_attempts": 3,
    "timeout_seconds": 30,
    "graceful_degradation": True
}
