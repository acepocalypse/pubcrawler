"""PubCrawler Sources

Individual source adapters for harvesting academic publications.
Each source adapter provides a fetch() function that returns List[Publication].
"""

from . import google_scholar
from . import scopus
from . import wos

__all__ = ["google_scholar", "scopus", "wos"]
