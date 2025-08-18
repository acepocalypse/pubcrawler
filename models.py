"""pubcrawler.models
~~~~~~~~~~~~~~~~~~~
This module defines the canonical data structures used throughout the pubcrawler pipeline.
Every harvesting module (e.g., from Google Scholar, Scopus, Web of Science) must
return data that conforms to these schemas. This ensures consistency and allows
for seamless aggregation and deduplication.

The core components are:

▸ Publication (dataclass)
    The canonical schema for a single publication record. It includes essential
    bibliographic metadata like title, authors, year, DOI, etc.

▸ Author (dataclass)
    Represents an author, primarily used for search queries and metadata.

By standardizing on these structures, the pipeline can treat different data
sources interchangeably, simplifying the overall workflow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Publication:
    """
    Canonical schema for a single publication.

    All harvesting modules MUST return a list of these objects.
    The fields are designed to be a common denominator across major
    bibliographic databases.
    
    Attributes
    ----------
    title : str
        The publication title (original casing preserved).
    authors : List[str]
        List of author names.
    journal : Optional[str]
        Journal or venue name.
    year : Optional[int]
        Publication year.
    doi : Optional[str]
        Digital Object Identifier (automatically normalized).
    issn : Optional[str]
        International Standard Serial Number.
    source : str
        Source database name (e.g., "Google Scholar", "Scopus", "WoS").
    citations : Optional[int]
        Number of citations (defaults to 0).
    url : Optional[str]
        Direct URL to the publication.
    """
    title: str
    authors: List[str]
    journal: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    issn: Optional[str] = None
    source: str = ""
    citations: Optional[int] = 0
    url: Optional[str] = None

    def __post_init__(self) -> None:
        """Normalize DOI after initialization."""
        # Sanitize DOI to be lowercase and remove common prefixes/suffixes
        if self.doi:
            self.doi = self.doi.lower().strip()
            if self.doi.startswith("http"):
                # Extract DOI from URL
                self.doi = self.doi.split("doi.org/")[-1]
            # Remove common file extensions
            if self.doi.endswith(('.pdf', '.html', '.xml')):
                self.doi = '.'.join(self.doi.split('.')[:-1])
        
        # Only strip whitespace from title, preserve original case and content
        if self.title:
            self.title = self.title.strip()
        
        # Ensure citations is at least 0
        if self.citations is None:
            self.citations = 0


@dataclass
class Author:
    """
    Represents an author for searching and metadata.
    
    Attributes
    ----------
    first_name : str
        Author's first name.
    last_name : str
        Author's last name.
    affiliation : Optional[str]
        Institutional affiliation.
    gs_id : Optional[str]
        Google Scholar ID.
    scopus_id : Optional[str]
        Scopus Author ID.
    wos_id : Optional[str]
        Web of Science ResearcherID.
    orcid_id : Optional[str]
        ORCID ID.
    """
    first_name: str
    last_name: str
    affiliation: Optional[str] = None
    gs_id: Optional[str] = None      # Google Scholar ID
    scopus_id: Optional[str] = None  # Scopus Author ID
    wos_id: Optional[str] = None     # Web of Science ResearcherID
    orcid_id: Optional[str] = None   # ORCID ID
    
    @property
    def full_name(self) -> str:
        """Return the full name as 'First Last'."""
        return f"{self.first_name} {self.last_name}"
    
    @property
    def last_first_name(self) -> str:
        """Return the name as 'Last, First' (common in academic searches)."""
        return f"{self.last_name}, {self.first_name}"
