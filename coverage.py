"""pubcrawler.coverage
~~~~~~~~~~~~~~~~~~~~
Index coverage analysis for academic publications.

This module provides functionality to analyze which databases index each publication
and identify coverage gaps. It helps researchers understand the visibility of their
work across different academic indexing services.

Key features:
- Track source coverage per publication
- Identify indexing gaps
- Generate coverage reports
- Provide recommendations for improving visibility
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
import pandas as pd

try:
    from .models import Publication
except ImportError:
    # Fallback for when running as script
    from models import Publication


class IndexingSource(Enum):
    """Enumeration of supported indexing sources."""
    GOOGLE_SCHOLAR = "Google Scholar"
    SCOPUS = "Scopus"
    WOS = "Web of Science"
    ORCID = "ORCID"


@dataclass
class CoverageFlags:
    """Coverage flags for a single publication."""
    
    # Which sources index this publication
    indexed_in: Set[IndexingSource] = field(default_factory=set)
    
    # Missing from these sources
    missing_from: Set[IndexingSource] = field(default_factory=set)
    
    # Coverage score (0.0 to 1.0)
    coverage_score: float = 0.0
    
    # Coverage category
    coverage_category: str = "unknown"
    
    @property
    def is_complete_coverage(self) -> bool:
        """True if indexed in all supported sources."""
        return len(self.indexed_in) == len(IndexingSource)
    
    @property
    def is_partial_coverage(self) -> bool:
        """True if indexed in some but not all sources."""
        return 0 < len(self.indexed_in) < len(IndexingSource)
    
    @property
    def is_missing_coverage(self) -> bool:
        """True if not found in any sources."""
        return len(self.indexed_in) == 0
    
    @property
    def indexed_sources_str(self) -> str:
        """Human-readable string of indexed sources."""
        if not self.indexed_in:
            return "None"
        return ", ".join(source.value for source in self.indexed_in)
    
    @property
    def missing_sources_str(self) -> str:
        """Human-readable string of missing sources."""
        if not self.missing_from:
            return "None"
        return ", ".join(source.value for source in self.missing_from)


@dataclass
class EnhancedPublication(Publication):
    """Publication with coverage analysis."""
    
    # Coverage information
    coverage: CoverageFlags = field(default_factory=CoverageFlags)
    
    # Source-specific metadata
    source_metadata: Dict[IndexingSource, Dict] = field(default_factory=dict)
    
    # Matching information (for deduplication tracking)
    matched_publications: List[Publication] = field(default_factory=list)
    canonical_source: Optional[IndexingSource] = None


class CoverageAnalyzer:
    """Analyzes index coverage across academic databases."""
    
    def __init__(self, available_sources: Optional[List[IndexingSource]] = None):
        """
        Initialize coverage analyzer.
        
        Parameters
        ----------
        available_sources : Optional[List[IndexingSource]]
            List of sources that were actually queried. If None, assumes all sources.
        """
        self.available_sources = available_sources or list(IndexingSource)
        
        # Map both enum values and actual source names used in publications
        self.source_map = {
            # Enum values (from available_sources in app.py)
            "Google Scholar": IndexingSource.GOOGLE_SCHOLAR,
            "Scopus": IndexingSource.SCOPUS,
            "Web of Science": IndexingSource.WOS,
            "ORCID": IndexingSource.ORCID,
            
            # Actual source names used in publications
            "Google Scholar": IndexingSource.GOOGLE_SCHOLAR,  # Same as enum
            "Scopus": IndexingSource.SCOPUS,  # Same as enum
            "WoS": IndexingSource.WOS,  # Different from enum!
            "ORCID": IndexingSource.ORCID,  # Same as enum
            
            # Handle variations and case issues
            "google scholar": IndexingSource.GOOGLE_SCHOLAR,
            "scopus": IndexingSource.SCOPUS,
            "wos": IndexingSource.WOS,
            "web of science": IndexingSource.WOS,
            "orcid": IndexingSource.ORCID,
        }
    
    def analyze_coverage(self, publications: List[Publication]) -> List[EnhancedPublication]:
        """
        Analyze coverage for a list of publications.
        
        Parameters
        ----------
        publications : List[Publication]
            Publications from aggregation pipeline.
            
        Returns
        -------
        List[EnhancedPublication]
            Publications with coverage analysis.
        """
        # Group publications by identity (DOI or title+year)
        publication_groups = self._group_publications_by_identity(publications)
        
        enhanced_publications = []
        
        for group_key, group_pubs in publication_groups.items():
            enhanced_pub = self._create_enhanced_publication(group_pubs)
            enhanced_publications.append(enhanced_pub)
        
        return enhanced_publications
    
    def _group_publications_by_identity(self, publications: List[Publication]) -> Dict[str, List[Publication]]:
        """Group publications by their identity (DOI or normalized title+year)."""
        groups = {}
        
        for pub in publications:
            # Primary key: DOI (if available)
            if pub.doi:
                key = f"doi:{pub.doi.lower().strip()}"
            else:
                # Fallback: normalized title + year
                title_key = pub.title.lower().strip() if pub.title else "unknown"
                year_key = str(pub.year) if pub.year else "unknown"
                key = f"title_year:{title_key}:{year_key}"
            
            if key not in groups:
                groups[key] = []
            groups[key].append(pub)
        
        return groups
    
    def _create_enhanced_publication(self, group_publications: List[Publication]) -> EnhancedPublication:
        """Create an enhanced publication from a group of duplicate publications."""
        # Choose the canonical publication (highest citations, or first if tied)
        canonical = max(group_publications, key=lambda p: (p.citations or 0, p.year or 0))
        
        # Determine which sources this publication appears in
        indexed_sources = set()
        source_metadata = {}
        
        for pub in group_publications:
            # Handle merged sources (e.g., "Multiple sources: Google Scholar (45 cites), Scopus (52 cites)")
            if pub.source and 'multiple sources:' in pub.source.lower():
                # Parse merged source string
                source_part = pub.source.split(':', 1)[1].strip() if ':' in pub.source else pub.source
                
                # Extract individual sources with case-insensitive matching
                import re
                
                # Match patterns like "Google Scholar (45 cites)" or "Google Scholar"
                pattern = r'([\w\s]+?)(?:\s*\([\d\s]*cites?\))?(?:,|$)'
                matches = re.findall(pattern, source_part, re.IGNORECASE)
                
                for match in matches:
                    source_name = match.strip()
                    source_lower = source_name.lower()
                    
                    # Map to enum using case-insensitive lookup
                    mapped_source = None
                    for key, enum_val in self.source_map.items():
                        if key.lower() == source_lower:
                            mapped_source = enum_val
                            break
                    
                    if mapped_source:
                        indexed_sources.add(mapped_source)
                        if mapped_source not in source_metadata:
                            source_metadata[mapped_source] = {
                                'citations': pub.citations,
                                'url': pub.url,
                                'journal': pub.journal,
                                'authors': pub.authors,
                                'year': pub.year,
                            }
            else:
                # Handle single source - use case-insensitive matching
                pub_source = pub.source.strip() if pub.source else ""
                mapped_source = None
                
                # Try exact match first, then case-insensitive
                if pub_source in self.source_map:
                    mapped_source = self.source_map[pub_source]
                else:
                    pub_source_lower = pub_source.lower()
                    for key, enum_val in self.source_map.items():
                        if key.lower() == pub_source_lower:
                            mapped_source = enum_val
                            break
                
                if mapped_source:
                    indexed_sources.add(mapped_source)
                    source_metadata[mapped_source] = {
                        'citations': pub.citations,
                        'url': pub.url,
                        'journal': pub.journal,
                        'authors': pub.authors,
                        'year': pub.year,
                    }
        
        # Convert available_sources (which might be strings) to enums
        available_source_enums = set()
        if self.available_sources:
            for src in self.available_sources:
                if isinstance(src, IndexingSource):
                    available_source_enums.add(src)
                elif isinstance(src, str):
                    # Map string to enum using case-insensitive lookup
                    src_lower = src.lower()
                    for key, enum_val in self.source_map.items():
                        if key.lower() == src_lower:
                            available_source_enums.add(enum_val)
                            break
        else:
            available_source_enums = set(IndexingSource)
        
        # Determine missing sources
        missing_sources = available_source_enums - indexed_sources
        
        # Calculate coverage score
        coverage_score = len(indexed_sources) / len(available_source_enums) if available_source_enums else 0.0
        
        # Determine coverage category
        coverage_category = self._get_coverage_category(coverage_score, len(indexed_sources))
        
        # Create coverage flags
        coverage = CoverageFlags(
            indexed_in=indexed_sources,
            missing_from=missing_sources,
            coverage_score=coverage_score,
            coverage_category=coverage_category
        )
        
        # Create enhanced publication
        enhanced_pub = EnhancedPublication(
            title=canonical.title,
            authors=canonical.authors,
            journal=canonical.journal,
            year=canonical.year,
            doi=canonical.doi,
            issn=canonical.issn,
            source=canonical.source,
            citations=canonical.citations,
            url=canonical.url,
            coverage=coverage,
            source_metadata=source_metadata,
            matched_publications=group_publications,
            canonical_source=self.source_map.get(canonical.source)
        )
        
        return enhanced_pub
    
    def _get_coverage_category(self, score: float, indexed_count: int) -> str:
        """Determine coverage category based on score and count."""
        if score == 1.0:
            return "Complete Coverage"
        elif score >= 0.67:
            return "Good Coverage"
        elif score >= 0.33:
            return "Partial Coverage"
        elif indexed_count > 0:
            return "Limited Coverage"
        else:
            return "No Coverage"


class CoverageReporter:
    """Generates coverage reports and recommendations."""
    
    def __init__(self):
        self.analyzer = CoverageAnalyzer()
    
    def generate_coverage_report(self, enhanced_publications: List[EnhancedPublication]) -> Dict:
        """
        Generate a comprehensive coverage report.
        
        Parameters
        ----------
        enhanced_publications : List[EnhancedPublication]
            Publications with coverage analysis.
            
        Returns
        -------
        Dict
            Comprehensive coverage report.
        """
        total_pubs = len(enhanced_publications)
        
        if total_pubs == 0:
            return {"error": "No publications to analyze"}
        
        # Overall statistics
        coverage_scores = [pub.coverage.coverage_score for pub in enhanced_publications]
        avg_coverage = sum(coverage_scores) / len(coverage_scores)
        
        # Coverage category distribution
        category_counts = {}
        for pub in enhanced_publications:
            category = pub.coverage.coverage_category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Source-specific statistics
        source_stats = {}
        for source in IndexingSource:
            indexed_count = sum(1 for pub in enhanced_publications 
                              if source in pub.coverage.indexed_in)
            missing_count = sum(1 for pub in enhanced_publications 
                               if source in pub.coverage.missing_from)
            
            source_stats[source.value] = {
                "indexed_count": indexed_count,
                "missing_count": missing_count,
                "coverage_rate": indexed_count / total_pubs if total_pubs > 0 else 0.0
            }
        
        # Gap analysis
        gaps = self._analyze_gaps(enhanced_publications)
        
        # Recommendations
        recommendations = self._generate_recommendations(enhanced_publications, source_stats)
        
        return {
            "summary": {
                "total_publications": total_pubs,
                "average_coverage": avg_coverage,
                "complete_coverage_count": category_counts.get("Complete Coverage", 0),
                "no_coverage_count": category_counts.get("No Coverage", 0)
            },
            "coverage_distribution": category_counts,
            "source_statistics": source_stats,
            "gap_analysis": gaps,
            "recommendations": recommendations,
            "detailed_publications": self._get_detailed_publication_list(enhanced_publications)
        }
    
    def _analyze_gaps(self, enhanced_publications: List[EnhancedPublication]) -> Dict:
        """Analyze coverage gaps across sources."""
        gaps = {
            "missing_from_google_scholar": [],
            "missing_from_scopus": [],
            "missing_from_wos": [],
            "single_source_only": [],
            "high_impact_missing": []  # High citation papers missing from sources
        }
        
        for pub in enhanced_publications:
            title_short = pub.title[:60] + "..." if len(pub.title) > 60 else pub.title
            pub_info = {
                "title": title_short,
                "year": pub.year,
                "citations": pub.citations,
                "doi": pub.doi,
                "indexed_in": pub.coverage.indexed_sources_str
            }
            
            # Check specific source gaps
            if IndexingSource.GOOGLE_SCHOLAR in pub.coverage.missing_from:
                gaps["missing_from_google_scholar"].append(pub_info)
            
            if IndexingSource.SCOPUS in pub.coverage.missing_from:
                gaps["missing_from_scopus"].append(pub_info)
            
            if IndexingSource.WOS in pub.coverage.missing_from:
                gaps["missing_from_wos"].append(pub_info)
            
            # Single source publications
            if len(pub.coverage.indexed_in) == 1:
                gaps["single_source_only"].append(pub_info)
            
            # High impact papers with gaps
            if (pub.citations or 0) >= 10 and not pub.coverage.is_complete_coverage:
                gaps["high_impact_missing"].append(pub_info)
        
        return gaps
    
    def _generate_recommendations(self, enhanced_publications: List[EnhancedPublication], 
                                 source_stats: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Overall coverage recommendations
        total_pubs = len(enhanced_publications)
        complete_coverage = sum(1 for pub in enhanced_publications 
                               if pub.coverage.is_complete_coverage)
        
        if complete_coverage / total_pubs < 0.5:
            recommendations.append(
                "🔍 Less than 50% of publications have complete coverage. "
                "Consider submitting missing publications to underrepresented databases."
            )
        
        # Source-specific recommendations
        for source_name, stats in source_stats.items():
            if stats["coverage_rate"] < 0.7:
                recommendations.append(
                    f"📈 {source_name} coverage is low ({stats['coverage_rate']:.1%}). "
                    f"Consider submitting {stats['missing_count']} missing publications."
                )
        
        # High-impact publication recommendations
        high_impact_gaps = [pub for pub in enhanced_publications 
                           if (pub.citations or 0) >= 10 and not pub.coverage.is_complete_coverage]
        
        if high_impact_gaps:
            recommendations.append(
                f"⭐ {len(high_impact_gaps)} high-impact publications have coverage gaps. "
                "Prioritize these for maximum visibility improvement."
            )
        
        # DOI recommendations
        no_doi_count = sum(1 for pub in enhanced_publications if not pub.doi)
        if no_doi_count > 0:
            recommendations.append(
                f"🔗 {no_doi_count} publications lack DOIs. "
                "DOIs improve discoverability and cross-database matching."
            )
        
        return recommendations
    
    def _get_detailed_publication_list(self, enhanced_publications: List[EnhancedPublication]) -> List[Dict]:
        """Get detailed publication list for the report."""
        detailed_list = []
        
        for pub in enhanced_publications:
            detailed_list.append({
                "title": pub.title[:80] + "..." if len(pub.title) > 80 else pub.title,
                "year": pub.year,
                "citations": pub.citations,
                "doi": pub.doi,
                "coverage_score": pub.coverage.coverage_score,
                "coverage_category": pub.coverage.coverage_category,
                "indexed_in": pub.coverage.indexed_sources_str,
                "missing_from": pub.coverage.missing_sources_str,
                "source_count": len(pub.coverage.indexed_in)
            })
        
        # Sort by coverage score (ascending) then citations (descending)
        detailed_list.sort(key=lambda x: (x["coverage_score"], -(x["citations"] or 0)))
        
        return detailed_list
    
    def print_coverage_summary(self, report: Dict) -> None:
        """Print a formatted coverage summary to console."""
        print("\n" + "="*80)
        print("📊 PUBLICATION INDEX COVERAGE ANALYSIS")
        print("="*80)
        
        summary = report["summary"]
        print(f"\n📈 Overview:")
        print(f"  • Total Publications: {summary['total_publications']}")
        print(f"  • Average Coverage: {summary['average_coverage']:.1%}")
        print(f"  • Complete Coverage: {summary['complete_coverage_count']} publications")
        print(f"  • No Coverage: {summary['no_coverage_count']} publications")
        
        print(f"\n🎯 Coverage Distribution:")
        for category, count in report["coverage_distribution"].items():
            percentage = count / summary['total_publications'] * 100
            print(f"  • {category}: {count} ({percentage:.1f}%)")
        
        print(f"\n🗃️ Source Statistics:")
        for source, stats in report["source_statistics"].items():
            print(f"  • {source}:")
            print(f"    - Indexed: {stats['indexed_count']} ({stats['coverage_rate']:.1%})")
            print(f"    - Missing: {stats['missing_count']}")
        
        print(f"\n💡 Recommendations:")
        for i, recommendation in enumerate(report["recommendations"], 1):
            print(f"  {i}. {recommendation}")
        
        # Show top gaps
        gaps = report["gap_analysis"]
        if gaps["high_impact_missing"]:
            print(f"\n⚠️ High-Impact Publications with Coverage Gaps:")
            for pub in gaps["high_impact_missing"][:3]:  # Top 3
                print(f"  • ({pub['year']}) [{pub['citations']} cites] {pub['title']}")
                print(f"    Missing from: {set(s.value for s in IndexingSource) - set(pub['indexed_in'].split(', '))}")
        
        print("\n" + "="*80)


def analyze_publication_coverage(publications: List[Publication], 
                                available_sources: Optional[List[str]] = None) -> Dict:
    """
    Main function to analyze publication coverage across databases.
    
    Parameters
    ----------
    publications : List[Publication]
        Publications from the aggregation pipeline.
    available_sources : Optional[List[str]]
        List of source names that were actually queried.
        
    Returns
    -------
    Dict
        Comprehensive coverage analysis report.
    """
    # Create a temporary analyzer to get the improved source mapping
    temp_analyzer = CoverageAnalyzer()
    
    # Convert source names to enums using improved mapping
    source_enums = []
    if available_sources:
        for source_name in available_sources:
            # Try case-insensitive lookup
            mapped_source = None
            source_name_lower = source_name.lower()
            for key, enum_val in temp_analyzer.source_map.items():
                if key.lower() == source_name_lower:
                    mapped_source = enum_val
                    break
            
            if mapped_source and mapped_source not in source_enums:
                source_enums.append(mapped_source)
    
    # Analyze coverage
    analyzer = CoverageAnalyzer(source_enums if source_enums else None)
    enhanced_publications = analyzer.analyze_coverage(publications)
    
    # Generate report
    reporter = CoverageReporter()
    report = reporter.generate_coverage_report(enhanced_publications)
    
    return report


def print_coverage_report(publications: List[Publication], 
                         available_sources: Optional[List[str]] = None) -> None:
    """
    Analyze and print coverage report to console.
    
    Parameters
    ----------
    publications : List[Publication]
        Publications from the aggregation pipeline.
    available_sources : Optional[List[str]]
        List of source names that were actually queried.
    """
    report = analyze_publication_coverage(publications, available_sources)
    
    reporter = CoverageReporter()
    reporter.print_coverage_summary(report)
