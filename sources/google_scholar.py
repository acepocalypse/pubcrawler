"""pubcrawler.sources.google_scholar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Google Scholar harvesting module for the **pubcrawler** pipeline.

▸ Public entry‑point
    fetch(gs_id, first_name, last_name, **options) -> list[Publication]
    ----------------------------------------------------------------------------------
    • Wraps Google Scholar profile and publication scraping.
    • Uses Selenium with undetected Chrome driver for web scraping.
    • Handles anti-bot detection and CAPTCHA avoidance.
    • Converts the scraped data into the pipeline's *canonical* schema
      (see models.Publication).
    • Returns `List[Publication]` ready for aggregation.
"""

from __future__ import annotations

import re
import time
import random
import logging
from typing import Optional, Dict, List, Union
from pathlib import Path
from datetime import datetime
from collections import deque

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, StaleElementReferenceException
)
from webdriver_manager.chrome import ChromeDriverManager

import undetected_chromedriver as uc
from fake_useragent import UserAgent

# Simple fix to ignore Windows handle errors in undetected_chromedriver
def _patched_quit(self):
    """Patched quit method that ignores Windows handle errors"""
    try:
        self._original_quit()
    except OSError as e:
        if "WinError 6" in str(e) or "The handle is invalid" in str(e):
            pass  # Ignore Windows handle errors
        else:
            raise

# Apply the patch
if hasattr(uc.Chrome, 'quit') and not hasattr(uc.Chrome, '_original_quit'):
    uc.Chrome._original_quit = uc.Chrome.quit
    uc.Chrome.quit = _patched_quit

# ----- pubcrawler core ------------------------------------------------------
try:
    from ..models import Publication
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models import Publication

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

__all__ = ["fetch"]

# ═══════════════════════════════════════════════════════════════════════════
# Core Classes
# ═══════════════════════════════════════════════════════════════════════════

class ScrapingConfig:
    """Configuration class for pipeline integration"""
    
    def __init__(self, **kwargs):
        self.headless = kwargs.get('headless', False)
        self.turbo_mode = kwargs.get('turbo_mode', True)
        self.max_publications_detail = kwargs.get('max_publications_detail', 500)
        
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class ScrapingResult:
    """Standardized result class for pipeline integration"""
    
    def __init__(self, success: bool = False, data: pd.DataFrame = None, 
                 metadata: Dict = None, error: str = None):
        self.success = success
        self.data = data if data is not None else pd.DataFrame()
        self.metadata = metadata if metadata is not None else {}
        self.error = error
        self.timestamp = datetime.now()

class AdaptiveHumanBehaviorSimulator:
    """Optimized human behavior simulation with adaptive timing"""
    
    def __init__(self):
        self.last_action_time = time.time()
        self.session_start = time.time()
        self.consecutive_actions = 0
        self.last_break_time = time.time()
        
    def assess_risk_level(self) -> str:
        """Assess current risk level based on activity patterns"""
        session_duration = time.time() - self.session_start
        time_since_break = time.time() - self.last_break_time
        
        if session_duration < 120 or time_since_break < 60:
            return "low"
        elif session_duration > 600 or self.consecutive_actions > 30:
            return "high"
        else:
            return "medium"
    
    def human_delay(self, min_sec: float = 0.3, max_sec: float = 1.5):
        """Optimized adaptive delay based on risk assessment"""
        risk_level = self.assess_risk_level()
        
        if risk_level == "low":
            min_sec *= 0.5
            max_sec *= 0.6
        elif risk_level == "high":
            min_sec *= 1.5
            max_sec *= 2.0
        
        min_sec = max(0.2, min_sec)
        max_sec = max(0.5, max_sec)
        
        delay = random.uniform(min_sec, max_sec)
        if random.random() < 0.05:
            delay += random.uniform(0.1, 0.2)
        
        time.sleep(delay)
        self.last_action_time = time.time()
        self.consecutive_actions += 1
        
    def should_take_break(self) -> bool:
        """Optimized break logic"""
        session_duration = time.time() - self.session_start
        time_since_break = time.time() - self.last_break_time
        
        if session_duration > 900 and time_since_break > 300:
            return random.random() < 0.2
        return random.random() < 0.01
        
    def take_break(self):
        """Take a short break"""
        break_duration = random.uniform(3, 10)
        logger.info(f"Taking a quick break for {break_duration:.1f} seconds...")
        time.sleep(break_duration)
        self.last_break_time = time.time()
        self.consecutive_actions = 0

class OptimizedCaptchaHandler:
    """Optimized CAPTCHA detection with caching"""
    
    def __init__(self, driver, behavior_sim: AdaptiveHumanBehaviorSimulator):
        self.driver = driver
        self.behavior_sim = behavior_sim
        self.captcha_encounter_count = 0
        self.last_check_url = None
        self.last_check_result = None
        
    def detect_captcha(self) -> bool:
        """Optimized CAPTCHA detection with caching"""
        try:
            current_url = self.driver.current_url
            
            if current_url == self.last_check_url:
                return self.last_check_result
                
            patterns = ['recaptcha', '/sorry/', 'captcha', 'unusual traffic', 'verification']
            
            if any(pattern in current_url.lower() for pattern in patterns):
                self.last_check_url = current_url
                self.last_check_result = True
                return True
                
            if any(pattern in self.driver.title.lower() for pattern in patterns):
                self.last_check_url = current_url
                self.last_check_result = True
                return True
                
            self.last_check_url = current_url
            self.last_check_result = False
            return False
            
        except:
            return False
            
    def avoid_captcha_strategy(self) -> bool:
        """Optimized CAPTCHA avoidance"""
        logger.info("Implementing quick CAPTCHA avoidance...")
        
        delay = min(120, 30 * (1.5 ** self.captcha_encounter_count))
        logger.info(f"Waiting {delay} seconds before retry...")
        time.sleep(delay)
        
        if self.captcha_encounter_count > 1:
            self.driver.delete_all_cookies()
            
        return True

class OptimizedStealthDriver:
    def __init__(self):
        self.behavior_sim = AdaptiveHumanBehaviorSimulator()
        self.ua = UserAgent()

    def create_driver(self, headless: bool = False) -> uc.Chrome:
        """Create optimized undetected Chrome driver"""
        options = uc.ChromeOptions()

        # Anti-detection
        user_agent = self.ua.random
        options.add_argument(f'--user-agent={user_agent}')
        options.add_argument('--disable-blink-features=AutomationControlled')

        # Performance flags
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-web-security')

        # Preferences
        prefs = {
            'profile.default_content_setting_values.notifications': 2,
            'profile.default_content_settings.popups': 0,
            'profile.managed_default_content_settings.images': 1,
            'credentials_enable_service': False,
            'profile.password_manager_enabled': False,
        }
        options.add_experimental_option('prefs', prefs)

        if headless:
            options.add_argument('--headless=new')

        try:
            driver = uc.Chrome(options=options, version_main=None)
            self._inject_stealth_js(driver)
            driver.set_page_load_timeout(20)
            driver.implicitly_wait(3)
            logger.info("Successfully created optimized Chrome driver")
            return driver
        except Exception as e:
            logger.error(f"Failed to create undetected-chromedriver: {e}")
            return self._create_fallback_driver(headless)

    def _create_fallback_driver(self, headless: bool = False):
        """Create fallback driver using regular Selenium"""
        logger.info("Creating fallback driver with regular Selenium...")
        options = Options()

        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option('useAutomationExtension', False)
        options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        options.add_argument('--window-size=1920,1080')
        options.page_load_strategy = 'eager'

        if headless:
            options.add_argument('--headless=new')

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        self._inject_stealth_js(driver)
        driver.set_page_load_timeout(20)
        driver.implicitly_wait(3)
        return driver
            
    def _inject_stealth_js(self, driver):
        """Inject essential stealth JavaScript"""
        stealth_js = """
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        window.chrome = {runtime: {}};
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3].map(n => ({
                name: `Plugin ${n}`,
                filename: `plugin${n}.dll`,
                length: 1
            }))
        });
        """
        
        try:
            driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': stealth_js})
        except:
            try:
                driver.execute_script(stealth_js)
            except:
                pass

class OptimizedScholarScraper:
    """Optimized scraper with batching"""
    
    def __init__(self):
        self.stealth_driver = OptimizedStealthDriver()
        self.behavior_sim = AdaptiveHumanBehaviorSimulator()
        self.driver = None
        self.captcha_handler = None
        self._cleanup_called = False
        
    def initialize_driver(self, headless: bool = False):
        """Initialize driver"""
        self.driver = self.stealth_driver.create_driver(headless)
        self.captcha_handler = OptimizedCaptchaHandler(self.driver, self.behavior_sim)
        self._warm_up()
        return self.driver
        
    def _warm_up(self):
        """Minimal browser warm-up"""
        logger.info("Quick browser warm-up...")
        try:
            self.driver.get("https://scholar.google.com")
            self.behavior_sim.human_delay(1, 2)
            
            # Handle cookies if present
            try:
                accept_btn = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Accept')]")
                if accept_btn.is_displayed():
                    accept_btn.click()
            except:
                pass
        except Exception as e:
            logger.debug(f"Warm-up error (non-critical): {e}")
            
    def navigate_with_retry(self, url: str, max_attempts: int = 2) -> bool:
        """Navigate with retry logic"""
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    self.behavior_sim.human_delay(2, 4)
                    
                self.driver.get(url)
                
                WebDriverWait(self.driver, 10).until(
                    lambda d: d.execute_script("return document.readyState") == "complete"
                )
                
                if self.captcha_handler.detect_captcha():
                    logger.warning("CAPTCHA detected!")
                    self.captcha_handler.captcha_encounter_count += 1
                    
                    if not self.captcha_handler.avoid_captcha_strategy():
                        return False
                    continue
                    
                return True
                
            except TimeoutException:
                if attempt == max_attempts - 1:
                    return False
                continue
                
        return False
        
    def scrape_publications(self, author_url: str) -> pd.DataFrame:
        """Scrape all publications from author page"""
        if not self.navigate_with_retry(author_url):
            logger.error("Failed to navigate to author page")
            return pd.DataFrame()
            
        # Wait for profile page
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#gsc_prf_in"))
            )
        except TimeoutException:
            logger.error("Profile page did not load")
            return pd.DataFrame()
            
        self._load_all_publications()
        return self._extract_publication_data()
        
    def _load_all_publications(self):
        """Load all publications by clicking 'Show more' button"""
        loaded_count = 0
        consecutive_failures = 0
        
        while consecutive_failures < 3:
            try:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                self.behavior_sim.human_delay(0.3, 0.8)
                
                show_more_btn = self.driver.find_element(By.ID, "gsc_bpf_more")
                
                if not show_more_btn.is_displayed() or not show_more_btn.is_enabled():
                    break
                    
                self.driver.execute_script("arguments[0].click();", show_more_btn)
                
                loaded_count += 1
                consecutive_failures = 0
                
                if loaded_count % 5 == 0:
                    logger.info(f"Loaded {loaded_count} batches...")
                    
                # Adaptive delay based on risk
                risk = self.behavior_sim.assess_risk_level()
                if risk == "low":
                    self.behavior_sim.human_delay(0.5, 1.0)
                elif risk == "high":
                    self.behavior_sim.human_delay(1.5, 2.5)
                    if self.captcha_handler.detect_captcha():
                        logger.warning("CAPTCHA during loading")
                        break
                else:
                    self.behavior_sim.human_delay(0.8, 1.5)
                    
            except (NoSuchElementException, StaleElementReferenceException):
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    break
                self.behavior_sim.human_delay(0.5, 1.0)
            except Exception as e:
                logger.debug(f"Error loading more: {e}")
                break
                
        logger.info(f"Finished loading {loaded_count} batches")
        
    def _extract_publication_data(self) -> pd.DataFrame:
        """Extract all publication data using JavaScript"""
        rows = self.driver.find_elements(By.CSS_SELECTOR, "tr.gsc_a_tr")
        logger.info(f"Extracting {len(rows)} publications...")
        
        extraction_script = """
        return Array.from(document.querySelectorAll('tr.gsc_a_tr')).map(row => {
            const titleElem = row.querySelector('a.gsc_a_at');
            const grayDivs = row.querySelectorAll('div.gs_gray');
            const citationsElem = row.querySelector('a.gsc_a_ac');
            const yearElem = row.querySelector('span.gsc_a_h, span.gsc_a_y');
            
            return {
                title: titleElem ? titleElem.textContent : '',
                puburl: titleElem ? titleElem.href : '',
                authors: grayDivs[0] ? grayDivs[0].textContent : '',
                journal: grayDivs[1] ? grayDivs[1].textContent : '',
                citations: citationsElem ? citationsElem.textContent : '0',
                year: yearElem ? yearElem.textContent : ''
            };
        });
        """
        
        try:
            records = self.driver.execute_script(extraction_script)
            
            # Add citations_numeric for all records
            for record in records:
                citations_text = record.get('citations', '0')
                record['citations_numeric'] = int(citations_text) if citations_text.isdigit() else 0
                
            df = pd.DataFrame(records)
            
            if df.empty:
                df = pd.DataFrame(columns=['title', 'puburl', 'authors', 'journal', 'citations', 'year', 'citations_numeric'])
                
            logger.info(f"Successfully extracted {len(records)} publications")
            return df
            
        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")
            return pd.DataFrame()
                    
    def cleanup(self):
        """Clean up resources"""
        if self._cleanup_called:
            return
            
        self._cleanup_called = True
        
        if self.driver:
            try:
                # Close windows and quit
                try:
                    for handle in self.driver.window_handles:
                        self.driver.switch_to.window(handle)
                        self.driver.close()
                except:
                    pass
                
                try:
                    self.driver.service.process.terminate()
                except:
                    pass
                
                self.driver.quit()
                
            except Exception as e:
                logger.debug(f"Error during cleanup: {e}")
            finally:
                self.driver = None

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass

# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def extract_doi_from_links(links: List[str]) -> Optional[str]:
    """Extract DOI from a list of links"""
    if not links:
        return None
        
    doi_pattern = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"
    
    for link in links:
        if isinstance(link, str):
            match = re.search(doi_pattern, link, re.IGNORECASE)
            if match:
                doi = match.group(0).lower()
                doi = re.sub(r'\.(pdf|html|xml)$', '', doi)
                return doi.strip('.')
                
    return None

def _standardize_gs_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Google Scholar DataFrame columns"""
    if df.empty:
        return pd.DataFrame(columns=[
            "title", "authors", "journal", "year", "doi", "issn", "source", 
            "citations", "url", "volume", "issue", "page_start", "page_end", 
            "page_count", "publisher", "links", "gs_id", "work_type"
        ])
    
    # Ensure citations_numeric exists
    if 'citations_numeric' not in df.columns:
        if 'citations' in df.columns:
            df['citations_numeric'] = pd.to_numeric(df['citations'], errors='coerce').fillna(0).astype(int)
        else:
            df['citations_numeric'] = 0
    
    # Map to standard columns
    standardized = pd.DataFrame()
    standardized["title"] = df.get("title", "").astype(str).str.strip().str.lower()
    standardized["authors"] = df.get("authors", "").apply(
        lambda x: [name.strip() for name in str(x).split(',') if name.strip()] 
        if pd.notna(x) and x != '' else []
    )
    standardized["journal"] = df.get("journal", None)
    standardized["year"] = df.get("year", None)
    standardized["doi"] = df.get("doi", None)
    standardized["issn"] = None
    standardized["source"] = "Google Scholar"
    standardized["citations"] = df['citations_numeric']
    standardized["url"] = df.get("puburl", None)
    
    # Set remaining columns to None
    for col in ["volume", "issue", "page_start", "page_end", "page_count", "publisher", "links", "gs_id", "work_type"]:
        standardized[col] = df.get(col, None)
    
    return standardized

# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline Interface
# ═══════════════════════════════════════════════════════════════════════════

def fetch(
    gs_id: str,
    first_name: str,
    last_name: str,
    *,
    max_publications_detail: int = 100,
    headless: bool = False,
    turbo_mode: bool = True,
) -> List[Publication]:
    """
    Harvest publications for an author from Google Scholar.

    Parameters
    ----------
    gs_id : str
        Google Scholar author ID.
    first_name, last_name : str
        Author details for metadata.
    max_publications_detail : int, default 100
        Maximum number of detailed records to fetch.
    headless : bool, default False
        Whether to run browser in headless mode.
    turbo_mode : bool, default True
        Enable speed optimizations.
    """
    author_info = {
        'gs_id': gs_id,
        'first_name': first_name,
        'last_name': last_name
    }
    
    config = ScrapingConfig(
        max_publications_detail=max_publications_detail,
        headless=headless,
        turbo_mode=turbo_mode
    )
    
    result = scrape_scholar_profile(author_info, config)
    
    if not result.success:
        logger.error(f"Google Scholar scraping failed: {result.error}")
        return []
    
    # Convert DataFrame to Publication objects
    pubs: List[Publication] = []
    for _, row in result.data.iterrows():
        # Parse authors list
        authors = []
        if isinstance(row.get('authors'), str):
            authors = [name.strip() for name in row['authors'].split(',') if name.strip()]
        elif isinstance(row.get('authors'), list):
            authors = row['authors']
        
        pubs.append(
            Publication(
                title=str(row.get('title', '')).strip().lower(),
                authors=authors,
                journal=str(row.get('journal', '')) if pd.notna(row.get('journal')) else None,
                year=int(row['year']) if pd.notna(row.get('year')) and str(row['year']).isdigit() else None,
                doi=str(row.get('doi', '')) if pd.notna(row.get('doi')) else None,
                issn=None,
                source="Google Scholar",
                citations=int(row.get('citations', 0)),
                url=str(row.get('url', '')) if pd.notna(row.get('url')) else None,
            )
        )
    
    return sorted(pubs, key=lambda p: (p.year or 0, p.citations or 0), reverse=True)

def scrape_scholar_profile(
    author_info: Union[Dict, str],
    config: ScrapingConfig = None
) -> ScrapingResult:
    """
    Pipeline-friendly Google Scholar scraping function
    
    Args:
        author_info: Dictionary with author details or GS ID string
        config: ScrapingConfig object
        
    Returns:
        ScrapingResult object with standardized output
    """
    start_time = time.time()
    
    # Parse author information
    if isinstance(author_info, str):
        gs_id = author_info
        first_name = "Unknown"
        last_name = "Author"
    elif isinstance(author_info, dict):
        gs_id = author_info.get('gs_id')
        first_name = author_info.get('first_name', 'Unknown')
        last_name = author_info.get('last_name', 'Author')
    else:
        return ScrapingResult(
            success=False, 
            error="Invalid author_info format. Expected string or dict."
        )
    
    if not gs_id:
        return ScrapingResult(success=False, error="Google Scholar ID is required")
    
    if config is None:
        config = ScrapingConfig()
    
    # Setup metadata
    metadata = {
        'gs_id': gs_id,
        'first_name': first_name,
        'last_name': last_name,
        'scraping_start': datetime.now().isoformat(),
        'config': config.to_dict()
    }
    
    author_url = f"https://scholar.google.com/citations?hl=en&user={gs_id}"
    
    logger.info(f"Starting scraping for {first_name} {last_name} (ID: {gs_id})")
    logger.info(f"Turbo mode: {'ON' if config.turbo_mode else 'OFF'}")
    
    scraper = OptimizedScholarScraper()
    
    try:
        scraper.initialize_driver(headless=config.headless)
        
        # Scrape publications
        publications_df = scraper.scrape_publications(author_url)
        
        if publications_df.empty:
            return ScrapingResult(
                success=False,
                metadata=metadata,
                error="No publications found for this author"
            )
        
        # Standardize columns
        publications_df = _standardize_gs_columns(publications_df)
        
        # Update metadata
        total_time = time.time() - start_time
        metadata.update({
            'publications_found': len(publications_df),
            'total_scraping_time': total_time,
            'scraping_end': datetime.now().isoformat(),
            'success': True
        })
        
        logger.info(f"✅ Scraping completed in {total_time:.1f} seconds")
        logger.info(f"📊 Found {len(publications_df)} publications")
        
        return ScrapingResult(
            success=True,
            data=publications_df,
            metadata=metadata
        )
        
    except Exception as e:
        error_msg = f"Scraping failed: {str(e)}"
        logger.error(error_msg)
        
        metadata.update({
            'error': error_msg,
            'total_scraping_time': time.time() - start_time,
            'scraping_end': datetime.now().isoformat(),
            'success': False
        })
        
        return ScrapingResult(success=False, metadata=metadata, error=error_msg)
        
    finally:
        try:
            scraper.cleanup()
            time.sleep(0.5)
        except Exception as cleanup_error:
            logger.debug(f"Cleanup error (non-critical): {cleanup_error}")
        finally:
            del scraper

# ═══════════════════════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        example_publications = fetch(
            gs_id='MTPuzvEAAAAJ',
            first_name='Morgan',
            last_name='Furze',
            max_publications_detail=500,
            headless=False,
            turbo_mode=True
        )
        
        print(f"\n✅ Fetched {len(example_publications)} publications from Google Scholar")
        if example_publications:
            print("--- Top 5 publications ---")
            for i, pub in enumerate(example_publications[:5]):
                print(f"{i+1}. ({pub.year}) [{pub.citations} cites] {pub.title[:80]}...")
                if pub.doi:
                    print(f"   DOI: {pub.doi}")
        else:
            print("❌ No publications found")
            
    except Exception as e:
        print(f"Error during scraping: {e}")
    finally:
        import gc
        gc.collect()
        time.sleep(1)