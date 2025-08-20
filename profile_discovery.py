"""Profile Discovery Module

Automatically discover researcher profiles across academic databases based on name and affiliation.

This module requires several third-party libraries:
    pip install requests rapidfuzz beautifulsoup4
    pip install undetected-chromedriver fake-useragent selenium webdriver-manager

Features:
    - Google Scholar profile discovery using robust scraping
    - ORCID and Scopus discovery via stable APIs
    - Profile verification with publication samples
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests
from rapidfuzz import fuzz

# Optional dependencies for Google Scholar scraping
try:
    import undetected_chromedriver as uc
    from fake_useragent import UserAgent
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.common.exceptions import TimeoutException, WebDriverException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    from webdriver_manager.chrome import ChromeDriverManager
    SCRAPING_DEPS_AVAILABLE = True
except ImportError:
    SCRAPING_DEPS_AVAILABLE = False

# Local imports with fallbacks
try:
    from sources import google_scholar, orcid, scopus, wos
    from models import Author
except ImportError:
    # Dummy implementations for testing
    class DummyPublication:
        def __init__(self, title): 
            self.title = title
    
    class DummySourceModule:
        def fetch(self, **kwargs): 
            return [DummyPublication(f"Sample Title {i+1}") for i in range(3)]
    
    google_scholar = orcid = scopus = wos = DummySourceModule()
    
    class Author: 
        pass

# Configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GS_COOKIES_PATH = os.path.join(BASE_DIR, ".gs_search_cookies.json")
DEFAULT_USER_AGENT = "ProfileDiscoveryBot/1.0"
MIN_NAME_SIMILARITY = 0.9
MAX_SEARCH_ATTEMPTS = 3
MAX_DRIVER_TIMEOUT = 45

# Data Models
@dataclass
class ProfileCandidate:
    source: str
    profile_id: str
    name: str
    affiliation: Optional[str] = None
    confidence_score: float = 0.0
    sample_publications: List[str] = field(default_factory=list)
    profile_url: Optional[str] = None
    verification_info: Optional[Dict] = field(default_factory=dict)

@dataclass
class DiscoveryResult:
    success: bool
    candidates: List[ProfileCandidate] = field(default_factory=list)
    error: Optional[str] = None

# Base Discovery Class
class BaseDiscovery:
    """Base class with common functionality for all discovery services."""
    
    def _calculate_confidence(self, query_name: str, result_name: str, 
                            query_aff: Optional[str] = None, result_aff: Optional[str] = None) -> float:
        """Calculate confidence score based on name and affiliation matching."""
        if not result_name:
            return 0.0
            
        # Calculate name similarity
        name_score = fuzz.ratio(query_name.lower(), result_name.lower()) / 100.0
        
        # Filter out poor name matches early
        if name_score < MIN_NAME_SIMILARITY:
            return 0.0
            
        # Start with name score weighted at 70%
        confidence = name_score * 0.7
        
        # Add affiliation score if available (30% weight)
        if query_aff and result_aff:
            affil_score = fuzz.partial_ratio(query_aff.lower(), result_aff.lower()) / 100.0
            confidence += affil_score * 0.3
        elif query_aff and not result_aff:
            # Small penalty for missing affiliation when expected
            confidence *= 0.9

        # Enforce minimum confidence for name-only matches (no affiliation provided)
        if (not query_aff or not str(query_aff).strip()) and name_score >= MIN_NAME_SIMILARITY:
            confidence = max(confidence, 0.75)
            
        return min(1.0, max(0.0, confidence))
    
    def _dedupe_and_sort_candidates(self, candidates: List[ProfileCandidate]) -> List[ProfileCandidate]:
        """Deduplicate candidates by profile_id, keeping highest confidence."""
        best_by_id = {}
        for candidate in candidates:
            profile_id = candidate.profile_id
            if (profile_id not in best_by_id or 
                candidate.confidence_score > best_by_id[profile_id].confidence_score):
                best_by_id[profile_id] = candidate
        
        return sorted(best_by_id.values(), key=lambda x: x.confidence_score, reverse=True)
    
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
        break_duration = random.uniform(2.5, 5.5)
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
        # Exponential backoff with jitter
        base = 15
        delay = min(180, base * (2 ** self.captcha_encounter_count)) + random.uniform(0.5, 2.5)
        logger.info(f"Waiting {delay:.1f} seconds before retry...")
        time.sleep(delay)
        if self.captcha_encounter_count > 0:
            try:
                self.driver.delete_all_cookies()
            except:
                pass
        # Stop after a few hits to avoid triggering more blocks
        return self.captcha_encounter_count < 3

class OptimizedStealthDriver:
    def __init__(self):
        self.behavior_sim = AdaptiveHumanBehaviorSimulator()
        self.ua = UserAgent()

    def create_driver(self, headless: bool = True) -> uc.Chrome:
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
        options.add_argument('--lang=en-US,en;q=0.9')

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
            options.add_argument('--no-first-run')
            options.add_argument('--disable-default-apps')

        try:
            driver = uc.Chrome(options=options, headless=headless, version_main=None)
            self._inject_stealth_js(driver)
            driver.set_page_load_timeout(20)
            driver.implicitly_wait(3)
            logger.info(f"Successfully created optimized Chrome driver (headless={headless})")
            return driver
        except Exception as e:
            logger.error(f"Failed to create undetected-chromedriver: {e}")
            return self._create_fallback_driver(headless)

    def _create_fallback_driver(self, headless: bool = True):
        """Create fallback driver using regular Selenium"""
        logger.info("Creating fallback driver with regular Selenium...")
        options = Options()

        options.add_argument('--disable-blink-features=AutomationControlled')
        # FIX: guard experimental options to avoid "unrecognized chrome option" crashes
        try:
            options.add_experimental_option('useAutomationExtension', False)
        except Exception as e:
            logger.debug(f"Ignoring unsupported experimental option useAutomationExtension: {e}")
        try:
            options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        except Exception as e:
            logger.debug(f"Ignoring unsupported experimental option excludeSwitches: {e}")
        options.add_argument('--window-size=1920,1080')
        options.page_load_strategy = 'eager'

        if headless:
            options.add_argument('--headless=new')
            options.add_argument('--no-first-run')
            options.add_argument('--disable-default-apps')

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        self._inject_stealth_js(driver)
        driver.set_page_load_timeout(20)
        driver.implicitly_wait(3)
        logger.info(f"Created fallback driver (headless={headless})")
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

# Discovery Classes
class GoogleScholarDiscovery(BaseDiscovery):
    """Discovers profiles by searching directly on scholar.google.com."""

    def __init__(self, api_keys: Dict[str, str] = None):
        if not SCRAPING_DEPS_AVAILABLE:
            raise ImportError("Selenium dependencies are required.")

        api_keys = api_keys or {}
        proxies = api_keys.get('proxies') or []
        self.proxy_pool = proxies if isinstance(proxies, list) else ([proxies] if proxies else [])
        self.stealth_driver = OptimizedStealthDriver()

    def search_profiles(self, first_name: str, last_name: str, affiliation: str = None) -> List[ProfileCandidate]:
        """Search for profiles with and without affiliation using a single driver session."""
        full_name = f"{first_name} {last_name}".strip()
        all_candidates = []

        # Prepare search queries
        queries = self._prepare_search_queries(full_name, affiliation)

        # Try all queries with a single driver session
        for attempt in range(1, MAX_SEARCH_ATTEMPTS + 1):
            driver = None
            proxy = self._get_proxy_for_attempt(attempt)

            logger.info(f"Starting Google Scholar search session (attempt {attempt}/{MAX_SEARCH_ATTEMPTS})")

            try:
                driver = self._create_and_setup_driver(proxy)

                if not self._is_driver_responsive(driver):
                    raise WebDriverException("Driver is not responsive")

                # Navigate to Scholar once and handle consent
                self._navigate_to_scholar(driver)
                self._handle_consent(driver, "//button[contains(text(), 'Accept')]")

                # Execute all queries with the same driver
                session_candidates = []
                for query_idx, (query, search_type) in enumerate(queries):
                    logger.info(f"Executing search query {query_idx + 1}/{len(queries)}: {search_type}")
                    
                    try:
                        candidates = self._perform_single_search(driver, query, full_name, affiliation, search_type)
                        if candidates is not None:
                            session_candidates.extend(candidates)
                            logger.info(f"Found {len(candidates)} candidates for {search_type}")
                        
                        # Delay between queries (but not after the last one)
                        if query_idx < len(queries) - 1:
                            time.sleep(random.uniform(1, 2))
                            
                    except Exception as e:
                        logger.warning(f"Query failed for {search_type}: {e}")
                        if self._is_captcha_page(driver):
                            logger.warning("CAPTCHA detected, ending session")
                            break

                # If we got any candidates from this session, we're done
                if session_candidates:
                    all_candidates.extend(session_candidates)
                    logger.info(f"Session successful. Found {len(session_candidates)} total candidates.")
                    break

            except Exception as e:
                self._handle_search_error(e, attempt, "session")

                if self._should_stop_retrying(str(e)):
                    break

                if attempt < MAX_SEARCH_ATTEMPTS:
                    self._wait_before_retry(attempt)
            finally:
                self._cleanup_driver(driver)

        if not all_candidates:
            logger.warning(f"All {MAX_SEARCH_ATTEMPTS} session attempts failed")

        return self._dedupe_and_sort_candidates(all_candidates)

    def _perform_single_search(self, driver, query: str, full_name: str, 
                              affiliation: str, search_type: str) -> Optional[List[ProfileCandidate]]:
        """Perform a single search query using existing driver."""
        try:
            # Navigate back to main search page if needed
            if "/search" in driver.current_url or "/scholar" not in driver.current_url:
                driver.get("https://scholar.google.com/")
                time.sleep(random.uniform(1, 2))

            # Execute search
            self._execute_search(driver, query)

            # Check for CAPTCHA
            if self._is_captcha_page(driver):
                raise WebDriverException("CAPTCHA detected after search submission")

            if not self._wait_for_search_results(driver):
                logger.debug(f"No search results found for {search_type}")
                return []

            return self._extract_profiles_from_results(driver, full_name, affiliation)

        except Exception as e:
            logger.warning(f"Error in single search for {search_type}: {e}")
            raise

    def _handle_consent(self, driver, xpath: str):
        """Handle consent pop-ups on Google/Scholar."""
        try:
            consent_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )
            driver.execute_script("arguments[0].scrollIntoView();", consent_button)
            time.sleep(0.3)
            consent_button.click()
            logger.info("Consent button clicked.")
            time.sleep(random.uniform(1.0, 1.5))
        except TimeoutException:
            logger.debug("Consent form not found.")
        except Exception as e:
            logger.debug(f"Consent handling error: {e}")

    def _prepare_search_queries(self, full_name: str, affiliation: str) -> List[tuple]:
        """Prepare list of search queries to try."""
        queries = [(full_name, "without affiliation")]
        if affiliation:
            queries.append((f"{full_name} {affiliation}", "with affiliation"))
        return queries

    # ...existing code... (remove _attempt_search_with_retries method as it's no longer needed)

    def _get_proxy_for_attempt(self, attempt: int) -> Optional[str]:
        """Get proxy for the given attempt number."""
        if self.proxy_pool and attempt <= len(self.proxy_pool):
            return self.proxy_pool[attempt - 1]
        return None

    def _create_and_setup_driver(self, proxy: Optional[str]):
        """Create and setup a WebDriver instance using OptimizedStealthDriver."""
        driver = self.stealth_driver.create_driver(headless=True)
        return driver

    def _is_driver_responsive(self, driver) -> bool:
        """Check if the driver is still responsive."""
        try:
            if not driver:
                return False
            driver.current_url  # Simple test operation
            return True
        except Exception:
            return False

    def _handle_search_error(self, error: Exception, attempt: int, search_type: str):
        """Handle and log search errors appropriately."""
        error_msg = str(error)
        logger.warning(f"Attempt {attempt} failed for {search_type}: {error_msg}")

        if self._is_window_error(error_msg):
            logger.warning("Window/session error detected, will retry with fresh driver")

    def _should_stop_retrying(self, error_msg: str) -> bool:
        """Determine if we should stop retrying based on error type."""
        if self._is_fatal_error(error_msg):
            logger.error(f"Fatal error detected, stopping retries: {error_msg}")
            return True
        return False

    def _wait_before_retry(self, attempt: int):
        """Wait before retrying with exponential backoff."""
        delay = min(60, 10 * attempt + random.uniform(3, 8))
        logger.info(f"Waiting {delay:.1f}s before retrying...")
        time.sleep(delay)

    def _is_fatal_error(self, error_msg: str) -> bool:
        """Check if error is fatal and retries won't help."""
        fatal_patterns = [
            "selenium dependencies are required",
            "failed to create any chrome driver",
            "chromedriver", "chrome binary", "permission denied"
        ]
        return any(pattern in error_msg.lower() for pattern in fatal_patterns)

    def _is_window_error(self, error_msg: str) -> bool:
        """Check if error is related to window/session management."""
        window_patterns = [
            "target window already closed", "web view not found",
            "session info: chrome", "no such window", "chrome not reachable",
            "session deleted", "invalid session id"
        ]
        return any(pattern in error_msg.lower() for pattern in window_patterns)

    def _cleanup_driver(self, driver):
        """Enhanced driver cleanup with multiple fallback strategies."""
        if not driver:
            return

        try:
            if self._is_driver_responsive(driver):
                driver.quit()
                logger.debug("Driver quit successfully")
                return
        except Exception as e:
            logger.debug(f"Graceful quit failed: {e}")

        try:
            driver.close()
            driver.quit()
            logger.debug("Driver force closed successfully")
        except Exception as e:
            logger.debug(f"Force close failed: {e}")

        time.sleep(random.uniform(1, 2))

    def _navigate_to_scholar(self, driver):
        logger.debug("Navigating to Google Scholar...")
        driver.get("https://scholar.google.com/")

        if not self._wait_for_page_load(driver):
            raise WebDriverException("Scholar page failed to load properly")

        time.sleep(random.uniform(1.5, 1.75))

    def _execute_search(self, driver, query: str):
        logger.debug("Finding search box...")
        search_box = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.NAME, "q"))
        )
        if not search_box.is_displayed():
            raise WebDriverException("Search box is not visible")

        self._paste_text(driver, search_box, query)
        self._submit_search(driver)

    def _submit_search(self, driver):
        search_button = driver.find_element(By.ID, "gs_hdr_tsb")
        if not search_button.is_enabled():
            time.sleep(1)
        search_button.click()
        time.sleep(random.uniform(2.0, 3.5))

    def _extract_profiles_from_results(self, driver, full_name: str, affiliation: str) -> List[ProfileCandidate]:
        """Extract inline profile candidates only, without navigation."""
        return self._parse_general_results_for_profiles(driver.page_source, full_name, affiliation)

    def _wait_for_page_load(self, driver, timeout: int = 10) -> bool:
        try:
            WebDriverWait(driver, timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            return True
        except TimeoutException:
            return False

    def _wait_for_search_results(self, driver, timeout: int = 10) -> bool:
        try:
            WebDriverWait(driver, timeout).until(
                EC.any_of(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#gs_res_ccl")),
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".gs_r"))
                )
            )
            return True
        except TimeoutException:
            return False

    def _is_captcha_page(self, driver) -> bool:
        try:
            patterns = ['recaptcha', '/sorry/', 'captcha', 'unusual traffic']
            if any(p in driver.current_url.lower() for p in patterns):
                return True
            if any(p in driver.title.lower() for p in patterns):
                return True
            return False
        except Exception:
            return False

    def _paste_text(self, driver, element, text: str):
        try:
            element.clear()
            element.send_keys(text)
        except Exception as e:
            logger.warning(f"Error pasting text: {e}")

    def _parse_general_results_for_profiles(self, page_source: str, full_name: str, affiliation: str) -> List[ProfileCandidate]:
        """Parse inline profile blocks in Google Scholar search results (no navigation)."""
        from bs4 import BeautifulSoup
        import re

        candidates = []
        soup = BeautifulSoup(page_source, "html.parser")

        for profile_div in soup.select("div.gs_r table"):
            try:
                name_el = profile_div.select_one("h4.gs_rt2 a")
                if not name_el:
                    continue

                profile_url = name_el.get("href", "")
                profile_id = None
                m = re.search(r"user=([A-Za-z0-9_-]+)", profile_url)
                if m:
                    profile_id = m.group(1)

                name = name_el.get_text(strip=True)

                affil_el = profile_div.select_one("div.gs_nph")
                affiliation_text = affil_el.get_text(" ", strip=True) if affil_el else None

                email_el = profile_div.find(string=lambda s: s and "Verified email" in s)
                email_text = email_el.strip() if email_el else None

                cited_el = profile_div.find(string=lambda s: s and "Cited by" in s)
                cited_text = cited_el.strip() if cited_el else None

                confidence = self._calculate_confidence(full_name, name, affiliation, affiliation_text)
                if email_text:
                    confidence += 0.1
                if cited_text:
                    try:
                        num_cites = int(re.search(r"(\d+)", cited_text).group(1))
                        if num_cites > 1000:
                            confidence += 0.1
                    except Exception:
                        pass

                candidate = ProfileCandidate(
                    source="Google Scholar",
                    profile_id=profile_id or "",
                    name=name,
                    affiliation=affiliation_text,
                    confidence_score=min(confidence, 1.0),
                    sample_publications=[],
                    profile_url=f"https://scholar.google.com{profile_url}" if profile_url else None,
                    verification_info={
                        "email": email_text,
                        "cited_by": cited_text
                    }
                )
                candidates.append(candidate)
            except Exception as e:
                logger.debug(f"Error parsing profile block: {e}")

        return candidates

class ORCIDDiscovery(BaseDiscovery):
    """Discover ORCID profiles using the public search API."""
    
    def __init__(self):
        self.search_base_url = "https://pub.orcid.org/v3.0/search"

    def search_profiles(self, first_name: str, last_name: str, affiliation: str = None) -> List[ProfileCandidate]:
        all_candidates = []
        
        # Try both searches: without and with affiliation
        queries_to_try = [
            [f'given-names:{first_name.strip()}', f'family-name:{last_name.strip()}']
        ]
        
        if affiliation:
            queries_to_try.append([
                f'given-names:{first_name.strip()}', 
                f'family-name:{last_name.strip()}',
                f'affiliation-org-name:{affiliation.strip()}'
            ])
        
        for query_idx, query_parts in enumerate(queries_to_try):
            search_type = "without affiliation" if query_idx == 0 else "with affiliation"
            candidates = self._perform_search(query_parts, search_type, f"{first_name} {last_name}", affiliation)
            all_candidates.extend(candidates)
        
        return self._dedupe_and_sort_candidates(all_candidates)

    def _perform_search(self, query_parts: List[str], search_type: str, 
                       query_name: str, affiliation: Optional[str]) -> List[ProfileCandidate]:
        """Perform ORCID search with given query parts."""
        params = {'q': " AND ".join(query_parts), 'rows': 20}
        headers = {'Accept': 'application/json', 'User-Agent': DEFAULT_USER_AGENT}
        
        try:
            logger.debug(f"ORCID search {search_type}: {params['q']}")
            response = requests.get(self.search_base_url, headers=headers, params=params, timeout=20)
            response.raise_for_status()
            
            try:
                data = response.json()
            except ValueError:
                logger.error(f"ORCID discovery failed ({search_type}): response is not JSON")
                return []
            
            candidates = self._parse_results(data or {}, query_name, affiliation)
            logger.info(f"Found {len(candidates)} ORCID candidates {search_type}")
            return candidates
            
        except requests.RequestException as e:
            logger.error(f"ORCID discovery failed ({search_type}): {e}")
            return []

    def _parse_results(self, data: Dict, query_name: str, query_affiliation: Optional[str]) -> List[ProfileCandidate]:
        candidates = []
        entries = data.get('result', [])
        if not isinstance(entries, list):
            entries = [entries] if entries else []

        for result in entries:
            try:
                # Extract ORCID ID
                orcid_identifier = result.get('orcid-identifier', {})
                orcid_id = orcid_identifier.get('path')
                if not orcid_id:
                    continue

                # Extract name
                person = result.get('person', {})
                name_info = person.get('name', {})
                given_names = name_info.get('given-names', {})
                family_name = name_info.get('family-name', {})
                given = given_names.get('value', '') if given_names else ''
                family = family_name.get('value', '') if family_name else ''
                profile_name = f"{given} {family}".strip() or query_name

                # Extract affiliation
                profile_affiliation = self._extract_affiliation(person)
                confidence = self._calculate_confidence(query_name, profile_name, query_affiliation, profile_affiliation)

                candidates.append(ProfileCandidate(
                    source="ORCID",
                    profile_id=orcid_id,
                    name=profile_name,
                    affiliation=profile_affiliation,
                    confidence_score=confidence,
                    profile_url=f"https://orcid.org/{orcid_id}"
                ))

            except Exception as e:
                logger.warning(f"Error parsing ORCID result: {e}")
                continue

        logger.info(f"Parsed {len(candidates)} ORCID candidates")
        return sorted(candidates, key=lambda x: x.confidence_score, reverse=True)

    def _extract_affiliation(self, person: Dict) -> Optional[str]:
        """Extract affiliation from ORCID person data."""
        affiliations = person.get('affiliation-group', [])
        for aff_group in affiliations:
            summaries = aff_group.get('summaries', [])
            for summary in summaries:
                emp_summary = summary.get('employment-summary', {})
                if emp_summary:
                    organization = emp_summary.get('organization', {})
                    org_name = organization.get('name')
                    if org_name:
                        return org_name
        return None

class ScopusDiscovery(BaseDiscovery):
    """Discover Scopus Author IDs using the Author Search API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.search_base_url = "https://api.elsevier.com/content/search/author"
    
    def search_profiles(self, first_name: str, last_name: str, affiliation: str = None) -> List[ProfileCandidate]:
        if not self.api_key:
            logger.warning("Scopus API key not provided, skipping discovery.")
            return []
        
        all_candidates = []
        
        # Try both searches: without and with affiliation
        queries_to_try = [
            [f'AUTHFIRST("{first_name}")', f'AUTHLASTNAME("{last_name}")']
        ]
        
        if affiliation:
            queries_to_try.append([
                f'AUTHFIRST("{first_name}")', 
                f'AUTHLASTNAME("{last_name}")',
                f'AFFIL("{affiliation}")'
            ])
        
        for query_idx, query_parts in enumerate(queries_to_try):
            search_type = "without affiliation" if query_idx == 0 else "with affiliation"
            candidates = self._perform_search(query_parts, search_type, f"{first_name} {last_name}", affiliation)
            all_candidates.extend(candidates)
        
        return self._dedupe_and_sort_candidates(all_candidates)

    def _perform_search(self, query_parts: List[str], search_type: str, 
                       query_name: str, affiliation: Optional[str]) -> List[ProfileCandidate]:
        """Perform Scopus search with given query parts."""
        params = {'query': " AND ".join(query_parts), 'count': 10}
        headers = {'X-ELS-APIKey': self.api_key, 'Accept': 'application/json'}

        try:
            logger.debug(f"Scopus search {search_type}: {params['query']}")
            response = requests.get(self.search_base_url, headers=headers, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            candidates = self._parse_results(data, query_name, affiliation)
            logger.info(f"Found {len(candidates)} Scopus candidates {search_type}")
            return candidates
            
        except requests.RequestException as e:
            logger.error(f"Scopus discovery failed ({search_type}): {e}")
            return []

    def _parse_results(self, data: Dict, query_name: str, query_affiliation: Optional[str]) -> List[ProfileCandidate]:
        candidates = []
        for entry in data.get('search-results', {}).get('entry', []):
            try:
                author_id = entry.get('dc:identifier', '').replace('AUTHOR_ID:', '')
                if not author_id:
                    continue

                p_name = entry.get('preferred-name', {})
                profile_name = f"{p_name.get('given-name', '')} {p_name.get('surname', '')}".strip()
                profile_affiliation = entry.get('affiliation-current', {}).get('affiliation-name')
                doc_count = int(entry.get('document-count', 0))
                
                confidence = self._calculate_scopus_confidence(query_name, profile_name, 
                                                             query_affiliation, profile_affiliation, doc_count)
                
                candidates.append(ProfileCandidate(
                    source="Scopus",
                    profile_id=author_id,
                    name=profile_name,
                    affiliation=profile_affiliation,
                    confidence_score=confidence,
                    profile_url=f"https://www.scopus.com/authid/detail.uri?authorId={author_id}",
                    verification_info={'document_count': doc_count}
                ))
            except Exception as e:
                logger.warning(f"Error parsing Scopus result: {e}")

        return sorted(candidates, key=lambda x: x.confidence_score, reverse=True)

    def _calculate_scopus_confidence(self, query_name: str, result_name: str, 
                                   query_aff: Optional[str], result_aff: Optional[str], doc_count: int) -> float:
        """Calculate confidence score including document count."""
        name_score = fuzz.ratio(query_name.lower(), result_name.lower()) / 100.0
        
        if name_score < MIN_NAME_SIMILARITY:
            return 0.0
            
        confidence = name_score * 0.6
        
        # Add affiliation score if available
        if query_aff and result_aff:
            affil_score = fuzz.partial_ratio(query_aff.lower(), result_aff.lower()) / 100.0
            confidence += affil_score * 0.3
        
        # Add document count bonus (up to 0.1)
        confidence += min(0.1, doc_count / 100.0)

        # Enforce minimum confidence for name-only matches (no affiliation provided)
        if (not query_aff or not str(query_aff).strip()) and name_score >= MIN_NAME_SIMILARITY:
            confidence = max(confidence, 0.75)
        
        return min(1.0, confidence)

class WOSDiscovery(BaseDiscovery):
    """Placeholder for Web of Science Author ID discovery."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def search_profiles(self, first_name: str, last_name: str, affiliation: str = None) -> List[ProfileCandidate]:
        if not self.api_key:
            return []
        logger.info("Web of Science author discovery is not yet implemented.")
        return []

# Main Service
class ProfileDiscoveryService:
    """Main service for discovering researcher profiles across all sources."""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        self.sources = {}
        
        self._initialize_sources()
    
    def _initialize_sources(self):
        """Initialize all available discovery sources."""
        if SCRAPING_DEPS_AVAILABLE:
            self.sources["Google Scholar"] = GoogleScholarDiscovery(self.api_keys)
        else:
            logger.warning("Google Scholar scraping disabled (dependencies not found).")
        
        self.sources["ORCID"] = ORCIDDiscovery()
        
        if self.api_keys.get('scopus_api_key'):
            self.sources["Scopus"] = ScopusDiscovery(self.api_keys['scopus_api_key'])
        
        if self.api_keys.get('wos_api_key'):
            self.sources["Web of Science"] = WOSDiscovery(self.api_keys['wos_api_key'])
    
    def discover_all_profiles(self, first_name: str, last_name: str, 
                            affiliation: str = None) -> DiscoveryResult:
        """Discover profiles from all available sources."""
        all_candidates, errors = [], []
        
        for name, service in self.sources.items():
            try:
                candidates = service.search_profiles(first_name, last_name, affiliation)
                all_candidates.extend(candidates)
                logger.info(f"Found {len(candidates)} candidate(s) from {name}")
            except Exception as e:
                error_msg = f"{name} discovery failed: {e}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)
        
        all_candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        return DiscoveryResult(
            success=not errors,
            candidates=all_candidates,
            error="; ".join(errors) if errors else None
        )
    
    def verify_profile(self, candidate: ProfileCandidate, max_publications: int = 3) -> ProfileCandidate:
        """Verify a profile by fetching sample publications."""
        fetch_map = {
            "Google Scholar": (google_scholar.fetch, {
                'gs_id': candidate.profile_id,
                'max_publications_detail': max_publications
            }),
            "ORCID": (orcid.fetch, {
                'orcid_id': candidate.profile_id,
                'max_records': max_publications,
                **self.api_keys
            }),
            "Scopus": (scopus.fetch, {
                'author_id': candidate.profile_id,
                'max_records': max_publications,
                **self.api_keys
            }),
        }
        
        if candidate.source not in fetch_map:
            return candidate

        fetch_func, kwargs = fetch_map[candidate.source]
        try:
            publications = fetch_func(**kwargs)
            candidate.sample_publications = [pub.title for pub in publications[:max_publications]]
            candidate.verification_info['verified_at'] = time.time()
            candidate.verification_info['sample_count'] = len(candidate.sample_publications)
        except Exception as e:
            logger.warning(f"Profile verification failed for {candidate.source} {candidate.profile_id}: {e}")
        
        return candidate

def discover_researcher_profiles(
    first_name: str, 
    last_name: str, 
    affiliation: str = None, 
    api_keys: Dict[str, str] = None, 
    verify_profiles: bool = False
) -> DiscoveryResult:
    """
    Discover researcher profiles across all academic databases.
    
    Args:
        first_name: Researcher's first name
        last_name: Researcher's last name
        affiliation: Optional institutional affiliation
        api_keys: Dictionary of API keys for various services
        verify_profiles: Whether to fetch sample publications for verification
    
    Returns:
        DiscoveryResult object containing candidates and status
    """
    service = ProfileDiscoveryService(api_keys)
    result = service.discover_all_profiles(first_name, last_name, affiliation)
    
    if verify_profiles and result.success:
        for i, candidate in enumerate(result.candidates[:5]):
            result.candidates[i] = service.verify_profile(candidate)
    
    return result


def main():
    """Example usage of the profile discovery service."""
    # Load API keys if available
    api_keys = {}
    config_path = "config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                api_keys = json.load(f)
            logger.info("Loaded API keys from config.json")
        except Exception as e:
            logger.warning(f"Failed to load config.json: {e}")

    # Example: Discover profiles for Geoffrey Hinton
    result = discover_researcher_profiles(
        first_name="Geoffrey",
        last_name="Hinton",
        affiliation="University of Toronto",
        api_keys=api_keys,
        verify_profiles=False
    )

    # Display results
    if result.success or result.candidates:
        print(f"\n--- Discovery Complete: Found {len(result.candidates)} total candidates ---")
        for i, candidate in enumerate(result.candidates[:5], 1):
            print(f"\n{i}. {candidate.source} Profile")
            print(f"   Name:         {candidate.name}")
            print(f"   Affiliation:  {candidate.affiliation or 'N/A'}")
            print(f"   Confidence:   {candidate.confidence_score:.2f}")
            print(f"   Profile URL:  {candidate.profile_url or 'N/A'}")
    else:
        print(f"\n--- Discovery Failed ---\nError: {result.error}")


if __name__ == "__main__":
    main()