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
from typing import Dict, List, Optional, Callable
import itertools
import inspect

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

    # Patch undetected_chromedriver quit to avoid WinError 6 on Windows
    def _patched_quit(self):
        try:
            self._original_quit()
        except OSError as e:
            if "WinError 6" in str(e) or "The handle is invalid" in str(e):
                pass
            else:
                raise

    if hasattr(uc.Chrome, 'quit') and not hasattr(uc.Chrome, '_original_quit'):
        uc.Chrome._original_quit = uc.Chrome.quit
        uc.Chrome.quit = _patched_quit

    # Suppress "could not detect version_main" warning from undetected_chromedriver
    import logging as _logging
    _uc_logger = _logging.getLogger("undetected_chromedriver")
    class _UCSuppressFilter(_logging.Filter):
        def filter(self, record):
            msg = str(record.getMessage())
            if "could not detect version_main" in msg:
                return False
            return True
    _uc_logger.addFilter(_UCSuppressFilter())

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
MIN_NAME_SIMILARITY = 0.85
MAX_SEARCH_ATTEMPTS = 2
MAX_DRIVER_TIMEOUT = 45
MAX_PUBLICATION_SAMPLES = 3

# Toggle headless/headful mode here
HEADLESS_MODE = True  # Set to True for headless, False for headful

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
    # New: total number of publications (if known)
    publication_count: Optional[int] = None

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
            # Use both partial_ratio and token_sort_ratio for better affiliation matching
            affil_score1 = fuzz.partial_ratio(query_aff.lower(), result_aff.lower()) / 100.0
            affil_score2 = fuzz.token_sort_ratio(query_aff.lower(), result_aff.lower()) / 100.0
            affil_score = max(affil_score1, affil_score2)
            confidence += affil_score * 0.3
            
            # Bonus for strong affiliation matches (e.g., "Purdue University" in both)
            if affil_score >= 0.8:
                confidence += 0.1
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

class OptimizedStealthDriver:
    def __init__(self):
        self.behavior_sim = AdaptiveHumanBehaviorSimulator()
        self.ua = UserAgent()

    def create_driver(self, headless: bool = None, proxy: Optional[str] = None) -> uc.Chrome:
        """Create optimized undetected Chrome driver with stealth & optional proxy"""
        options = uc.ChromeOptions()

        # --- Anti-detection flags & UA ---
        user_agent = self.ua.random
        options.add_argument(f'--user-agent={user_agent}')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-web-security')
        options.add_argument('--lang=en-US,en;q=0.9')
        options.add_argument('--password-store=basic')
        options.add_argument('--enable-features=NetworkService,NetworkServiceInProcess')

        # --- Optional proxy ---
        if proxy:
            options.add_argument(f'--proxy-server={proxy}')

        # --- Preferences ---
        prefs = {
            'profile.default_content_setting_values.notifications': 2,
            'profile.default_content_settings.popups': 0,
            'profile.managed_default_content_settings.images': 1,
            'credentials_enable_service': False,
            'profile.password_manager_enabled': False,
        }
        options.add_experimental_option('prefs', prefs)

        # Always use global toggle
        headless = HEADLESS_MODE if headless is None else headless

        try:
            driver = uc.Chrome(options=options, headless=headless, version_main=None)
            self._post_boot_stealth(driver)
            driver.set_page_load_timeout(20)
            driver.implicitly_wait(3)
            return driver
        except Exception as e:
            logger.error(f"Failed to create undetected-chromedriver: {e}")
            return self._create_fallback_driver(headless, proxy)

    def _create_fallback_driver(self, headless: bool = None, proxy: Optional[str] = None):
        logger.info("Creating fallback driver with regular Selenium...")
        options = Options()
        options.add_argument('--disable-blink-features=AutomationControlled')
        try:
            options.add_experimental_option('useAutomationExtension', False)
        except Exception:
            pass
        try:
            options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        except Exception:
            pass
        options.add_argument('--window-size=1920,1080')
        options.page_load_strategy = 'eager'
        if proxy:
            options.add_argument(f'--proxy-server={proxy}')
        # Always use global toggle
        headless = HEADLESS_MODE if headless is None else headless
        # Add headless argument when needed
        if headless:
            options.add_argument('--headless=new')
            options.add_argument('--no-first-run')
            options.add_argument('--disable-default-apps')
        else:
            options.add_argument('--no-first-run')
            options.add_argument('--disable-default-apps')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        self._post_boot_stealth(driver)
        driver.set_page_load_timeout(20)
        driver.implicitly_wait(3)
        return driver

    def _post_boot_stealth(self, driver):
        """Stealth JS, language/platform/vendor, timezone, & cookie jar setup"""
        # --- JS stealth shims ---
        stealth_js = """
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        window.chrome = {runtime: {}};
        Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3] });
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US','en'] });
        Object.defineProperty(navigator, 'platform', { get: () => 'Win32' });
        Object.defineProperty(navigator, 'vendor', { get: () => 'Google Inc.' });
        try {
          const getParameter = WebGLRenderingContext.prototype.getParameter;
          WebGLRenderingContext.prototype.getParameter = function(parameter) {
            if (parameter === 37445) return 'Intel Open Source Technology Center';
            if (parameter === 37446) return 'Mesa DRI Intel(R) UHD Graphics';
            return getParameter.call(this, parameter);
          };
        } catch(e) {}
        """
        try:
            driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': stealth_js})
        except Exception:
            try:
                driver.execute_script(stealth_js)
            except Exception:
                pass

        # --- Timezone spoof (match en-US) ---
        try:
            driver.execute_cdp_cmd('Emulation.setTimezoneOverride', {'timezoneId': 'America/Los_Angeles'})
        except Exception:
            pass

        # --- Cookie jar: pre-load from disk if present (reduces consent prompts) ---
        try:
            if os.path.exists(GS_COOKIES_PATH):
                with open(GS_COOKIES_PATH, 'r', encoding='utf-8') as f:
                    cookies = json.load(f)
                driver.get("https://scholar.google.com/")
                for ck in cookies:
                    try:
                        driver.add_cookie(ck)
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"Cookie preload skipped: {e}")

    # Persist cookies for next run
    def save_cookies(self, driver):
        try:
            cookies = driver.get_cookies()
            with open(GS_COOKIES_PATH, 'w', encoding='utf-8') as f:
                json.dump(cookies, f)
        except Exception as e:
            logger.debug(f"Cookie save skipped: {e}")


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
        self._last_driver = None  # Store last successful driver for reuse

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

                # --- NEW: Wait for page load before responsiveness check ---
                try:
                    self._wait_for_page_load(driver, timeout=12)
                    time.sleep(random.uniform(0.5, 1.0))  # Short warm-up
                except Exception:
                    pass

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
                    # Store driver for potential reuse instead of cleaning up
                    try:
                        self.stealth_driver.save_cookies(driver)
                    except Exception:
                        pass
                    self._last_driver = driver
                    driver = None  # Prevent cleanup in finally block
                    break

            except Exception as e:
                self._handle_search_error(e, attempt, "session")

                if self._should_stop_retrying(str(e)):
                    break

                if attempt < MAX_SEARCH_ATTEMPTS:
                    self._wait_before_retry(attempt)
            finally:
                # Only cleanup if we're not storing the driver for reuse
                if driver is not None:
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

    def _handle_consent(self, driver, xpath_primary: str):
        """Robust consent handler for different regions/locales"""
        XPATHS = [
            xpath_primary,  # caller's hint
            "//button[.//span[contains(translate(., 'ACEPTAR', 'aceptar'), 'acept')]]",
            "//button[contains(translate(., 'OK', 'ok'), 'ok')]",
            "//button[contains(., 'I agree')]",
            "//button[contains(., 'Accept all')]",
            "//div[@role='dialog']//button[contains(., 'Accept')]",
        ]
        try:
            for xp in XPATHS:
                try:
                    btn = WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.XPATH, xp)))
                    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
                    time.sleep(0.25)
                    btn.click()
                    time.sleep(random.uniform(0.8, 1.3))
                    break
                except TimeoutException:
                    continue
        except Exception as e:
            logger.debug(f"Consent handling benign error: {e}")
        finally:
            # Save any consent cookies for reuse
            try:
                self.stealth_driver.save_cookies(driver)
            except Exception:
                pass

    def _prepare_search_queries(self, full_name: str, affiliation: str) -> List[tuple]:
        """Prepare list of search queries to try."""
        queries = [(full_name, "without affiliation")]
        if affiliation:
            queries.append((f"{full_name} {affiliation}", "with affiliation"))
        return queries

    def _get_proxy_for_attempt(self, attempt: int) -> Optional[str]:
        """Get proxy for the given attempt number."""
        if self.proxy_pool and attempt <= len(self.proxy_pool):
            return self.proxy_pool[attempt - 1]
        return None

    def _create_and_setup_driver(self, proxy: Optional[str]):
        # Always use global toggle for headless/headful
        driver = self.stealth_driver.create_driver(headless=HEADLESS_MODE, proxy=proxy)
        # Minimal warm-up
        try:
            driver.get("https://scholar.google.com/")
            time.sleep(random.uniform(1.2, 1.8))
        except Exception:
            pass
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
        driver.get("https://scholar.google.com/")
        if not self._wait_for_page_load(driver):
            raise WebDriverException("Scholar page failed to load properly")
        # Consent gets handled right after (below)
        time.sleep(random.uniform(1.2, 1.6))

    def _execute_search(self, driver, query: str):
        # Find & type like a human; avoid programmatic submit if button disabled
        search_box = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.NAME, "q")))
        if not search_box.is_displayed():
            raise WebDriverException("Search box is not visible")

        # Clear, type with slight jitter
        try:
            search_box.clear()
        except Exception:
            pass

        for chunk in [query]:  # simple, but ready for per-word typing if needed
            search_box.send_keys(chunk)
            time.sleep(random.uniform(0.08, 0.16))

        # Prefer header button, fallback to Enter
        try:
            btn = driver.find_element(By.ID, "gs_hdr_tsb")
            if not btn.is_enabled():
                time.sleep(0.4)
            btn.click()
        except Exception:
            from selenium.webdriver.common.keys import Keys
            search_box.send_keys(Keys.ENTER)

        time.sleep(random.uniform(1.8, 2.8))

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
            cur = (driver.current_url or "").lower()
            title = (driver.title or "").lower()
            hit = any(p in cur for p in patterns) or any(p in title for p in patterns)
            if hit and hasattr(self, 'captcha_handler'):  # if wired from scraper pattern
                self.captcha_handler.captcha_encounter_count += 1
            return hit
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

                # New: try to populate publication_count for discovery tab
                pub_count = self._extract_inline_publication_count(profile_div)
                estimated = False
                if not pub_count and profile_id:
                    pub_count = self._fetch_gs_publication_count(profile_id)
                    estimated = bool(pub_count)  # this count is a minimum estimate if present
                if pub_count is not None:
                    candidate.publication_count = pub_count
                    candidate.verification_info["publication_count"] = pub_count
                    if estimated:
                        candidate.verification_info["publication_count_is_min"] = True

                candidates.append(candidate)
            except Exception as e:
                logger.debug(f"Error parsing profile block: {e}")

        return candidates

    # New helpers: publication count extraction for Google Scholar
    def _extract_inline_publication_count(self, profile_div) -> Optional[int]:
        """Best-effort parse of publication count if inline text shows it."""
        import re
        try:
            text = profile_div.get_text(" ", strip=True)
            m = re.search(r'\bPublications?\s*:?\s*(\d+)\b', text, re.IGNORECASE)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return None

    def _fetch_gs_publication_count(self, profile_id: str) -> Optional[int]:
        """Lightweight HTTP fetch to count visible rows on profile page as a minimum estimate."""
        try:
            from bs4 import BeautifulSoup
            headers = {'User-Agent': DEFAULT_USER_AGENT}
            # Only first page: cstart=0, pagesize=20 (do not attempt to load more)
            url = (
                f"https://scholar.google.com/citations?"
                f"user={profile_id}&hl=en&view_op=list_works&sortby=pubdate&cstart=0&pagesize=20"
            )
            resp = requests.get(url, headers=headers, timeout=8)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            rows = soup.select("tr.gsc_a_tr")
            if rows:
                return len(rows)
        except Exception as e:
            logger.debug(f"GS publication count fetch failed for {profile_id}: {e}")
        return None

class ORCIDDiscovery(BaseDiscovery):
    """Discover ORCID profiles using the public search API."""
    
    def __init__(self):
        self.search_base_url = "https://pub.orcid.org/v3.0/search"

    def search_profiles(self, first_name: str, last_name: str, affiliation: str = None) -> List[ProfileCandidate]:
        """Optimized ORCID search with combined query approach."""
        all_candidates = []
        
        # Start with affiliation-based search if available, then fall back to name-only
        if affiliation:
            # Try comprehensive search first
            query_parts = [
                f'given-names:{first_name.strip()}',
                f'family-name:{last_name.strip()}',
                f'affiliation-org-name:{affiliation.strip()}'
            ]
            candidates = self._perform_search(query_parts, "with affiliation", f"{first_name} {last_name}", affiliation)
            all_candidates.extend(candidates)
            
            # If we got high-confidence matches, skip name-only search
            if any(c.confidence_score >= 0.9 for c in candidates):
                logger.info("High-confidence ORCID match found with affiliation, skipping name-only search")
                return self._dedupe_and_sort_candidates(all_candidates)
        
        # Name-only search (always do this if no high-confidence affiliation match)
        query_parts = [f'given-names:{first_name.strip()}', f'family-name:{last_name.strip()}']
        candidates = self._perform_search(query_parts, "name only", f"{first_name} {last_name}", affiliation)
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

                # Extract other candidate names (credit-name and other-names) for matching
                other_names = self._extract_other_names(person)

                # Extract affiliation (very noisy in ORCID; tiny influence)
                profile_affiliation = self._extract_affiliation(person)

                # New: fetch works count before computing confidence (for small boost)
                pub_count = self._fetch_orcid_publication_count(orcid_id)

                # ORCID-specific confidence scoring (name-first, minimal affiliation)
                confidence = self._calculate_orcid_confidence(
                    query_name=query_name,
                    result_name=profile_name,
                    query_aff=query_affiliation,
                    result_aff=profile_affiliation,
                    works_count=pub_count,
                    other_names=other_names
                )

                candidate = ProfileCandidate(
                    source="ORCID",
                    profile_id=orcid_id,
                    name=profile_name,
                    affiliation=profile_affiliation,
                    confidence_score=confidence,
                    profile_url=f"https://orcid.org/{orcid_id}"
                )

                if pub_count is not None:
                    candidate.publication_count = pub_count
                    candidate.verification_info["publication_count"] = pub_count

                candidates.append(candidate)

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

    def _extract_other_names(self, person: Dict) -> List[str]:
        names: List[str] = []
        try:
            name_info = person.get('name', {}) or {}
            credit = name_info.get('credit-name') or {}
            credit_val = (credit.get('value') or credit.get('content') or "").strip()
            if credit_val:
                names.append(credit_val)

            other_names = (person.get('other-names') or {}).get('other-name') or []
            if isinstance(other_names, dict):
                other_names = [other_names]
            for on in other_names:
                val = (on.get('value') or on.get('content') or "").strip()
                if val:
                    names.append(val)
        except Exception:
            pass
        return names

    def _calculate_orcid_confidence(
        self,
        query_name: str,
        result_name: str,
        query_aff: Optional[str],
        result_aff: Optional[str],
        works_count: Optional[int] = None,
        other_names: Optional[List[str]] = None
    ) -> float:
        other_names = other_names or []
        try:
            # Best name score across primary and alternate names
            base_names = [result_name] + [n for n in other_names if n]
            best_name = 0.0
            q = (query_name or "").lower()
            for n in base_names:
                best_name = max(best_name, fuzz.ratio(q, (n or "").lower()) / 100.0)

            # Gate: require a solid name match
            if best_name < MIN_NAME_SIMILARITY:
                return 0.0

            # Heavily weight name match
            confidence = best_name * 0.93

            # Very small affiliation contribution due to ORCID noisiness
            if query_aff and result_aff:
                aff = max(
                    fuzz.partial_ratio(query_aff.lower(), result_aff.lower()) / 100.0,
                    fuzz.token_set_ratio(query_aff.lower(), result_aff.lower()) / 100.0
                )
                confidence += aff * 0.03
                if aff >= 0.8:
                    confidence += 0.01

            # Small works-count bump (log-scale)
            if isinstance(works_count, int) and works_count > 0:
                import math
                confidence += min(0.04, math.log10(works_count + 1) * 0.03)

            # Strong baseline for name-only matches when no affiliation provided
            if (not query_aff or not str(query_aff).strip()) and best_name >= MIN_NAME_SIMILARITY:
                confidence = max(confidence, 0.88)

            return float(min(1.0, max(0.0, confidence)))
        except Exception:
            return 0.0

    def _fetch_orcid_publication_count(self, orcid_id: str) -> Optional[int]:
        """Return the number of unique works (groups) for an ORCID iD."""
        url = f"https://pub.orcid.org/v3.0/{orcid_id}/works"
        headers = {'Accept': 'application/json', 'User-Agent': DEFAULT_USER_AGENT}
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            groups = data.get('group', [])
            if isinstance(groups, list):
                return len(groups)
        except Exception as e:
            logger.debug(f"ORCID works count fetch failed for {orcid_id}: {e}")
        return None
    
class ScopusDiscovery(BaseDiscovery):
    """Discover Scopus Author IDs using the Author Search API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.search_base_url = "https://api.elsevier.com/content/search/author"
    
    def search_profiles(self, first_name: str, last_name: str, affiliation: str = None) -> List[ProfileCandidate]:
        """Optimized Scopus search with early termination and sample fetching during discovery."""
        if not self.api_key:
            logger.warning("Scopus API key not provided, skipping discovery.")
            return []
        
        all_candidates = []
        
        # Start with affiliation-based search if available
        if affiliation:
            query_parts = [
                f'AUTHFIRST("{first_name}")', 
                f'AUTHLASTNAME("{last_name}")',
                f'AFFIL("{affiliation}")'
            ]
            candidates = self._perform_search(query_parts, "with affiliation", f"{first_name} {last_name}", affiliation)
            all_candidates.extend(candidates)
            
            # If we got high-confidence matches, skip name-only search
            if any(c.confidence_score >= 0.9 for c in candidates):
                logger.info("High-confidence Scopus match found with affiliation, skipping name-only search")
                return self._dedupe_and_sort_candidates(all_candidates)
        
        # Name-only search
        query_parts = [f'AUTHFIRST("{first_name}")', f'AUTHLASTNAME("{last_name}")']
        candidates = self._perform_search(query_parts, "name only", f"{first_name} {last_name}", affiliation)
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
                
                candidate = ProfileCandidate(
                    source="Scopus",
                    profile_id=author_id,
                    name=profile_name,
                    affiliation=profile_affiliation,
                    confidence_score=confidence,
                    profile_url=f"https://www.scopus.com/authid/detail.uri?authorId={author_id}",
                    verification_info={'document_count': doc_count},
                    publication_count=doc_count
                )

                # Try to fetch sample publications during discovery
                sample_pubs = self._fetch_scopus_sample_publications(author_id)
                if sample_pubs:
                    candidate.sample_publications = sample_pubs
                    candidate.verification_info['sample_count'] = len(sample_pubs)
                else:
                    logger.debug(f"No sample publications fetched for Scopus author {author_id}")

                candidates.append(candidate)
            except Exception as e:
                logger.warning(f"Error parsing Scopus result: {e}")

        return sorted(candidates, key=lambda x: x.confidence_score, reverse=True)

    def _fetch_scopus_sample_publications(self, author_id: str) -> List[str]:
        """Fetch sample publications via the canonical scopus.fetch(scopus_id, api_key)."""
        try:
            if not self.api_key:
                return []
            pubs = scopus.fetch(scopus_id=author_id, api_key=self.api_key, page_batch=MAX_PUBLICATION_SAMPLES)
            titles = [getattr(p, "title", str(p)) for p in pubs if getattr(p, "title", None)]
            return titles[:MAX_PUBLICATION_SAMPLES]
        except Exception as e:
            logger.debug(f"Failed to fetch Scopus sample publications via scopus.fetch for {author_id}: {e}")
            return []
        
    def _calculate_scopus_confidence(self, query_name: str, result_name: str, 
                                    query_aff: Optional[str], result_aff: Optional[str], doc_count: int) -> float:
        """Calculate confidence score with emphasis on name matching, minimal affiliation weight."""
        name_score = fuzz.ratio(query_name.lower(), result_name.lower()) / 100.0
        
        if name_score < MIN_NAME_SIMILARITY:
            return 0.0
            
        # Heavy emphasis on name matching (90% weight)
        confidence = name_score * 0.9
        
        # Minimal affiliation weight (5%) since Scopus often has partial/department info
        if query_aff and result_aff:
            affil_score = fuzz.partial_ratio(query_aff.lower(), result_aff.lower()) / 100.0
            confidence += affil_score * 0.05
        
        # Document count bonus (up to 5%)
        confidence += min(0.05, doc_count / 100.0)

        # High baseline for strong name matches when no affiliation provided
        if (not query_aff or not str(query_aff).strip()) and name_score >= MIN_NAME_SIMILARITY:
            confidence = max(confidence, 0.85)
        
        return min(1.0, confidence)

class WOSDiscovery(BaseDiscovery):
    """Discover Web of Science ResearcherIDs via the Starter (documents) API."""
    BASE_URL = "https://api.clarivate.com/apis/wos-starter/v1/documents"

    def __init__(self, api_key: str):
        self.api_key = api_key

    # ---------- public ----------
    def search_profiles(self, first_name: str, last_name: str, affiliation: str = None) -> List[ProfileCandidate]:
        if not self.api_key:
            return []

        full_name = f"{first_name} {last_name}".strip()
        headers = {"X-ApiKey": self.api_key, "Accept": "application/json", "User-Agent": DEFAULT_USER_AGENT}

        # Query variations: with and without affiliation (try a few syntaxes)
        q_variants: List[tuple[str, str]] = []
        if affiliation:
            q_variants.extend([
                (f'AU=("{last_name}, {first_name}") AND AD=("{affiliation}")', "with affiliation (AD)"),
                (f'AU=("{last_name}, {first_name}") AND OG=("{affiliation}")', "with affiliation (OG)"),
            ])
        q_variants.append((f'AU=("{last_name}, {first_name}")', "name only"))

        # Collect hits across variations until we have “enough”
        seen_uids = set()
        all_hits: List[dict] = []
        for q, label in q_variants:
            hits = self._collect_hits(headers, q, limit_per_page=50, max_total=200)
            # Deduplicate by WoS UID across variants
            new_hits = [h for h in hits if h.get("uid") not in seen_uids]
            for h in new_hits:
                uid = h.get("uid")
                if uid:
                    seen_uids.add(uid)
            all_hits.extend(new_hits)

            # Early stop if we already have a good number of hits
            if len(all_hits) >= 120:
                break

        # Mine ResearcherIDs from authors, aggregate by ResearcherID
        candidates = self._extract_candidates_from_hits(all_hits, full_name, affiliation)

        # Final dedupe/sort using common helper
        return self._dedupe_and_sort_candidates(candidates)

    # ---------- internals ----------
    def _collect_hits(self, headers: Dict[str, str], q: str, *, limit_per_page: int = 50, max_total: int = 200) -> List[dict]:
        """Paginate Starter API results for a given query with simple retry/backoff."""
        all_hits: List[dict] = []
        page = 1
        backoff = 1.5

        while len(all_hits) < max_total:
            remaining = max_total - len(all_hits)
            limit = max(1, min(limit_per_page, remaining))
            params = {"q": q, "db": "WOS", "limit": limit, "page": page}

            try:
                resp = requests.get(self.BASE_URL, headers=headers, params=params, timeout=(5, 30))
            except requests.RequestException:
                time.sleep(backoff)
                backoff = min(30.0, backoff * 1.7)
                break

            if resp.status_code == 200:
                data = resp.json() or {}
                hits = data.get("hits", []) or []
                if not hits:
                    break
                all_hits.extend(hits)
                if len(hits) < limit:
                    break  # last page
                page += 1
                time.sleep(0.25)
                continue

            # Handle common transient / auth issues gracefully
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(30.0, backoff * 1.7)
                continue
            else:
                # Non-retryable
                break

        return all_hits

    def _normalize_name_for_wos_comparison(self, first_name: str, last_name: str) -> str:
        """Convert input name to WOS format: 'Last, F' or 'Last, FI' for comparison."""
        first_clean = first_name.strip()
        last_clean = last_name.strip()
        
        if not first_clean or not last_clean:
            return f"{last_clean}, {first_clean}"
        
        # Create variations: "Last, F" and "Last, FI" (first + middle initial if available)
        initials = first_clean[0].upper()
        if len(first_clean.split()) > 1:
            # Handle middle names/initials
            parts = first_clean.split()
            if len(parts[1]) == 1 or (len(parts[1]) == 2 and parts[1].endswith('.')):
                # Already an initial
                initials += parts[1].replace('.', '').upper()
            else:
                # Full middle name, take first letter
                initials += parts[1][0].upper()
        
        return f"{last_clean}, {initials}"

    def _matches_target_researcher(self, wos_author_name: str, target_first: str, target_last: str) -> bool:
        """Check if WOS author name matches our target researcher using intelligent fuzzy matching."""
        if not wos_author_name or not target_first or not target_last:
            return False
        
        wos_name = wos_author_name.strip()
        target_normalized = self._normalize_name_for_wos_comparison(target_first, target_last)
        
        # Direct match check
        if wos_name.upper() == target_normalized.upper():
            return True
        
        # Parse WOS name format: "Last, Initials"
        if ',' not in wos_name:
            return False
        
        wos_parts = wos_name.split(',', 1)
        if len(wos_parts) != 2:
            return False
        
        wos_last = wos_parts[0].strip()
        wos_initials = wos_parts[1].strip().replace('.', '').replace(' ', '').upper()
        
        target_last_clean = target_last.strip()
        target_first_clean = target_first.strip()
        
        # Last name must match closely (high threshold for last names)
        last_name_score = fuzz.ratio(wos_last.lower(), target_last_clean.lower()) / 100.0
        if last_name_score < 0.9:  # Very strict on last name matching
            return False
        
        # Check if initials match
        target_initials = target_first_clean[0].upper()
        if len(target_first_clean.split()) > 1:
            # Add middle initial if available
            middle_part = target_first_clean.split()[1]
            if middle_part:
                target_initials += middle_part[0].upper()
        
        # Initials matching logic
        if not wos_initials or not target_initials:
            return last_name_score >= 0.95  # Very high bar if no initials
        
        # Check if WOS initials start with our target initials
        if target_initials.startswith(wos_initials) or wos_initials.startswith(target_initials):
            return True
        
        # Check if first initial matches at minimum
        if wos_initials[0] == target_initials[0]:
            return True
        
        return False

    # NEW: WOS-specific confidence scoring that rewards "Last, Initial(s)" matches and minimizes affiliation reliance.
    def _calculate_wos_confidence(
        self,
        wos_author_name: str,
        target_first: str,
        target_last: str,
        query_aff: Optional[str],
        rec_aff_text: Optional[str]
    ) -> float:
        """
        Compute a confidence score tailored for WOS author names which use 'Last, Initials' format
        and often lack stable profile-level affiliation. Relies primarily on:
          - strict last-name match,
          - initials match quality,
          - tiny optional bump for coarse record-level affiliation overlap.
        """
        try:
            # Early gate using our existing name guard
            if not self._matches_target_researcher(wos_author_name, target_first, target_last):
                return 0.0

            # Parse WOS name format: "Last, Initials"
            parts = wos_author_name.split(',', 1)
            wos_last = parts[0].strip() if len(parts) == 2 else wos_author_name.strip()
            wos_initials = parts[1].strip().replace('.', '').replace(' ', '').upper() if len(parts) == 2 else ""

            tgt_first = (target_first or "").strip()
            tgt_last = (target_last or "").strip()
            if not tgt_last or not tgt_first:
                return 0.0

            # Last-name similarity (very strict)
            last_sim = fuzz.ratio(wos_last.lower(), tgt_last.lower()) / 100.0
            if last_sim < 0.9:
                return 0.0

            # Build target initials (first + optional middle)
            t_parts = tgt_first.split()
            target_initials = (t_parts[0][0] if t_parts and t_parts[0] else "").upper()
            if len(t_parts) > 1 and t_parts[1]:
                target_initials += t_parts[1][0].upper()

            # Grade initials match quality
            base = 0.0
            if wos_initials and target_initials:
                if wos_initials == target_initials:
                    base = 0.92
                elif wos_initials.startswith(target_initials) or target_initials.startswith(wos_initials):
                    base = 0.90
                elif wos_initials[0] == target_initials[0]:
                    base = 0.86
                else:
                    base = 0.0
            else:
                # If initials unavailable on either side, rely on strong last name only
                base = 0.85 if last_sim >= 0.95 else 0.82

            # Tiny, optional bump for affiliation overlap (record-level, very noisy)
            bump = 0.0
            if query_aff and rec_aff_text:
                aff_score = max(
                    fuzz.partial_ratio(query_aff.lower(), rec_aff_text.lower()) / 100.0,
                    fuzz.token_set_ratio(query_aff.lower(), rec_aff_text.lower()) / 100.0
                )
                if aff_score >= 0.7:
                    bump += 0.03
                elif aff_score >= 0.5:
                    bump += 0.015

            return min(1.0, base + bump)
        except Exception:
            return 0.0

    # NEW: modest confidence boost based on how many documents we saw for this RID.
    def _boost_confidence_with_count(self, conf: float, doc_count: int) -> float:
        try:
            import math
            if doc_count <= 0:
                return conf
            # Log-scale bump up to ~0.08 for prolific presence across hits
            bump = min(0.08, math.log10(doc_count + 1) * 0.05)
            return min(1.0, conf + bump)
        except Exception:
            return conf

    def _extract_candidates_from_hits(self, hits: List[dict], query_name: str, query_aff: Optional[str]) -> List[ProfileCandidate]:
        """
        Build candidates keyed by ResearcherID. Confidence is based on name
        similarity (+ optional coarse affiliation signal from record-level addresses).
        Now includes intelligent name filtering for WOS format.
        """
        # Parse target name for filtering
        name_parts = query_name.split()
        if len(name_parts) >= 2:
            target_first = name_parts[0]
            target_last = " ".join(name_parts[1:])
        else:
            target_first = query_name
            target_last = ""

        # Coarse record-level affiliation text for confidence (best-effort)
        def record_aff_text(rec: dict) -> Optional[str]:
            try:
                addrs = rec.get("addresses", {})
                orgs = []
                # Starter responses may have "organizations" with "displayName"
                for a in addrs.get("organizations", []) or []:
                    dn = a.get("displayName") or a.get("name")
                    if dn:
                        orgs.append(dn)
                if not orgs:
                    # Sometimes it's nested differently
                    for addr in addrs.get("addresses", []) or []:
                        for org in addr.get("organizations", []) or []:
                            dn = org.get("displayName") or org.get("name")
                            if dn:
                                orgs.append(dn)
                if orgs:
                    # join unique org names
                    uniq = list(dict.fromkeys(orgs))
                    return " ; ".join(uniq[:5])
            except Exception:
                pass
            return None

        # Aggregate per ResearcherID
        by_rid: Dict[str, Dict] = {}
        for rec in hits:
            authors = (rec.get("names") or {}).get("authors") or []
            rec_aff_text = record_aff_text(rec)
            title = rec.get("title") or ""

            for a in authors:
                name = (a.get("wosStandard") or a.get("displayName") or "").strip()
                rid = (a.get("researcherId") or "").strip()
                orcid = (a.get("orcidId") or "").strip() or None

                if not rid or not name:
                    continue

                # NEW: filter and score using WOS-specific confidence
                if not self._matches_target_researcher(name, target_first, target_last):
                    continue
                conf = self._calculate_wos_confidence(name, target_first, target_last, query_aff, rec_aff_text)

                slot = by_rid.get(rid)
                if not slot:
                    slot = {
                        "rid": rid,
                        "name": name,
                        "best_aff": rec_aff_text,
                        "best_conf": conf,
                        "titles": [],
                        "count": 0,
                        "orcids": set(),
                    }
                    by_rid[rid] = slot

                # Aggregate
                slot["count"] += 1
                if title:
                    if len(slot["titles"]) < MAX_PUBLICATION_SAMPLES:
                        slot["titles"].append(title)
                if orcid:
                    slot["orcids"].add(orcid)
                if conf > slot["best_conf"]:
                    slot["best_conf"] = conf
                if not slot["best_aff"] and rec_aff_text:
                    slot["best_aff"] = rec_aff_text

        # Convert to ProfileCandidate list
        candidates: List[ProfileCandidate] = []
        for rid, slot in by_rid.items():
            profile_url = f"https://www.webofscience.com/wos/author/record/{rid}"
            # NEW: apply doc-count-based boost at the end
            boosted_conf = self._boost_confidence_with_count(slot["best_conf"], slot["count"])
            candidate = ProfileCandidate(
                source="Web of Science",
                profile_id=rid,
                name=slot["name"],
                affiliation=slot["best_aff"],
                confidence_score=min(1.0, boosted_conf),
                sample_publications=slot["titles"],
                profile_url=profile_url,
                verification_info={
                    "orcid_ids": sorted(slot["orcids"]) if slot["orcids"] else None,
                    "notes": "Discovered via Starter API documents; profile URL constructed from ResearcherID."
                },
                publication_count=slot["count"],
            )
            candidates.append(candidate)

        candidates.sort(key=lambda c: (c.confidence_score, c.publication_count or 0), reverse=True)
        return candidates

# Main Service
class ProfileDiscoveryService:
    """Main service for discovering researcher profiles across all sources."""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        self.sources = {}
        self._gs_driver = None  # Store Google Scholar driver for reuse
        
        self._initialize_sources()
    
    def _initialize_sources(self):
        """Initialize all available discovery sources in optimized order."""
        # Reorder sources for efficiency: fast APIs first, then scraping, WOS last
        self.sources["ORCID"] = ORCIDDiscovery()
        
        if self.api_keys.get('scopus_api_key'):
            self.sources["Scopus"] = ScopusDiscovery(self.api_keys['scopus_api_key'])
        
        if SCRAPING_DEPS_AVAILABLE:
            self.sources["Google Scholar"] = GoogleScholarDiscovery(self.api_keys)
        else:
            logger.warning("Google Scholar scraping disabled (dependencies not found).")
        
        # Move WOS last since it often returns too many low-quality candidates
        if self.api_keys.get('wos_api_key'):
            self.sources["Web of Science"] = WOSDiscovery(self.api_keys['wos_api_key'])
    
    def discover_all_profiles(self, first_name: str, last_name: str, 
                            affiliation: str = None,
                            on_progress: Optional[Callable[[float, str], None]] = None) -> DiscoveryResult:
        """Discover profiles from all available sources - always search all sources, reporting progress."""
        def _progress(p: float, msg: str):
            try:
                if on_progress:
                    on_progress(max(0.0, min(100.0, float(p))), str(msg))
            except Exception:
                pass

        all_candidates, errors = [], []

        total_sources = max(1, len(self.sources))
        completed = 0
        _progress(2, "Initializing discovery...")

        for name, service in self.sources.items():
            try:
                _progress(5 + (65 * (completed / total_sources)), f"Searching {name}...")
                candidates = service.search_profiles(first_name, last_name, affiliation)
                
                # Store Google Scholar driver for potential reuse
                if name == "Google Scholar" and hasattr(service, '_last_driver'):
                    self._gs_driver = service._last_driver
                    service._last_driver = None  # Transfer ownership
                
                # Filter WOS candidates to reduce noise
                if name == "Web of Science":
                    # Include high-confidence OR strong last-name matches (>= 0.88)
                    candidates = [
                        c for c in candidates
                        if (c.confidence_score > 0.8) or self._wos_last_name_match(c, last_name, threshold=0.88)
                    ]
                    # Keep top few for brevity
                    candidates = candidates[:5]
                    logger.info(f"Filtered to {len(candidates)} WOS candidates")
                
                all_candidates.extend(candidates)
                logger.info(f"Found {len(candidates)} candidate(s) from {name}")
                
            except Exception as e:
                error_msg = f"{name} discovery failed: {e}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)
            finally:
                completed += 1
                _progress(5 + (65 * (completed / total_sources)), f"Finished {name} ({completed}/{total_sources})")
        
        _progress(75, "Aggregating results...")
        all_candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        return DiscoveryResult(
            success=not errors,
            candidates=all_candidates,
            error="; ".join(errors) if errors else None
        )
    
    def verify_profile(self, candidate: ProfileCandidate, max_publications: int = 3) -> ProfileCandidate:
        """Verify a profile by fetching sample publications, avoiding redundant API calls."""
        max_pubs = min(MAX_PUBLICATION_SAMPLES, max(1, int(max_publications or 0)))
        
        # Skip verification if we already have sample publications from discovery
        if candidate.sample_publications and len(candidate.sample_publications) >= max_pubs:
            logger.info(f"Skipping verification for {candidate.source} {candidate.profile_id} - already has {len(candidate.sample_publications)} samples")
            candidate.verification_info['verification_skipped'] = 'already_has_samples'
            return candidate

        # NEW: For Google Scholar, reuse the shared driver from discovery to avoid spawning a new one.
        if candidate.source == "Google Scholar":
            titles = self._fetch_gs_publications_via_shared_driver(candidate.profile_id, max_pubs)
            if titles:
                candidate.sample_publications = titles[:max_pubs]
                candidate.verification_info['verified_at'] = time.time()
                candidate.verification_info['sample_count'] = len(candidate.sample_publications)
                candidate.verification_info['gs_fetch_method'] = 'shared_driver'
            else:
                # Do not create a new driver here; keep a note for transparency.
                candidate.verification_info['verification_skipped'] = 'gs_shared_driver_unavailable_or_failed'
            return candidate

        fetch_map = {
            "ORCID": (orcid.fetch, {
                'orcid_id': candidate.profile_id,
                'max_records': max_pubs,
                'max_results': max_pubs,
                'limit': max_pubs,
            }),
            "Scopus": (scopus.fetch, {
                'scopus_id': candidate.profile_id,
                'api_key': self.api_keys.get('scopus_api_key'),
                'page_batch': max_pubs,
            }),
        }
        if candidate.source not in fetch_map:
            return candidate

        fetch_func, kwargs = fetch_map[candidate.source]
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            publications = self._invoke_with_supported_kwargs(fetch_func, kwargs)
            if isinstance(publications, list):
                pub_objs = publications[:max_pubs]
            else:
                pub_objs = list(itertools.islice(publications, max_pubs))
            candidate.sample_publications = [getattr(pub, "title", str(pub)) for pub in pub_objs]
            candidate.verification_info['verified_at'] = time.time()
            candidate.verification_info['sample_count'] = len(candidate.sample_publications)
        except Exception as e:
            logger.warning(f"Profile verification failed for {candidate.source} {candidate.profile_id}: {e}")

        return candidate

    # NEW: Fetch Google Scholar publications using the shared driver from discovery
    def _fetch_gs_publications_via_shared_driver(self, profile_id: str, max_pubs: int = 3) -> List[str]:
        # Only proceed if Selenium stack is available and a driver is present
        if not SCRAPING_DEPS_AVAILABLE or not self._gs_driver:
            return []
        driver = self._gs_driver
        try:
            # Simple responsiveness check
            try:
                _ = driver.current_url
            except Exception:
                return []

            # Only first page of works; no pagination, no "Show more" clicks
            url = (
                f"https://scholar.google.com/citations?"
                f"user={profile_id}&hl=en&view_op=list_works&sortby=pubdate&cstart=0&pagesize=20"
            )
            if url not in (driver.current_url or ""):
                driver.get(url)

            # Quick CAPTCHA detection
            try:
                cur = (driver.current_url or "").lower()
                title = (driver.title or "").lower()
                if any(p in cur or p in title for p in ['recaptcha', '/sorry/', 'captcha', 'unusual traffic']):
                    return []
            except Exception:
                pass

            # Wait for publication rows to render and scrape titles (first page only)
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "tr.gsc_a_tr"))
                )
            except Exception:
                time.sleep(1)

            elems = driver.find_elements(By.CSS_SELECTOR, "tr.gsc_a_tr .gsc_a_t a")
            titles: List[str] = []
            for el in elems:
                t = (el.text or "").strip()
                if t:
                    titles.append(t)
                if len(titles) >= max_pubs:
                    break
            return titles[:max_pubs]
        except Exception as e:
            logger.debug(f"GS shared driver fetch failed for {profile_id}: {e}")
            return []

    def __del__(self):
        """Cleanup Google Scholar driver on service destruction."""
        if self._gs_driver:
            try:
                self._gs_driver.quit()
            except Exception:
                pass

    def _invoke_with_supported_kwargs(self, func, kwargs: Dict):
        """Call func with only the kwargs it supports to avoid unexpected-kwarg errors."""
        try:
            sig = inspect.signature(func)
            params = sig.parameters
            # If function accepts **kwargs, pass everything
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                return func(**kwargs)
            allowed = {k: v for k, v in kwargs.items() if k in params}
            return func(**allowed)
        except (ValueError, TypeError):
            # Fallback for callables without inspectable signature
            return func(**kwargs)
        
    # NEW: helper to allow WOS last-name-based inclusion
    def _wos_last_name_match(self, candidate: ProfileCandidate, target_last: str, threshold: float = 0.88) -> bool:
        try:
            if not candidate or not candidate.name or not target_last:
                return False
            cand_last = self._extract_last_name(candidate.name)
            if not cand_last:
                return False
            sim = fuzz.ratio(cand_last.lower(), target_last.lower()) / 100.0
            return sim >= threshold
        except Exception:
            return False

    def _extract_last_name(self, display_name: str) -> str:
        """Best-effort last-name extraction from 'Last, FI' or 'First Middle Last'."""
        name = (display_name or "").strip()
        if not name:
            return ""
        if ',' in name:
            return name.split(',', 1)[0].strip()
        parts = name.split()
        return parts[-1].strip() if parts else ""

def discover_researcher_profiles(
    first_name: str, 
    last_name: str, 
    affiliation: str = None, 
    api_keys: Dict[str, str] = None, 
    verify_profiles: bool = False,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> DiscoveryResult:
    """
    Discover researcher profiles across all academic databases.

    Args:
        first_name: Researcher's first name
        last_name: Researcher's last name
        affiliation: Optional institutional affiliation
        api_keys: Dictionary of API keys for various services
        verify_profiles: Whether to fetch sample publications for verification (up to 3)
    """
    def _progress(p: float, msg: str):
        try:
            if on_progress:
                on_progress(max(0.0, min(100.0, float(p))), str(msg))
        except Exception:
            pass

    service = ProfileDiscoveryService(api_keys)
    result = service.discover_all_profiles(first_name, last_name, affiliation, on_progress=_progress)

    if verify_profiles and result.success:
        # Verify top candidates only, prioritizing highest confidence scores
        # Limit to top 5 to avoid excessive verification calls
        top_candidates = sorted(result.candidates, key=lambda x: x.confidence_score, reverse=True)[:5]
        total = max(1, len(top_candidates))
        for i, candidate in enumerate(top_candidates):
            # Find the original candidate in the list and update it
            _progress(80 + (20 * ((i) / total)), f"Verifying candidates ({i+1}/{total})...")
            for j, orig_candidate in enumerate(result.candidates):
                if (orig_candidate.source == candidate.source and 
                    orig_candidate.profile_id == candidate.profile_id):
                    result.candidates[j] = service.verify_profile(candidate, MAX_PUBLICATION_SAMPLES)
                    break
        _progress(100, "Discovery complete")

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
               api_keys= api_keys,
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
