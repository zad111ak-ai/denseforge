"""
DenseForge ↔ Harvest Integration Bridge

Optional connector: DenseForge can use Harvest as its "web fetcher"
for scraping, extraction, and compliance checking.

Architecture:
    DenseForge (standalone)  →  works alone, no Harvest needed
    DenseForge + Harvest     →  can auto-scrape URLs from queries
                                 compliance checks, Cloudflare bypass

Usage:
    # Auto-detect (if Harvest installed)
    from denseforge.integrations.harvest import HarvestBridge
    bridge = HarvestBridge()
    if bridge.available:
        result = await bridge.scrape("https://example.com")
        contacts = await bridge.contacts("https://company.com/about")
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class HarvestBridge:
    """Bridge between DenseForge and Harvest.

    Detects if Harvest is installed and provides a clean API
    for DenseForge to use Harvest's scraping capabilities.
    """

    def __init__(self):
        self._scraper = None
        self._available = False

        try:
            from harvest.core import Scraper

            self._scraper = Scraper(use_stealth=True)
            self._available = True
            logger.info("Harvest bridge: connected")
        except ImportError:
            logger.debug("Harvest not installed — bridge disabled")
        except Exception as e:
            logger.warning(f"Harvest bridge init failed: {e}")

    @property
    def available(self) -> bool:
        return self._available

    async def scrape(self, url: str, selector: Optional[str] = None,
                     extraction: str = "markdown") -> Optional[dict]:
        """Scrape a web page using Harvest."""
        if not self._available:
            return None
        try:
            return await self._scraper.scrape(url, selector=selector, extraction=extraction)
        except Exception as e:
            logger.error(f"Harvest scrape failed: {e}")
            return None

    async def contacts(self, url: str) -> Optional[dict]:
        """Extract contacts from a web page."""
        if not self._available:
            return None
        try:
            from harvest.contacts import ContactCollector
            return await ContactCollector().collect(url)
        except Exception as e:
            logger.error(f"Harvest contacts failed: {e}")
            return None

    async def detect_api(self, url: str) -> Optional[dict]:
        """Detect hidden APIs on a web page."""
        if not self._available:
            return None
        try:
            from harvest.api_detector import APIDetector
            return await APIDetector().detect(url)
        except Exception as e:
            logger.error(f"Harvest API detection failed: {e}")
            return None

    async def batch_scrape(self, urls: list[str],
                           selector: Optional[str] = None) -> Optional[list[dict]]:
        """Scrape multiple URLs in parallel."""
        if not self._available:
            return None
        try:
            import asyncio
            tasks = [self._scraper.scrape(u, selector=selector) for u in urls]
            return list(await asyncio.gather(*tasks, return_exceptions=False))
        except Exception as e:
            logger.error(f"Harvest batch failed: {e}")
            return None
