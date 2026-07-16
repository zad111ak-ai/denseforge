"""Carbon-Aware Scheduler — defer compute to low-carbon periods."""
import time
from typing import Optional
from loguru import logger


class CarbonAwareScheduler:
    """Defer queries to low-carbon-intensity periods."""

    def __init__(self, carbon_api_key: Optional[str] = None, region: str = "EU"):
        self.carbon_api_key = carbon_api_key
        self.region = region
        self._total_queries = 0
        self._deferred_queries = 0
        self._intensity_cache: dict = {}

    def should_defer_query(self, estimated_wh: float = 0.5) -> dict:
        """Check if query should be deferred based on carbon intensity."""
        self._total_queries += 1
        intensity = self._get_current_intensity()

        if intensity > 500:  # gCO2/kWh — high carbon
            self._deferred_queries += 1
            return {
                "decision": "defer",
                "reason": f"High carbon intensity: {intensity} gCO2/kWh",
                "current_intensity": intensity,
                "suggested_time": self._estimate_low_carbon_time(),
            }
        return {"decision": "proceed", "current_intensity": intensity}

    def _get_current_intensity(self) -> float:
        """Get current carbon intensity (mock or API)."""
        if self.carbon_api_key:
            try:
                import httpx
                resp = httpx.get(
                    f"https://api.electricitymap.org/v3/carbon-intensity",
                    params={"zone": self.region},
                    headers={"auth-token": self.carbon_api_key},
                    timeout=10,
                )
                if resp.status_code == 200:
                    return resp.json().get("carbonIntensity", 300)
            except Exception:
                pass
        # Fallback: time-based heuristic (more green during night)
        hour = time.gmtime().tm_hour
        if 0 <= hour <= 6:
            return 200  # Night = lower
        elif 12 <= hour <= 18:
            return 450  # Day = higher
        return 350

    def _estimate_low_carbon_time(self) -> str:
        hour = time.gmtime().tm_hour
        if hour >= 18 or hour < 6:
            return "now"
        return "00:00-06:00 UTC"

    def generate_eu_ai_act_report(self) -> dict:
        return {
            "region": self.region,
            "total_queries": self._total_queries,
            "deferred_queries": self._deferred_queries,
            "deferral_rate": self._deferred_queries / max(self._total_queries, 1),
        }

    def stats(self) -> dict:
        return self.generate_eu_ai_act_report()
