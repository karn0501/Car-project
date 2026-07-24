"""
Phase 9/10: B2B API Key Metering & Quota Tracking Module.
Provides multi-tiered API key validation, quota management, and real-time usage metrics.
Tiers:
- FREE: 60 requests / min
- PRO: 500 requests / min
- ENTERPRISE: Unlimited
"""

import os
import json
import time
from typing import Dict, Any, Optional


USAGE_FILE = os.path.abspath("models/api_usage.json")


# Pre-configured B2B API Keys
API_KEYS_DB = {
    "car-prediction-api-key-2026": {"tier": "FREE", "limit_per_min": 60, "owner": "Public Demo"},
    "pro-key-dealer-998877": {"tier": "PRO", "limit_per_min": 500, "owner": "CarDekho Partner"},
    "enterprise-key-bank-112233": {"tier": "ENTERPRISE", "limit_per_min": 10000, "owner": "HDFC Auto Loans"}
}


class APIMeteringManager:
    """
    Manages API key validation, tier-based rate limits, and request volume statistics.
    """

    def __init__(self):
        self.usage_data = self._load_usage()

    def _load_usage(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(USAGE_FILE):
            try:
                with open(USAGE_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_usage(self):
        try:
            os.makedirs(os.path.dirname(USAGE_FILE), exist_ok=True)
            with open(USAGE_FILE, "w") as f:
                json.dump(self.usage_data, f, indent=2)
        except Exception:
            pass

    def validate_and_record(self, api_key: str) -> Dict[str, Any]:
        """
        Validates API key, checks minute quota, and records usage.
        """
        key_info = API_KEYS_DB.get(api_key)
        if not key_info:
            return {"valid": False, "reason": "Invalid API key"}

        now = time.time()
        current_minute = int(now // 60)

        if api_key not in self.usage_data:
            self.usage_data[api_key] = {
                "total_requests": 0,
                "tier": key_info["tier"],
                "owner": key_info["owner"],
                "minute_window": current_minute,
                "minute_requests": 0
            }

        data = self.usage_data[api_key]

        # Reset minute window if new minute
        if data["minute_window"] != current_minute:
            data["minute_window"] = current_minute
            data["minute_requests"] = 0

        # Check quota
        limit = key_info["limit_per_min"]
        if data["minute_requests"] >= limit:
            return {
                "valid": False,
                "reason": f"Rate limit exceeded for tier {key_info['tier']} ({limit} req/min)",
                "limit": limit,
                "used": data["minute_requests"]
            }

        # Increment counters
        data["minute_requests"] += 1
        data["total_requests"] += 1
        self._save_usage()

        return {
            "valid": True,
            "tier": key_info["tier"],
            "owner": key_info["owner"],
            "total_requests": data["total_requests"],
            "remaining_this_minute": limit - data["minute_requests"]
        }

    def get_key_metrics(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Returns current usage metrics for an API key."""
        key_info = API_KEYS_DB.get(api_key)
        if not key_info:
            return None

        usage = self.usage_data.get(api_key, {"total_requests": 0, "minute_requests": 0})
        return {
            "api_key": api_key[:8] + "...",
            "tier": key_info["tier"],
            "owner": key_info["owner"],
            "limit_per_minute": key_info["limit_per_min"],
            "used_this_minute": usage.get("minute_requests", 0),
            "total_lifetime_requests": usage.get("total_requests", 0)
        }
