"""Domain availability checking tools.

Enable via config: tools.domain_checker_enabled: true
Requires env:     WHOISJSON_API_KEY  (falls back to local python-whois)
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import requests
import whois
from langchain_core.tools import tool


def _whoisjson_check(domain: str) -> Dict:
    api_key = os.getenv("WHOISJSON_API_KEY", "")
    url = f"https://whoisjson.com/api/v1/domain-availability?domain={domain}"
    try:
        res = requests.get(url, headers={"Authorization": f"Token={api_key}"}, timeout=5)
        data = res.json()
        if "available" in data:
            return {"available": data["available"], "source": "whoisjson"}
        if "result" in data and not data["result"]:
            return {"available": True, "source": "whoisjson"}
        return {"available": False, "source": "whoisjson"}
    except Exception as e:
        return {"error": str(e), "source": "whoisjson"}


def _local_whois_check(domain: str) -> Dict:
    try:
        w = whois.whois(domain)
        return {"available": not bool(w.domain_name), "source": "local_whois"}
    except Exception:
        return {"available": True, "source": "local_whois"}


def _query_domain(domain: str) -> str:
    result = _whoisjson_check(domain)
    if "error" not in result:
        status = "可註冊" if result["available"] else "已註冊"
        return f"{domain} → {status} (via {result['source']})"
    fallback = _local_whois_check(domain)
    status = "可註冊" if fallback["available"] else "已註冊"
    return f"{domain} → {status} (fallback local whois)"


def build_domain_tools() -> list:
    @tool
    def check_domain_available(domain: str) -> str:
        """Check if a domain name is available for registration.

        Input:  domain name (e.g. example.com)
        Output: available / registered + data source
        """
        return _query_domain(domain)

    @tool
    def check_domains_bulk(domains: list) -> str:
        """Check availability of multiple domain names in parallel."""
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_query_domain, domains))
        return "\n".join(results)

    return [check_domain_available, check_domains_bulk]
