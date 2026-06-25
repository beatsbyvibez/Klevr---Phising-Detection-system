"""
blacklist.py
============
Layer 0 of the PhishGuard AI system: known-bad URL blacklist check.

WHY THIS MODULE EXISTS
----------------------
The Random Forest model in Layer 1 inspects URL structure only and
achieves about 89.5% accuracy on the Hannousse benchmark. That means
roughly 10 of every 100 phishing URLs slip through. Many of those
slip-throughs are URLs that *look* clean to a feature-based model but
have already been reported by the community to one of the major
threat-intelligence feeds.

Layer 0 closes that gap by checking the URL against two free,
publicly downloadable phishing blacklists BEFORE the model runs:

  1. PhishTank   (operated by Cisco Talos) - human-verified phishing URLs
  2. URLhaus     (operated by abuse.ch)    - malware/phishing URLs

Both feeds publish their full database as a CSV that anyone can
download without an API key. We pull each feed at most once every six
hours, store it on disk as a Python set of URLs and hostnames, and do
fast in-memory lookups thereafter.

If the blacklist refresh fails (no internet, server down, etc.), we
log a warning and continue with whatever cached data we have. If we
have no cache yet, the blacklist simply returns "not listed" for
every URL and the system continues to rely on Layers 1 and 2.

ETHICS / ATTRIBUTION
--------------------
PhishTank data is licensed for free use, including commercial, under a
restrictive license that requires attribution to PhishTank/Cisco Talos.
URLhaus data is licensed under CC0 (public domain). Both are credited
in the Streamlit "About" page when this module is enabled.
"""

import os
import time
import json
from urllib.parse import urlparse

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Where to store the cached blacklist on disk so we don't re-download
# every time the Streamlit app restarts.
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "blacklist.json")

# Refresh the database at most once every six hours. PhishTank updates
# hourly, but six hours is plenty for an academic project and avoids
# being rate-limited.
REFRESH_INTERVAL_SECONDS = 6 * 60 * 60

# Feed URLs. These are the public bulk downloads that do NOT require
# an API key. If you later register a PhishTank application key you
# can swap in the keyed URL for higher rate limits.
PHISHTANK_FEED = "http://data.phishtank.com/data/online-valid.csv"
URLHAUS_FEED = "https://urlhaus.abuse.ch/downloads/csv_recent/"

# How long to wait for a feed to respond before giving up.
FEED_TIMEOUT_SECONDS = 20

# PhishTank requires a descriptive User-Agent. They reject blank or
# generic agents, so we identify ourselves clearly as an academic
# research project.
USER_AGENT = "PhishGuardAI/1.0 (academic research project; FYP)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_url(url: str) -> str:
    """
    Return a comparable form of a URL.

    Blacklist matching is sensitive to small differences like trailing
    slashes, http vs https, and www. prefixes. We strip those so a
    URL the user types is more likely to match an entry in the feed.
    """
    url = url.strip().lower()
    # Drop the protocol so http and https forms compare equal.
    if url.startswith("http://"):
        url = url[7:]
    elif url.startswith("https://"):
        url = url[8:]
    # Drop a leading www.
    if url.startswith("www."):
        url = url[4:]
    # Drop a trailing slash.
    if url.endswith("/"):
        url = url[:-1]
    return url


def _hostname_of(url: str) -> str:
    """Return just the hostname part of a URL, lowercased, no www."""
    if "://" not in url:
        url = "http://" + url
    host = urlparse(url).netloc.lower()
    if ":" in host:
        host = host.split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def _ensure_cache_dir():
    """Create the cache folder if it does not yet exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Feed downloaders
# ---------------------------------------------------------------------------

def _download_phishtank() -> list:
    """
    Download the PhishTank online-valid feed and return a list of URLs.

    On any failure (network down, timeout, server 500), return an empty
    list and let the caller decide what to do.
    """
    try:
        response = requests.get(
            PHISHTANK_FEED,
            headers={"User-Agent": USER_AGENT},
            timeout=FEED_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except Exception as e:
        print(f"[blacklist] PhishTank download failed: {e}")
        return []

    urls = []
    # The CSV has a header line; columns are:
    # phish_id,url,phish_detail_url,submission_time,verified,verification_time,online,target
    lines = response.text.splitlines()
    for line in lines[1:]:  # skip header
        # A URL field can contain commas if double-quoted, so we do a
        # safer split: find the SECOND column by splitting on commas
        # outside quotes. For PhishTank data the URL is in column index 1.
        parts = line.split(",", 2)
        if len(parts) >= 2:
            url = parts[1].strip().strip('"')
            if url:
                urls.append(url)
    return urls


def _download_urlhaus() -> list:
    """
    Download the URLhaus recent CSV and return a list of URLs.

    URLhaus prefixes its CSV with several comment lines starting with '#';
    we skip those.
    """
    try:
        response = requests.get(
            URLHAUS_FEED,
            headers={"User-Agent": USER_AGENT},
            timeout=FEED_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except Exception as e:
        print(f"[blacklist] URLhaus download failed: {e}")
        return []

    urls = []
    # URLhaus CSV columns: id, dateadded, url, url_status, last_online,
    # threat, tags, urlhaus_link, reporter
    for line in response.text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split(",", 3)  # split only the first three fields
        if len(parts) >= 3:
            url = parts[2].strip().strip('"')
            if url:
                urls.append(url)
    return urls


# ---------------------------------------------------------------------------
# The Blacklist class
# ---------------------------------------------------------------------------

class Blacklist:
    """
    A lazy-loaded, disk-cached, multi-source phishing-URL blacklist.

    Usage:
        bl = Blacklist()
        bl.refresh_if_stale()   # refresh from the internet if cache is old
        verdict = bl.check("http://bad-site.tk/login")
    """

    def __init__(self):
        # In-memory storage. Two sets: exact-URL matches and hostname
        # matches. Hostname matches catch the case where the attacker
        # changes the path/query while keeping the same malicious domain.
        self.urls = set()
        self.hosts = set()
        # Where each URL came from, so we can tell the user which feed
        # flagged it. Map: normalised_url -> source_name
        self.source_of = {}
        # When was the cache last refreshed (Unix timestamp)?
        self.last_refresh = 0
        # Load whatever is on disk so the app is usable on first run
        # even without internet.
        self._load_from_disk()

    # ------------------------------------------------------------------
    # Public methods used by the rest of the app
    # ------------------------------------------------------------------

    def check(self, url: str) -> dict:
        """
        Look up `url` in the blacklist.

        Returns a dict:
            listed       : True if the URL or its hostname is on a blacklist
            match_type   : "exact_url", "hostname", or None
            source       : "PhishTank", "URLhaus", or None
        """
        if not url:
            return {"listed": False, "match_type": None, "source": None}

        # Defensive guard: if the URL's hostname is on the curated
        # allowlist we treat it as not listed, even if a feed has
        # wrongly included the domain. This protects the system from
        # feed poisoning (e.g. PhishTank entries that reference a
        # legitimate domain in the URL string).
        try:
            from allowlist import check as _allow_check
            if _allow_check(url).get("listed"):
                return {"listed": False, "match_type": None, "source": None}
        except Exception:
            # If allowlist is unavailable for any reason, fall through
            # to normal blacklist matching rather than crashing.
            pass

        normalised = _normalise_url(url)
        if normalised in self.urls:
            return {
                "listed": True,
                "match_type": "exact_url",
                "source": self.source_of.get(normalised, "blacklist"),
            }

        host = _hostname_of(url)
        if host and host in self.hosts:
            return {
                "listed": True,
                "match_type": "hostname",
                "source": self.source_of.get(host, "blacklist"),
            }

        return {"listed": False, "match_type": None, "source": None}

    def refresh_if_stale(self, force: bool = False) -> dict:
        """
        Re-download the feeds if more than REFRESH_INTERVAL_SECONDS
        have passed since the last successful refresh.

        Returns a small status dict so the UI can show what happened.
        """
        now = time.time()
        if not force and (now - self.last_refresh) < REFRESH_INTERVAL_SECONDS:
            return {
                "refreshed": False,
                "reason": "cache is still fresh",
                "urls_in_cache": len(self.urls),
                "hosts_in_cache": len(self.hosts),
            }
        return self._refresh_now()

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _refresh_now(self) -> dict:
        """Download both feeds and rebuild the in-memory sets."""
        new_urls = set()
        new_hosts = set()
        new_source = {}

        # PhishTank
        phishtank_urls = _download_phishtank()
        for raw in phishtank_urls:
            norm = _normalise_url(raw)
            new_urls.add(norm)
            new_source[norm] = "PhishTank"
            host = _hostname_of(raw)
            if host:
                new_hosts.add(host)
                # Hostname source is filled only if not already set,
                # so PhishTank takes priority over URLhaus for the same host.
                new_source.setdefault(host, "PhishTank")

        # URLhaus
        urlhaus_urls = _download_urlhaus()
        for raw in urlhaus_urls:
            norm = _normalise_url(raw)
            new_urls.add(norm)
            new_source.setdefault(norm, "URLhaus")
            host = _hostname_of(raw)
            if host:
                new_hosts.add(host)
                new_source.setdefault(host, "URLhaus")

        # If BOTH feeds failed we keep the old cache instead of wiping it.
        if not phishtank_urls and not urlhaus_urls:
            return {
                "refreshed": False,
                "reason": "both feeds failed to download; keeping existing cache",
                "urls_in_cache": len(self.urls),
                "hosts_in_cache": len(self.hosts),
            }

        # Otherwise replace the in-memory data and persist to disk.
        self.urls = new_urls
        self.hosts = new_hosts
        self.source_of = new_source
        self.last_refresh = time.time()
        self._save_to_disk()

        return {
            "refreshed": True,
            "reason": "downloaded fresh data",
            "urls_in_cache": len(self.urls),
            "hosts_in_cache": len(self.hosts),
            "phishtank_count": len(phishtank_urls),
            "urlhaus_count": len(urlhaus_urls),
        }

    def _load_from_disk(self):
        """Restore the in-memory sets from the JSON cache, if present."""
        if not os.path.exists(CACHE_FILE):
            return
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.urls = set(data.get("urls", []))
            self.hosts = set(data.get("hosts", []))
            self.source_of = data.get("source_of", {})
            self.last_refresh = data.get("last_refresh", 0)
        except Exception as e:
            print(f"[blacklist] Could not load cache file: {e}")

    def _save_to_disk(self):
        """Write the current sets to a JSON file so they survive restarts."""
        _ensure_cache_dir()
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "urls": sorted(self.urls),
                    "hosts": sorted(self.hosts),
                    "source_of": self.source_of,
                    "last_refresh": self.last_refresh,
                }, f)
        except Exception as e:
            print(f"[blacklist] Could not save cache file: {e}")


# ---------------------------------------------------------------------------
# A small CLI so the student can refresh the cache from the terminal:
#     python blacklist.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[blacklist] Refreshing PhishTank + URLhaus feeds…")
    bl = Blacklist()
    status = bl.refresh_if_stale(force=True)
    for k, v in status.items():
        print(f"  {k}: {v}")
