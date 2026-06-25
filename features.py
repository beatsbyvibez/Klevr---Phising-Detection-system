"""
features.py
===========
Academic-grade feature extraction for phishing URL detection.
Expands from 7 baseline features to 26 well-documented features
covering lexical, host-based, and structural URL characteristics.

Feature categories:
  1. Lexical / length-based   (URL, domain, path, query characteristics)
  2. Character composition     (digits, special chars, entropy)
  3. Structural / syntax       (subdomains, path depth, ports, protocols)
  4. Content / keyword-based   (suspicious tokens, brand names, TLD)
  5. Obfuscation signals       (IP in URL, URL shorteners, hex encoding)
"""

import re
import math
import tldextract
from urllib.parse import urlparse, parse_qs


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "banking", "paypal", "password", "credential", "ebay", "amazon", "apple",
    "microsoft", "google", "netflix", "free", "lucky", "prize", "winner",
    "click", "urgent", "support", "helpdesk", "webscr", "wallet",
    "suspended", "unusual", "activity", "billing", "invoice",
]

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "is.gd",
    "cli.gs", "yfrog.com", "migre.me", "ff.im", "su.pr", "twurl.nl",
    "snipurl.com", "short.to", "budurl.com", "ping.fm", "post.ly",
    "just.as", "bkite.com", "snipr.com", "fic.kr", "loopt.us", "doiop.com",
    "short.ie", "kl.am", "wp.me", "rubyurl.com", "om.ly", "to.ly",
    "rb.gy", "cutt.ly", "shorturl.at", "tiny.cc",
}

SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".club",
    ".online", ".site", ".info", ".biz", ".work", ".click", ".link",
    ".live", ".download", ".zip", ".review", ".country", ".kim",
    ".science", ".party", ".gdn", ".racing", ".date", ".win",
}

TRUSTED_BRANDS = [
    # Major financial / payment brands
    "paypal", "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc",
    "barclays", "santander", "wise", "venmo", "cashapp", "zelle",
    "stripe", "paystack", "flutterwave", "remita", "interswitch",

    # Major Nigerian banks (very commonly impersonated)
    "gtbank", "gtb", "firstbank", "accessbank", "zenithbank", "uba",
    "stanbic", "fcmb", "sterling", "fidelity", "kuda", "opay", "palmpay",

    # Major tech and social platforms
    "facebook", "google", "amazon", "apple", "microsoft", "netflix",
    "instagram", "twitter", "linkedin", "ebay", "yahoo",
    "telegram", "whatsapp", "tiktok", "discord", "snapchat", "signal",
    "youtube", "tumblr", "pinterest", "reddit", "twitch",

    # Email and cloud
    "gmail", "outlook", "hotmail", "icloud", "office365", "dropbox",
    "onedrive", "gdrive",

    # Crypto-related (heavily impersonated for drainers)
    "binance", "coinbase", "metamask", "uniswap", "opensea", "kraken",
    "trustwallet", "ledger", "trezor", "phantom",

    # E-commerce common to Nigerian users
    "jumia", "konga", "amazonng", "aliexpress",

    # Major AI platforms (newer impersonation targets)
    "openai", "chatgpt", "anthropic", "claude", "gemini",

    # Telecoms (Nigerian)
    "mtn", "airtel", "glo", "9mobile",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string as a measure of randomness."""
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def _has_ip_address(url: str) -> int:
    """Return 1 if URL contains a raw IPv4 address."""
    ipv4_pattern = re.compile(
        r"(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])"
    )
    return 1 if ipv4_pattern.search(url) else 0


def _has_hex_encoding(url: str) -> int:
    """Return 1 if URL contains percent-encoded (hex) characters beyond normal."""
    hex_matches = re.findall(r"%[0-9a-fA-F]{2}", url)
    return 1 if len(hex_matches) > 2 else 0


# ---------------------------------------------------------------------------
# Core feature extraction
# ---------------------------------------------------------------------------

def extract_features(url: str) -> dict:
    """
    Extract 26 features from a URL and return them as an ordered dict.

    Parameters
    ----------
    url : str
        Raw URL string (with or without scheme).

    Returns
    -------
    dict
        Feature name -> numeric value mapping.
    """
    # Normalise: ensure scheme is present for parsing
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed   = urlparse(url)
    ext      = tldextract.extract(url)
    domain   = parsed.netloc.lower()
    path     = parsed.path
    query    = parsed.query
    full_url = url.lower()

    # -----------------------------------------------------------------------
    # 1. Lexical / length-based features
    # -----------------------------------------------------------------------
    f_url_length      = len(url)
    f_domain_length   = len(domain)
    f_path_length     = len(path)
    f_query_length    = len(query) if query else 0
    f_num_params      = len(parse_qs(query)) if query else 0

    # -----------------------------------------------------------------------
    # 2. Character composition features
    # -----------------------------------------------------------------------
    f_dot_count       = url.count(".")
    f_hyphen_count    = url.count("-")
    f_at_count        = url.count("@")          # @ forces browser to use post-@ portion
    f_slash_count     = path.count("/")          # URL path depth
    f_double_slash    = 1 if "//" in path else 0 # Redirect trick: http://legit.com//evil.com
    f_underscore_count = url.count("_")
    f_question_mark   = 1 if "?" in url else 0
    f_equal_sign      = url.count("=")
    f_ampersand_count = url.count("&")

    # Digit ratio: % of the full URL that is numeric
    digit_count       = sum(c.isdigit() for c in url)
    f_digit_ratio     = round(digit_count / max(len(url), 1), 4)

    # Shannon entropy of the domain (high entropy → random-looking domain)
    f_domain_entropy  = round(_shannon_entropy(ext.domain), 4)

    # -----------------------------------------------------------------------
    # 3. Structural / syntax features
    # -----------------------------------------------------------------------
    subdomain_part    = ext.subdomain
    f_subdomain_count = subdomain_part.count(".") + 1 if subdomain_part else 0
    f_subdomain_len   = len(subdomain_part)

    # Non-standard port (anything other than 80/443)
    port = parsed.port
    f_has_port        = 1 if port and port not in (80, 443) else 0

    # -----------------------------------------------------------------------
    # 4. Protocol / obfuscation features
    # -----------------------------------------------------------------------
    f_has_https       = 1 if parsed.scheme == "https" else 0
    f_has_ip          = _has_ip_address(url)
    f_hex_encoding    = _has_hex_encoding(url)

    # URL shortener detected
    domain_only = (ext.domain + "." + ext.suffix).lower()
    f_is_shortener    = 1 if domain_only in SHORTENER_DOMAINS else 0

    # -----------------------------------------------------------------------
    # 5. Content / keyword features
    # -----------------------------------------------------------------------
    f_keyword_count   = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in full_url)
    f_brand_in_subdomain = int(
        any(brand in subdomain_part.lower() for brand in TRUSTED_BRANDS)
    )  # Phishing trick: paypal.malicious.com
    f_suspicious_tld  = int(
        any(full_url.endswith(tld) or ("." + ext.suffix) == tld
            for tld in SUSPICIOUS_TLDS)
    )

    # -----------------------------------------------------------------------
    # Assemble in a consistent order (same order used by the model)
    # -----------------------------------------------------------------------
    features = {
        # Lexical
        "url_length":           f_url_length,
        "domain_length":        f_domain_length,
        "path_length":          f_path_length,
        "query_length":         f_query_length,
        "num_params":           f_num_params,
        # Character composition
        "dot_count":            f_dot_count,
        "hyphen_count":         f_hyphen_count,
        "at_count":             f_at_count,
        "slash_count":          f_slash_count,
        "double_slash":         f_double_slash,
        "underscore_count":     f_underscore_count,
        "has_question_mark":    f_question_mark,
        "equal_sign_count":     f_equal_sign,
        "ampersand_count":      f_ampersand_count,
        "digit_ratio":          f_digit_ratio,
        "domain_entropy":       f_domain_entropy,
        # Structural
        "subdomain_count":      f_subdomain_count,
        "subdomain_length":     f_subdomain_len,
        "has_port":             f_has_port,
        # Protocol / obfuscation
        "has_https":            f_has_https,
        "has_ip_address":       f_has_ip,
        "has_hex_encoding":     f_hex_encoding,
        "is_shortener":         f_is_shortener,
        # Content / keyword
        "keyword_count":        f_keyword_count,
        "brand_in_subdomain":   f_brand_in_subdomain,
        "suspicious_tld":       f_suspicious_tld,
    }
    return features


def feature_names() -> list:
    """Return the canonical ordered list of feature names."""
    dummy = extract_features("http://example.com")
    return list(dummy.keys())


# ---------------------------------------------------------------------------
# Human-readable feature descriptions (used in the Streamlit UI)
# ---------------------------------------------------------------------------

FEATURE_DESCRIPTIONS = {
    "url_length":           "Total URL length (longer URLs are often suspicious)",
    "domain_length":        "Length of the domain name",
    "path_length":          "Length of the URL path",
    "query_length":         "Length of the query string",
    "num_params":           "Number of query parameters",
    "dot_count":            "Number of dots '.' in the URL",
    "hyphen_count":         "Number of hyphens '-' in the URL",
    "at_count":             "Presence of '@' symbol (used to mislead browsers)",
    "slash_count":          "Number of slashes in the URL path (depth)",
    "double_slash":         "Double '//' in path (redirect obfuscation trick)",
    "underscore_count":     "Number of underscores in the URL",
    "has_question_mark":    "Presence of '?' (query string indicator)",
    "equal_sign_count":     "Number of '=' signs (parameter value pairs)",
    "ampersand_count":      "Number of '&' signs (multiple parameters)",
    "digit_ratio":          "Proportion of the URL that is numeric characters",
    "domain_entropy":       "Shannon entropy of the domain name (higher = more random)",
    "subdomain_count":      "Number of subdomains present",
    "subdomain_length":     "Total length of the subdomain portion",
    "has_port":             "Non-standard port detected in the URL",
    "has_https":            "HTTPS protocol is used (secure connection)",
    "has_ip_address":       "Raw IP address used instead of domain name",
    "has_hex_encoding":     "Excessive percent/hex encoding in the URL",
    "is_shortener":         "URL shortening service detected (hides real destination)",
    "keyword_count":        "Count of suspicious keywords (login, verify, paypal…)",
    "brand_in_subdomain":   "Trusted brand name appears in subdomain (spoofing attempt)",
    "suspicious_tld":       "Top-level domain associated with high phishing activity",
}
