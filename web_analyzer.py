"""
web_analyzer.py
===============
Layer 2 of the PhishGuard AI system: page content analysis.

WHY THIS MODULE EXISTS
----------------------
Layer 1 (features.py + the Random Forest model) inspects only the URL
string. It catches phishing URLs that LOOK suspicious — raw IP addresses,
suspicious TLDs like .tk, brand names hidden in subdomains, and so on.

However, some modern attacks bypass Layer 1 because the URL itself looks
completely normal. The classic example is a crypto wallet drainer: the
URL might be a short, clean link to "swap-rewards.io" with HTTPS, no
suspicious keywords, no IP address. URL features alone cannot detect it.
The malicious behaviour only appears when you load the page and look at
the HTML and JavaScript inside.

This module fetches the actual web page and runs FOUR simple checks.
Each check returns 0 (clean) or 1 (suspicious). The total is a
"suspicion score" between 0 and 4. A score of 2 or more means the page
has multiple red flags and the user should be warned even if Layer 1
called it safe.

SAFETY NOTE
-----------
This module uses the `requests` library, which only downloads the HTML
text. It NEVER executes JavaScript. That means visiting a drainer page
through this module cannot drain a wallet, because no wallet is connected.
We only read the page source code as text and look for patterns.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keywords that indicate the page wants to talk to a crypto wallet.
# These appear inside the page's JavaScript when it asks the browser to
# connect to MetaMask or a similar wallet extension.
WALLET_PATTERNS = [
    "window.ethereum",      # the standard wallet object exposed by MetaMask
    "web3.",                # the Web3 JavaScript library
    "metamask",             # direct mention of the wallet
    "connect wallet",       # the button text on most drainer sites
    "walletconnect",        # the WalletConnect protocol
    "eth_requestaccounts",  # the function called to access wallet accounts
]

# Sites that legitimately use wallet connections. If the page domain ends
# with one of these, finding wallet keywords is expected and not a warning.
KNOWN_CRYPTO_DOMAINS = {
    "metamask.io",
    "uniswap.org",
    "opensea.io",
    "coinbase.com",
    "binance.com",
    "kraken.com",
    "ethereum.org",
}

# How long to wait for the page to load (seconds). Phishing pages are
# often slow because they live on shared hosting.
REQUEST_TIMEOUT = 8

# We send a normal browser User-Agent so most sites don't block us.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_domain(url: str) -> str:
    """Return the lowercase hostname of a URL, stripping any leading 'www.'."""
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _fetch_page(url: str):
    """
    Download the HTML of `url`.

    Returns a tuple (response, error_message). If the page can be fetched
    successfully, error_message is None. If anything goes wrong, response
    is None and error_message describes the problem.
    """
    headers = {"User-Agent": USER_AGENT}
    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        return response, None
    except requests.exceptions.Timeout:
        return None, "The page took too long to respond"
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to the website"
    except requests.exceptions.TooManyRedirects:
        return None, "Too many redirects (this is itself suspicious)"
    except Exception as e:
        return None, f"Unexpected error while fetching page: {e}"


# ---------------------------------------------------------------------------
# The four checks — each one is short and easy to explain to a panel
# ---------------------------------------------------------------------------

def check_wallet_keywords(html: str, page_domain: str) -> int:
    """
    CHECK 1 — Does the page try to talk to a crypto wallet?

    We search the raw HTML/JavaScript text for known wallet-API keywords.
    If any are present AND the page is not on a recognised crypto site,
    that is a strong indicator of a wallet-drainer attack.
    """
    # If the domain is a known legitimate crypto platform, ignore.
    if any(page_domain.endswith(d) for d in KNOWN_CRYPTO_DOMAINS):
        return 0

    html_lower = html.lower()
    for keyword in WALLET_PATTERNS:
        if keyword in html_lower:
            return 1
    return 0


def check_form_action_mismatch(soup: BeautifulSoup, page_domain: str) -> int:
    """
    CHECK 2 — Does any form on the page submit data to a different domain?

    Legitimate sites usually post their login forms back to themselves.
    A form on `bank-login-page.com` that submits to `attacker.ru` is the
    most common credential-harvesting pattern.
    """
    for form in soup.find_all("form"):
        action = form.get("action", "").strip()
        # Empty or relative actions stay on the same domain - safe.
        if not action or not action.startswith("http"):
            continue
        action_domain = _get_domain(action)
        if action_domain and action_domain != page_domain:
            return 1
    return 0


def check_redirect_chain(response) -> int:
    """
    CHECK 3 — How many redirects did it take to reach the final page?

    The `requests` library stores every redirect in `response.history`.
    Three or more redirects, especially across different domains, is a
    well-known phishing tactic used to hide the real destination.
    """
    return 1 if len(response.history) >= 3 else 0


def check_hidden_iframe(soup: BeautifulSoup, page_domain: str) -> int:
    """
    CHECK 4 — Is the page hiding an iframe that loads external content?

    Attackers sometimes inject a tiny invisible iframe pointing to a
    malicious site. The user sees a legitimate-looking page but the
    iframe silently runs the attack.
    """
    for iframe in soup.find_all("iframe"):
        src = iframe.get("src", "")
        if not src or not src.startswith("http"):
            continue

        iframe_domain = _get_domain(src)
        if iframe_domain == page_domain:
            continue  # Same-domain iframe — not suspicious

        # Check if the iframe is hidden by CSS or by zero dimensions.
        style = (iframe.get("style") or "").lower()
        width = str(iframe.get("width", "")).strip()
        height = str(iframe.get("height", "")).strip()
        is_hidden = (
            "display:none" in style.replace(" ", "")
            or "visibility:hidden" in style.replace(" ", "")
            or width == "0"
            or height == "0"
        )
        if is_hidden:
            return 1
    return 0


# ---------------------------------------------------------------------------
# The public entry point used by app.py
# ---------------------------------------------------------------------------

def analyze_page(url: str) -> dict:
    """
    Run all four Layer-2 checks against `url` and return a result dict.

    Keys returned:
        success          : True if the page could be fetched and analysed
        error            : None, or a human-readable error string
        wallet_keywords  : 0 or 1  (check 1 result)
        form_mismatch    : 0 or 1  (check 2 result)
        redirect_chain   : 0 or 1  (check 3 result)
        hidden_iframe    : 0 or 1  (check 4 result)
        suspicion_score  : 0..4    (sum of the four checks)
        final_url        : the URL after any redirects
        redirect_count   : how many redirect hops occurred
    """
    # Default result dictionary; we'll fill in fields as we go.
    result = {
        "success": False,
        "error": None,
        "wallet_keywords": 0,
        "form_mismatch": 0,
        "redirect_chain": 0,
        "hidden_iframe": 0,
        "suspicion_score": 0,
        "final_url": url,
        "redirect_count": 0,
    }

    # Step 1: fetch the page. If this fails we return early.
    response, error = _fetch_page(url)
    if error:
        result["error"] = error
        return result

    # Step 2: parse the HTML so we can search inside <form>, <iframe> etc.
    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        result["error"] = f"Could not parse the HTML: {e}"
        return result

    # Step 3: run the four checks.
    page_domain = _get_domain(response.url)
    result["success"]         = True
    result["final_url"]       = response.url
    result["redirect_count"]  = len(response.history)
    result["wallet_keywords"] = check_wallet_keywords(response.text, page_domain)
    result["form_mismatch"]   = check_form_action_mismatch(soup, page_domain)
    result["redirect_chain"]  = check_redirect_chain(response)
    result["hidden_iframe"]   = check_hidden_iframe(soup, page_domain)

    # Step 4: combine the four 0/1 flags into a single 0..4 score.
    result["suspicion_score"] = (
        result["wallet_keywords"]
        + result["form_mismatch"]
        + result["redirect_chain"]
        + result["hidden_iframe"]
    )
    return result


# Human-readable explanations used by the Streamlit UI to display results.
CHECK_DESCRIPTIONS = {
    "wallet_keywords":
        "Page requests crypto-wallet access (wallet-drainer indicator)",
    "form_mismatch":
        "A login form on the page submits data to a different domain",
    "redirect_chain":
        "The URL bounced through 3 or more redirects before the final page",
    "hidden_iframe":
        "The page hides an iframe loading content from another domain",
}
