"""
allowlist.py
============
A curated list of well-known legitimate domains.

WHY THIS MODULE EXISTS
----------------------
The Random Forest model in Layer 1 reaches about 89.5% accuracy on the
Hannousse benchmark. That means roughly one in ten URLs is misclassified.
A particularly damaging case is when a legitimate URL the user knows is
safe (Google, GitHub, Claude, their own bank) gets flagged as phishing.
That kind of false positive destroys user trust in the whole system far
more than a missed phishing URL does.

This module is the symmetric complement to the blacklist. PhishTank and
URLhaus tell us what is definitely BAD. This list tells us what is
definitely GOOD. When the user submits a URL whose registered domain is
on this allowlist, the system trusts it without running the model.

HOW IT WORKS
------------
The list contains REGISTERED DOMAINS only, not full URLs. For example,
"google.com" matches https://google.com, https://www.google.com,
https://mail.google.com/inbox, and https://drive.google.com/file/123.
Matching is done on the suffix of the hostname so subdomains are
included automatically.

WHY THIS IS DEFENDABLE IN A PANEL
---------------------------------
1. Allowlisting is a standard security practice. Operating-system
   anti-malware, browser safe-browsing, and corporate firewalls all use
   allowlists alongside blocklists.
2. The list is small (about 60 domains) and every entry is auditable.
3. It only short-circuits in the direction of TRUST, not danger. It
   cannot cause a phishing URL to be classified as safe, because phishing
   sites cannot use one of these registered domains as their hostname.
4. The list reflects the system's known limitation (89.5% accuracy) and
   the deliberate engineering choice to handle the false-positive cases
   with a known-good list rather than trying to make the model larger.
"""

from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# The allowlist itself
# ---------------------------------------------------------------------------
# Each entry is a registered domain. Subdomains of these are included
# automatically. The list is deliberately small and conservative so every
# entry can be justified.

ALLOWED_DOMAINS = {

    # === Search and major tech platforms ===
    "google.com",
    "youtube.com",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "reddit.com",
    "tiktok.com",
    "snapchat.com",
    "pinterest.com",
    "whatsapp.com",
    "telegram.org",

    # === Email providers ===
    "gmail.com",
    "outlook.com",
    "yahoo.com",
    "hotmail.com",
    "icloud.com",
    "protonmail.com",

    # === Operating systems and cloud ===
    "microsoft.com",
    "apple.com",
    "windows.com",
    "live.com",
    "office.com",
    "aws.amazon.com",
    "cloud.google.com",
    "azure.com",

    # === E commerce and payments ===
    "amazon.com",
    "ebay.com",
    "aliexpress.com",
    "jumia.com.ng",
    "konga.com",
    "paypal.com",
    "stripe.com",
    "paystack.com",
    "flutterwave.com",

    # === Streaming and media ===
    "netflix.com",
    "spotify.com",
    "primevideo.com",
    "hulu.com",
    "twitch.tv",

    # === Developer and AI platforms ===
    "github.com",
    "gitlab.com",
    "stackoverflow.com",
    "claude.ai",
    "anthropic.com",
    "openai.com",
    "chatgpt.com",
    "huggingface.co",
    "kaggle.com",
    "colab.research.google.com",

    # === News and reference ===
    "wikipedia.org",
    "bbc.com",
    "bbc.co.uk",
    "cnn.com",
    "reuters.com",
    "nytimes.com",
    "theguardian.com",
    "punchng.com",
    "vanguardngr.com",

    # === Education ===
    "calebuniversity.edu.ng",
    "unilag.edu.ng",
    "ui.edu.ng",
    "coursera.org",
    "edx.org",
    "khanacademy.org",

    # === Nigerian banks ===
    "gtbank.com",
    "firstbanknigeria.com",
    "accessbankplc.com",
    "zenithbank.com",
    "uba.com",
    "stanbicibtc.com",
}


# ---------------------------------------------------------------------------
# Public lookup function
# ---------------------------------------------------------------------------

def _hostname_of(url: str) -> str:
    """Extract the lowercase hostname from a URL, stripping any leading 'www.'."""
    if "://" not in url:
        url = "http://" + url
    host = urlparse(url).netloc.lower()
    if ":" in host:
        host = host.split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def check(url: str) -> dict:
    """
    Check whether the URL's hostname is on the allowlist.

    A URL is considered allowed if its hostname EQUALS an allowlist entry
    OR is a subdomain of one. So `mail.google.com` matches `google.com`,
    but `google.com.attacker.tk` does NOT (the attacker.tk is the real
    registered domain, not google.com).

    Returns
    -------
    dict with keys:
        listed         : True if the hostname is on the allowlist
        matched_domain : the allowlist entry that matched, or None
    """
    if not url:
        return {"listed": False, "matched_domain": None}

    host = _hostname_of(url)
    if not host:
        return {"listed": False, "matched_domain": None}

    # Direct match — the URL's hostname IS an allowlist entry.
    if host in ALLOWED_DOMAINS:
        return {"listed": True, "matched_domain": host}

    # Subdomain match — the URL's hostname ends with ".allowlistentry".
    # The leading dot prevents google.com.attacker.tk from matching.
    for allowed in ALLOWED_DOMAINS:
        if host.endswith("." + allowed):
            return {"listed": True, "matched_domain": allowed}

    return {"listed": False, "matched_domain": None}
