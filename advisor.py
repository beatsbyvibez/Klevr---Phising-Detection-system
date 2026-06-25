"""
advisor.py
==========
The PhishGuard advisor: turns numerical scan results into plain-English
explanations and recommendations for the user.

WHY THIS MODULE EXISTS
----------------------
Layers 0, 1, and 2 produce data: a True/False blacklist hit, a 0-1
probability score, a list of flagged feature names, and a 0-4 Layer 2
suspicion score. Raw numbers are hard for non-technical users to act
on. This module reads those results and writes a short natural-language
advisory explaining what was found, why it matters, and what the user
should do next.

HOW IT WORKS
------------
This is a rule-based template engine — NOT a large language model.
Every advisory is built from pre-written sentences that are chosen and
combined based on the scan results. This means:

  * Same scan results always produce the same advisory (deterministic)
  * No external API calls, no API keys, no internet needed
  * Every word the user sees can be traced back to a specific rule
  * The whole module is testable and defendable in a panel

It is essentially an extension of the user interface — the part of the
system that translates technical output into something the user can act
on. The "AI" in the advisor's voice comes from the model's prediction
upstream, not from any language generation here.
"""

from typing import Iterator
import time


# ---------------------------------------------------------------------------
# Feature explainers
# ---------------------------------------------------------------------------
# For each of the 26 features extracted by features.py, we keep a short
# human-readable phrase. When a feature has an unusual value, the advisor
# can cite the explanation directly.

FEATURE_EXPLANATIONS = {
    "url_length":          "the URL is unusually long for a legitimate site",
    "domain_length":       "the domain name is unusually long",
    "path_length":         "the URL path is unusually long",
    "query_length":        "the query string is unusually long",
    "has_ip_address":      "it uses a raw IP address instead of a domain name",
    "at_count":            'it contains an "@" symbol, a known URL-spoofing trick',
    "double_slash":        "it uses double slashes in the path, a redirect trick",
    "has_hex_encoding":    "parts of the URL are hex-encoded to hide their contents",
    "subdomain_count":     "it has many subdomains, which can hide the real owner",
    "brand_in_subdomain":  "a famous brand name appears inside a subdomain, a classic impersonation pattern",
    "suspicious_tld":      "it uses a top-level domain (like .tk or .xyz) known for phishing abuse",
    "is_shortener":        "it goes through a URL shortener, hiding the final destination",
    "domain_entropy":      "the domain name looks randomly generated rather than meaningful",
    "keyword_count":       "it contains alarming words like 'verify', 'login', or 'confirm'",
    "has_https":           "it does not use HTTPS encryption",
    "digit_ratio":         "the domain contains an unusual number of digits",
    "hyphen_count":        "the domain contains an unusual number of hyphens",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_top_suspicious_features(features: dict, top_n: int = 3) -> list:
    """
    From the feature dict, find which features have unusual / suspicious
    values and return a list of their plain-English explanations.

    'Unusual' here means binary 1-flags are tripped or numeric counts
    are above a sensible threshold. This is a heuristic that mirrors
    what the Random Forest model itself learns.
    """
    flagged = []

    # Binary flag features — any 1 means the flag is tripped
    binary_features = [
        "has_ip_address", "double_slash", "has_hex_encoding",
        "brand_in_subdomain", "suspicious_tld", "is_shortener",
    ]
    for f in binary_features:
        if features.get(f, 0) == 1:
            flagged.append(FEATURE_EXPLANATIONS.get(f))

    # @-symbol count
    if features.get("at_count", 0) >= 1:
        flagged.append(FEATURE_EXPLANATIONS["at_count"])

    # No HTTPS is the same idea, but inverted
    if features.get("has_https", 1) == 0:
        flagged.append(FEATURE_EXPLANATIONS["has_https"])

    # Numeric features with sensible thresholds
    if features.get("url_length", 0) > 100:
        flagged.append(FEATURE_EXPLANATIONS["url_length"])
    if features.get("subdomain_count", 0) >= 3:
        flagged.append(FEATURE_EXPLANATIONS["subdomain_count"])
    if features.get("keyword_count", 0) >= 2:
        flagged.append(FEATURE_EXPLANATIONS["keyword_count"])
    if features.get("domain_entropy", 0) > 3.5:
        flagged.append(FEATURE_EXPLANATIONS["domain_entropy"])
    if features.get("digit_ratio", 0) > 0.3:
        flagged.append(FEATURE_EXPLANATIONS["digit_ratio"])
    if features.get("hyphen_count", 0) >= 3:
        flagged.append(FEATURE_EXPLANATIONS["hyphen_count"])

    # Drop any None values that slipped in if a feature was missing
    flagged = [f for f in flagged if f]

    return flagged[:top_n]


def _join_clauses(items: list) -> str:
    """Join a list into a comma-separated, comma-and-and phrase."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# ---------------------------------------------------------------------------
# Step descriptions — used by the live progress strip in the UI
# ---------------------------------------------------------------------------

STEPS = [
    ("Checking PhishTank and URLhaus blacklists",     "blacklist"),
    ("Extracting 26 structural features from the URL", "features"),
    ("Running the Random Forest model",                "model"),
    ("Running Layer 2 deep page scan",                 "layer2"),
]


# ---------------------------------------------------------------------------
# The main advisor function
# ---------------------------------------------------------------------------

def build_advisory(
    url: str,
    bl_result: dict,
    label: str,
    conf_phishing: float,
    features: dict,
    layer2: dict | None = None,
    al_result: dict | None = None,
) -> str:
    """
    Build a complete, plain-English advisory string from the scan results.

    Parameters
    ----------
    url            : the URL the user submitted
    bl_result      : dict from blacklist.check()  has 'listed', 'source', 'match_type'
    label          : 'Phishing' or 'Safe' from the ML model
    conf_phishing  : float 0..1, model's phishing probability
    features       : dict of 26 features from features.extract_features()
    layer2         : optional dict from web_analyzer.analyze_page(), or None
    al_result      : optional dict from allowlist.check(), or None

    Returns
    -------
    A multi-paragraph advisory string ready to display to the user.
    """
    paragraphs = []
    al_listed = bool(al_result and al_result.get("listed"))

    # ----- PARAGRAPH 1: the headline finding -----
    if bl_result.get("listed"):
        match_word = (
            "the exact URL is on the list"
            if bl_result.get("match_type") == "exact_url"
            else "the host domain is on the list"
        )
        paragraphs.append(
            f"This URL is a **confirmed phishing page**. It is listed on "
            f"**{bl_result['source']}**, a threat-intelligence feed where "
            f"security researchers manually verify reported phishing URLs, "
            f"and {match_word}."
        )
    elif al_listed:
        paragraphs.append(
            f"This URL is **trusted**. Its hostname matches "
            f"**{al_result['matched_domain']}** on my known-good allowlist, "
            f"a curated list of major legitimate domains. Phishing sites "
            f"cannot operate from these domains, so the verdict is taken "
            f"directly from the allowlist."
        )
    elif label == "Phishing":
        conf_pct = round(conf_phishing * 100, 1)
        paragraphs.append(
            f"My machine learning model has flagged this URL as **likely phishing** "
            f"with a confidence of **{conf_pct}%**. It is not yet on the public "
            f"blacklists, but its structural pattern matches known phishing URLs "
            f"in the training data."
        )
    else:
        conf_pct = round((1 - conf_phishing) * 100, 1)
        paragraphs.append(
            f"This URL appears **safe**. It is not on the PhishTank or URLhaus "
            f"blacklists, and my machine learning model scored it as legitimate "
            f"with **{conf_pct}%** confidence."
        )

    # ----- PARAGRAPH 2: why, cite specific features -----
    flagged = _pick_top_suspicious_features(features)
    if bl_result.get("listed") and flagged:
        # Blacklist listed AND model agrees with reasons
        paragraphs.append(
            f"My model independently agreed, scoring it as "
            f"{round(conf_phishing * 100, 1)}% phishing. The main reasons: "
            f"{_join_clauses(flagged)}."
        )
    elif al_listed:
        # On the allowlist, optionally note any minor flags but reassure
        if flagged:
            paragraphs.append(
                f"The model did notice a few patterns it sometimes associates "
                f"with phishing ({_join_clauses(flagged)}), but the allowlist "
                f"takes priority because this is a verified legitimate domain."
            )
        else:
            paragraphs.append(
                "No suspicious patterns were detected by the model either. "
                "Both layers of the system agree this URL is safe."
            )
    elif label == "Phishing" and flagged:
        paragraphs.append(
            f"The features that influenced this decision: {_join_clauses(flagged)}."
        )
    elif label == "Safe" and not flagged:
        paragraphs.append(
            "None of the 26 features I extracted look unusual. The domain is "
            "well-formed, of normal length, uses HTTPS, and does not match "
            "any known impersonation pattern."
        )
    elif label == "Safe" and flagged:
        paragraphs.append(
            f"A few mild flags were raised ({_join_clauses(flagged)}), but "
            f"not enough to push the model past the phishing threshold. "
            f"If you have any doubt, double-check the sender or context."
        )

    # ----- PARAGRAPH 3: Layer 2 findings, if run -----
    if layer2 and layer2.get("success"):
        score = layer2.get("suspicion_score", 0)
        layer2_flags = []
        if layer2.get("wallet_keywords") == 1:
            layer2_flags.append("the page tries to access a crypto wallet without being a known crypto platform")
        if layer2.get("form_mismatch") == 1:
            layer2_flags.append("a login form on the page submits data to a different domain")
        if layer2.get("redirect_chain") == 1:
            layer2_flags.append("the URL bounced through three or more redirects before settling")
        if layer2.get("hidden_iframe") == 1:
            layer2_flags.append("the page hides an iframe loading external content")

        if score == 0:
            paragraphs.append(
                "The Layer 2 deep page scan completed cleanly. No wallet "
                "access requests, no form mismatches, no excessive redirects, "
                "and no hidden iframes."
            )
        elif score == 1:
            paragraphs.append(
                f"The Layer 2 deep page scan raised **one indicator**: "
                f"{layer2_flags[0]}. This alone could be coincidence, but "
                f"combine it with anything else suspicious and you should "
                f"avoid the site."
            )
        else:  # score >= 2
            paragraphs.append(
                f"The Layer 2 deep page scan raised **{score} indicators**: "
                f"{_join_clauses(layer2_flags)}. Multiple independent red "
                f"flags on the page itself make this very likely malicious."
            )

    # ----- PARAGRAPH 4: recommended action -----
    if bl_result.get("listed") or label == "Phishing":
        paragraphs.append(
            "**Recommended action:** do not open this link. If you received "
            "it by email or message, treat the sender as suspicious. A real "
            "account from this organisation would never link here. If you "
            "have already clicked it and entered credentials, change those "
            "passwords immediately and watch the account for unauthorised "
            "activity."
        )
    else:
        paragraphs.append(
            "**Recommended action:** the URL looks fine, but always sanity-"
            "check the context. If something about the message that sent "
            "you here feels off, verify with the sender through a separate "
            "channel before logging in or sharing personal information."
        )

    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Typewriter streaming helper
# ---------------------------------------------------------------------------

def typewriter_stream(text: str, delay_per_word: float = 0.025) -> Iterator[str]:
    """
    Yield the text word-by-word with a small pause between words, so the
    Streamlit UI can display it as if a person is typing it out.

    Parameters
    ----------
    text            : the full advisory text to stream
    delay_per_word  : seconds to wait between yielding words (0.025 ≈ ChatGPT pace)
    """
    accumulated = ""
    words = text.split(" ")
    for i, word in enumerate(words):
        accumulated += word
        # Add the trailing space except on the very last word
        if i < len(words) - 1:
            accumulated += " "
        yield accumulated
        time.sleep(delay_per_word)
