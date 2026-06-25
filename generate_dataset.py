"""
generate_dataset.py
===================
Generates a realistic synthetic labelled URL dataset for training.
Produces urls.csv with columns: url, label (0=safe, 1=phishing)

Usage:
    python generate_dataset.py --samples 10000 --output urls.csv

This script is for academic demonstration when you don't have a
pre-labelled dataset. For production use, replace with real datasets
such as PhishTank, OpenPhish, or the UCI Phishing dataset.
"""

import random
import argparse
import csv
import string

# ─── Legitimate domain pools ────────────────────────────────────────────────
LEGITIMATE_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "twitter.com", "amazon.com",
    "wikipedia.org", "reddit.com", "instagram.com", "linkedin.com", "github.com",
    "stackoverflow.com", "microsoft.com", "apple.com", "netflix.com", "spotify.com",
    "dropbox.com", "slack.com", "zoom.us", "paypal.com", "ebay.com",
    "nytimes.com", "bbc.com", "cnn.com", "forbes.com", "techcrunch.com",
    "medium.com", "quora.com", "pinterest.com", "twitch.tv", "discord.com",
    "cloudflare.com", "digitalocean.com", "aws.amazon.com", "azure.microsoft.com",
    "caleb.edu.ng", "unilag.edu.ng", "ui.edu.ng", "oau.edu.ng", "lasu.edu.ng",
]

SAFE_PATHS = [
    "/", "/about", "/contact", "/products", "/services", "/blog", "/news",
    "/help", "/faq", "/support", "/home", "/search", "/profile", "/settings",
    "/category/tech", "/article/how-to-stay-safe", "/docs/api", "/pricing",
    "/download", "/team", "/careers", "/login", "/signup", "/dashboard",
]

SAFE_QUERIES = [
    "", "?q=python+tutorial", "?id=12345", "?page=2", "?ref=homepage",
    "?category=news", "?lang=en", "?tab=overview", "?sort=asc", "",
    "?utm_source=email", "?v=3.2.1", "", "?search=machine+learning", "",
]

# ─── Phishing domain pools ───────────────────────────────────────────────────
PHISHING_TLDS = [".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top",
                 ".online", ".site", ".info", ".click", ".live"]

PHISHING_KEYWORDS = [
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "banking", "paypal", "password", "ebay", "amazon", "apple", "microsoft",
    "google", "netflix", "free", "prize", "winner", "click-here", "urgent",
    "support", "helpdesk", "webscr", "wallet", "suspended", "billing",
]

BRAND_NAMES = [
    "paypal", "apple", "microsoft", "google", "amazon", "netflix", "facebook",
    "instagram", "twitter", "linkedin", "ebay", "chase", "wellsfargo",
]

SUSPICIOUS_PATHS = [
    "/login", "/signin", "/verify", "/confirm", "/update-account",
    "/secure/login", "/account/verify", "/webscr", "/billing/update",
    "/password-reset", "/suspended/reactivate", "/unusual-activity",
    "/login.php", "/signin.php", "/verify.html", "/account-locked",
    "/free-prize", "/claim-reward", "/winner", "/support/ticket",
]

SHORTENER_URLS = [
    "http://bit.ly/3xPhish1",
    "http://tinyurl.com/fakepaypal",
    "http://rb.gy/abcdef",
    "http://cutt.ly/phishlink",
    "http://short.to/steal",
    "http://is.gd/malicious",
    "http://ow.ly/Xk8m50",
    "http://t.co/FakeBank99",
]


def _rand_string(length=8):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _rand_ip():
    return f"{random.randint(1,254)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


def generate_safe_url():
    domain = random.choice(LEGITIMATE_DOMAINS)
    path   = random.choice(SAFE_PATHS)
    query  = random.choice(SAFE_QUERIES)
    scheme = "https://" if random.random() > 0.05 else "http://"
    return scheme + domain + path + query


def generate_phishing_url():
    strategy = random.choices(
        ["typosquat", "subdomain_brand", "ip_host", "keyword_path",
         "shortener", "random_domain", "hex_encoded"],
        weights=[20, 25, 15, 20, 5, 10, 5],
        k=1
    )[0]

    if strategy == "typosquat":
        brand  = random.choice(BRAND_NAMES)
        tld    = random.choice(PHISHING_TLDS)
        suffix = random.choice(["-secure", "-login", "-verify", "-update",
                                "-account", "0", "1", "-official"])
        path   = random.choice(SUSPICIOUS_PATHS)
        return f"http://{brand}{suffix}{tld}{path}"

    elif strategy == "subdomain_brand":
        brand  = random.choice(BRAND_NAMES)
        tld    = random.choice(PHISHING_TLDS)
        host   = _rand_string(random.randint(5, 12))
        path   = random.choice(SUSPICIOUS_PATHS)
        return f"http://{brand}.{host}{tld}{path}"

    elif strategy == "ip_host":
        ip   = _rand_ip()
        path = random.choice(SUSPICIOUS_PATHS)
        qs   = f"?redirect={random.choice(BRAND_NAMES)}.com"
        return f"http://{ip}{path}{qs}"

    elif strategy == "keyword_path":
        kw1  = random.choice(PHISHING_KEYWORDS)
        kw2  = random.choice(PHISHING_KEYWORDS)
        tld  = random.choice(PHISHING_TLDS)
        base = _rand_string(random.randint(6, 14))
        path = random.choice(SUSPICIOUS_PATHS)
        qs   = f"?user={_rand_string()}&token={_rand_string(16)}&redirect=1"
        return f"http://{base}{tld}/{kw1}/{kw2}{path}{qs}"

    elif strategy == "shortener":
        return random.choice(SHORTENER_URLS)

    elif strategy == "random_domain":
        tld  = random.choice(PHISHING_TLDS)
        host = _rand_string(random.randint(12, 22))
        path = random.choice(SUSPICIOUS_PATHS)
        qs   = f"?id={_rand_string(20)}&src=email"
        return f"http://{host}{tld}{path}{qs}"

    elif strategy == "hex_encoded":
        brand = random.choice(BRAND_NAMES)
        tld   = random.choice(PHISHING_TLDS)
        path  = "/login%2Everify%2Ephp"
        qs    = f"?u%73er={_rand_string()}&p%61ss={_rand_string(12)}"
        return f"http://{brand}-{_rand_string(5)}{tld}{path}{qs}"


def generate_dataset(n_samples: int, output_path: str):
    """Generate balanced phishing/safe URL dataset."""
    half     = n_samples // 2
    urls     = []
    labels   = []

    print(f"[INFO] Generating {half:,} safe URLs…")
    for _ in range(half):
        urls.append(generate_safe_url())
        labels.append(0)

    print(f"[INFO] Generating {half:,} phishing URLs…")
    for _ in range(half):
        urls.append(generate_phishing_url())
        labels.append(1)

    # Shuffle
    combined = list(zip(urls, labels))
    random.shuffle(combined)
    urls, labels = zip(*combined)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        for u, l in zip(urls, labels):
            writer.writerow([u, l])

    print(f"[INFO] Dataset saved: {output_path}  ({n_samples:,} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000,
                        help="Total number of URL samples to generate")
    parser.add_argument("--output",  default="urls.csv",
                        help="Output CSV path")
    args = parser.parse_args()
    generate_dataset(args.samples, args.output)
