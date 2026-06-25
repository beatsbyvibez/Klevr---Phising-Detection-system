# PhishGuard AI — Phishing URL Detection System
### Final Year Project | Caleb University, Imota, Lagos
**Student:** Demilade Ogunlade | **Matric No:** 22/11227  
**Supervisor:** Prof. M. K. Aregbesola | **Session:** 2024/2025

---

## Project Overview
PhishGuard AI is an AI-powered phishing URL detection system that uses
machine learning (Random Forest) to classify URLs as safe or phishing
in real time. The system analyses 26 lexical, structural, and content-based
features extracted purely from the URL string — no external API calls,
DNS lookups, or internet access required at inference time.

---

## File Structure
```
phishing_detector/
├── app.py                  ← Main Streamlit web application
├── features.py             ← 26-feature URL extraction module
├── train_model.py          ← Model training + evaluation script
├── generate_dataset.py     ← Synthetic dataset generator (for demo)
├── logger.py               ← Prediction history logger
├── requirements.txt        ← Python dependencies
├── urls.csv                ← Training dataset (generated or your own)
├── model.pkl               ← Trained Random Forest model (after training)
├── feature_cols.pkl        ← Feature column order (after training)
├── model_report.txt        ← Evaluation metrics report (after training)
├── confusion_matrix.png    ← Confusion matrix plot (after training)
├── roc_curve.png           ← ROC curve plot (after training)
├── feature_importance.png  ← Feature importance chart (after training)
└── prediction_log.csv      ← Auto-generated prediction history log
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2A. Use your own dataset (recommended)
Your CSV must have a `url` column and a `label` column (0=safe, 1=phishing):
```bash
python train_model.py --dataset your_data.csv
```

### 2B. Use the synthetic demo dataset
```bash
python generate_dataset.py --samples 10000 --output urls.csv
python train_model.py --dataset urls.csv
```

### 3. Launch the web app
```bash
streamlit run app.py
```

---

## The 26 Features

| # | Feature | Category | Description |
|---|---------|----------|-------------|
| 1 | url_length | Lexical | Total URL character length |
| 2 | domain_length | Lexical | Length of domain name |
| 3 | path_length | Lexical | Length of URL path |
| 4 | query_length | Lexical | Length of query string |
| 5 | num_params | Lexical | Number of query parameters |
| 6 | dot_count | Character | Number of dots in URL |
| 7 | hyphen_count | Character | Number of hyphens in URL |
| 8 | at_count | Character | Presence of @ symbol (browser redirect trick) |
| 9 | slash_count | Character | Path depth (slash count) |
| 10 | double_slash | Character | Double // in path (redirect obfuscation) |
| 11 | underscore_count | Character | Number of underscores |
| 12 | has_question_mark | Character | Query string present |
| 13 | equal_sign_count | Character | Number of = signs |
| 14 | ampersand_count | Character | Number of & signs |
| 15 | digit_ratio | Character | Proportion of digits in URL |
| 16 | domain_entropy | Character | Shannon entropy of domain name |
| 17 | subdomain_count | Structural | Number of subdomains |
| 18 | subdomain_length | Structural | Total subdomain length |
| 19 | has_port | Structural | Non-standard port in URL |
| 20 | has_https | Protocol | HTTPS protocol used |
| 21 | has_ip_address | Obfuscation | Raw IP address instead of domain |
| 22 | has_hex_encoding | Obfuscation | Excessive percent/hex encoding |
| 23 | is_shortener | Obfuscation | URL shortening service detected |
| 24 | keyword_count | Content | Count of suspicious keywords |
| 25 | brand_in_subdomain | Content | Trusted brand spoofed in subdomain |
| 26 | suspicious_tld | Content | High-risk top-level domain |

---

## Model Architecture
- **Algorithm:** Random Forest Classifier
- **Estimators:** 300 decision trees
- **Class weights:** Balanced (handles imbalanced datasets)
- **Validation:** 80/20 train-test split + 5-fold stratified cross-validation
- **Metrics:** Accuracy, Precision, Recall, F1 Score, ROC-AUC

---

## App Pages
| Page | Description |
|------|-------------|
| Single URL Analysis | Classify one URL, see confidence score, feature breakdown, suspicious signals |
| Batch URL Analysis | Upload CSV of URLs, download classified results |
| Model Performance | View accuracy metrics, confusion matrix, ROC curve, feature importance |
| Prediction Log | Full history of all predictions with export option |
| About | Project documentation and research context |

---

## Using a Real Dataset (Recommended for FYP)
For a stronger academic result, use one of these public datasets:
- **PhishTank** — https://www.phishtank.com/developer_info.php
- **OpenPhish** — https://openphish.com/
- **UCI Phishing dataset** — https://archive.ics.uci.edu/dataset/967
- **Kaggle Phishing URL dataset** — search "phishing URL dataset" on Kaggle

Format your CSV as:
```
url,label
https://google.com,0
http://paypal-verify.tk/login,1
```

---

## Notes
- The 100% accuracy on the synthetic dataset is expected — the data was
  algorithmically generated with deterministic patterns. On real-world
  datasets expect 94 to 98% accuracy.
- The model runs entirely offline after training. No API keys needed.
- prediction_log.csv is auto-created on first use.
