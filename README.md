Klevr — AI-Based Phishing URL Detection System
Final Year Project | B.Sc. Cyber Security

Caleb University, Imota, Lagos | Session 2025/2026

Student: Olubo Demilade Subomi | Matric: 22/11227

Supervisor: Prof. M. K. Aregbesola
Live Demo
Deployed Application:

https://klevr---phising-detection-system-mce38ocvuwxknndwphhdxu.streamlit.app/
GitHub Repository:

https://github.com/beatsbyvibez/Klevr---Phising-Detection-system
Project Overview
Klevr is a three-layer phishing URL detection system that combines threat-intelligence blacklists, a curated allowlist, and a Random Forest machine learning classifier to identify malicious URLs in real time. The system provides plain-English advisory explanations alongside every verdict, making it accessible to non-technical users.
The name Klevr is a deliberate stylisation of the word "clever", reflecting the system's use of intelligent, well-engineered detection rather than brute-force computation.
Detection Architecture
Klevr processes every submitted URL through the following layers in order:
Layer 0a — Known-Good Allowlist

A curated set of approximately 60 trusted domain names. If the submitted URL matches a trusted domain, a Safe verdict is returned immediately at 99% confidence. This layer prevents false positives on major legitimate sites regardless of what the downstream layers detect.
Layer 0b — Threat-Intelligence Blacklist

Aggregates data from two public threat-intelligence feeds:

PhishTank (Cisco Talos) — human-verified phishing URLs
URLhaus (abuse.ch) — recent malware and phishing URLs

The cache refreshes automatically every six hours. A blacklist hit returns a Confirmed Phishing verdict at 99% confidence, but only if the allowlist has not already cleared the URL.
Layer 1 — Random Forest Classifier

A scikit-learn RandomForestClassifier with 300 decision trees trained on 26 lexical and structural features extracted from the URL string. Trained on the Hannousse and Yahiouche (2021) benchmark dataset of 11,430 labelled URLs. No network request is required to classify a URL at this layer.
Layer 2 — Deep Page Content Scanner (Optional)

Fetches the live page content and inspects four indicators: form destination mismatch, hidden iframes, excessive redirects, and wallet-draining JavaScript. Produces a suspicion score from 0 to 4. Opt-in only via a checkbox in the interface.
Advisory Module

Rule-based templating engine that generates plain-English explanations of every verdict, streamed to the interface character by character. Always concludes with a concrete recommended action.
Model Performance
Trained and evaluated on the Hannousse and Yahiouche (2021) dataset using an 80/20 stratified train-test split:
MetricValueAccuracy89.3%Precision88.1%Recall90.8%F1 Score89.4%ROC-AUC0.9588CV F1 (5-fold)0.8932 +/- 0.0026Features used26Training samples9,144 URLsTest samples2,286 URLs
Project Structure
Klevr---Phising-Detection-system/
├── app.py                  Main Streamlit application
├── allowlist.py            Layer 0a curated trusted domains
├── blacklist.py            Layer 0b PhishTank + URLhaus integration
├── features.py             26-feature URL extractor
├── advisor.py              Rule-based advisory engine
├── logger.py               Prediction history logging
├── web_analyzer.py         Layer 2 page content scanner
├── train_model.py          Model training script
├── prepare_dataset.py      Dataset preparation utilities
├── model.pkl               Trained Random Forest model (serialised)
├── feature_cols.pkl        Feature column ordering for inference
├── confusion_matrix.png    Confusion matrix plot
├── roc_curve.png           ROC curve plot
├── feature_importance.png  Feature importance chart
├── model_report.txt        Full classification report
├── requirements.txt        Python dependencies
├── config.toml             Streamlit configuration
└── README.md               This file
Installation and Local Setup
Requirements: Python 3.9 or higher
Step 1 — Clone the repository
bashgit clone https://github.com/beatsbyvibez/Klevr---Phising-Detection-system.git
cd Klevr---Phising-Detection-system
Step 2 — Install dependencies
bashpip install -r requirements.txt
Step 3 — Run the application
bashstreamlit run app.py
The application will open at http://localhost:8501
Note: The blacklist cache will populate automatically on first run by downloading the PhishTank and URLhaus feeds. This requires an active internet connection and may take 30 to 60 seconds on first launch.
Retraining the Model
bashpython train_model.py --dataset your_dataset.csv
The dataset must contain a url column and a label column where 1 indicates phishing and 0 indicates legitimate.
Dependencies
streamlit>=1.32.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
joblib>=1.2.0
requests>=2.28.0
tldextract>=3.4.0
beautifulsoup4>=4.11.0
lxml>=4.9.0
Dataset
The model was trained on the Hannousse and Yahiouche (2021) phishing URL benchmark dataset, a balanced set of 11,430 labelled URLs accompanied by 87 pre-extracted features.
Citation: Hannousse, A., and Yahiouche, S. (2021). Towards benchmark datasets for machine learning based website phishing detection: An experimental study. Engineering Applications of Artificial Intelligence, 104, 104347. https://doi.org/10.1016/j.engappai.2021.104347
Threat Intelligence Sources
PhishTank: http://www.phishtank.com — Operated by Cisco Talos. Licensed for free use with attribution.
URLhaus: https://urlhaus.abuse.ch — Operated by abuse.ch. Licensed under CC0 (public domain).
Academic Context
This project was submitted in partial fulfilment of the requirements for the award of the Bachelor of Science degree in Cyber Security at the College of Computing and Information Sciences, Caleb University, Imota, Lagos, Nigeria, in the academic session 2025/2026.
Supervisor: Professor M. K. Aregbesola

Department of Computer Science

Caleb University, Imota, Lagos
Licence
This project is released for academic and educational purposes. The source code may be used freely for non-commercial research.
