# Dataset Documentation

## Overview
This study utilizes a multi-industry resume dataset containing **13,389 resumes** categorized across **43 distinct job domains** (including Software Development, Data Science, Civil Engineering, Healthcare, Sales, Management, and HR).

## Data Schema
Each record in the primary dataset consists of:
- `ID`: Unique candidate identifier.
- `Resume_str`: Unstructured text representation of candidate work history, education, and skills.
- `Category`: Domain label (e.g., `Java Developer`, `React Developer`, `Consultant`).

## Privacy & PII Handling
To prevent automated bias and ensure ethical privacy compliance, all resumes undergo strict preprocessing via `src/preprocessing.py`, masking:
- Email addresses (`[EMAIL_REDACTED]`)
- Phone numbers (`[PHONE_REDACTED]`)
- Web URLs (`[URL_REDACTED]`)

## Local Setup
Due to GitHub file size constraints, full multi-gigabyte datasets are stored externally.
To run full-scale evaluation:
1. Download `Dataset.csv` from the project release/storage.
2. Place `Dataset.csv` into this `data/` directory.
3. Use `data/sample_resumes.json` for rapid local testing and CI/CD validation.
