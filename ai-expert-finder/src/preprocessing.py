"""
Text Preprocessing and Privacy-Preserving PII Masking Module.
"""

import re
from typing import Dict, Any


class ResumePreprocessor:
    """Handles cleaning, normalization, and privacy-preserving PII redaction for resumes."""

    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
    PHONE_PATTERN = re.compile(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}')
    URL_PATTERN = re.compile(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_+.~#?&/=]*)')
    LOREM_IPSUM = re.compile(r'lorem\s+ipsum.*?(?=\n\n|\Z)', re.IGNORECASE | re.DOTALL)

    def __init__(self, mask_pii: bool = True):
        self.mask_pii = mask_pii

    def redact_pii(self, text: str) -> str:
        """Masks emails, phone numbers, and web URLs to preserve candidate privacy."""
        if not text:
            return ""
        text = self.EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)
        text = self.PHONE_PATTERN.sub("[PHONE_REDACTED]", text)
        text = self.URL_PATTERN.sub("[URL_REDACTED]", text)
        return text

    def clean_text(self, text: str) -> str:
        """Removes placeholder text, redundant whitespace, and artifacts."""
        if not text:
            return ""
        text = self.LOREM_IPSUM.sub("", text)
        text = re.sub(r'[\r\t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ ]{2,}', ' ', text)
        return text.strip()

    def process(self, raw_resume: str) -> str:
        """Executes full preprocessing pipeline."""
        cleaned = self.clean_text(raw_resume)
        if self.mask_pii:
            cleaned = self.redact_pii(cleaned)
        return cleaned
