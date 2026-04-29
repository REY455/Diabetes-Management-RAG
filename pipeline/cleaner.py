import re

def clean_text(text):
    # Fix broken spacing like "haveahighincidence"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Fix hyphen words
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

    # Remove garbage characters
    text = re.sub(r'[^a-zA-Z0-9.,;:\-\s]', '', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()