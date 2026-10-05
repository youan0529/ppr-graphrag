"""Shared entity-name normalization."""
import unicodedata


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())
