"""
NLP Analyzer — text preprocessing, TF-IDF classification, N-gram suggestions.
No NLTK or scikit-learn required.
"""

import re
import math
import json
from collections import defaultdict


# ──────────────────────────────────────────────────────────────────────────────
# 1. Text Preprocessor
# ──────────────────────────────────────────────────────────────────────────────

STOPWORDS = {
    "a", "an", "the", "is", "in", "on", "at", "to", "of", "and", "or",
    "for", "with", "by", "from", "as", "it", "its", "this", "that", "be",
    "are", "was", "were", "has", "have", "had", "do", "does", "did", "not",
    "but", "if", "so", "up", "out", "no", "we", "you", "he", "she", "they",
}

SUFFIXES = ["ing", "tion", "ness", "ment", "er", "ed", "ly", "al", "ic", "ity"]


class TextPreprocessor:
    def preprocess(self, text: str) -> list[str]:
        """Return list of stemmed tokens from raw text."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = text.split()
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        tokens = [self._stem(t) for t in tokens]
        return tokens

    def _stem(self, word: str) -> str:
        for suffix in SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word


# ──────────────────────────────────────────────────────────────────────────────
# 2. TF-IDF Classifier
# ──────────────────────────────────────────────────────────────────────────────

CATEGORY_SEEDS = {
    "Food & Beverage": [
        "food", "drink", "juice", "sugar", "organ", "calori", "protein",
        "vitam", "fruit", "water", "milk", "chees", "bread", "meat",
        "chocol", "natur", "fresh", "orang", "veget", "grain",
    ],
    "Electronics": [
        "usb", "cabl", "batteri", "bluetooth", "wireless", "adapt",
        "voltag", "charger", "electr", "digit", "smart", "devic",
        "laptop", "phone", "camera", "screen",
    ],
    "Books & Media": [
        "isbn", "book", "author", "edit", "publish", "novel", "guid",
        "chapter", "read", "fiction", "scienc", "histor", "encyclop",
    ],
    "Health & Beauty": [
        "shampoo", "cream", "lotion", "vitam", "supplement", "tablet",
        "capsul", "skin", "hair", "soap", "toothpast", "medic", "pharmac",
    ],
    "Clothing": [
        "cotton", "size", "fabric", "wash", "medium", "larg", "small",
        "color", "cloth", "shirt", "pant", "dress", "wool", "poliest",
    ],
    "URL / Web": [
        "http", "https", "www", "com", "net", "org", "html", "php",
        "web", "site", "link", "url", "page",
    ],
    "WiFi Credential": [
        "wifi", "ssid", "wpa", "wep", "password", "network", "connect",
    ],
}


class TFIDFClassifier:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.corpus: list[list[str]] = []
        # Pre-process seed keywords
        self._seeds = {
            cat: set(kws) for cat, kws in CATEGORY_SEEDS.items()
        }

    def classify(self, text: str, product_info: dict = None) -> tuple[str, float]:
        """Return (category, confidence_0_to_1)."""
        low = text.lower()

        # Obvious patterns first
        if any(p in low for p in ["http", "https", "www."]):
            return "URL / Web", 1.0
        if any(p in low for p in ["wifi", "ssid", "wpa", "wep"]):
            return "WiFi Credential", 1.0

        # If product info available, use its categories
        if product_info and product_info.get("categories"):
            cats = product_info["categories"].lower()
            if any(k in cats for k in ["food", "drink", "beverage", "snack", "dairy", "meat", "fruit", "vegetable", "water", "juice", "milk", "cheese", "bread", "cereal"]):
                return "Food & Beverage", 0.95
            if any(k in cats for k in ["electronic", "cable", "battery", "phone"]):
                return "Electronics", 0.95
            if any(k in cats for k in ["beauty", "cosmetic", "shampoo", "cream", "soap"]):
                return "Health & Beauty", 0.95
            return "Food & Beverage", 0.80  # most barcodes are food

        # Pure numeric = likely retail product barcode → Food & Beverage
        if re.match(r"^\d+$", text.strip()):
            return "Food & Beverage", 0.70
        
        

        tokens = self.preprocessor.preprocess(text)
        if not tokens:
            return "Food & Beverage", 0.50

        best_cat, best_score = "Food & Beverage", 0.0
        for cat, seeds in self._seeds.items():
            overlap = sum(1 for t in tokens if t in seeds)
            score = overlap / max(len(tokens), 1)
            if score > best_score:
                best_score, best_cat = score, cat

        return best_cat, round(max(min(best_score, 1.0), 0.5), 2)

    def add_to_corpus(self, text: str):
        tokens = self.preprocessor.preprocess(text)
        if tokens:
            self.corpus.append(tokens)

    def similar_past_scans(self, text: str, top_k: int = 3) -> list[str]:
        """Simple cosine similarity against corpus."""
        tokens = set(self.preprocessor.preprocess(text))
        if not tokens or not self.corpus:
            return []
        scores = []
        for doc in self.corpus:
            doc_set = set(doc)
            inter = tokens & doc_set
            sim = len(inter) / (math.sqrt(len(tokens)) * math.sqrt(len(doc_set)) + 1e-9)
            scores.append((sim, " ".join(doc)))
        scores.sort(reverse=True)
        return [s[1] for s in scores[:top_k] if s[0] > 0]


# ──────────────────────────────────────────────────────────────────────────────
# 3. N-Gram Language Model
# ──────────────────────────────────────────────────────────────────────────────

class NGramModel:
    def __init__(self, n: int = 2):
        self.n = n
        self.counts: dict = defaultdict(lambda: defaultdict(int))
        self.vocab: set = set()

    def train(self, tokens: list[str]):
        self.vocab.update(tokens)
        for i in range(len(tokens) - self.n + 1):
            ctx = tuple(tokens[i: i + self.n - 1])
            word = tokens[i + self.n - 1]
            self.counts[ctx][word] += 1

    def predict_next(self, tokens: list[str], top_k: int = 3) -> list[str]:
        if len(tokens) < self.n - 1 or not self.counts:
            return []
        ctx = tuple(tokens[-(self.n - 1) :])
        dist = self.counts.get(ctx, {})
        if not dist:
            return []
        sorted_words = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:top_k]]

    def perplexity(self, tokens: list[str]) -> float:
        if len(tokens) < self.n or not self.vocab:
            return float("inf")
        V = len(self.vocab)
        log_prob = 0.0
        count = 0
        for i in range(self.n - 1, len(tokens)):
            ctx = tuple(tokens[i - self.n + 1 : i])
            word = tokens[i]
            c_ctx_word = self.counts[ctx].get(word, 0)
            c_ctx = sum(self.counts[ctx].values())
            prob = (c_ctx_word + 1) / (c_ctx + V + 1)
            log_prob += math.log(prob)
            count += 1
        return round(math.exp(-log_prob / max(count, 1)), 2)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Product Lookup (Open Food Facts — free, no API key needed)
# ──────────────────────────────────────────────────────────────────────────────


import re
import requests



def lookup_product(barcode_data: str) -> dict:
    
    clean = re.sub(r"\D", "", barcode_data)
    if len(clean) < 8:

        return {}

    try:
        r = requests.get(
            f"https://world.openfoodfacts.org/api/v0/product/{clean}.json",
            headers={"User-Agent": "SmartScanner/1.0"},
            timeout=5,
        )
        if r.status_code == 200:
            d = r.json()
            if d.get("status") == 1:
                p = d["product"]
                return {
                    "name":        p.get("product_name", ""),
                    "brand":       p.get("brands", ""),
                    "categories":  p.get("categories", ""),
                    "nutriscore":  p.get("nutriscore_grade", "").upper(),
                    "ingredients": p.get("ingredients_text", "")[:200],
                    "image_url":   p.get("image_url", ""),
                    "quantity":    p.get("quantity", ""),
                }
    except Exception as e:
        print(f"[Open Food Facts] Error: {e}")

    return {}
