"""
Unit tests — run with: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from nlp.analyzer import TextPreprocessor, TFIDFClassifier, NGramModel
from core.preprocessor import Preprocessor


# ── TextPreprocessor ──────────────────────────────────────────────────────────
def test_lowercase():
    tp = TextPreprocessor()
    tokens = tp.preprocess("ORGANIC Orange JUICE")
    assert all(t == t.lower() for t in tokens)

def test_stopword_removal():
    tp = TextPreprocessor()
    tokens = tp.preprocess("this is a bottle of juice")
    assert "this" not in tokens and "is" not in tokens and "a" not in tokens

def test_stemming():
    tp = TextPreprocessor()
    tokens = tp.preprocess("running nutrition")
    assert "runn" in tokens or any("nutri" in t for t in tokens)


# ── TFIDFClassifier ───────────────────────────────────────────────────────────
def test_food_classify():
    clf = TFIDFClassifier()
    cat, conf = clf.classify("Organic Orange Juice Natural Vitamin C calories")
    assert cat == "Food & Beverage"

def test_electronics_classify():
    clf = TFIDFClassifier()
    cat, conf = clf.classify("USB cable bluetooth wireless adapter battery")
    assert cat == "Electronics"

def test_url_classify():
    clf = TFIDFClassifier()
    cat, conf = clf.classify("https://www.example.com/product/123")
    assert cat == "URL / Web"


# ── NGramModel ────────────────────────────────────────────────────────────────
def test_predict_next():
    model = NGramModel(n=2)
    tokens = ["organ", "orang", "juic", "natur", "vitam", "organ", "orang"]
    model.train(tokens)
    preds = model.predict_next(["organ"])
    assert isinstance(preds, list)

def test_perplexity():
    model = NGramModel(n=2)
    tokens = ["organ", "orang", "juic", "vitam", "protein"]
    model.train(tokens)
    pp = model.perplexity(tokens)
    assert isinstance(pp, float) and pp > 0


# ── Preprocessor ─────────────────────────────────────────────────────────────
def test_gaussian_kernel():
    p = Preprocessor(sigma=1.4, kernel_size=7)
    k = p._make_gaussian_kernel()
    assert abs(k.sum() - 1.0) < 1e-5
    assert k.shape == (1, 7)

def test_full_pipeline():
    p = Preprocessor()
    fake_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    processed, debug = p.process(fake_frame)
    assert processed.shape == (240, 320)
    assert "fft" in debug and "edges" in debug
