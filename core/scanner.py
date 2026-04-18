"""
BarcodeScanner — OpenCV-only barcode/QR scanner.

Decode pipeline (ordered fastest → most aggressive):
  Stage 1 — Academic preprocessing (4 techniques from class):
              raw gray · Gaussian blur (1D conv) · FFT bandpass ·
              edge-closed · adaptive threshold · CLAHE · Otsu
  Stage 2 — Upscaling  ×2 / ×3 / ×4  (small or distant codes)
  Stage 3 — Rotations  90° / 180° / 270°
  Stage 4 — ROI search via Sobel gradient (barcode localisation)
  Stage 5 — Tiled sliding window (code in corner of very large image)

Each stage exits immediately when any code is found.
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

from core.preprocessor import Preprocessor
from nlp.analyzer import TextPreprocessor, TFIDFClassifier, NGramModel, lookup_product


HISTORY_FILE   = Path("data/scan_history.json")
HISTORY_FILE.parent.mkdir(exist_ok=True)
DECODE_EVERY_N = 5   # webcam: process 1 in every N frames


# ── Singleton detectors ───────────────────────────────────────────────────────

_QR_DETECTOR = cv2.QRCodeDetector()
try:
    _BARCODE_DETECTOR = cv2.barcode.BarcodeDetector()
except AttributeError:
    _BARCODE_DETECTOR = None

# Singleton Preprocessor (holds cached CLAHE + Gaussian kernel)
_PREP = Preprocessor()


# ── Core detector ─────────────────────────────────────────────────────────────

def _decode_gray(gray: np.ndarray) -> list[dict]:
    """
    Run QR + 1D barcode detectors on a grayscale image.
    Returns list of dicts: {data, type, polygon}.
    """
    if gray is None or gray.size == 0:
        return []
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    results = []

    # QR codes
    try:
        ok, decoded_list, points, _ = _QR_DETECTOR.detectAndDecodeMulti(gray)
        if ok and decoded_list:
            for i, data in enumerate(decoded_list):
                if data:
                    poly = [(int(p[0]), int(p[1])) for p in points[i]] \
                           if points is not None and i < len(points) else []
                    results.append({"data": data, "type": "QR_CODE", "polygon": poly})
    except Exception:
        pass

    # 1D barcodes
    if _BARCODE_DETECTOR is not None:
        try:
            ok2, dec_list, type_list, pts2 = _BARCODE_DETECTOR.detectAndDecodeMulti(gray)
            if ok2 and dec_list:
                for i, data in enumerate(dec_list):
                    if data:
                        btype = str(type_list[i]) \
                                if type_list is not None and i < len(type_list) else "BARCODE"
                        poly = [(int(p[0]), int(p[1])) for p in pts2[i]] \
                               if pts2 is not None and i < len(pts2) else []
                        results.append({"data": data, "type": btype, "polygon": poly})
        except Exception:
            pass

    return results


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

def _upscale(img: np.ndarray, factor: int) -> np.ndarray:
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

def _scale_poly(poly: list, factor: float) -> list:
    return [(int(x / factor), int(y / factor)) for x, y in poly]

def _rotate_img(img: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:  return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def _rotate_poly_back(poly: list, angle: int, w: int, h: int) -> list:
    if not poly:
        return poly
    cx, cy   = w / 2, h / 2
    rad      = -np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    out = []
    for x, y in poly:
        x -= cx; y -= cy
        out.append((int(x * cos_a - y * sin_a + cx),
                    int(x * sin_a + y * cos_a + cy)))
    return out

def _majority_vote(candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    counts = Counter(c["data"] for c in candidates)
    seen, out = set(), []
    for c in sorted(candidates, key=lambda x: -counts[x["data"]]):
        if c["data"] not in seen:
            seen.add(c["data"])
            out.append(c)
    return out


# ── Barcode ROI localisation (Sobel gradient method) ─────────────────────────

def _find_barcode_rois(gray: np.ndarray) -> list[tuple]:
    """
    Localise probable barcode regions using horizontal Sobel dominance.
    Barcodes have strong horizontal gradient (many vertical lines).
    Returns list of (x, y, w, h).
    """
    sobelx = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    sobely = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    diff   = cv2.subtract(sobelx, sobely)
    _, diff = cv2.threshold(diff, 0, 255, cv2.THRESH_TOZERO)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed  = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)
    closed  = cv2.erode(closed, None, iterations=3)
    closed  = cv2.dilate(closed, None, iterations=6)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ih, iw = gray.shape[:2]
    rois   = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 400 or area > iw * ih * 0.95:
            continue
        pad = 20
        rois.append((max(0, x-pad), max(0, y-pad),
                     min(iw, x+cw+pad) - max(0, x-pad),
                     min(ih, y+ch+pad) - max(0, y-pad)))
    return rois


# ── Main decode pipeline ──────────────────────────────────────────────────────

def try_decode_image(frame: np.ndarray) -> list[dict]:
    """
    Full decode pipeline. Exits as soon as any code is found.

    Stage 1 — academic preprocessing (fastest, no upscaling)
    Stage 2 — upscaling ×2/3/4
    Stage 3 — rotations 90/180/270° (+ upscale)
    Stage 4 — Sobel ROI search
    Stage 5 — sliding window (very large images only)
    """
    if frame is None:
        return []

    orig_h, orig_w = frame.shape[:2]
    gray = _to_gray(frame)
    all_candidates: list[dict] = []

    # ── Stage 1: Academic preprocessing ──────────────────────────────────────
    # Techniques from class, ordered by speed:
    #   raw · Gaussian (1D conv) · FFT bandpass · edge-closed · adaptive · CLAHE · Otsu
    _, otsu_img  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sharpen_k    = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

    stage1 = [
        ("raw",      gray),
        ("gaussian", _PREP.gaussian_blur(gray)),          # 1D continuous convolution
        ("fft",      _PREP.fft_bandpass(gray)),           # frequency domain filtering
        ("edges",    _PREP.edge_detect(gray)),            # edge detection (Canny+close)
        ("adaptive", _PREP.adaptive_thresh(gray)),        # adaptive threshold
        ("clahe",    _PREP.clahe(gray)),                  # contrast enhancement
        ("otsu",     otsu_img),                           # global threshold
        ("sharpen",  cv2.filter2D(gray, -1, sharpen_k)), # sharpening (unsharp mask)
    ]
    for _name, proc in stage1:
        for r in _decode_gray(proc):
            all_candidates.append(r)

    if all_candidates:
        return _majority_vote(all_candidates)

    # ── Stage 2: Upscaling ────────────────────────────────────────────────────
    for scale in [2, 3, 4]:
        up = _upscale(gray, scale)
        for r in _decode_gray(up):
            r["polygon"] = _scale_poly(r["polygon"], scale)
            all_candidates.append(r)
        for r in _decode_gray(_PREP.gaussian_blur(up)):
            r["polygon"] = _scale_poly(r["polygon"], scale)
            all_candidates.append(r)

    if all_candidates:
        return _majority_vote(all_candidates)

    # ── Stage 3: Rotations ────────────────────────────────────────────────────
    for angle in [90, 180, 270]:
        rg  = _rotate_img(gray, angle)
        rh, rw = rg.shape[:2]
        for proc in [rg, _PREP.gaussian_blur(rg), _PREP.clahe(rg)]:
            for r in _decode_gray(proc):
                r["polygon"] = _rotate_poly_back(r["polygon"], angle, rw, rh)
                all_candidates.append(r)
        for scale in [2, 3]:
            up = _upscale(rg, scale)
            for r in _decode_gray(up):
                poly = _scale_poly(r["polygon"], scale)
                r["polygon"] = _rotate_poly_back(poly, angle, rw, rh)
                all_candidates.append(r)

    if all_candidates:
        return _majority_vote(all_candidates)

    # ── Stage 4: ROI-based search ─────────────────────────────────────────────
    for (rx, ry, rw, rh) in _find_barcode_rois(gray):
        roi = gray[ry:ry+rh, rx:rx+rw]
        if roi.size == 0:
            continue
        for proc in [roi, _PREP.gaussian_blur(roi), _PREP.clahe(roi)]:
            for scale in [1, 2, 3]:
                img_s = _upscale(proc, scale) if scale > 1 else proc
                for r in _decode_gray(img_s):
                    poly = _scale_poly(r["polygon"], scale)
                    r["polygon"] = [(x + rx, y + ry) for x, y in poly]
                    all_candidates.append(r)

    if all_candidates:
        return _majority_vote(all_candidates)

    # ── Stage 5: Sliding window (very large images) ───────────────────────────
    if orig_w > 1500 or orig_h > 1500:
        tile, step = 1000, 500
        for ty in range(0, orig_h, step):
            for tx in range(0, orig_w, step):
                t = gray[ty:min(ty+tile, orig_h), tx:min(tx+tile, orig_w)]
                if t.shape[0] < 50 or t.shape[1] < 50:
                    continue
                for proc in [t, _PREP.gaussian_blur(t), _PREP.clahe(t)]:
                    for r in _decode_gray(proc):
                        r["polygon"] = [(x + tx, y + ty) for x, y in r["polygon"]]
                        all_candidates.append(r)

    return _majority_vote(all_candidates)


# ── BarcodeScanner ────────────────────────────────────────────────────────────

class BarcodeScanner:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.text_prep    = TextPreprocessor()
        self.classifier   = TFIDFClassifier()
        self.bigram       = NGramModel(n=2)
        self.trigram      = NGramModel(n=3)
        self.history: list[dict]       = self._load_history()
        self.seen_codes: set[str]      = set()
        self._frame_n                  = 0
        self._last_results: list[dict] = []

    def _load_history(self) -> list[dict]:
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE) as f:
                    data = json.load(f)
                for entry in data:
                    tokens = self.text_prep.preprocess(entry["data"])
                    self.bigram.train(tokens)
                    self.trigram.train(tokens)
                    self.classifier.add_to_corpus(entry["data"])
                return data
            except Exception:
                pass
        return []

    def save_history(self):
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.history, f, indent=2)

    def decode_frame(self, frame: np.ndarray) -> list[dict]:
        self._frame_n += 1
        if self._frame_n % DECODE_EVERY_N != 0:
            return self._last_results
        raw_codes = try_decode_image(frame)
        results   = []
        for code in raw_codes:
            data = code["data"]
            if data in self.seen_codes:
                continue
            self.seen_codes.add(data)
            result = self._analyze(data, code)
            self.history.append(result)
            results.append(result)
            tokens = self.text_prep.preprocess(data)
            self.bigram.train(tokens)
            self.trigram.train(tokens)
            self.classifier.add_to_corpus(data)
        self._last_results = results if results else []
        return self._last_results

    def _analyze(self, data: str, code: dict) -> dict:
        product_info: dict = {}
        api_done = threading.Event()

        def _fetch():
            r = lookup_product(data)
            if r:
                product_info.update(r)
            api_done.set()

        threading.Thread(target=_fetch, daemon=True).start()

        tokens               = self.text_prep.preprocess(data)
        category, confidence = self.classifier.classify(data)

        if category == "Unknown" and data.isdigit() and len(data) in (8, 12, 13):
            category, confidence = "Food & Beverage", 0.5

        suggestions = self.bigram.predict_next(tokens) or self.trigram.predict_next(tokens)
        perplexity  = self.bigram.perplexity(tokens)

        api_done.wait(timeout=4.0)

        if product_info:
            parts    = [product_info.get(k, "") for k in ("name", "brand", "categories", "ingredients")]
            nlp_text = " ".join(p for p in parts if p)
            if nlp_text.strip():
                tokens               = self.text_prep.preprocess(nlp_text)
                category, confidence = self.classifier.classify(nlp_text, product_info)
                suggestions          = self.bigram.predict_next(tokens) or self.trigram.predict_next(tokens)
                perplexity           = self.bigram.perplexity(tokens)

        return {
            "timestamp":   datetime.now().isoformat(),
            "type":        code.get("type", "UNKNOWN"),
            "data":        data,
            "category":    category,
            "confidence":  confidence,
            "keywords":    tokens[:6],
            "suggestions": suggestions,
            "perplexity":  perplexity,
            "product":     product_info,
            "polygon":     code.get("polygon", []),
        }