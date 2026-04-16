"""
BarcodeScanner — production-grade OpenCV-only barcode/QR scanner.

Decode pipeline per frame (ordered fastest → most aggressive):
  1.  Raw grayscale
  2.  Sharpened
  3.  1D Gaussian blur       (Continuous Convolution lecture)
  4.  CLAHE contrast boost
  5.  FFT band-pass          (Frequency Domain Filtering lecture)
  6.  Adaptive threshold     (Edge Detection lecture)
  7.  Otsu threshold
  8.  Denoised
  9.  Morphological gradient
  10. 2× / 3× / 4× upscale  (small/distant codes)
  11. Rotation 90 / 180 / 270°
  12. Sharpened + rotations
  13. CLAHE + rotations
  14. ROI search via barcode-gradient detection (shelf/multi-product images)
  15. Tiled sliding-window   (code in one corner of large image)
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
DECODE_EVERY_N = 5


# ── OpenCV detectors ──────────────────────────────────────────────────────────

_QR_DETECTOR = cv2.QRCodeDetector()
try:
    _BARCODE_DETECTOR = cv2.barcode.BarcodeDetector()
except AttributeError:
    _BARCODE_DETECTOR = None


def _decode_gray(gray: np.ndarray) -> list[dict]:
    """Run both detectors on a grayscale image. Returns list of result dicts."""
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
                    poly = [(int(p[0]), int(p[1])) for p in points[i]] if points is not None and i < len(points) else []
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
                        btype = str(type_list[i]) if type_list is not None and i < len(type_list) else "BARCODE"
                        poly = [(int(p[0]), int(p[1])) for p in pts2[i]] if pts2 is not None and i < len(pts2) else []
                        results.append({"data": data, "type": btype, "polygon": poly})
        except Exception:
            pass

    return results


# ── Preprocessing helpers ─────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

def _sharpen(gray: np.ndarray) -> np.ndarray:
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(gray, -1, k)

def _clahe(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)

def _otsu(gray: np.ndarray) -> np.ndarray:
    _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out

def _adaptive(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)

def _denoised(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=10)

def _morph_gradient(gray: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k)

def _rotate_img(img: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:  return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def _upscale(img: np.ndarray, factor: int) -> np.ndarray:
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

def _scale_poly(poly: list, factor: float) -> list:
    return [(int(x / factor), int(y / factor)) for x, y in poly]

def _rotate_poly_back(poly: list, angle: int, w: int, h: int) -> list:
    """Rotate polygon points back to original image coordinates."""
    if not poly:
        return poly
    cx, cy = w / 2, h / 2
    rad = -np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    out = []
    for x, y in poly:
        x -= cx; y -= cy
        rx = x * cos_a - y * sin_a + cx
        ry = x * sin_a + y * cos_a + cy
        out.append((int(rx), int(ry)))
    return out


def _find_barcode_rois(gray: np.ndarray) -> list[tuple]:
    """
    Use gradient analysis to locate probable barcode/QR regions.
    Returns list of (x, y, w, h) bounding boxes.
    """
    # Horizontal gradient dominates in barcodes (many vertical lines)
    sobelx = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    sobely = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    diff = cv2.subtract(sobelx, sobely)
    _, diff = cv2.threshold(diff, 0, 255, cv2.THRESH_TOZERO)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=3)
    closed = cv2.dilate(closed, None, iterations=6)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    ih, iw = gray.shape[:2]
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        # Filter: reasonable size, not the whole image
        if area < 400 or area > (iw * ih * 0.95):
            continue
        # Add padding
        pad = 20
        x1 = max(0, x - pad);  y1 = max(0, y - pad)
        x2 = min(iw, x+cw+pad); y2 = min(ih, y+ch+pad)
        rois.append((x1, y1, x2 - x1, y2 - y1))
    return rois


# ── Main decode pipeline ──────────────────────────────────────────────────────

def _majority_vote(candidates: list[dict]) -> list[dict]:
    """
    When multiple preprocessing stages produce results, deduplicate and
    prefer codes seen most often (majority vote for confidence).
    """
    if not candidates:
        return []
    counts = Counter(c["data"] for c in candidates)
    seen = set()
    out = []
    # Sort by frequency descending, preserve first polygon seen
    for c in sorted(candidates, key=lambda x: -counts[x["data"]]):
        if c["data"] not in seen:
            seen.add(c["data"])
            out.append(c)
    return out


def try_decode_image(frame: np.ndarray) -> list[dict]:
    """
    Full production pipeline — tries every known technique until codes are found.
    Returns deduplicated list of decoded results with polygon coordinates
    mapped back to the original frame.
    """
    if frame is None:
        return []

    orig_h, orig_w = frame.shape[:2]
    gray = _to_gray(frame)
    all_candidates: list[dict] = []

    # ── Stage 1: Basic preprocessing (fast) ───────────────────────────────────
    basic = [
        ("raw",       gray),
        ("sharpen",   _sharpen(gray)),
        ("gaussian",  _gaussian_blur(gray)),   # 1D convolution from lecture
        ("clahe",     _clahe(gray)),
        ("fft",       _fft_bandpass(gray)),    # freq domain from lecture
        ("adaptive",  _adaptive(gray)),        # edge detection from lecture
        ("otsu",      _otsu(gray)),
        ("denoise",   _denoised(gray)),
        ("gradient",  _morph_gradient(gray)),
    ]
    for name, proc in basic:
        results = _decode_gray(proc)
        for r in results:
            all_candidates.append(r)

    if all_candidates:
        return _majority_vote(all_candidates)

    # ── Stage 2: Upscaling (small/distant codes) ──────────────────────────────
    for scale in [2, 3, 4]:
        up = _upscale(gray, scale)
        results = _decode_gray(up)
        for r in results:
            r["polygon"] = _scale_poly(r["polygon"], scale)
            all_candidates.append(r)
        # also try sharpen + upscale
        results2 = _decode_gray(_upscale(_sharpen(gray), scale))
        for r in results2:
            r["polygon"] = _scale_poly(r["polygon"], scale)
            all_candidates.append(r)

    if all_candidates:
        return _majority_vote(all_candidates)

    # ── Stage 3: Rotations ────────────────────────────────────────────────────
    for angle in [90, 180, 270]:
        rot_gray  = _rotate_img(gray, angle)
        rot_sharp = _rotate_img(_sharpen(gray), angle)
        rot_clahe = _rotate_img(_clahe(gray), angle)
        rh, rw = rot_gray.shape[:2]

        for proc in [rot_gray, rot_sharp, rot_clahe]:
            results = _decode_gray(proc)
            for r in results:
                r["polygon"] = _rotate_poly_back(r["polygon"], angle, rw, rh)
                all_candidates.append(r)

        # rotated + upscale
        for scale in [2, 3]:
            up = _upscale(rot_gray, scale)
            results = _decode_gray(up)
            for r in results:
                poly = _scale_poly(r["polygon"], scale)
                r["polygon"] = _rotate_poly_back(poly, angle, rw, rh)
                all_candidates.append(r)

    if all_candidates:
        return _majority_vote(all_candidates)

    # ── Stage 4: ROI-based search (shelf/multi-product images) ────────────────
    rois = _find_barcode_rois(gray)
    for (rx, ry, rw, rh) in rois:
        roi_gray = gray[ry:ry+rh, rx:rx+rw]
        if roi_gray.size == 0:
            continue
        for proc in [roi_gray, _sharpen(roi_gray), _clahe(roi_gray)]:
            for scale in [1, 2, 3]:
                img_s = _upscale(proc, scale) if scale > 1 else proc
                results = _decode_gray(img_s)
                for r in results:
                    poly = _scale_poly(r["polygon"], scale)
                    r["polygon"] = [(x + rx, y + ry) for x, y in poly]
                    all_candidates.append(r)
                # also try rotations of ROI
                for angle in [90, 180, 270]:
                    rot = _rotate_img(img_s, angle)
                    sh, sw = rot.shape[:2]
                    results2 = _decode_gray(rot)
                    for r in results2:
                        poly = _rotate_poly_back(r["polygon"], angle, sw, sh)
                        poly = _scale_poly(poly, scale)
                        r["polygon"] = [(x + rx, y + ry) for x, y in poly]
                        all_candidates.append(r)

    if all_candidates:
        return _majority_vote(all_candidates)

    # ── Stage 5: Sliding window (code in corner of very large image) ──────────
    if orig_w > 1500 or orig_h > 1500:
        tile_size = 1000
        step      = 500
        for ty in range(0, orig_h, step):
            for tx in range(0, orig_w, step):
                tile = gray[ty:min(ty+tile_size, orig_h), tx:min(tx+tile_size, orig_w)]
                if tile.shape[0] < 50 or tile.shape[1] < 50:
                    continue
                for proc in [tile, _sharpen(tile), _clahe(tile)]:
                    results = _decode_gray(proc)
                    for r in results:
                        r["polygon"] = [(x + tx, y + ty) for x, y in r["polygon"]]
                        all_candidates.append(r)

    return _majority_vote(all_candidates)


# ── Preprocessor wrappers (use class methods) ─────────────────────────────────
_PREPROCESSOR = None

def _get_preprocessor():
    global _PREPROCESSOR
    if _PREPROCESSOR is None:
        _PREPROCESSOR = Preprocessor()
    return _PREPROCESSOR

def _gaussian_blur(gray):
    return _get_preprocessor().gaussian_blur(gray)

def _fft_bandpass(gray):
    return _get_preprocessor().fft_bandpass(gray)


# ── BarcodeScanner class ──────────────────────────────────────────────────────

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

    # ── History ───────────────────────────────────────────────────────────────
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

    # ── Frame decode (webcam) ─────────────────────────────────────────────────
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

    # ── NLP analysis ──────────────────────────────────────────────────────────
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
                category, confidence = self.classifier.classify(nlp_text)
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

    def draw_overlay(self, frame: np.ndarray, results: list[dict]) -> np.ndarray:
        return frame
