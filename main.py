"""
Smart Barcode & QR Code Scanner
================================
Usage:
  python main.py                   # webcam mode
  python main.py --image path.jpg  # single image mode

Webcam keys:
  Q — quit
  P — toggle pipeline overlay (academic techniques, same window)
  S — save scan history
"""

import argparse
import threading
import time

import cv2
import numpy as np

from core.scanner import BarcodeScanner, try_decode_image, DECODE_EVERY_N
from core.preprocessor import Preprocessor

# ── Palette ────────────────────────────────────────────────────────────────────
C_GREEN  = (50,  210,  80)
C_CYAN   = (200, 200,  20)
C_WHITE  = (255, 255, 255)
C_YELLOW = (30,  210, 255)
C_GRAY   = (150, 150, 150)
C_DARK   = (18,   18,  18)
C_DIM    = (60,   60,  60)
C_RED    = (60,   60, 220)


# ═══════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def draw_box(img: np.ndarray, pts: list, color=C_GREEN, thickness=3):
    """Draw a polygon with corner accent ticks."""
    if not pts or len(pts) < 2:
        return
    arr = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [arr], True, color, thickness, cv2.LINE_AA)

    # Corner L-shaped ticks
    n = len(pts)
    accent = 16
    for i in range(n):
        p  = np.array(pts[i],           dtype=float)
        p1 = np.array(pts[(i-1) % n],   dtype=float)
        p2 = np.array(pts[(i+1) % n],   dtype=float)
        for nb in [p1, p2]:
            d = nb - p
            norm = np.linalg.norm(d)
            if norm > 0:
                d = d / norm * accent
                cv2.line(img, tuple(p.astype(int)),
                         tuple((p + d).astype(int)),
                         C_WHITE, thickness + 1, cv2.LINE_AA)


def _put(img, text, x, y, color=C_WHITE, scale=0.50, bold=False):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2 if bold else 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
#  INFO PANEL  (left strip, 300 px wide)
# ═══════════════════════════════════════════════════════════════════════════════

PANEL_W = 300

def draw_info_panel(img: np.ndarray, results: list[dict], fps: float = None):
    """
    Render a semi-transparent info panel on the left side of *img* (in-place).
    Shows FPS, then one block per detected code with all NLP fields.
    """
    h = img.shape[0]

    # Semi-transparent dark background
    overlay = img[:, :PANEL_W].copy()
    cv2.rectangle(overlay, (0, 0), (PANEL_W, h), C_DARK, -1)
    cv2.addWeighted(overlay, 0.75, img[:, :PANEL_W], 0.25, 0, img[:, :PANEL_W])
    cv2.line(img, (PANEL_W, 0), (PANEL_W, h), C_DIM, 1)

    y   = 22
    lh  = 24        # line height
    pad = 8         # left padding
    max_y = h - lh

    def line(text, color=C_WHITE, scale=0.48, bold=False):
        nonlocal y
        if y > max_y:
            return
        _put(img, text, pad, y, color, scale, bold)
        y += lh

    def separator():
        nonlocal y
        if y > max_y:
            return
        cv2.line(img, (pad, y - 4), (PANEL_W - pad, y - 4), C_DIM, 1)
        y += 6

    # ── FPS ──
    if fps is not None:
        line(f"FPS  {int(fps)}", C_YELLOW, 0.56, bold=True)
        separator()

    if not results:
        line("Searching...", C_GRAY, 0.46)
        return

    for idx, r in enumerate(results):
        if y > max_y:
            break

        code_type = r.get("type", "?")
        color     = C_GREEN if "QR" in code_type else C_CYAN

        # Type badge
        line(f"[{idx+1}]  {code_type}", color, 0.54, bold=True)

        # Raw data (wrap at ~34 chars)
        data = r.get("data", "")
        for i in range(0, min(len(data), 102), 34):
            line("  " + data[i:i+34], C_WHITE, 0.44)
        if len(data) > 102:
            line("  ...", C_GRAY, 0.42)

        # NLP fields
        cat  = r.get("category", "")
        conf = r.get("confidence", 0)
        line(f"  Category : {cat}", C_GRAY, 0.44)
        line(f"  Conf.    : {conf*100:.0f}%", C_GRAY, 0.44)

        kw = ", ".join(r.get("keywords", [])[:4])
        if kw:
            line(f"  Keywords : {kw[:32]}", C_GRAY, 0.42)

        sugg = r.get("suggestions", [])
        if sugg:
            line(f"  Suggests : {' '.join(sugg[:3])[:28]}", C_GRAY, 0.42)

        perp = r.get("perplexity", None)
        if perp and perp != float("inf"):
            line(f"  Perplexity: {perp:.1f}", C_GRAY, 0.42)

        # Product info (from Open Food Facts)
        prod = r.get("product", {})
        if prod.get("name"):
            line(f"  Product  : {prod['name'][:28]}", C_YELLOW, 0.46)
        if prod.get("brand"):
            line(f"  Brand    : {prod['brand'][:28]}", C_YELLOW, 0.44)
        if prod.get("nutriscore"):
            line(f"  Nutriscore: {prod['nutriscore']}", C_YELLOW, 0.44)
        if prod.get("quantity"):
            line(f"  Qty      : {prod['quantity'][:20]}", C_GRAY, 0.42)

        separator()


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE OVERLAY  (right 2/3 of the same window, 4×2 grid)
# ═══════════════════════════════════════════════════════════════════════════════

_PREP  = Preprocessor()

PIPE_STAGES = [
    # (label,  function)
    ("1. Grayscale",         lambda g: g),
    ("2. Gaussian (1D conv)",lambda g: _PREP.gaussian_blur(g)),
    ("3. FFT Bandpass",      lambda g: _PREP.fft_bandpass(g)),
    ("4. Edge Detection",    lambda g: _PREP.edge_detect(g)),
    ("5. Adaptive Thresh",   lambda g: _PREP.adaptive_thresh(g)),
    ("6. CLAHE",             lambda g: _PREP.clahe(g)),
    ("7. Otsu Threshold",    lambda g: cv2.threshold(g, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
    ("8. Sharpen",           lambda g: cv2.filter2D(g, -1,
                                        np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],
                                        dtype=np.float32))),
]


def build_pipeline_overlay(frame: np.ndarray,
                            cell_w: int = 213,
                            cell_h: int = 160) -> np.ndarray:
    """
    Build a 4-column × 2-row grid of pipeline stages.
    Returns a BGR image of size (cell_h*2, cell_w*4).
    """
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
    cols   = 4
    grid_w = cell_w * cols
    grid_h = cell_h * 2
    grid   = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for i, (label, fn) in enumerate(PIPE_STAGES):
        r = i // cols
        c = i  % cols
        try:
            stage = fn(gray)
        except Exception:
            stage = gray
        thumb = cv2.resize(stage, (cell_w, cell_h))
        if thumb.ndim == 2:
            thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)

        x0, y0 = c * cell_w, r * cell_h
        grid[y0:y0+cell_h, x0:x0+cell_w] = thumb

        # Label bar
        cv2.rectangle(grid, (x0, y0), (x0+cell_w, y0+20), (20, 20, 20), -1)
        cv2.putText(grid, label, (x0+4, y0+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

    # Grid lines
    for i in range(1, cols):
        cv2.line(grid, (i*cell_w, 0), (i*cell_w, grid_h), (55, 55, 55), 1)
    cv2.line(grid, (0, cell_h), (grid_w, cell_h), (55, 55, 55), 1)

    return grid


def composite_pipeline(base: np.ndarray, pipe_grid: np.ndarray) -> np.ndarray:
    """
    Place the pipeline grid into the RIGHT portion of *base* (after the panel).
    Returns a new image the same size as *base*.
    """
    out    = base.copy()
    bh, bw = out.shape[:2]
    # Available area: x from PANEL_W to bw
    avail_w = bw - PANEL_W
    avail_h = bh

    # Scale grid to fit
    gh, gw = pipe_grid.shape[:2]
    scale  = min(avail_w / gw, avail_h / gh)
    new_w  = int(gw * scale)
    new_h  = int(gh * scale)
    resized = cv2.resize(pipe_grid, (new_w, new_h))

    # Centre vertically
    y_off = (avail_h - new_h) // 2
    x_off = PANEL_W

    out[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  PRINT TO TERMINAL
# ═══════════════════════════════════════════════════════════════════════════════

def print_result(r: dict):
    print("\n" + "=" * 52)
    print(f"  Type     : {r.get('type')}")
    print(f"  Data     : {r.get('data')}")
    print(f"  Category : {r.get('category')}  ({r.get('confidence',0)*100:.0f}%)")
    print(f"  Keywords : {', '.join(r.get('keywords', []))}")
    prod = r.get("product", {})
    if prod.get("name"):
        print(f"  Product  : {prod.get('name')}  |  {prod.get('brand','')}")
        if prod.get("nutriscore"):
            print(f"  Nutriscore: {prod['nutriscore']}")
    print("=" * 52)


# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_image(image_path: str):
    scanner = BarcodeScanner()
    frame   = cv2.imread(image_path)

    if frame is None:
        print(f"[ERROR] Cannot open: {image_path}")
        return

    raw_codes = try_decode_image(frame)
    results   = []
    for code in raw_codes:
        r = scanner._analyze(code["data"], code)
        results.append(r)
        print_result(r)

    if not results:
        print("[INFO] No code detected.")

    # Draw boxes on a copy
    display = frame.copy()
    for r in results:
        color = C_GREEN if "QR" in r.get("type", "") else C_CYAN
        draw_box(display, r.get("polygon"), color)

    # Resize so the full image fits in ~900 px height, then add panel
    max_h  = 720
    dh, dw = display.shape[:2]
    if dh > max_h:
        scale   = max_h / dh
        display = cv2.resize(display, (int(dw*scale), int(dh*scale)))

    # Expand canvas left by PANEL_W
    canvas = np.zeros((display.shape[0], display.shape[1] + PANEL_W, 3), dtype=np.uint8)
    canvas[:, PANEL_W:] = display
    draw_info_panel(canvas, results)

    cv2.imshow("Scanner — Result", canvas)
    print("\nP = pipeline view  |  any other key = close")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("p"):
            pipe  = build_pipeline_overlay(frame)
            combo = composite_pipeline(canvas, pipe)
            cv2.imshow("Scanner — Result", combo)
        else:
            break

    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════════
#  WEBCAM MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_webcam():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # keep capture buffer minimal

    scanner = BarcodeScanner()

    # ── Background decode thread ──────────────────────────────────────────────
    _lock         = threading.Lock()
    _decode_buf   = [None]   # latest frame posted for decoding
    _last_results = []
    _decode_req   = threading.Event()
    _running      = [True]

    def _decode_worker():
        while _running[0]:
            _decode_req.wait(timeout=0.1)
            _decode_req.clear()
            with _lock:
                f = _decode_buf[0]
            if f is None:
                continue
            raw = try_decode_image(f)
            new = []
            for code in raw:
                r = scanner._analyze(code["data"], code)
                new.append(r)
                print_result(r)
            with _lock:
                _last_results.clear()
                _last_results.extend(new)

    t = threading.Thread(target=_decode_worker, daemon=True)
    t.start()

    frame_id  = 0
    prev_time = time.time()
    fps       = 0.0
    show_pipe = False

    print("Scanner running")
    print("  Q = quit  |  P = toggle pipeline overlay  |  S = save history")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Post every Nth frame to the decode worker (non-blocking)
        if frame_id % DECODE_EVERY_N == 0:
            with _lock:
                _decode_buf[0] = frame.copy()
            _decode_req.set()

        with _lock:
            results = list(_last_results)

        # ── Compose display ───────────────────────────────────────────────────
        # Canvas = PANEL_W (info) + frame width
        cam_h, cam_w = frame.shape[:2]
        canvas = np.zeros((cam_h, PANEL_W + cam_w, 3), dtype=np.uint8)
        canvas[:, PANEL_W:] = frame

        if show_pipe:
            # Build pipeline grid and composite into right portion
            pipe  = build_pipeline_overlay(frame,
                                           cell_w=cam_w  // 4,
                                           cell_h=cam_h  // 2)
            canvas = composite_pipeline(canvas, pipe)
        else:
            # Draw bounding boxes (on the camera portion)
            for r in results:
                color = C_GREEN if "QR" in r.get("type", "") else C_CYAN
                pts   = r.get("polygon")
                if pts:
                    # Shift x by PANEL_W since camera is offset in canvas
                    shifted = [(x + PANEL_W, y) for x, y in pts]
                    draw_box(canvas, shifted, color)

        # FPS
        now  = time.time()
        fps  = 0.85 * fps + 0.15 / max(now - prev_time, 1e-6)
        prev_time = now

        draw_info_panel(canvas, results, fps)

        cv2.imshow("Scanner", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            show_pipe = not show_pipe
        elif key == ord("s"):
            scanner.save_history()
            print("[INFO] History saved.")

    _running[0] = False
    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Smart Barcode & QR Scanner")
    parser.add_argument("--image", "-i", help="Path to image file (omit for webcam)")
    args = parser.parse_args()

    if args.image:
        run_image(args.image)
    else:
        run_webcam()


if __name__ == "__main__":
    main()