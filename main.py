"""
Smart Barcode & QR Code Scanner
================================
Usage:
  python main.py                   # webcam mode
  python main.py --image photo.jpg # scan a single image

Controls (webcam mode):
  q  — quit
  s  — save scan history
  d  — toggle debug view
"""

import argparse
import math
import sys
import time

import cv2
import numpy as np

from core.scanner import BarcodeScanner, try_decode_image
from core.preprocessor import Preprocessor

# ── Colors BGR ────────────────────────────────────────────────────────────────
GREEN  = (0, 220, 80)
CYAN   = (255, 220, 0)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
DARK   = (20,  20,  20)
ORANGE = (0,  140, 255)
RED    = (50,  50, 220)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_corner_box(img, pts, color=GREEN, thickness=3, clen=30):
    if not pts or len(pts) < 4:
        return
    poly = np.array(pts, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(poly)
    segs = [
        ((x,   y),   (x+clen, y)),   ((x,   y),   (x,   y+clen)),
        ((x+w, y),   (x+w-clen, y)), ((x+w, y),   (x+w, y+clen)),
        ((x,   y+h), (x+clen, y+h)), ((x,   y+h), (x,   y+h-clen)),
        ((x+w, y+h), (x+w-clen, y+h)), ((x+w, y+h), (x+w, y+h-clen)),
    ]
    for p1, p2 in segs:
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    cv2.polylines(img, [poly], True, color, 1, cv2.LINE_AA)


def draw_info_panel(img, result, x, y):
    prod  = result.get("product", {})
    lines = [
        ("TYPE",     result.get("type", "?")),
        ("DATA",     result.get("data", "")[:30]),
        ("CATEGORY", f"{result.get('category','?')}  {result.get('confidence',0)*100:.0f}%"),
        ("KEYWORDS", ", ".join(result.get("keywords", [])[:4])),
    ]
    if prod.get("name"):       lines.append(("NAME",       prod["name"][:28]))
    if prod.get("brand"):      lines.append(("BRAND",      prod["brand"][:28]))
    if prod.get("nutriscore"): lines.append(("NUTRISCORE", prod["nutriscore"]))
    if prod.get("quantity"):   lines.append(("QUANTITY",   prod["quantity"]))

    pad, lh, pw = 8, 22, 265
    ph = pad * 2 + len(lines) * lh
    H, W = img.shape[:2]
    px = min(x + 10, W - pw - 5)
    py = max(y - 10, 40)
    if py + ph > H:
        py = H - ph - 5

    overlay = img.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), DARK, -1)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)
    cv2.rectangle(img, (px, py), (px + pw, py + ph), GREEN, 1, cv2.LINE_AA)
    cv2.rectangle(img, (px, py), (px + pw, py + lh), GREEN, -1)
    cv2.putText(img, "SMART SCANNER", (px + 6, py + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLACK, 1, cv2.LINE_AA)

    for i, (label, value) in enumerate(lines):
        ty = py + lh + 4 + i * lh
        cv2.putText(img, label,      (px + 6,  ty + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, CYAN,  1, cv2.LINE_AA)
        cv2.putText(img, str(value), (px + 85, ty + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)


def draw_hud(img, scan_count, fps):
    H, W = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W, 36), DARK, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.putText(img, f"SCANS: {scan_count}",    (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 1, cv2.LINE_AA)
    cv2.putText(img, "Q:Quit  S:Save  D:Debug", (W//2 - 95, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"FPS: {fps:.0f}",         (W - 90, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN,  1, cv2.LINE_AA)


def draw_finder(img, t):
    H, W  = img.shape[:2]
    cx, cy = W // 2, H // 2
    size   = 120
    pulse  = int(3 + 3 * abs(math.sin(t * 2)))
    alpha  = abs(math.sin(t * 2))
    color  = (0, int(150 + 70 * alpha), int(60 + 20 * alpha))
    segs = [
        ((cx-size, cy-size), (cx-size+40, cy-size)), ((cx-size, cy-size), (cx-size, cy-size+40)),
        ((cx+size, cy-size), (cx+size-40, cy-size)), ((cx+size, cy-size), (cx+size, cy-size+40)),
        ((cx-size, cy+size), (cx-size+40, cy+size)), ((cx-size, cy+size), (cx-size, cy+size-40)),
        ((cx+size, cy+size), (cx+size-40, cy+size)), ((cx+size, cy+size), (cx+size, cy+size-40)),
    ]
    for p1, p2 in segs:
        cv2.line(img, p1, p2, color, pulse, cv2.LINE_AA)
    sy = cy - size + int(2 * size * ((math.sin(t * 3) + 1) / 2))
    cv2.line(img, (cx - size, sy), (cx + size, sy), color, 1, cv2.LINE_AA)
    cv2.putText(img, "Point at a barcode or QR code",
                (cx - 135, cy + size + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def print_result(r):
    print("\n" + "═" * 55)
    print("  ✅  NEW SCAN DETECTED")
    print("═" * 55)
    print(f"  Type       : {r['type']}")
    print(f"  Data       : {r['data']}")
    print(f"  Category   : {r['category']}  ({r['confidence']*100:.0f}%)")
    print(f"  Keywords   : {', '.join(r['keywords']) or '—'}")
    print(f"  Suggestions: {', '.join(r['suggestions']) or '—'}")
    print(f"  Perplexity : {r['perplexity']}")
    prod = r.get("product", {})
    if prod:
        print("  ── Product Info ──────────────────────")
        for k in ("name", "brand", "nutriscore", "quantity", "countries"):
            if prod.get(k):
                print(f"  {k.capitalize():<12}: {str(prod[k])[:80]}")
        if prod.get("ingredients"):
            print(f"  Ingredients: {prod['ingredients'][:120]}…")
    print("═" * 55)


# ── Image mode ────────────────────────────────────────────────────────────────
def run_image(image_path: str):
    scanner = BarcodeScanner()
    frame   = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Could not open image: {image_path}")
        sys.exit(1)

    print(f"🔍 Scanning {image_path} ({frame.shape[1]}x{frame.shape[0]})…")
    print("   Running full detection pipeline — please wait…")

    raw_codes = try_decode_image(frame)

    results = []
    for code in raw_codes:
        result = scanner._analyze(code["data"], code)
        results.append(result)
        print_result(result)

    if not results:
        print("⚠️  No barcode or QR code found.")
    else:
        scanner.history.extend(results)
        scanner.save_history()

    # Draw annotated image
    annotated = frame.copy()
    # Resize for display if too large
    dh, dw = annotated.shape[:2]
    max_display = 1200
    scale = min(max_display / dw, max_display / dh, 1.0)
    display = cv2.resize(annotated, (int(dw*scale), int(dh*scale))) if scale < 1 else annotated.copy()

    for r in results:
        pts = r.get("polygon")
        if pts and scale < 1:
            pts = [(int(x*scale), int(y*scale)) for x,y in pts]
        draw_corner_box(display, pts)
        x = pts[0][0] if pts else 10
        y = pts[0][1] if pts else 80
        draw_info_panel(display, r, x, y)

    if not results:
        cv2.putText(display, "NO CODE FOUND", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, RED, 2, cv2.LINE_AA)

    cv2.imshow("Smart Scanner", display)
    print("\nPress any key to close…")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Webcam mode ───────────────────────────────────────────────────────────────
def run_webcam():
    scanner      = BarcodeScanner()
    preprocessor = Preprocessor()
    cap          = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("📷  Webcam started.  Q=quit  S=save  D=debug")

    show_debug = False
    fps        = 0.0
    prev_time  = time.time()
    t_start    = time.time()
    printed    = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now       = time.time()
        fps       = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
        prev_time = now
        t         = now - t_start

        results = scanner.decode_frame(frame)

        for r in results:
            if r["data"] not in printed:
                print_result(r)
                printed.add(r["data"])

        display = frame.copy()

        if not results:
            draw_finder(display, t)
        else:
            for r in results:
                pts = r.get("polygon")
                draw_corner_box(display, pts)
                x = pts[0][0] if pts else 10
                y = pts[0][1] if pts else 80
                draw_info_panel(display, r, x, y)

        draw_hud(display, len(scanner.history), fps)

        if show_debug:
            _, dbg    = preprocessor.process(frame)
            small_fft  = cv2.resize(dbg["fft"],   (200, 150))
            small_edge = cv2.resize(dbg["edges"],  (200, 150))
            fft_bgr    = cv2.cvtColor(small_fft,  cv2.COLOR_GRAY2BGR)
            edge_bgr   = cv2.cvtColor(small_edge, cv2.COLOR_GRAY2BGR)
            cv2.putText(fft_bgr,  "FFT",   (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)
            cv2.putText(edge_bgr, "EDGES", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)
            display[36:336, 0:200] = np.vstack([fft_bgr, edge_bgr])

        cv2.imshow("Smart Barcode & QR Scanner", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            scanner.save_history()
            print("💾 Saved.")
        elif key == ord("d"):
            show_debug = not show_debug

    scanner.save_history()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Done. {len(scanner.history)} unique code(s) scanned.")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", type=str, default=None)
    args = parser.parse_args()
    if args.image:
        run_image(args.image)
    else:
        run_webcam()


if __name__ == "__main__":
    main()
