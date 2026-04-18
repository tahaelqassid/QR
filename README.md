# Smart Barcode & QR Code Scanner 🔍

A Computer Vision + NLP barcode/QR scanner.  
**100 % OpenCV** — no pyzbar, no zxing.  
Supports **webcam live scanning** and **image file scanning**.

---

## ⚡ Quick Start

### Step 1 — Install Python packages

```bash
pip install -r requirements.txt
```

> Requires Python 3.10+.  
> Uses `opencv-contrib-python` (includes `cv2.barcode.BarcodeDetector`).

---

### Step 2 — Run it

**Webcam mode (live scanner):**
```bash
python main.py
```

**Scan an image file:**
```bash
python main.py --image path/to/photo.jpg
```

---

## 🎮 Controls (webcam mode)

| Key | Action |
|-----|--------|
| `Q` | Quit and save scan history |
| `S` | Save scan history now |
| `D` | Toggle debug panel (FFT / Edge view) |

---

## 🧠 Computer Vision Pipeline

Each decode attempt walks through four stages (matching course lectures):

| Stage | Lecture | OpenCV call |
|-------|---------|-------------|
| 1. Raw grayscale | — | `cv2.cvtColor` |
| 2. 1D Gaussian blur | **1D Continuous Convolution** | `cv2.filter2D` (separable) |
| 3. FFT band-pass | **Frequency Domain Filtering** | `np.fft.fft2 / ifft2` |
| 4. Edge threshold | **Edge Detection** | `cv2.Canny` + `adaptiveThreshold` |
| 5. 2× upscale | — | `cv2.resize` |

The first stage that produces a successful decode wins.

---

## 📦 What it does

- Detects **QR codes** (`cv2.QRCodeDetector`)
- Detects **EAN-13, EAN-8, UPC-A, Code 128** and more (`cv2.barcode.BarcodeDetector`)
- Applies the full CV pipeline: Gaussian → FFT → Edge detection
- Runs **NLP analysis**: TF-IDF classification, N-gram suggestions
- Looks up product info from **Open Food Facts** (no API key needed)
- Saves full **scan history** to `data/scan_history.json`

---

## 📁 Project Structure

```
smart_scanner/
├── main.py                  ← Entry point
├── requirements.txt
├── core/
│   ├── preprocessor.py      ← CV pipeline (Gaussian, FFT, Canny)
│   └── scanner.py           ← BarcodeScanner — OpenCV decode + NLP
├── nlp/
│   └── analyzer.py          ← TextPreprocessor, TF-IDF, N-gram, product lookup
├── data/
│   └── scan_history.json    ← Auto-generated after first scan
└── tests/
    └── test_pipeline.py
```

---

## ❓ Troubleshooting

| Problem | Fix |
|---------|-----|
| `AttributeError: module 'cv2' has no attribute 'barcode'` | Install **opencv-contrib-python**, not plain opencv-python |
| `No barcode found` in image mode | Ensure good lighting; code fully in frame |
| Webcam not opening | Try `python main.py --image test.jpg` first |
 

 ########## 
    sudo mv /usr/local/include/google /usr/local/include/google_bak
    sudo mv /usr/local/include/absl /usr/local/include/absl_bak 
 then install openncv and after that restorethem 

 sudo mv /usr/local/include/google_bak /usr/local/include/google
sudo mv /usr/local/include/absl_bak /usr/local/include/absl
