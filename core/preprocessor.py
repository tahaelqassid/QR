"""
Computer Vision Preprocessing Pipeline
=======================================
Implements exactly the four techniques studied in class:

  1. 1D Continuous Convolution  — Gaussian separable filter
  2. Frequency Domain Filtering — FFT band-pass (remove DC + high-freq noise)
  3. Edge Detection             — Canny + morphological closing
  4. Adaptive Thresholding      — local binarisation (edge-aware)

Each method is self-contained. The `process()` method runs all four
and returns a dict of debug images for the pipeline visualizer.
"""

import cv2
import numpy as np


class Preprocessor:
    """
    Applies the four CV techniques from lecture, in order:
      gaussian_blur  → 1D separable convolution
      fft_bandpass   → frequency-domain bandpass mask
      edge_detect    → Canny + morphological closing
      adaptive_thresh→ local binarisation
    """

    def __init__(self, sigma: float = 1.4, kernel_size: int = 7):
        self.sigma       = sigma
        self.kernel_size = kernel_size
        self._kernel     = self._build_gaussian_kernel()
        self._clahe      = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # ── 1. 1D Continuous Convolution ─────────────────────────────────────────
    def _build_gaussian_kernel(self) -> np.ndarray:
        """
        Build a 1-D Gaussian kernel (row vector).
        Discrete form of the continuous Gaussian:
            g(x) = exp(-x² / 2σ²)
        Normalised so sum = 1.  Applied as separable H then V passes.
        """
        half = self.kernel_size // 2
        x    = np.arange(-half, half + 1, dtype=np.float32)
        g    = np.exp(-(x ** 2) / (2 * self.sigma ** 2))
        g   /= g.sum()
        return g.reshape(1, -1)   # row vector for cv2.filter2D

    def gaussian_blur(self, gray: np.ndarray) -> np.ndarray:
        """
        Separable 1-D Gaussian convolution.

        Theory (1D continuous convolution):
            (f * g)(x) = ∫ f(τ) g(x − τ) dτ

        Implementation: two cv2.filter2D passes (horizontal then vertical).
        Separability of the 2D Gaussian means this is equivalent to the
        full 2D convolution but runs in O(k·N) instead of O(k²·N).
        """
        out = cv2.filter2D(gray, -1, self._kernel)      # horizontal pass
        out = cv2.filter2D(out,  -1, self._kernel.T)    # vertical pass
        return out

    # ── 2. Frequency Domain Filtering ────────────────────────────────────────
    def fft_bandpass(self, gray: np.ndarray) -> np.ndarray:
        """
        FFT → shift DC to centre → annular band-pass mask → IFFT.

        Theory (Frequency Domain Filtering):
            F(u,v) = FFT{ f(x,y) }
            G(u,v) = H(u,v) · F(u,v)      ← multiplication in freq domain
            g(x,y) = IFFT{ G(u,v) }

        The annular mask H keeps mid-frequencies (the stripe pattern of a
        barcode) and removes:
          • DC / very-low freq  → flat background illumination
          • Very-high freq      → sensor noise, fine texture

        r_min / r_max are in pixels from the centre of the spectrum.
        """
        H, W   = gray.shape[:2]
        f      = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)

        cy, cx = H // 2, W // 2
        r_min  = max(8,  min(H, W) // 20)
        r_max  = min(H, W) // 2 - 10
        y_idx, x_idx = np.ogrid[:H, :W]
        r      = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
        mask   = ((r >= r_min) & (r <= r_max)).astype(np.float32)

        filtered = np.fft.ifft2(np.fft.ifftshift(fshift * mask))
        result   = np.abs(filtered).astype(np.uint8)
        return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

    # ── 3. Edge Detection ─────────────────────────────────────────────────────
    def edge_detect(self, gray: np.ndarray) -> np.ndarray:
        """
        Canny edge detector + morphological closing.

        Theory (Edge Detection):
            1. Smooth with Gaussian to reduce noise
            2. Compute gradient magnitude & direction (Sobel)
            3. Non-maximum suppression → thin edges
            4. Hysteresis thresholding → keep strong/connected edges

        Morphological CLOSE (dilate then erode) connects the short bar-edges
        of a 1D barcode into solid blobs the detector can lock onto.
        """
        blurred = self.gaussian_blur(gray)
        edges   = cv2.Canny(blurred, 50, 150)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        return closed

    # ── 4. Adaptive Thresholding (edge-aware binarisation) ───────────────────
    def adaptive_thresh(self, gray: np.ndarray) -> np.ndarray:
        """
        Gaussian-blur then adaptive (local) threshold.

        Each pixel's threshold = weighted mean of its neighbourhood − C.
        Handles uneven illumination that defeats global Otsu thresholding.
        Used to produce a clean binary image for the barcode decoder under
        challenging real-world lighting.
        """
        blurred = self.gaussian_blur(gray)
        return cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    
    def clahe(self, gray: np.ndarray) -> np.ndarray:

        """
        Contrast Limited Adaptive Histogram Equalization (CLAHE).

        Enhances local contrast, especially useful for low-light or
        low-contrast barcode regions.
        """
        
        return self._clahe.apply(gray)

    # ── Full pipeline: run all four + return debug images ─────────────────────
    def process(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Run the complete pipeline.

        Returns
        -------
        best : np.ndarray
            The Gaussian-blurred image (best balance for decoding).
        debug : dict
            Keys: 'gray', 'gaussian', 'fft', 'edges', 'adaptive'
            All are single-channel uint8 for display / further processing.
        """
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        gaussian = self.gaussian_blur(gray)
        fft_img  = self.fft_bandpass(gray)
        edges    = self.edge_detect(gray)
        adaptive = self.adaptive_thresh(gray)

        debug = {
            "gray":     gray,
            "gaussian": gaussian,
            "fft":      fft_img,
            "edges":    edges,
            "adaptive": adaptive,
        }
        return gaussian, debug