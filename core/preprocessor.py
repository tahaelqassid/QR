"""
Computer Vision preprocessing pipeline.

Implements the three techniques from class:
  1. 1D Continuous Convolution  — gaussian_blur()
  2. Frequency Domain Filtering — fft_bandpass()
  3. Edge Detection             — edge_detect() / edge_based_threshold()
"""

import numpy as np
import cv2


class Preprocessor:
    def __init__(self, sigma: float = 1.4, kernel_size: int = 7):
        self.sigma       = sigma
        self.kernel_size = kernel_size
        self._kernel     = self._make_gaussian_kernel()

    # ── 1. 1D Gaussian kernel (Continuous Convolution) ────────────────────────
    def _make_gaussian_kernel(self) -> np.ndarray:
        half = self.kernel_size // 2
        x    = np.arange(-half, half + 1, dtype=np.float32)
        g    = np.exp(-(x ** 2) / (2 * self.sigma ** 2))
        return (g / g.sum()).reshape(1, -1)   # row vector → separable kernel

    def gaussian_blur(self, gray: np.ndarray) -> np.ndarray:
        """
        Separable 1D Gaussian convolution (horizontal then vertical).
        Mirrors the 1D continuous convolution from lecture:
            (f * g)(x) = integral f(tau) g(x - tau) d(tau)
        implemented discretely via cv2.filter2D.
        """
        blurred = cv2.filter2D(gray, -1, self._kernel)      # horizontal pass
        blurred = cv2.filter2D(blurred, -1, self._kernel.T) # vertical pass
        return blurred

    # ── 2. FFT band-pass filter (Frequency Domain Filtering) ──────────────────
    def fft_bandpass(self, gray: np.ndarray) -> np.ndarray:
        """
        FFT → shift to centre → apply annular band-pass mask → IFFT.
        Removes DC (low-freq background) and very high-freq noise,
        keeping the mid-frequency stripes that make up a barcode.
        """
        H, W    = gray.shape
        f       = np.fft.fft2(gray.astype(np.float32))
        fshift  = np.fft.fftshift(f)

        cy, cx  = H // 2, W // 2
        r_min   = 10
        r_max   = min(H, W) // 2 - 20
        y, x    = np.ogrid[:H, :W]
        r       = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask    = ((r >= r_min) & (r <= r_max)).astype(np.float32)

        filtered = np.fft.ifft2(np.fft.ifftshift(fshift * mask))
        result   = np.abs(filtered).astype(np.uint8)
        return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

    # ── 3a. Canny edge detection ───────────────────────────────────────────────
    def edge_detect(self, gray: np.ndarray) -> np.ndarray:
        """
        Canny edge detector followed by morphological closing.
        Closing connects the short bar-edges of a 1D barcode into solid blobs
        that barcode detectors can lock onto.
        """
        edges  = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        return closed

    # ── 3b. Edge-based adaptive threshold (for decoding pipeline) ─────────────
    def edge_based_threshold(self, gray: np.ndarray) -> np.ndarray:
        """
        Gaussian blur → Canny edges → adaptive threshold.
        Produces a clean binary image that helps OpenCV's detector
        read barcodes in challenging lighting conditions.
        """
        blurred = self.gaussian_blur(gray)
        # Adaptive threshold guided by local gradient (edge-aware binarisation)
        thresh  = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    # ── 4. Full debug pipeline ─────────────────────────────────────────────────
    def process(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Returns (processed_gray, debug_images_dict).
        processed_gray is the blurred image — best balance for decoding.
        debug_images_dict exposes all four intermediate images for the D key.
        """
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        blurred = self.gaussian_blur(gray)
        fft_img = self.fft_bandpass(blurred)
        edges   = self.edge_detect(blurred)

        debug = {
            "gray":    gray,
            "blurred": blurred,
            "fft":     fft_img,
            "edges":   edges,
        }
        return blurred, debug

    # ── 5. Hybrid overlay ─────────────────────────────────────────────────────
    @staticmethod
    def hybrid_overlay(frame: np.ndarray, annotation: np.ndarray, alpha: float = 0.75) -> np.ndarray:
        """Blend two images — used for the hybrid-image overlay in debug mode."""
        return cv2.addWeighted(frame, alpha, annotation, 1 - alpha, 0)
