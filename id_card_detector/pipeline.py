from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import imutils
import numpy as np
from skimage import exposure


@dataclasses.dataclass
class ExtractionResult:
    """Container for all extracted card artifacts."""

    card: np.ndarray
    binary_card: np.ndarray
    portrait: Optional[np.ndarray]
    text_regions: List[np.ndarray]

    def save(self, output_dir: Path) -> None:
        """Persist all extracted pieces to ``output_dir``.

        Args:
            output_dir: destination folder. It is created when missing.
        """

        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "card.png"), self.card)
        cv2.imwrite(str(output_dir / "card_binary.png"), self.binary_card)

        if self.portrait is not None:
            cv2.imwrite(str(output_dir / "portrait.png"), self.portrait)

        for idx, region in enumerate(self.text_regions):
            cv2.imwrite(str(output_dir / f"text_region_{idx+1:02d}.png"), region)


class IdCardDetector:
    """Detects and rectifies ID cards from cluttered backgrounds."""

    def __init__(self, target_width: int = 900, canny_sigma: float = 0.33) -> None:
        self.target_width = target_width
        self.canny_sigma = canny_sigma

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Locate the four-point contour of the ID card.

        Returns:
            A 4x2 float array describing the contour in the original image space.

        Raises:
            ValueError: if no plausible ID card contour is found.
        """

        resized, ratio = self._resize(image)
        preprocessed = self._preprocess(resized)
        edge_map = self._edge_map(preprocessed)

        contours = self._candidate_quadrilaterals(edge_map)
        h, w = resized.shape[:2]
        min_area = 0.2 * h * w
        best = None
        best_score = -np.inf

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if width == 0 or height == 0:
                continue
            aspect_ratio = max(width, height) / min(width, height)
            if not 1.3 <= aspect_ratio <= 2.0:
                continue

            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) != 4:
                continue

            solidity = area / cv2.contourArea(cv2.convexHull(contour))
            score = area * solidity
            if score > best_score:
                best_score = score
                best = approx

        if best is None:
            raise ValueError("No ID card contour found. Try a sharper input or different lighting.")

        contour = best.reshape(4, 2).astype("float32") / ratio
        return self._order_points(contour)

    def rectify(self, image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """Warp the detected contour into a top-down, canonical view."""

        (tl, tr, br, bl) = contour
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = int(max(width_a, width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = int(max(height_a, height_b))

        dst = np.array(
            [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(contour, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        return warped

    def _resize(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        resized = imutils.resize(image, width=self.target_width)
        ratio = image.shape[1] / float(resized.shape[1])
        return resized, ratio

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        gray = cv2.equalizeHist(gray)
        return gray

    def _edge_map(self, gray: np.ndarray) -> np.ndarray:
        v = np.median(gray)
        lower = int(max(0, (1.0 - self.canny_sigma) * v))
        upper = int(min(255, (1.0 + self.canny_sigma) * v))
        edges = cv2.Canny(gray, lower, upper)

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

        blended = cv2.bitwise_or(edges, gradient)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(blended, cv2.MORPH_CLOSE, kernel, iterations=2)
        closed = cv2.dilate(closed, kernel, iterations=1)
        return closed

    def _candidate_quadrilaterals(self, edge_map: np.ndarray) -> Iterable[np.ndarray]:
        contours = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
        return contours

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect


class ComponentExtractor:
    """Extracts the portrait and text groups from a rectified ID card image."""

    def __init__(self, min_text_height: int = 12) -> None:
        self.min_text_height = min_text_height
        self._face_detector = self._load_face_detector()

    def extract(self, card: np.ndarray) -> ExtractionResult:
        normalized = self._normalize(card)
        binary = self._binarize(normalized)
        portrait = self._extract_portrait(normalized)
        text_regions = self._extract_text_regions(binary)
        cropped_regions = [self._crop_region(normalized, box) for box in text_regions]

        return ExtractionResult(card=normalized, binary_card=binary, portrait=portrait, text_regions=cropped_regions)

    def _normalize(self, card: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(card, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        normalized = exposure.rescale_intensity(normalized, in_range=(0, 255), out_range=(0, 255)).astype(np.uint8)
        return normalized

    def _binarize(self, card: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
        binary = cv2.bitwise_not(binary)
        return binary

    def _extract_portrait(self, card: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
        portrait = None

        if self._face_detector is not None:
            faces = self._face_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(60, 60))
            if len(faces) > 0:
                x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                pad = int(0.05 * max(w, h))
                portrait = card[max(y - pad, 0) : y + h + pad, max(x - pad, 0) : x + w + pad]
        return portrait

    def _extract_text_regions(self, binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        morph = cv2.dilate(binary, kernel, iterations=2)
        morph = cv2.erode(morph, kernel, iterations=1)

        contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        boxes: List[Tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < self.min_text_height or w < 40:
                continue
            aspect_ratio = w / float(h)
            if aspect_ratio < 1.5:
                continue
            boxes.append((x, y, w, h))

        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        return boxes

    def _crop_region(self, image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = box
        pad = 4
        return image[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad]

    def _load_face_detector(self) -> Optional[cv2.CascadeClassifier]:
        try:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            classifier = cv2.CascadeClassifier(str(cascade_path))
            if classifier.empty():
                return None
            return classifier
        except Exception:
            return None


def process_image(image_path: Path, output_dir: Path) -> ExtractionResult:
    """End-to-end processing for a single image."""

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    detector = IdCardDetector()
    contour = detector.detect(image)
    warped = detector.rectify(image, contour)

    extractor = ComponentExtractor()
    result = extractor.extract(warped)
    result.save(output_dir)
    return result
