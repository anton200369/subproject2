# ID Card Detection and Component Extraction

This subproject provides a **ready-to-run** computer vision pipeline that detects an ID card placed on any background, rectifies it with a perspective transform, and extracts the key components (full card, high-contrast binary card, portrait crop, and consolidated text regions).

## Why this approach is robust
- **Edge+gradient fusion**: Combines adaptive Canny edges with Sobel gradients to recover the outer card contour even under low contrast or glare.
- **Geometric vetting**: Filters candidate contours by area, aspect ratio, convexity, and polygon count to avoid false positives from other rectangular objects.
- **Color normalization**: CLAHE in LAB space and intensity rescaling stabilize colors for consistent thresholding across lighting conditions.
- **Adaptive binarization**: Gaussian-adaptive thresholding paired with inversion keeps text crisp for OCR while suppressing textured backgrounds.
- **Face-aware portrait extraction**: Uses OpenCV’s Haar cascade (when available) to isolate the portrait; gracefully degrades if a face is not detected.

## Quickstart
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the extractor on a photo containing an ID card:
   ```bash
   python -m id_card_detector.cli path/to/photo.jpg --output artifacts/
   ```
3. Review the outputs in the chosen `--output` directory:
   - `card.png`: normalized, perspective-corrected card
   - `card_binary.png`: high-contrast binary card for OCR
   - `portrait.png`: cropped face area (when detected)
   - `text_region_*.png`: cropped text bands ready for OCR/ML processing

## Implementation notes
- The detector operates at a configurable working width (default 900 px) to balance speed and precision.
- Candidate rectangles are vetted by size (≥20% of the resized frame) and aspect ratio (1.3–2.0) to match typical ID proportions.
- Text grouping uses rectangular morphology tuned for horizontally aligned lines; adjust `min_text_height` in `ComponentExtractor` for unusually small fonts.
- If OpenCV’s Haar cascades are unavailable, portrait extraction is skipped while the rest of the pipeline continues to run.

## Extending for the highest marks
- Plug OCR directly on `card_binary.png` or each `text_region_*` crop using `pytesseract.image_to_data`.
- Add template-driven field parsing by measuring relative offsets inside the rectified card.
- Train a lightweight segmentation/face model and replace the Haar cascade path in `ComponentExtractor._load_face_detector`.

## Repository contents
- `id_card_detector/`: Python package implementing detection, rectification, and component extraction.
- `requirements.txt`: Dependency list.
- `Subproject2_2024.pdf`: Original assignment brief (kept for reference).
