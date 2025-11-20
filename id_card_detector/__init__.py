"""High accuracy ID card detection and component extraction pipeline."""

__all__ = [
    "IdCardDetector",
    "ComponentExtractor",
    "ExtractionResult",
    "process_image",
    "process_webcam",
]
__version__ = "1.0.0"

from .pipeline import ComponentExtractor, ExtractionResult, IdCardDetector, process_image, process_webcam
