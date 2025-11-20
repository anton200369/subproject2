"""High accuracy ID card detection and component extraction pipeline."""

__all__ = ["IdCardDetector", "ComponentExtractor", "ExtractionResult"]
__version__ = "1.0.0"

from .pipeline import IdCardDetector, ComponentExtractor, ExtractionResult
