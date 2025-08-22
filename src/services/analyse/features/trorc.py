from math import ceil
import logging
from pathlib import Path

from .paddle_ocr import PaddleOCRExtractor
from .feature_extractor import FeatureExtractor, ImageDataset


class TrOCR(FeatureExtractor):
    """
    OCR Feature Extractor using PaddleOCR (replacement for EasyOCR).

    This class maintains compatibility with the existing TrOCR interface
    while using PaddleOCR as the backend to avoid SSL certificate issues.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Initializing TrOCR with PaddleOCR backend")

        # Initialize PaddleOCR with Vietnamese and English support
        try:
            # Try multilingual first, fallback to English if issues
            self._reader = PaddleOCRExtractor(
                languages=["en"],  # Start with English, can be extended
                use_angle_cls=True,
                use_gpu=None,  # Auto-detect
            )
            self.logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise

    def get_image_features(self, image_paths, batch_size, callback):
        """
        Extract OCR features from images using PaddleOCR.

        This method maintains compatibility with the original EasyOCR interface
        while using PaddleOCR as the backend.
        """
        image_features = []
        image_paths = [str(x) for x in image_paths]
        num_batches = ceil(len(image_paths) / batch_size)

        self.logger.info(
            f"Processing {len(image_paths)} images in {num_batches} batches"
        )
        callback(0, num_batches, None)

        for b in range(num_batches):
            batch_paths = image_paths[b * batch_size : (b + 1) * batch_size]

            try:
                # Use PaddleOCR to process the batch
                batch_results = self._reader.get_image_features(
                    batch_paths, batch_size=len(batch_paths)
                )

                # Convert PaddleOCR results to format expected by the system
                for paddle_result in batch_results:
                    detected_texts = self._convert_paddle_to_legacy_format(
                        paddle_result
                    )
                    image_features.append(detected_texts)

            except Exception as e:
                self.logger.error(f"Error processing batch {b}: {e}")
                # Add empty results for failed batch
                for _ in batch_paths:
                    image_features.append([])

            callback(b + 1, num_batches, image_features)

        self.logger.info(
            f"OCR processing completed. Processed {len(image_features)} images."
        )
        return image_features

    def _convert_paddle_to_legacy_format(self, paddle_result):
        """
        Convert PaddleOCR results to the format expected by the legacy system.

        PaddleOCR format: [bbox, text, confidence, source]
        Legacy format: [normalized_bbox, text, confidence]
        """
        detected_texts = []

        for item in paddle_result:
            if len(item) >= 3:
                bbox = item[0]  # Bounding box coordinates
                text = item[1]  # Detected text
                confidence = item[2]  # Confidence score

                # Normalize bounding box coordinates (assume 640x360 target size for compatibility)
                # PaddleOCR returns absolute coordinates, we need to normalize them
                normalized_bbox = []

                try:
                    # Convert bbox to the expected format
                    if len(bbox) == 4:  # 4 corner points
                        for point in bbox:
                            if len(point) >= 2:
                                # Normalize coordinates (this is a simplified normalization)
                                # In practice, you might want to use actual image dimensions
                                normalized_point = [
                                    float(point[0]) / 640,  # Normalize x to 640 width
                                    float(point[1]) / 360,  # Normalize y to 360 height
                                ]
                                normalized_bbox.append(normalized_point)

                    # Create the legacy format
                    legacy_item = [normalized_bbox, text, float(confidence)]
                    detected_texts.append(legacy_item)

                except Exception as e:
                    self.logger.warning(f"Error converting bbox format: {e}")
                    continue

        return detected_texts

    def get_text_features(self, texts):
        """Extract features from text (placeholder implementation)"""
        return texts

    def to(self, device):
        """
        Move model to specified device.

        Args:
            device: Target device ('cpu', 'cuda', 'mps')
        """
        try:
            use_gpu = device in ["cuda", "mps"]
            self.logger.info(f"Moving OCR model to device: {device} (GPU: {use_gpu})")

            # Reinitialize PaddleOCR with new device setting
            self._reader = PaddleOCRExtractor(
                languages=["en"],  # Keep current language setting
                use_angle_cls=True,
                use_gpu=use_gpu,
            )

        except Exception as e:
            self.logger.error(f"Error moving model to device {device}: {e}")
            # Keep existing reader if device change fails
