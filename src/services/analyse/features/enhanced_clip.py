import logging
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from config import GlobalConfig

from .feature_extractor import FeatureExtractor, ImageDataset


class EnhancedCLIP(FeatureExtractor):
    """Enhanced CLIP with multiple models and optimization techniques"""

    def __init__(self, pretrained_models: Union[str, List[str]], use_ensemble=False):
        """
        Initialize Enhanced CLIP

        Args:
            pretrained_models: Single model name or list of model names
            use_ensemble: Whether to use ensemble of multiple models
        """
        self.logger = logging.getLogger(__name__)
        self.use_ensemble = use_ensemble

        if isinstance(pretrained_models, str):
            pretrained_models = [pretrained_models]

        self.models = []
        self.processors = []

        for model_name in pretrained_models:
            try:
                model = CLIPModel.from_pretrained(model_name)
                processor = CLIPProcessor.from_pretrained(model_name)
                model.eval()

                self.models.append(model)
                self.processors.append(processor)
                self.logger.info(f"Loaded CLIP model: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")

        if not self.models:
            raise ValueError("No models could be loaded")

        # Use first processor as primary
        self._processor = self.processors[0]

        # Feature dimension (assuming all models have same dimension)
        self.feature_dim = self.models[0].config.projection_dim

    def get_image_features(self, image_paths, batch_size, callback):
        """Extract image features with optimization"""
        dataset = ImageDataset(image_paths, self._processor)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=GlobalConfig.get("analyse", "num_workers") or 0,
            pin_memory=GlobalConfig.get("analyse", "pin_memory", False),
            persistent_workers=(
                True if GlobalConfig.get("analyse", "num_workers", 0) > 0 else False
            ),
        )

        all_features = []
        num_batches = len(dataloader)

        with torch.no_grad():
            callback(0, num_batches, None)

            for i, data in enumerate(dataloader):
                for key in data:
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].to(self.models[0].device)

                if self.use_ensemble and len(self.models) > 1:
                    batch_features = self._ensemble_image_features(data)
                else:
                    batch_features = self.models[0].get_image_features(**data)

                batch_features = F.normalize(batch_features, p=2, dim=1)

                all_features.append(batch_features)
                callback(i + 1, num_batches, None)

        image_features = torch.cat(all_features, dim=0)
        return image_features

    def get_text_features(self, texts):
        """Extract text features with optimization"""
        if isinstance(texts, str):
            texts = [texts]

        tokenized_input = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )

        for key in tokenized_input:
            if isinstance(tokenized_input[key], torch.Tensor):
                tokenized_input[key] = tokenized_input[key].to(self.models[0].device)

        with torch.no_grad():
            if self.use_ensemble and len(self.models) > 1:
                text_features = self._ensemble_text_features(tokenized_input)
            else:
                text_features = self.models[0].get_text_features(**tokenized_input)

            text_features = F.normalize(text_features, p=2, dim=1)

        return text_features

    def _ensemble_image_features(self, data):
        """Compute ensemble image features"""
        features_list = []

        for model in self.models:
            features = model.get_image_features(**data)
            features = F.normalize(features, p=2, dim=1)
            features_list.append(features)

        ensemble_features = torch.stack(features_list).mean(dim=0)

        ensemble_features = F.normalize(ensemble_features, p=2, dim=1)
        return ensemble_features

    def _ensemble_text_features(self, tokenized_input):
        """Compute ensemble text features"""
        features_list = []

        for model in self.models:
            features = model.get_text_features(**tokenized_input)
            features = F.normalize(features, p=2, dim=1)
            features_list.append(features)

        ensemble_features = torch.stack(features_list).mean(dim=0)

        # Re-normalize after averaging
        ensemble_features = F.normalize(ensemble_features, p=2, dim=1)
        return ensemble_features

    def to(self, device):
        """Move all models to device"""
        for model in self.models:
            model.to(device)


class MultiModalCLIP(FeatureExtractor):
    """CLIP with additional multimodal features"""

    def __init__(self, pretrained_model, use_temporal_features=True):
        self.base_clip = EnhancedCLIP(pretrained_model)
        self.use_temporal_features = use_temporal_features
        self.logger = logging.getLogger(__name__)

        # Additional feature extractors can be added here
        self.feature_dim = self.base_clip.feature_dim

        if use_temporal_features:
            # Add temporal f    eature dimension
            self.feature_dim += 64  # Additional temporal features

    def get_image_features(self, image_paths, batch_size, callback):
        """Extract enhanced multimodal features"""
        # Get base CLIP features
        clip_features = self.base_clip.get_image_features(
            image_paths, batch_size, callback
        )

        if not self.use_temporal_features:
            return clip_features

        # Add temporal features based on frame ordering
        temporal_features = self._extract_temporal_features(image_paths)

        # Concatenate features
        enhanced_features = torch.cat([clip_features, temporal_features], dim=1)
        return enhanced_features

    def get_text_features(self, texts):
        """Extract text features (same as base CLIP for now)"""
        base_features = self.base_clip.get_text_features(texts)

        if not self.use_temporal_features:
            return base_features

        # Pad with zeros for temporal dimension to match image features
        batch_size = base_features.shape[0]
        temporal_padding = torch.zeros(batch_size, 64, device=base_features.device)
        enhanced_features = torch.cat([base_features, temporal_padding], dim=1)

        return enhanced_features

    @staticmethod
    def _extract_temporal_features(image_paths):
        """Extract simple temporal features based on frame indices"""
        temporal_features = []

        for path in image_paths:
            # Extract frame number from filename (assuming format like 000001.jpg)
            frame_num = int(path.stem)

            # Create simple temporal encoding
            temporal_vec = np.zeros(64)

            # Encode frame position (normalized)
            temporal_vec[0] = frame_num / 100000.0  # Normalize frame number

            # Add some sinusoidal encodings for periodicity
            for i in range(1, 32):
                temporal_vec[i] = np.sin(frame_num / (10 ** (i % 4)))
                temporal_vec[i + 32] = np.cos(frame_num / (10 ** (i % 4)))

            temporal_features.append(temporal_vec)

        return torch.tensor(temporal_features, dtype=torch.float32)

    def to(self, device):
        self.base_clip.to(device)
