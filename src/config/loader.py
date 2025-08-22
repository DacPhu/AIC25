import os
import logging
from pathlib import Path
from typing import Any, Dict, Final, Optional

import yaml


class GlobalConfig:
    RELATED_CONFIG_FILE_PATH: Final[str] = "layout/config.yaml"
    _config = None
    _config_loaded = False
    _work_dir = None

    @classmethod
    def initialize(cls, work_dir: Path):
        """Initialize the GlobalConfig with a specific work directory"""
        cls._work_dir = Path(work_dir)
        cls._config_loaded = False
        cls._config = None
        if work_dir is None:
            cls._work_dir = Path.cwd()

    @classmethod
    def _load_config(cls):
        """Load configuration from the YAML file with enhanced error handling"""
        if cls._config_loaded:
            return

        logger = logging.getLogger(
            f'{".".join(__name__.split(".")[:-1])}.{cls.__name__}'
        )

        if cls._work_dir is None:
            work_dir = os.getenv("AIC25_WORK_DIR", ".")
            logger.warning(
                f"GlobalConfig not initialized. Using work directory from environment: {work_dir}"
            )
            cls.initialize(Path(work_dir))

        config_path = cls._work_dir / cls.RELATED_CONFIG_FILE_PATH

        if not config_path.exists():
            logger.warning(
                f'"{cls.RELATED_CONFIG_FILE_PATH}" not found in {cls._work_dir}. Creating default configuration.'
            )
            cls._create_default_config(config_path)

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cls._config = yaml.safe_load(f)

            if cls._config is None:
                logger.warning(
                    f'"{cls.RELATED_CONFIG_FILE_PATH}" is empty. Using default configuration.'
                )
                cls._config = {}

            logger.info(f"Configuration loaded from {config_path}")

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_path}: {e}")
            cls._config = {}
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            cls._config = {}

        cls._config_loaded = True

    @classmethod
    def get(cls, *args, default=None):
        """
        Get configuration value using dot notation

        Args:
            *args: Configuration path (e.g., 'database', 'host' for database.host)
            default: Default value if key is not found

        Returns:
            Configuration value or default
        """
        cls._load_config()

        if cls._config is None:
            return default

        try:
            result = cls._config
            for arg in args:
                if isinstance(result, dict) and arg in result:
                    result = result[arg]
                else:
                    return default
            return result
        except (KeyError, TypeError):
            return default

    @classmethod
    def set(cls, value: Any, *args):
        """
        Set configuration value using dot notation

        Args:
            value: Value to set
            *args: Configuration path
        """
        cls._load_config()

        if cls._config is None:
            cls._config = {}

        current = cls._config
        for key in args[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[args[-1]] = value

    @classmethod
    def update(cls, config_dict: Dict[str, Any], *path):
        """
        Update configuration with a dictionary

        Args:
            config_dict: Dictionary to merge
            *path: Optional path to update a specific section
        """
        cls._load_config()

        if cls._config is None:
            cls._config = {}

        if path:
            target = cls._config
            for key in path[:-1]:
                if key not in target:
                    target[key] = {}
                target = target[key]
            target[path[-1]] = config_dict
        else:
            cls._config.update(config_dict)

    @classmethod
    def save_config(cls, config_path: Optional[Path] = None):
        """Save current configuration to file"""
        if config_path is None:
            config_path = Path.cwd() / cls.RELATED_CONFIG_FILE_PATH

        if cls._config is None:
            return

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    cls._config, f, default_flow_style=False, indent=2, sort_keys=False
                )

            logger = logging.getLogger(cls.__name__)
            logger.info(f"Configuration saved to {config_path}")

        except Exception as e:
            logger = logging.getLogger(cls.__name__)
            logger.error(f"Failed to save configuration to {config_path}: {e}")

    @classmethod
    def reload_config(cls):
        """Force reload configuration from file"""
        cls._config_loaded = False
        cls._config = None
        cls._load_config()

    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Get the entire configuration as a dictionary"""
        cls._load_config()
        return cls._config.copy() if cls._config else {}

    @classmethod
    def has_key(cls, *args) -> bool:
        return cls.get(*args) is not None

    @classmethod
    def _create_default_config(cls, config_path: Path):
        """Create default configuration file"""
        default_config = {
            "max_workers_ratio": 0.8,
            "add": {
                "max_scene_length": 50,
                "keyframe_extraction": {
                    "method": "adaptive",
                    "adaptive_threshold": 0.3,
                    "max_frames_per_second": 2,
                    "min_frames_per_video": 10,
                    "max_frames_per_video": 1000,
                },
            },
            "analyse": {
                "features": [
                    {
                        "name": "clip",
                        "pretrained_model": "openai/clip-vit-base-patch16",
                        "batch_size": 16,
                        "use_ensemble": False,
                    },
                    {
                        "name": "ocr",
                        "batch_size": 8,
                        "confidence_threshold": 0.7,
                        "languages": ["vi", "en"],
                    },
                ],
                "num_workers": 2,
                "pin_memory": True,
                "mixed_precision": True,
                "keyframe_extraction": {
                    "method": "adaptive",
                    "adaptive_threshold": 0.3,
                    "max_frames_per_second": 2,
                    "min_frames_per_video": 10,
                    "max_frames_per_video": 1000,
                },
            },
            "faiss": {
                "index_type": "IVF",
                "nlist": 128,
                "nprobe": 8,
                "M": 32,
                "efConstruction": 400,
                "efSearch": 200,
                "m": 8,
                "nbits": 8,
                "fields": [
                    {
                        "field_name": "frame_id",
                        "datatype": "VARCHAR",
                        "max_length": 32,
                        "is_primary": True,
                    },
                    {"field_name": "clip", "datatype": "FLOAT_VECTOR", "dim": 512},
                    {"field_name": "ocr", "datatype": "JSON"},
                ],
            },
            "webui": {
                "features": [
                    {
                        "name": "clip",
                        "pretrained_model": "openai/clip-vit-base-patch16",
                        "batch_size": 16,
                    },
                    {
                        "name": "sentence_transformer",
                        "pretrained_model": "all-MiniLM-L6-v2",
                        "device": "auto",
                        "description": "Semantic text search with sentence transformers",
                    },
                ],
                "database": "faiss",
                "cache": {"enabled": True, "max_size": 10000, "ttl": 3600},
                "search": {
                    "default_nprobe": 16,
                    "max_results": 10000,
                    "rerank": {
                        "enabled": False,
                        "top_k": 100,
                        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    },
                },
            },
            "hardware": {
                "gpu": {
                    "enabled": True,
                    "mixed_precision": True,
                    "compile_model": False,
                },
                "cpu": {"num_threads": -1},
                "memory": {"prefetch_factor": 4, "persistent_workers": True},
            },
            "advanced": {
                "query_expansion": {
                    "enabled": False,
                    "synonyms_file": "synonyms.txt",
                    "expand_ratio": 0.3,
                },
                "prf": {
                    "enabled": False,
                    "top_k": 5,
                    "expansion_terms": 10,
                    "weight": 0.3,
                },
                "fusion": {
                    "enabled": True,
                    "weights": {"clip": 0.7, "ocr": 0.2, "temporal": 0.1},
                },
                "temporal": {
                    "enabled": True,
                    "window_size": 25,
                    "consistency_weight": 0.15,
                },
            },
            "monitoring": {
                "enabled": True,
                "metrics": [
                    "search_latency",
                    "index_size",
                    "recall_at_k",
                    "precision_at_k",
                ],
                "logging": {"level": "INFO", "file": "retrieval.log"},
            },
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                default_config, f, default_flow_style=False, indent=2, sort_keys=False
            )

        logger = logging.getLogger(cls.__name__)
        logger.info(f"Created default configuration file: {config_path}")

        cls._config = default_config
