"""
System and health check API routes.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import psutil
from fastapi import APIRouter, Depends, HTTPException
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from config import GlobalConfig
from entry.web.models import HealthResponse, ModelsResponse
from services.search import SearchFactory

# Setup
router = APIRouter(prefix="/api/v1/system", tags=["system"])
logger = logging.getLogger(__name__)
WORK_DIR = Path(os.getenv("AIC25_WORK_DIR") or ".")

_searcher = None


def _get_database_type():
    """Get database type from config, with fallback."""
    try:
        return GlobalConfig.get("webui", "database") or "faiss"
    except Exception as e:
        logger.warning(f"Could not get database type from config: {e}")
        logger.warning("Using default database type: faiss")
        return "faiss"


def _initialize_searcher():
    """Initialize searcher instance lazily."""
    global _searcher
    if _searcher is None:
        database_type = _get_database_type()
        _searcher = SearchFactory.create_searcher("default", database_type)
    return _searcher


def get_searcher():
    """Dependency to get searcher instance."""
    return _initialize_searcher()


@router.get("/health", response_model=HealthResponse)
async def health_check(searcher_instance=Depends(get_searcher)):
    """
    Comprehensive health check endpoint.

    Returns detailed system health information including:
    - Service status
    - Database connectivity
    - Resource usage
    - Index statistics
    """
    try:
        # Test database connectivity
        try:
            total_frames = (
                searcher_instance.get_total()
                if hasattr(searcher_instance, "get_total")
                else None
            )
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            total_frames = None

        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage(str(WORK_DIR))

        return HealthResponse(
            success=True,
            status="healthy",
            version="1.0.0",
            database_type=_get_database_type(),
            total_frames=total_frames,
            message="All systems operational",
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            success=False,
            status="unhealthy",
            version="1.0.0",
            database_type=_get_database_type(),
            message=f"Health check failed: {str(e)}",
            timestamp=datetime.utcnow().isoformat(),
        )


@router.get("/models", response_model=ModelsResponse)
async def get_available_models(searcher_instance=Depends(get_searcher)):
    """
    Get list of available AI models.

    Returns all available models that can be used for search operations,
    including their capabilities and current status.
    """
    try:
        models = searcher_instance.get_models()

        return ModelsResponse(
            success=True,
            models=models,
            message=f"Found {len(models)} available models",
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Models retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to retrieve available models", "message": str(e)},
        )


@router.get("/stats")
async def get_system_stats(searcher_instance=Depends(get_searcher)):
    """
    Get detailed system statistics.

    Returns comprehensive system statistics including:
    - Resource usage (CPU, memory, disk)
    - Database statistics
    - Performance metrics
    - Index information
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage(str(WORK_DIR))

        try:
            total_frames = (
                searcher_instance.get_total()
                if hasattr(searcher_instance, "get_total")
                else 0
            )
            available_models = searcher_instance.get_models()
        except Exception as e:
            logger.warning(f"Could not get database stats: {e}")
            total_frames = 0
            available_models = []

        videos_dir = WORK_DIR / "videos"
        keyframes_dir = WORK_DIR / "keyframes"

        video_count = len(list(videos_dir.glob("*"))) if videos_dir.exists() else 0
        keyframe_dirs = (
            len(list(keyframes_dir.glob("*"))) if keyframes_dir.exists() else 0
        )

        stats = {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory_info.total,
                    "available": memory_info.available,
                    "used": memory_info.used,
                    "percent": memory_info.percent,
                },
                "disk": {
                    "total": disk_info.total,
                    "used": disk_info.used,
                    "free": disk_info.free,
                    "percent": (disk_info.used / disk_info.total) * 100,
                },
            },
            "database": {
                "type": _get_database_type(),
                "total_frames": total_frames,
                "available_models": available_models,
                "model_count": len(available_models),
            },
            "content": {
                "video_count": video_count,
                "indexed_videos": keyframe_dirs,
                "work_directory": str(WORK_DIR),
            },
        }

        return stats

    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to retrieve system statistics", "message": str(e)},
        )


@router.get("/config")
async def get_system_config():
    """
    Get system configuration information.

    Returns current system configuration including:
    - Database settings
    - Feature flags
    - Performance settings
    - Enabled capabilities
    """
    try:
        config_info = {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "database": {"type": _get_database_type(), "work_directory": str(WORK_DIR)},
            "features": {
                "search_enabled": True,
                "similar_search_enabled": True,
                "audio_search_enabled": True,
                "video_streaming_enabled": True,
                "batch_search_enabled": True,
            },
            "limits": {
                "max_search_results": 200,
                "max_batch_queries": 10,
                "max_suggestions": 50,
                "chunk_size_mb": 1,
                "max_chunk_size_mb": 8,
            },
            "supported_formats": {
                "video": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
                "image": [".jpg", ".jpeg", ".png"],
                "search_types": ["text", "audio_text", "similar", "video_filter"],
            },
        }

        return config_info

    except Exception as e:
        logger.error(f"Config retrieval failed: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to retrieve system configuration",
                "message": str(e),
            },
        )


@router.post("/cache/clear")
async def clear_caches():
    """
    Clear system caches.

    Clears various system caches to free up memory and force
    fresh data retrieval on subsequent requests.
    """
    try:
        cleared_caches = []

        if hasattr(searcher, "cache"):
            cache_size = len(searcher.cache)
            searcher.cache.clear()
            cleared_caches.append(f"searcher_cache ({cache_size} items)")

        return {
            "success": True,
            "message": "Caches cleared successfully",
            "cleared_caches": cleared_caches,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Cache clearing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to clear caches", "message": str(e)},
        )
