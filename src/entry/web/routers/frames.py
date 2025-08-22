"""
Frame-related API routes.
"""

import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from starlette.status import HTTP_404_NOT_FOUND, HTTP_500_INTERNAL_SERVER_ERROR

from config import GlobalConfig
from entry.web.models import FrameData, FrameInfoResponse
from services.search import SearchFactory

router = APIRouter(prefix="/api/v1/frames", tags=["frames"])
LOGGER = logging.getLogger(__name__)
WORK_DIR = Path(os.getenv("AIC25_WORK_DIR") or ".")

_searcher = None


def _get_database_type():
    """Get database type from config, with fallback."""
    try:
        return GlobalConfig.get("webui", "database") or "faiss"
    except Exception as e:
        LOGGER.warning(f"Could not get database type from config: {e}")
        LOGGER.warning("Using default database type: faiss")
        return "faiss"


def _initialize_searcher():
    """Initialize the searcher instance lazily."""
    global _searcher
    if _searcher is None:
        database_type = _get_database_type()
        _searcher = SearchFactory.create_searcher("default", database_type)
    return _searcher


def get_searcher():
    """Dependency to get searcher instance."""
    return _initialize_searcher()


@router.get("/{video_id}/{frame_id}", response_model=FrameInfoResponse)
async def get_frame_info(
    request: Request,
    video_id: str,
    frame_id: str,
    searcher_instance=Depends(get_searcher),
):
    """
    Get detailed information about a specific frame.

    Returns comprehensive frame metadata including URIs, FPS, and availability status.
    """
    try:
        frame_full_id = f"{video_id}#{frame_id}"

        record = searcher_instance.get(frame_full_id)

        frame_uri = f"{request.base_url}api/files/keyframes/{video_id}/{frame_id}.jpg"
        video_uri = f"{request.base_url}api/stream/videos/{video_id}.mp4"

        fps = 25
        try:
            video_info_path = WORK_DIR / "videos_info" / f"{video_id}.json"
            if video_info_path.exists():
                with open(video_info_path, "r") as f:
                    fps = json.load(f).get("frame_rate", 25)
        except Exception as e:
            LOGGER.warning(f"Could not load FPS for {video_id}: {e}")

        frame_exists = len(record) > 0

        if frame_exists:
            frame_data = FrameData(
                id=frame_full_id,
                video_id=video_id,
                frame_id=frame_id,
                frame_uri=frame_uri,
                video_uri=video_uri,
                fps=fps,
            )
        else:
            frame_data = None

        return FrameInfoResponse(
            success=True,
            frame=frame_data,
            message="Frame found" if frame_exists else "Frame not found",
        )

    except Exception as e:
        LOGGER.error(f"Frame info retrieval failed: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to retrieve frame information",
                "message": str(e),
                "video_id": video_id,
                "frame_id": frame_id,
            },
        )


@router.get("/{video_id}/{frame_id}/image")
async def get_frame_image(
    video_id: str,
    frame_id: str,
    thumbnail: bool = Query(default=False, description="Return thumbnail version"),
):
    """
    Get the actual frame image file.

    Returns the frame image as a file response. Supports thumbnail option
    for smaller image sizes.
    """
    try:
        if thumbnail:
            file_path = WORK_DIR / "keyframes" / video_id / f"{frame_id}_thumb.jpg"
            if not file_path.exists():
                file_path = WORK_DIR / "keyframes" / video_id / f"{frame_id}.jpg"
        else:
            file_path = WORK_DIR / "keyframes" / video_id / f"{frame_id}.jpg"

        if not file_path.exists():
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail={
                    "error": "Frame image not found",
                    "video_id": video_id,
                    "frame_id": frame_id,
                },
            )

        return FileResponse(
            path=str(file_path),
            media_type="image/jpeg",
            filename=f"{video_id}_{frame_id}.jpg",
        )

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Frame image retrieval failed: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to retrieve frame image", "message": str(e)},
        )


@router.get("/{video_id}/frames")
async def list_video_frames(
    request: Request,
    video_id: str,
    offset: int = Query(default=0, ge=0, description="Frame offset"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum frames"),
    searcher_instance=Depends(get_searcher),
):
    """
    List all frames for a specific video.

    Returns a paginated list of all frames available for the specified video.
    """
    try:
        query = f"video:{video_id}"

        results = searcher_instance.search(
            query, "", offset, limit, 8, "clip", 10000, 1.0, 40, 250, None
        )

        frames = []
        for record in results["results"]:
            data = record["entity"]
            video_frame_str = data["frame_id"]
            _, frame_id = video_frame_str.split("#")

            frame_uri = (
                f"{request.base_url}api/files/keyframes/{video_id}/{frame_id}.jpg"
            )
            video_uri = f"{request.base_url}api/stream/videos/{video_id}.mp4"

            fps = 25
            try:
                video_info_path = WORK_DIR / "videos_info" / f"{video_id}.json"
                if video_info_path.exists():
                    with open(video_info_path, "r") as f:
                        fps = json.load(f).get("frame_rate", 25)
            except Exception as e:
                LOGGER.error(e)

            frames.append(
                FrameData(
                    id=video_frame_str,
                    video_id=video_id,
                    frame_id=frame_id,
                    frame_uri=frame_uri,
                    video_uri=video_uri,
                    fps=fps,
                )
            )

        return {
            "success": True,
            "video_id": video_id,
            "frames": frames,
            "total": results["total"],
            "offset": offset,
            "limit": limit,
            "has_more": offset + limit < results["total"],
        }

    except Exception as e:
        LOGGER.error(f"Video frames listing failed: {e}")
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail={
                "error": "Failed to list video frames",
                "message": str(e),
                "video_id": video_id,
            },
        )
