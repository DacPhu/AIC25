"""
Video-related API routes.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from starlette.status import HTTP_206_PARTIAL_CONTENT, HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE, \
    HTTP_500_INTERNAL_SERVER_ERROR, HTTP_404_NOT_FOUND

from entry.web.models import VideoInfo, VideoListResponse

# Setup
router = APIRouter(prefix="/api/v1/videos", tags=["videos"])
logger = logging.getLogger(__name__)
WORK_DIR = Path(os.getenv("AIC25_WORK_DIR") or ".")

# Video streaming configuration
CHUNK_SIZE = 1024 * 1024  # 1MB chunks
MAX_CHUNK_SIZE = 8 * 1024 * 1024  # 8MB max chunk


@router.get("/", response_model=VideoListResponse)
async def list_videos(
    offset: int = Query(default=0, ge=0, description="Video offset"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum videos"),
    search: Optional[str] = Query(default=None, description="Search in video names"),
):
    """
    List all available videos with metadata.

    Returns a paginated list of videos with their metadata including
    duration, frame rate, and file information.
    """
    try:
        videos = []
        videos_dir = WORK_DIR / "videos"
        videos_info_dir = WORK_DIR / "videos_info"

        if not videos_dir.exists():
            return VideoListResponse(
                success=True, videos=[], total=0, message="Videos directory not found"
            )

        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        video_files = []

        for ext in video_extensions:
            video_files.extend(videos_dir.glob(f"*{ext}"))

        if search:
            video_files = [f for f in video_files if search.lower() in f.stem.lower()]

        video_files.sort(key=lambda x: x.name)

        total_videos = len(video_files)
        paginated_files = video_files[offset : offset + limit]

        for video_file in paginated_files:
            video_id = video_file.stem

            video_info = VideoInfo(video_id=video_id, title=video_id, fps=25)

            info_file = videos_info_dir / f"{video_id}.json"
            if info_file.exists():
                try:
                    with open(info_file, "r") as f:
                        info_data = json.load(f)
                        video_info.fps = info_data.get("frame_rate", 25)
                        video_info.duration = info_data.get("duration")
                        video_info.total_frames = info_data.get("total_frames")
                except Exception as e:
                    logger.warning(f"Could not load info for {video_id}: {e}")

            try:
                video_info.file_size = video_file.stat().st_size
            except Exception as e:
                logger.warning(f"Could not get file size for {video_id}: {e}")

            videos.append(video_info)

        return VideoListResponse(success=True, videos=videos, total=total_videos)

    except Exception as e:
        logger.error(f"Video listing failed: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to list videos", "message": str(e)},
        )


@router.get("/{video_id}", response_model=VideoInfo)
async def get_video_info(video_id: str):
    """
    Get detailed information about a specific video.

    Returns comprehensive video metadata including duration, frame rate,
    file size, and other available information.
    """
    try:
        videos_dir = WORK_DIR / "videos"
        videos_info_dir = WORK_DIR / "videos_info"

        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        video_file = None

        for ext in video_extensions:
            potential_file = videos_dir / f"{video_id}{ext}"
            if potential_file.exists():
                video_file = potential_file
                break

        if not video_file:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail={"error": "Video not found", "video_id": video_id},
            )

        video_info = VideoInfo(video_id=video_id, title=video_id, fps=25)

        info_file = videos_info_dir / f"{video_id}.json"
        if info_file.exists():
            try:
                with open(info_file, "r") as f:
                    info_data = json.load(f)
                    video_info.fps = info_data.get("frame_rate", 25)
                    video_info.duration = info_data.get("duration")
                    video_info.total_frames = info_data.get("total_frames")
                    video_info.title = info_data.get("title", video_id)
            except Exception as e:
                logger.warning(f"Could not load info for {video_id}: {e}")

        try:
            video_info.file_size = video_file.stat().st_size
        except Exception as e:
            logger.warning(f"Could not get file size for {video_id}: {e}")

        return video_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video info retrieval failed: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to retrieve video information", "message": str(e)},
        )


@router.get("/{video_id}/stream")
async def stream_video(
    video_id: str,
    request: Request,
    range_header: Optional[str] = Header(None, alias="range"),
):
    """
    Stream video content with support for range requests.

    Supports HTTP range requests for efficient video streaming,
    allowing clients to seek to specific positions and stream
    only the required portions of the video.
    """
    try:
        videos_dir = WORK_DIR / "videos"

        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        video_file = None

        for ext in video_extensions:
            potential_file = videos_dir / f"{video_id}{ext}"
            if potential_file.exists():
                video_file = potential_file
                break

        if not video_file:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail={"error": "Video file not found", "video_id": video_id},
            )

        file_size = video_file.stat().st_size

        if range_header:
            try:
                # Parse range header: "bytes=start-end"
                range_match = range_header.replace("bytes=", "")
                ranges = range_match.split("-")
                start = int(ranges[0]) if ranges[0] else 0
                end = int(ranges[1]) if ranges[1] else file_size - 1

                if start >= file_size or end >= file_size or start > end:
                    return Response(
                        status_code=HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                        headers={"Content-Range": f"bytes */{file_size}"},
                    )

                chunk_size = end - start + 1
                if chunk_size > MAX_CHUNK_SIZE:
                    end = start + MAX_CHUNK_SIZE - 1
                    chunk_size = MAX_CHUNK_SIZE

                def iter_file_range():
                    with open(video_file, "rb") as f:
                        f.seek(start)
                        remaining = chunk_size
                        while remaining > 0:
                            read_size = min(CHUNK_SIZE, remaining)
                            data = f.read(read_size)
                            if not data:
                                break
                            remaining -= len(data)
                            yield data

                headers = {
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(chunk_size),
                    "Content-Type": "video/mp4",
                }

                return StreamingResponse(
                    iter_file_range(),
                    status_code=HTTP_206_PARTIAL_CONTENT,
                    headers=headers,
                )

            except (ValueError, IndexError) as e:
                logger.error(f"Invalid range header: {range_header}, error: {e}")

        def iter_file():
            with open(video_file, "rb") as f:
                while True:
                    data = f.read(CHUNK_SIZE)
                    if not data:
                        break
                    yield data

        headers = {
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
            "Content-Type": "video/mp4",
        }

        return StreamingResponse(iter_file(), headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video streaming failed: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to stream video", "message": str(e)},
        )


@router.get("/{video_id}/thumbnail")
async def get_video_thumbnail(video_id: str):
    """
    Get a thumbnail image for the video.

    Returns a representative thumbnail image for the video,
    typically the first frame or a generated preview image.
    """
    try:
        keyframes_dir = WORK_DIR / "keyframes" / video_id

        if not keyframes_dir.exists():
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail={"error": "No frames available for video", "video_id": video_id},
            )

        frame_files = list(keyframes_dir.glob("*.jpg"))
        if not frame_files:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail={
                    "error": "No frame images found for video",
                    "video_id": video_id,
                },
            )

        frame_files.sort()
        thumbnail_file = frame_files[0]

        from fastapi.responses import FileResponse

        return FileResponse(
            path=str(thumbnail_file),
            media_type="image/jpeg",
            filename=f"{video_id}_thumbnail.jpg",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Thumbnail retrieval failed: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to retrieve video thumbnail", "message": str(e)},
        )
