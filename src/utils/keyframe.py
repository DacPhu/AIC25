import subprocess
from pathlib import Path
from typing import Union


def get_keyframes_list(video_path: Union[str, Path]):
    ffprobe_cmd = (
        ["ffprobe", "-v", "quiet"]
        + [
            "-select_streams",
            "v",
            "-show_frames",
            "-show_entries",
            "frame=pict_type",
        ]
        + ["-of", "csv", str(video_path)]
    )
    res = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
    keyframes_list = res.stdout.strip().split("\n")
    keyframes_list = [x for x in keyframes_list if x.startswith("frame")]
    keyframes_list = [
        i for i, x in enumerate(keyframes_list) if x.startswith("frame,I")
    ]
    return keyframes_list
