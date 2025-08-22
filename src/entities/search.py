from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SearchResult:
    frame_id: str
    distance: float
    entity: Dict[str, Any]
