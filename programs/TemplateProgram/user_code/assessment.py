import numpy as np
from typing import Optional
from typing import List


__all__ = [
    "assess",
]


def assess(
    ideas: List[str],
    scores: List[float],
    infos: List[Optional[str]]
) -> float:
    
    database_score = np.max(np.array(scores))
    return database_score
