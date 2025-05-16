import numpy as np
from typing import Optional


__all__ = [
    "assess",
]


def assess(
    ideas: list[str],
    scores: list[float],
    info: list[Optional[str]]
) -> float:
    
    database_score = np.max(np.array(scores))
    return database_score
