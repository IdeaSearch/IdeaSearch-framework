from typing import Optional
import numpy as np


__all__ = [
    "evaluate",
]


evaluate_random_generator = np.random.default_rng()

def evaluate(
    idea: str,
)-> tuple[float, Optional[str]]:
    
    """
    对大语言模型生成的答案进行评估，返回分数和评语。

    Args:
        idea (str): 大语言模型生成的程序/文本。

    Returns:
        tuple[float, str]: 包含两个元素的元组：
            - float: 回答的评分（0~100）。
            - str: 对回答的简要评语或解释信息（可为 None）。
    """
    
    score = evaluate_random_generator.uniform(0.0, 100.0)
    info = "非常好！"
    return score, info
