import numpy as np
from typing import Optional
from typing import Tuple
from threading import Lock


__all__ = [
    "evaluate",
]


evaluate_random_generator = np.random.default_rng()
evaluate_upper_bound = 5.0
evaluate_lock = Lock()

def evaluate(
    idea: str,
)-> Tuple[float, Optional[str]]:
    
    """
    对大语言模型生成的答案进行评估，返回分数和评语。

    Args:
        idea (str): 大语言模型生成的程序/文本。

    Returns:
        Tuple[float, str]: 包含两个元素的元组：
            - float: 回答的评分（0~100）。
            - str: 对回答的简要评语或解释信息（可为 None）。
    """
    
    with evaluate_lock:
    
        global evaluate_upper_bound
        global evaluate_random_generator
        
        score = evaluate_random_generator.uniform(0.0, evaluate_upper_bound)
        evaluate_upper_bound = min(
            max(
                0,
                evaluate_upper_bound + evaluate_random_generator.uniform(-1.0, 4.0),
            ),
            100.0
        )
        info = "非常好！" if score >= 80.0 else "一般般！"
        return score, info
