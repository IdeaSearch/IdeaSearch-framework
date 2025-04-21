# please implement your evaluator here
def evaluate(llm_answer: str) -> tuple[float, str]:
    """
    对语言模型生成的答案进行评估，返回分数和评语。

    Args:
        llm_answer (str): 语言模型生成的回答文本。

    Returns:
        tuple[float, str]: 包含两个元素的元组：
            - float: 回答的评分（0~100）。
            - str: 对回答的简要评语或解释信息（可为 None）。
    """
    score = 100.00
    info = None
    return score, info


