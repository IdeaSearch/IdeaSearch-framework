import numpy as np


__all__ = [
    "calculate_chi_squared",
]


def calculate_chi_squared(
    predicted_data, 
    ground_truth_data, 
    errors
):
    
    """
    计算卡方值（χ²），用于衡量预测值与真实值之间的偏差程度。

    卡方值的定义为：
        χ² = Σ [(预测值 - 实际值) / 误差]²

    参数说明：
        predicted_data (可迭代对象)：预测值列表或数组。
        ground_truth_data (可迭代对象)：真实观测值列表或数组。
        errors (可迭代对象)：每个数据点的误差（标准差），必须全部为正数。

    返回值：
        float：计算得到的卡方值。

    异常：
        ValueError：当输入长度不一致，或误差中存在非正数时抛出。
    """
    
    predicted_data = np.asarray(predicted_data)
    ground_truth_data = np.asarray(ground_truth_data)
    errors = np.asarray(errors)

    if not (len(predicted_data) == len(ground_truth_data) == len(errors)):
        raise ValueError("预测值、真实值和误差数组的长度必须一致。")
    if np.any(errors <= 0):
        raise ValueError("所有误差值必须为正数且非零。")

    chi_squared = np.sum(((predicted_data - ground_truth_data) / errors) ** 2)
    return chi_squared
