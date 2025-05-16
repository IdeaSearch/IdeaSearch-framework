import numpy as np
from typing import Callable
from scipy.integrate import quad


__all__ = [
    "calculate_chi_squared",
    "compute_1d_func_integral",
]


def calculate_chi_squared(
    predicted_data, 
    ground_truth_data, 
    errors,
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


def compute_1d_func_integral(
    func: Callable[[float], float],
    start: float,
    end: float,
    absolute_error: float,
    relative_error: float,
)-> float:
    
    """
    计算一维函数在指定区间内的定积分。

    使用 SciPy 的 quad 方法进行数值积分，允许用户设置绝对误差和相对误差控制精度。

    参数：
        func (Callable[[float], float]): 被积函数，接受一个 float 类型参数并返回 float。
        start (float): 积分区间起点。
        end (float): 积分区间终点。
        absolute_error (float): 积分允许的绝对误差容限。
        relative_error (float): 积分允许的相对误差容限。

    返回值：
        float: 计算得到的积分值。
    """
    
    integral_result = quad(
        func = func, 
        a = start,
        b = end,
        epsabs = absolute_error, 
        epsrel = relative_error,
    )[0]
    
    return integral_result


if __name__ == "__main__":
    
    pass
