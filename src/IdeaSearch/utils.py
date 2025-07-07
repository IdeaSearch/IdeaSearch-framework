import os
import bisect
import numpy as np
from typing import List
from typing import Optional
from typing import Union
from typing import Sequence


__all__ = [
    "guarantee_path_exist",
    "append_to_file",
    "clear_file_content",
    "get_auto_markersize",
    "get_label",
    "default_assess_func",
    "make_boltzmann_choice",
]


def guarantee_path_exist(path):
    
    """
    确保给定路径存在，如果不存在则创建它。
    如果路径是文件，则创建文件所在的文件夹；
    如果路径是文件夹，则直接创建该文件夹。

    参数:
    path (str): 要检查或创建的路径。
    """

    # 获取文件的目录
    directory = os.path.dirname(path)

    # 检查目录是否存在
    if not os.path.exists(directory):
        # 创建目录
        os.makedirs(directory)

    # 如果路径是文件，确保文件存在
    if not os.path.exists(path):
        open(path, 'a').close()  # 创建空文件    


def append_to_file(
    file_path,
    content_str, 
    end="\n", 
    encoding = "utf-16"
)-> None:
    
    """
    将指定内容附加到文件末尾。

    参数:
    file_path (str): 目标文件的路径。
    content_str (str): 要写入文件的内容。
    end (str, optional): 内容结尾的字符，默认为换行符。
    encoding (str, optional): 编码方式，默认为utf-16
    """
    
    guarantee_path_exist(file_path)

    # 以追加模式打开文件并写入内容
    with open(file_path, "a", encoding=encoding) as file:
        file.write(content_str + end)
        
        
def clear_file_content(
    file_path, 
    encoding="utf-16"
):
    """
    清空指定文件的全部内容。
    
    若文件不存在，则静默跳过。

    参数:
        file_path (str): 目标文件的路径。
        encoding (str, optional): 编码方式，默认为 utf-16。
    """
    try:
        with open(file_path, "w", encoding=encoding):
            pass
    except FileNotFoundError:
        pass
    
    
def get_auto_markersize(
    point_num: int
)-> int:
        
    if point_num <= 20:
        auto_markersize = 8
    elif point_num <= 50:
        auto_markersize = 6
    elif point_num <= 100:
        auto_markersize = 4
    else:
        auto_markersize = 2
        
    return auto_markersize


def get_label(
    x: int, 
    thresholds: list[int], 
    labels: list[str]
) -> str:
    if not thresholds:
        raise ValueError("thresholds 列表不能为空")

    if len(labels) != len(thresholds) + 1:
        raise ValueError(
            f"labels 列表长度应比 thresholds 长 1，"
            f"但实际为 labels={len(labels)}, thresholds={len(thresholds)}"
        )
    
    index = bisect.bisect_right(thresholds, x)
    return labels[index]


def default_assess_func(
    ideas: List[str],
    scores: List[float],
    info: List[Optional[str]]
) -> float:
    
    database_score = np.max(np.array(scores))
    return database_score


boltzmann_random_generator = np.random.default_rng()

def make_boltzmann_choice(
    energies: Union[Sequence[float], np.ndarray],
    temperature: float,
    size: Optional[int] = None,
    replace: bool = False,
) -> Union[int, List[int]]:
    
    """
    从能量列表中根据 softmax(energy / temperature) 分布采样索引。

    特殊温度值：
        - temperature = np.inf: 高温极限，均匀采样；
        - temperature = 0.0: 低温极限，总是选择能量最高的索引（不是最低）；
    
    参数:
        energies: 能量列表。
        temperature: 温度参数。
        size: 返回样本数量，None 表示返回单个索引。
        replace: 是否放回采样。

    返回:
        单个索引，或多个索引组成的列表。
    """
    
    energies_ndarray = np.array(energies)
    num_energies = len(energies_ndarray)

    if temperature == np.inf:
        print("i am here~")
        # 高温：完全随机
        if size is None:
            return boltzmann_random_generator.choice(
                a = num_energies,
            )
        else:
            return boltzmann_random_generator.choice(
                a = num_energies,
                size = size,
                replace = replace,
            ).tolist()

    elif temperature == 0.0:
        # 低温极限：选择能量从高到低排序后的前 size 个索引
        sorted_indices = np.argsort(-energies_ndarray)

        if size is None:
            return int(sorted_indices[0])  # 确保返回 Python int
        else:
            return sorted_indices[:size].tolist()

    else:
        # 一般 softmax 采样
        logits = energies_ndarray / temperature
        max_logit = np.max(logits)
        probabilities = np.exp(logits - max_logit)
        probabilities /= np.sum(probabilities)

        if size is None:
            return boltzmann_random_generator.choice(
                a = num_energies,
                p = probabilities,
            )
        else:
            return boltzmann_random_generator.choice(
                a = num_energies,
                p = probabilities,
                size = size,
                replace = replace,
            ).tolist()