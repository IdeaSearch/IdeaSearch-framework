import os
import bisect


__all__ = [
    "guarantee_path_exist",
    "append_to_file",
    "clear_file_content",
    "get_auto_markersize",
    "get_label",
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

    # 以追加模式打开文件并写入内容
    with open(file_path, "a", encoding=encoding) as file:
        file.write(content_str + end)
        
def clear_file_content(
    file_path, 
    encoding="utf-16"):
    """
    清空指定文件的全部内容。

    参数:
    file_path (str): 目标文件的路径。
    encoding (str, optional): 编码方式，默认为utf-16。
    """

    # 以写入模式打开文件并立即关闭，清空原内容
    with open(file_path, "w", encoding=encoding):
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