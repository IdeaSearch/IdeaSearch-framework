import os


__all__ = [
    "guarantee_path_exist",
    "append_to_file",
    "clear_file_content",
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
    encoding = "utf-16"):
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