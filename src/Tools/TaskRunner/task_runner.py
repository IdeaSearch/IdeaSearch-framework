import re
import os
import shlex
import tempfile
import subprocess
import numpy as np
from typing import Optional
from threading import Lock


__all__ = [
    "get_tmp_file_path",
    "delete_file",
    "execute_python_script",
    "execute_command",
    "measure_perf_runtime",
]


tempfile_lock = Lock()

def get_tmp_file_path(
    suffix: Optional[str] = "",
    prefix: str = "tmp_",
    directory: Optional[str] = None,
) -> str:
    
    """
    在线程安全的环境中生成一个临时文件路径或临时目录。

    本函数会：
      - 如果 suffix 为 None，则创建一个临时目录并返回其路径；
      - 否则，在线程锁保护下生成一个唯一的临时文件路径（不创建文件）。

    Args:
        suffix (str or None): 文件后缀名；若为 None，则表示创建临时目录。
        prefix (str): 文件或目录前缀，默认 "tmp_"。
        directory (str): 保存路径，默认使用系统临时目录。

    Returns:
        str: 生成的临时文件路径或临时目录路径。
    """
    with tempfile_lock:
        if suffix is None:
            tmp_dir_path = tempfile.mkdtemp(
                prefix = prefix,
                dir = directory,
            )
            return tmp_dir_path
        else:
            tmp_file_path = tempfile.mktemp(
                suffix = suffix,
                prefix = prefix,
                dir = directory,
            )
            return tmp_file_path


def delete_file(
    path: str
) -> None:
    
    """
    删除指定的文件或空目录。

    本函数会：
      - 在线程锁保护下删除指定路径的文件或空目录。
    
    Args:
        path (str): 要删除的文件或空目录路径。
    """
    
    with tempfile_lock:
        if os.path.exists(path):
            if os.path.isdir(path):
                os.rmdir(path)
            else:
                os.remove(path)


def execute_command(
    command: str,
    timeout_seconds: int = 300,
    shell: bool = False,
) -> dict:
    
    """
    在线程安全的环境中执行一条 shell 命令，并捕获其输出与状态。

    本函数会：
      - 使用 subprocess 安全执行 shell 命令；
      - 捕获标准输出、错误输出和退出码；
      - 返回包含执行结果信息的字典。

    Args:
        command (str): 要执行的 shell 命令（字符串形式）。
        timeout_seconds (int): 最长允许的执行时间（单位：秒）。默认 300 秒。
        shell (bool): 是否启用 shell 模式。默认 False（推荐使用 False 提高安全性）。

    Returns:
        dict: 包含以下字段的执行结果信息：
            - success (bool): 是否成功执行（即退出码为 0）。
            - stdout (str): 命令的标准输出。
            - stderr (str): 命令的标准错误输出。
            - timeout (bool): 是否超时退出。
            - exit_code (int): 子进程的退出码。
            - exception (Optional[str]): 如果执行中发生异常，记录异常类型与信息，否则为 None。
    """

    result_info = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "timeout": False,
        "exit_code": None,
        "exception": None,
    }

    try:
        
        args = command if shell else shlex.split(command)

        process = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
            shell=shell,
        )

        result_info["stdout"] = process.stdout
        result_info["stderr"] = process.stderr
        result_info["exit_code"] = process.returncode
        result_info["success"] = (process.returncode == 0)

    except subprocess.TimeoutExpired as e:
        result_info["timeout"] = True
        result_info["exception"] = f"TimeoutExpired: {e}"

    except Exception as e:
        result_info["exception"] = f"{type(e).__name__}: {e}"

    return result_info


def execute_python_script(
    script_content: str,
    timeout_seconds: int = 300,
    python_command: str = "python",
) -> dict:
    
    """
    在线程安全的环境中临时生成并执行一段 Python 脚本，并捕获其输出与状态。

    本函数会：
      - 在线程锁保护下生成唯一的临时目录与 Python 脚本文件；
      - 执行该脚本，捕获标准输出、错误输出和退出码；
      - 删除临时文件与目录，保持文件系统整洁；
      - 返回包含执行结果信息的字典。

    Args:
        script_content (str): 要执行的 Python 脚本内容（字符串形式）。
        timeout_seconds (int): 最长允许的执行时间（单位：秒）。默认 300 秒。
        python_command (str): Python 可执行命令（如 "python" 或 "python3"）。默认 "python"。

    Returns:
        dict: 包含以下字段的执行结果信息：
            - success (bool): 是否成功执行（即退出码为 0）。
            - stdout (str): 脚本的标准输出。
            - stderr (str): 脚本的标准错误输出。
            - timeout (bool): 是否超时退出。
            - exit_code (int): 子进程的退出码。
            - exception (Optional[str]): 如果执行中发生异常，记录异常类型与信息，否则为 None。
    """
       
    tmp_file_path = get_tmp_file_path(
        suffix = ".py"
    )
        
    with open(tmp_file_path, "w", encoding="utf-8") as tmp_file:
        tmp_file.write(script_content)
        
    result_info =  execute_command(
        command = python_command + " " + tmp_file_path,
        timeout_seconds = timeout_seconds,
        shell = False,
    )
    
    delete_file(
        path = tmp_file_path
    )
    
    return result_info


def measure_perf_runtime(
    path: str,
    trial_num: int = 3
) -> float:
    """
    使用 Linux 的 perf 工具测量一个可执行文件的平均运行时间（秒）.

    本函数会：
      - 通过 perf 工具多次执行给定的 .out 可执行文件
      - 解析 stderr 输出中 "seconds time elapsed" 所对应的浮点数字符串
      - 返回所有成功测量结果的平均值（单位：秒）

    注意事项：
      - perf 可能需要较高权限（例如 root 或 perf_event_paranoid 设定）
      - perf 会向 stderr 输出统计信息，因此本函数通过 stderr 进行解析
      - 若全部测量失败（无法解析时间），将抛出 RuntimeError

    Args:
        path (str): 可执行文件的路径（例如 "./a.out"）
        trial_num (int): 执行次数，默认 3 次以获得更稳健的平均值

    Returns:
        float: 平均运行时间（单位：秒）
    """
    times = []
    pattern = r"([\d.]+)\s+seconds time elapsed"

    for _ in range(trial_num):
        result = subprocess.run(
            ["perf", "stat", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )
        match = re.search(pattern, result.stderr)
        if match:
            times.append(float(match.group(1)))

    if times:
        return float(np.mean(times))
    else:
        raise RuntimeError(f"测量可执行文件 {path} 的运行时间时全部失败，可能是权限不足或 perf 未安装.")