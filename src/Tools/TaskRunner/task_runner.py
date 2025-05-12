import os
import shlex
import tempfile
import threading
import subprocess


__all__ = [
    "execute_python_script",
    "execute_shell_command",
]


tempfile_lock = threading.Lock()


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
    
    result_info = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "timeout": False,
        "exit_code": None,
        "exception": None
    }

    try:
       
        with tempfile_lock:
            temp_dir = tempfile.mkdtemp()
            tmp_file_path = os.path.join(temp_dir, "script.py")
            
        with open(tmp_file_path, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(script_content)
       
        process = subprocess.run(
            [python_command, tmp_file_path],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
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
        
    finally:
        with tempfile_lock:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    return result_info


def execute_shell_command(
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

