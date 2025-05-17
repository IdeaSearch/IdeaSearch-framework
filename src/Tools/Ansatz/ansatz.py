import re
import ast
import astor
import random
import numpy as np
from typing import List
from typing import Tuple
from typing import TypeVar
from typing import Callable
from time import perf_counter
from scipy.optimize import minimize
from src.Tools.TaskRunner.task_runner import execute_python_script


__all__ = [
    "check_ansatz_format",
    "use_ansatz_random_trial",
    "use_ansatz_optimize",
]


def check_ansatz_format(
    expression: str,
    variables: list[str],
    functions: list[str],
) -> int:
    
    """
    检查输入的表达式是否符合预定义的拟设（ansatz）格式要求。

    本函数会：
    - 校验表达式中使用的运算符、变量、函数是否符合预定义要求；
    - 确保变量名、函数名、参数名称等符号合法，并且符合语法要求；
    - 校验表达式中的参数是否按规定编号且连续，不允许存在常数。

    参数：
        expression (str): 被检查的数学表达式字符串。
        variables (list[str]): 允许使用的变量名列表，表达式中的变量必须严格来自该列表。
        functions (list[str]): 允许使用的函数名列表，函数名必须为裸函数名，不带模块前缀。

    返回值：
        int: 
            - 如果表达式合法，返回最大参数编号（即 'paramN' 的 N 值）。
            - 如果表达式不合法，返回 0。

    注意：
        本函数会首先对 `variables` 和 `functions` 中的内容进行合法性校验，若包含非法名称（如带模块前缀的函数名），将抛出异常。
    """
    
    identifier_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    for name in variables + functions:
        if not identifier_pattern.fullmatch(name):
            raise ValueError(f"非法标识符名：'{name}'，应仅由字母、数字、下划线组成，不能包含点号等")

    if re.search(r"[^\w\s+\-*/(),]", expression):
        return 0

    try:
        tree = ast.parse(expression, mode="eval")
    except Exception:
        return 0

    used_names = set()
    used_funcs = set()

    param_indices = set()

    def visit(node):
        if isinstance(node, ast.BinOp) or isinstance(node, ast.UnaryOp):
            allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
            allowed_unops = (ast.UAdd, ast.USub)
            if isinstance(node, ast.BinOp):
                if not isinstance(node.op, allowed_binops):
                    raise ValueError("不支持的二元运算符")
            if isinstance(node, ast.UnaryOp):
                if not isinstance(node.op, allowed_unops):
                    raise ValueError("不支持的一元运算符")
            visit(node.operand if isinstance(node, ast.UnaryOp) else node.left)
            if isinstance(node, ast.BinOp):
                visit(node.right)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("函数调用形式非法")
            func_name = node.func.id
            if func_name not in functions:
                raise ValueError(f"调用了未注册的函数 '{func_name}'")
            used_funcs.add(func_name)
            for arg in node.args:
                visit(arg)
        elif isinstance(node, ast.Name):
            name = node.id
            used_names.add(name)
            if name.startswith("param"):
                match = re.fullmatch(r"param([1-9][0-9]*)", name)
                if not match:
                    raise ValueError(f"非法参数名称 '{name}'")
                param_indices.add(int(match.group(1)))
            elif name not in variables and name not in functions:
                raise ValueError(f"使用了非法变量或未注册函数 '{name}'")
        elif isinstance(node, ast.Constant):
            raise ValueError("表达式中不允许使用任何常数")
        elif isinstance(node, ast.Expr):
            visit(node.value)
        else:
            raise ValueError(f"表达式中包含不支持的语法节点类型：{type(node).__name__}")

    try:
        visit(tree.body)
    except Exception:
        return 0

    if param_indices:
        max_index = max(param_indices)
        if sorted(param_indices) != list(range(1, max_index + 1)):
            return 0
        return max_index
    else:
        return 0
    
    
UseNumericAnsatzReturnType = TypeVar("use_numeric_ansatz_return_type")

def use_ansatz_random_trial(
    ansatz: str,
    param_num: int,
    param_ranges: List[Tuple[float, float]],
    use_numeric_ansatz: Callable[[str], UseNumericAnsatzReturnType],
    trial_num: int,
    seed: int
) -> List[UseNumericAnsatzReturnType]:
    
    """
    在给定参数范围内进行多次随机尝试，替换参数后评估 ansatz 表达式，并返回全部评估结果。

    本函数会：
    - 使用固定随机种子，生成 trial_num 组参数；
    - 将每组参数替换进 ansatz 表达式中，构造可计算的完整表达式；
    - 调用用户提供的评估函数 use_numeric_ansatz 逐个执行表达式；
    - 返回所有评估结果构成的列表。

    参数：
        ansatz (str): 含有 param1、param2 等参数占位符的表达式字符串。
        param_num (int): 参数个数，必须为正整数。
        param_ranges (list[Tuple[float, float]]): 每个参数的取值范围，长度必须等于 param_num。
        use_numeric_ansatz (Callable[[str], UseNumericAnsatzReturnType]): 用于评估表达式的函数，接受字符串返回任意类型。
        trial_num (int): 随机尝试次数，即生成多少组参数进行评估。
        seed (int): 随机种子，用于保证参数采样过程可复现。

    返回值：
        list[UseNumericAnsatzReturnType]: 包含每次评估结果的列表，类型由 use_numeric_ansatz 的返回值决定。
    """
    
    random_generator = random.Random(seed)
    results = []

    for _ in range(trial_num):
        parameter_values = [
            random_generator.uniform(param_ranges[i][0], param_ranges[i][1])
            for i in range(param_num)
        ]

        expression = ansatz
        for i, value in enumerate(parameter_values):
            param_name = f"param{i + 1}"
            
            tree = ast.parse(expression, mode='eval')
            class ParamReplacer(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id == param_name:
                        return ast.Constant(value)
                    return node

            transformer = ParamReplacer()
            modified_tree = transformer.visit(tree)
            ast.fix_missing_locations(modified_tree)
            expression = astor.to_source(modified_tree).strip()

        result = use_numeric_ansatz(expression)
        results.append(result)

    return results


def use_ansatz_optimize(
    ansatz: str,
    param_num: int,
    param_ranges: List[Tuple[float, float]],
    use_numeric_ansatz: Callable[[str], float],
    trial_num: int,
    seed: int,
    do_minimize: bool = True
) -> float:
    
    """
    在给定参数范围内进行多次优化尝试，返回最优评估结果。

    本函数会：
    - 使用固定随机种子，生成 trial_num 个初始参数点；
    - 对每个初始点执行一次完整的优化过程（如梯度下降）；
    - 将每次优化所得的最终参数替换进 ansatz 表达式中进行评估；
    - 返回所有评估结果中最优（最小或最大）的一个。

    参数：
        ansatz (str): 含有 param1、param2 等参数占位符的表达式字符串。
        param_num (int): 参数个数，必须为正整数。
        param_ranges (list[Tuple[float, float]]): 每个参数的取值范围，长度必须等于 param_num。
        use_numeric_ansatz (Callable[[str], float]): 用于评估表达式的函数，接受完整表达式字符串，返回一个 float 值。
        trial_num (int): 优化尝试次数，即从随机参数范围内生成多少个初始点并进行优化。
        seed (int): 随机种子，用于保证初始化参数生成过程可复现。
        do_minimize (bool): 是否寻找最小值，默认为 True。若为 False，则寻找最大值。

    返回值：
        float: 所有优化尝试中最优的评估结果值。
    """
    
    random_generator = random.Random(seed)
    bounds = param_ranges
    best_result = None

    def build_objective():
        def objective(params):
            expression = ansatz
            for i, value in enumerate(params):
                param_name = f"param{i + 1}"

                tree = ast.parse(expression, mode='eval')

                class ParamReplacer(ast.NodeTransformer):
                    def visit_Name(self, node):
                        if node.id == param_name:
                            return ast.Constant(value)
                        return node

                transformer = ParamReplacer()
                modified_tree = transformer.visit(tree)
                ast.fix_missing_locations(modified_tree)
                expression = astor.to_source(modified_tree).strip()

            result = use_numeric_ansatz(expression)
            return result if do_minimize else -result

        return objective

    for _ in range(trial_num):
        x0 = [
            random_generator.uniform(param_ranges[i][0], param_ranges[i][1])
            for i in range(param_num)
        ]

        res = minimize(build_objective(), x0, bounds=bounds, method='L-BFGS-B')

        final_value = res.fun if do_minimize else -res.fun

        if best_result is None:
            best_result = final_value
        else:
            best_result = min(best_result, final_value) if do_minimize else max(best_result, final_value)

    return best_result
    
    
if __name__ == "__main__":

    # check_ansatz_format 和 use_ansatz 的使用例
    ansatz = "(x - param1) ** param2"
    variables = ["x"]
    functions = []

    ansatz_param_num = check_ansatz_format(
        expression = ansatz,
        variables = variables,
        functions = functions,
    )

    if ansatz_param_num:

        param_ranges = [
            (2.0, 4.0),
            (2.0, 4.0),
        ]

        def use_numeric_ansatz(numeric_ansatz):
            script = f"""if __name__ == "__main__":
    x = 5.0
    print({numeric_ansatz}, end = "")
"""
            run_script_result = execute_python_script(
                script_content = script,
                timeout_seconds = 1,
                python_command = "python",
            )

            if not run_script_result["success"]:
                return 100.0

            return float(run_script_result["stdout"])

        # 使用随机试验方式评估 ansatz
        start = perf_counter()

        ansatz_output_list = use_ansatz_random_trial(
            ansatz = ansatz,
            param_num = ansatz_param_num,
            param_ranges = param_ranges,
            use_numeric_ansatz = use_numeric_ansatz,
            trial_num = 300,
            seed = 114514,
        )

        end = perf_counter()
        print(f"【随机采样】拟设最小值：{np.min(ansatz_output_list)} （用时 {int(end-start)} 秒）")

        # 使用优化方式评估 ansatz
        start = perf_counter()

        ansatz_optimized_value = use_ansatz_optimize(
            ansatz = ansatz,
            param_num = ansatz_param_num,
            param_ranges = param_ranges,
            use_numeric_ansatz = use_numeric_ansatz,
            trial_num = 20,
            seed = 1919810,
            do_minimize = True,
        )

        end = perf_counter()
        print(f"【优化搜索】拟设最小值：{ansatz_optimized_value} （用时 {int(end-start)} 秒）")

    else:
        print("拟设格式有误！")
