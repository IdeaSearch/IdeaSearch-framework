import re
import ast
import random
from typing import List
from typing import Tuple
from typing import Callable


__all__ = [
    "check_ansatz_format",
    "use_ansatz",
]


def check_ansatz_format(
    expression: str,
    variables: list[str],
    functions: list[str],
) -> int:
    
    """
    检查输入的表达式是否符合预定义的拟设（ansatz）格式要求。

    参数说明：
    ----------
    expression : str
        被检查的数学表达式字符串。

    variables : list[str]
        表达式中允许使用的变量名列表，所有变量必须严格来自该列表。

    functions : list[str]
        表达式中允许使用的函数名列表，函数名必须为裸函数名（如 'sin'），
        不允许包含模块前缀（如 'np.sin'）。

    格式要求：
    ----------
    1. 表达式仅允许使用以下运算符：+、-、*、/、**，按 Python 默认优先级解析；
       其中 + 和 - 可作为一元运算符（unary operator）。

    2. 所有变量名必须来自 `variables` 列表，表达式中不允许出现其他变量。

    3. 表达式不得使用任何常数，允许的“参数”格式为 'para' 加正整数编号，例如 'para1'、'para2' 等；
       且必须从 para1 开始，连续编号，中间不允许跳过（如 para1、para3 是不合法的），但是允许重复
       （如可以出现两次 para1 ）。

    4. 只能调用 `functions` 列表中明确允许的函数。

    5. 表达式中只允许出现以下字符：英文字母、数字、下划线、加号、减号、星号、斜杠、
       括号和英文逗号（,）；不允许出现其他符号（如小数点、引号、汉字等）。

    注意：
    -----
    本函数也会首先对 `variables` 和 `functions` 中的内容进行合法性校验，
    若其中包含非法名称（如带模块前缀的函数名），将立即抛出异常。

    返回值：
    -------
    int
        如果表达式合法，返回其中符合格式的参数数量（参数最高至 'paraN' 则返回正整数N）；
        如果表达式不合法，返回 0。
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

    para_indices = set()

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
            if name.startswith("para"):
                match = re.fullmatch(r"para([1-9][0-9]*)", name)
                if not match:
                    raise ValueError(f"非法参数名称 '{name}'")
                para_indices.add(int(match.group(1)))
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

    if para_indices:
        max_index = max(para_indices)
        if sorted(para_indices) != list(range(1, max_index + 1)):
            return 0
        return max_index
    else:
        return 0
    
    
def use_ansatz(
    ansatz: str,
    para_num: int,
    para_ranges: List[Tuple[float, float]],
    numeric_ansatz_evaluate_func: Callable[[str], float],
    trial_num: int,
    seed: int
) -> List[float]:
    random_generator = random.Random(seed)
    results = []
    
    for _ in range(trial_num):
        parameter_values = [
            random_generator.uniform(para_ranges[i][0], para_ranges[i][1]) 
            for i in range(para_num)
        ]
        
        expression = ansatz
        
        for i, value in enumerate(parameter_values):
            param_name = f"para{i + 1}"
            expression = replace_param_with_value(expression, param_name, value)
        
        result = numeric_ansatz_evaluate_func(expression)
        results.append(result)
    
    return results

def replace_param_with_value(expression: str, param_name: str, value: float) -> str:
    tree = ast.parse(expression, mode='eval')
    
    class ParamReplacer(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == param_name:
                return ast.Constant(value)
            return node

    transformer = ParamReplacer()
    modified_tree = transformer.visit(tree)
    ast.fix_missing_locations(modified_tree)
    
    return compile(modified_tree, filename="<ast>", mode="eval")
    
    
if __name__ == "__main__":
    
    print(check_ansatz_format(
        expression = "sqrt(x) + para1 / para1 + cos(para2 * x)",
        variables = ["x"],
        functions = ["sqrt", "cos"]
    ))