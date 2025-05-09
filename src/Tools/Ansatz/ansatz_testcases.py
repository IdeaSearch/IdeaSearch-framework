from collections import namedtuple


__all__ = [
    "check_format_testcases",
]


CheckFormatCase = namedtuple("CheckFormatCase", ["expression", "variables", "functions", "expected"])

check_format_testcases = [
    
    # ✅ 合法的复杂表达式
    CheckFormatCase(
        expression = "para1 * x + sin(para2) - para3 / (y + para4) + cos(z)",
        variables = ["x", "y", "z"],
        functions = ["sin", "cos"],
        expected = 4,
    ),
    
    # ✅ 合法的复杂表达式
    CheckFormatCase(
        expression = "-para1 * x + +para2 - sin(para3 + para4 * y)",
        variables = ["x", "y"],
        functions = ["sin"],
        expected = 4,
    ),
    
    # ✅ 合法的复杂表达式
    CheckFormatCase(
        expression = "log(para1) + sqrt(para2) + para3**x",
        variables = ["x"],
        functions = ["log", "sqrt"],
        expected = 3,
    ),
    
    # ✅ 合法的复杂表达式
    CheckFormatCase(
        expression = "abs(para1 - para2) + exp(-para3 * x)",
        variables = ["x"],
        functions = ["abs", "exp"],
        expected = 3,
    ),

    # ❌ 错误的参数编号（跳号）
    CheckFormatCase(
        expression = "para1 + para3",
        variables = [],
        functions = [],
        expected = 0,
    ),

    # ❌ 使用常数
    CheckFormatCase(
        expression = "para1 * 2",
        variables = [],
        functions = [],
        expected = 0,
    ),

    # ❌ 使用未列出的变量
    CheckFormatCase(
        expression = "para1 * x + y",
        variables = ["x"],
        functions = [],
        expected = 0,
    ),

    # ❌ 使用未列出的函数
    CheckFormatCase(
        expression = "para1 * x + tan(para2)",
        variables = ["x"],
        functions = ["sin", "cos"],
        expected = 0,
    ),

    # ❌ 使用非法字符
    CheckFormatCase(
        expression = "para1 * x + y$",
        variables = ["x", "y"],
        functions = [],
        expected = 0,
    ),

    # ❌ 使用模块前缀的函数名
    CheckFormatCase(
        expression = "np.sin(para1) + para2",
        variables = [],
        functions = ["sin"],
        expected = 0,
    ),

    # ✅ 合法的一元运算符组合
    CheckFormatCase(
        expression = "-para1 + +para2 * x",
        variables = ["x"],
        functions = [],
        expected = 2,
    ),

    # ✅ 合法的函数嵌套
    CheckFormatCase(
        expression = "sin(cos(para1)) + log(sqrt(para2))",
        variables = [],
        functions = ["sin", "cos", "log", "sqrt"],
        expected = 2,
    ),

    # ❌ para编号不是从1开始
    CheckFormatCase(
        expression = "para0 + para1",
        variables = [],
        functions = [],
        expected = 0,
    ),

    # ✅ para编号重复
    CheckFormatCase(
        expression = "para1 + para1 + para2",
        variables = [],
        functions = [],
        expected = 2,
    ),

    # ✅ 单变量、单函数、单参数
    CheckFormatCase(
        expression = "sin(para1 * x)",
        variables = ["x"],
        functions = ["sin"],
        expected = 1,
    ),

    # ❌ 空表达式
    CheckFormatCase(
        expression = "",
        variables = [],
        functions = [],
        expected = 0,
    ),

    # ✅ 嵌套括号与操作符混用
    CheckFormatCase(
        expression = "((para1 + para2) * x) / (y + para3) - sin(para4)",
        variables = ["x", "y"],
        functions = ["sin"],
        expected = 4,
    ),
    
    # ✅ 合法：带多个变量和嵌套函数
    CheckFormatCase(
        expression = "exp(para1 * x) + sin(para2 * y) + cos(para3 * z)",
        variables = ["x", "y", "z"],
        functions = ["exp", "sin", "cos"],
        expected = 3,
    ),

    # ❌ 非法：出现小数点
    CheckFormatCase(
        expression = "para1 * 3.14 + x",
        variables = ["x"],
        functions = [],
        expected = 0,
    ),

    # ✅ 合法：仅使用一元运算符
    CheckFormatCase(
        expression = "-para1 + +para2",
        variables = [],
        functions = [],
        expected = 2,
    ),

    # ❌ 非法：使用双下划线（不影响合法性字符，但可作为边界测试）
    CheckFormatCase(
        expression = "para1__ + para2",
        variables = [],
        functions = [],
        expected = 0,
    ),

    # ✅ 合法：函数嵌套，复杂组合
    CheckFormatCase(
        expression = "log(exp(sin(para1))) + para2 * x - para3 / y",
        variables = ["x", "y"],
        functions = ["log", "exp", "sin"],
        expected = 3,
    ),

    # ❌ 非法：非法字符 @
    CheckFormatCase(
        expression = "para1 @ x",
        variables = ["x"],
        functions = [],
        expected = 0,
    ),

    # ❌ 非法：函数名为非法标识符
    CheckFormatCase(
        expression = "cos(para1) + x",
        variables = ["x"],
        functions = ["cosine"],  # 未包含 'cos'
        expected = 0,
    ),

    # ✅ 合法：变量、函数、参数混合使用
    CheckFormatCase(
        expression = "sin(x) + para1 * cos(y) + para2",
        variables = ["x", "y"],
        functions = ["sin", "cos"],
        expected = 2,
    ),

    # ✅ 合法：变量名为长名字，函数混合使用
    CheckFormatCase(
        expression = "sin(long_variable_name) + para1 + para2 * cos(another_var)",
        variables = ["long_variable_name", "another_var"],
        functions = ["sin", "cos"],
        expected = 2,
    ),

    # ❌ 非法：para编号不从1开始
    CheckFormatCase(
        expression = "para2 + para3 + para4",
        variables = [],
        functions = [],
        expected = 0,
    ),

    # ❌ 非法：para编号缺失（para1, para2, para4）
    CheckFormatCase(
        expression = "para1 + para2 + para4",
        variables = [],
        functions = [],
        expected = 0,
    ),

    # ✅ 合法：大量连续 para
    CheckFormatCase(
        expression = "para1 + para2 + para3 + para4 + para5 + para6",
        variables = [],
        functions = [],
        expected = 6,
    ),
    
    # ✅ 合法：多变量函数 + 嵌套表达式
    CheckFormatCase(
        expression = "pow(para1 + para2, para3 + para4) + x * y - para5",
        variables = ["x", "y"],
        functions = ["pow"],
        expected = 5,
    ),

    # ✅ 合法：三层函数嵌套与多变量混用
    CheckFormatCase(
        expression = "log(sqrt(abs(para1 + x))) + para2 * y - para3 / z",
        variables = ["x", "y", "z"],
        functions = ["log", "sqrt", "abs"],
        expected = 3,
    ),

    # ❌ 非法：函数使用未注册名（pow未列出）
    CheckFormatCase(
        expression = "pow(para1, para2) + para3",
        variables = [],
        functions = ["exp", "log"],
        expected = 0,
    ),

    # ❌ 非法：函数中嵌套非法字符
    CheckFormatCase(
        expression = "log(sqrt(para1 + 3.14))",
        variables = [],
        functions = ["log", "sqrt"],
        expected = 0,
    ),

    # ✅ 合法：复杂括号与多层组合
    CheckFormatCase(
        expression = "(((para1 + para2))) * ((x + y)) - cos(para3) + sin(z)",
        variables = ["x", "y", "z"],
        functions = ["cos", "sin"],
        expected = 3,
    ),

    # ❌ 非法：para编号重复 + 未列函数
    CheckFormatCase(
        expression = "tan(para1 + para1) + para2",
        variables = [],
        functions = ["sin"],
        expected = 0,
    ),

    # ✅ 合法：三元组合函数 + 复杂操作
    CheckFormatCase(
        expression = "log(abs(sin(para1) + cos(para2))) * para3 + sqrt(para4)",
        variables = [],
        functions = ["log", "abs", "sin", "cos", "sqrt"],
        expected = 4,
    ),

    # ✅ 合法：变量 + para 混写， para 乱序
    CheckFormatCase(
        expression = "x * para1 + y * para3 + sin(para2)",
        variables = ["x", "y"],
        functions = ["sin"],
        expected = 3,
    ),

    # ✅ 合法：复合嵌套，三变量三函数
    CheckFormatCase(
        expression = "exp(sin(x) + cos(y)) + log(para1 + para2 * z)",
        variables = ["x", "y", "z"],
        functions = ["exp", "sin", "cos", "log"],
        expected = 2,
    ),

    # ❌ 非法：para后缀带下划线
    CheckFormatCase(
        expression = "para1_ + para2",
        variables = [],
        functions = [],
        expected = 0,
    ),
    
    # ✅ 合法
    CheckFormatCase(
        expression = "power(para1 + x, para2) * para3 + sqrt(log(exp(para4 + x)))",
        variables = ["x"],
        functions = ["power", "sqrt", "log", "exp"],
        expected = 4,
    ),
    
    # ❌ 非法
    CheckFormatCase(
        expression = "power(para1 + x, para2) * para3 + sqrt(log(exp(para4 + x))) + x // para5",
        variables = ["x"],
        functions = ["power", "sqrt", "log", "exp"],
        expected = 0,
    ),
    
    # ❌ 非法
    CheckFormatCase(
        expression = "power(para1 + x, para2) * para3 + sqrt(log(exp(para4 + x))) + 1",
        variables = ["x"],
        functions = ["power", "sqrt", "log", "exp"],
        expected = 0,
    )
]
