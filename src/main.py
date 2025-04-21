
import programs.FeynmanIntegralReduction.evaluator.evaluator as FeynmanIntegralReduction
import programs.SysYCompilerTest.evaluator.evaluator as SysYCompilerTest
from src.API4LLMs.get_answer import get_answer
from src.FunSearch.interface import FunSearchInterface


if __name__ == "__main__":
    
    # prompt = prologue section + examples section + epilogue section
    FunSearchInterface(
        program_name =  "FeynmanIntegralReduction",
        samplers_num = 10,
        evaluators_num = 10,
        prologue_section = (
            r"</think>"
            "<我们正在尝试通过生成一个ansatz并对其变分来寻找最适合制备模拟规范场初态的量子线路>\n\n"
            "需要实现的函数框架：\n"
            "from qiskit import QuantumCircuit, transpile\n"
            "from qiskit.circuit import Parameter,ParameterVector\n"
            "\t# 这里应该是ansatz实现\n"
            "#def do_vqe():\n"
            "\tvqe = VQE(estimator=estimator,ansatz=qc_ansatz, optimizer=optimizer,callback=callback)\n"
            "\tresult = vqe.compute_minimum_eigenvalue(operator=H)\n"
            "\treturn result.eigenvalue\n"
            "# 其他相关函数结构...\n"
        ),
        examples_num = 3,
        generate_num = 1,
        model = "Deepseek_V3",
        epilogue_section = (
            "\n# 生成要求：\n"
            "# 1. 必须正确定义生成函数\n"
            "# 2. 函数输出应具有物理合理性（纠缠，转动等）\n"
            "# 3. 避免直接复制示例，至少完成以下一项（泛化之前的代码，简化之前的代码，进化之前的代码，创建独特的实现）\n"
            "# 4. 确保代码包含必要的数学运算库导入\n"
            "# 5. 注意代码参数尽可能的简洁性和可拓展性\n"
            "# 6. 将最终代码包裹在Python代码块中\n"
            r"</answer>"
        ),
        max_interaction_num = 10,
        evaluate_func = FeynmanIntegralReduction.evaluate,
    )
    
    