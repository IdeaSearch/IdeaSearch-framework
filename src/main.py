
import programs.FeynmanIntegralReduction.evaluator.evaluator as FeynmanIntegralReduction
import programs.SysYCompilerTest.evaluator.evaluator as SysYCompilerTest
from src.API4LLMs.get_answer import get_answer
from src.FunSearch.interface import FunSearchInterface


if __name__ == "__main__":
    
    # prompt = prologue section + examples section + epilogue section
    FunSearchInterface(
        program_name =  "FeynmanIntegralReduction",
        samplers_num = 3,
        evaluators_num = 3,
        prologue_section = (
            "嗯嗯阿訇收到后哇否吧v啊欧巴v"
        ),
        examples_num = 3,
        generate_num = 1,
        model = "Deepseek_V3",
        epilogue_section = (
            "asicubveubvoebobnaiobwobawobgaobvowbjob"
        ),
        max_interaction_num = 9,
        evaluate_func = FeynmanIntegralReduction.evaluate,
    )
    
    