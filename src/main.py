from programs.SysYCompilerTest.evaluator.evaluator import evaluate as SysYCompilerTest_evaluate
from programs.SysYCompilerTest.prompt import prologue_section as SysYCompilerTest_prologue_section
from programs.SysYCompilerTest.prompt import epilogue_section as SysYCompilerTest_epilogue_section
from src.API4LLMs.get_answer import get_answer
from src.FunSearch.interface import FunSearchInterface


if __name__ == "__main__":
    
    # prompt = prologue section + examples section + epilogue section
    FunSearchInterface(
        program_name =  "SysYCompilerTest",
        samplers_num = 3,
        evaluators_num = 3,
        prologue_section = SysYCompilerTest_prologue_section,
        examples_num = 3,
        generate_num = 1,
        model = "Deepseek_V3",
        epilogue_section = SysYCompilerTest_epilogue_section,
        max_interaction_num = 9,
        evaluate_func = SysYCompilerTest_evaluate,
    )
    
    