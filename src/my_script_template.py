from programs.TemplateProgram.evaluator.evaluator import evaluate as TemplateProgram_evaluate
from programs.TemplateProgram.prompt import prologue_section as TemplateProgram_prologue_section
from programs.TemplateProgram.prompt import epilogue_section as TemplateProgram_epilogue_section
from src.FunSearch.interface import FunSearchInterface


if __name__ == "__main__":
    
    # prompt = prologue section + examples section + epilogue section
    FunSearchInterface(
        program_name =  "TemplateProgram",
        samplers_num = 5,
        evaluators_num = 5,
        prologue_section = TemplateProgram_prologue_section,
        examples_num = 3,
        generate_num = 2,
        model = "Deepseek_V3",
        epilogue_section = TemplateProgram_epilogue_section,
        max_interaction_num = 300,
        evaluate_func = TemplateProgram_evaluate,
    )
    
    