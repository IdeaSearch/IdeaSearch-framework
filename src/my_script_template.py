from programs.TemplateProgram.evaluator.evaluator import evaluate as TemplateProgram_evaluate
from programs.TemplateProgram.prompt import prologue_section as TemplateProgram_prologue_section
from programs.TemplateProgram.prompt import epilogue_section as TemplateProgram_epilogue_section
from src.IdeaSearch.interface import IdeaSearchInterface


from programs.SysYCompilerTest.evaluator.evaluator import evaluate as SysYCompilerTest_evaluate
from programs.SysYCompilerTest.prompt import prologue_section as SysYCompilerTest_prologue_section
from programs.SysYCompilerTest.prompt import epilogue_section as SysYCompilerTest_epilogue_section
from src.IdeaSearch.interface import IdeaSearchInterface


def main()-> None:
    
    # Things you 【must】 modify
    program_name = "TemplateProgram"
    prologue_section = TemplateProgram_prologue_section
    epilogue_section = TemplateProgram_epilogue_section
    evaluate_func = TemplateProgram_evaluate
    
    # Algorithm parameters you may change
    samplers_num = 5
    sample_temperature = 50.0
    evaluators_num = samplers_num
    examples_num = 3
    generate_num = 2
    model = "Deepseek_V3"
    model_temperature = 1.0
    initialization_cleanse_threshold = 1.0
    delete_when_initial_cleanse = True
    evaluator_handle_threshold = 0.0
    
    # Max interaction num
    max_interaction_num = 5
    
    # Diary path
    diary_path = "src/diary.txt"
    
    # Start IdeaSearch
    # prompt = prologue section + examples section + epilogue section
    IdeaSearchInterface(
        program_name =  program_name,
        samplers_num = samplers_num,
        sample_temperature = sample_temperature,
        evaluators_num = evaluators_num,
        prologue_section = prologue_section,
        examples_num = examples_num,
        generate_num = generate_num,
        model = model,
        model_temperature = model_temperature,
        epilogue_section = epilogue_section,
        max_interaction_num = max_interaction_num,
        evaluate_func = evaluate_func,
        diary_path = diary_path,
        initialization_cleanse_threshold = initialization_cleanse_threshold,
        delete_when_initial_cleanse = delete_when_initial_cleanse,
        evaluator_handle_threshold = evaluator_handle_threshold,
    )


if __name__ == "__main__":
    
    main()
    