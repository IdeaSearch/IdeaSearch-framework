from programs.TemplateProgram.evaluator.evaluator import evaluate as TemplateProgram_evaluate
from programs.TemplateProgram.prompt import prologue_section as TemplateProgram_prologue_section
from programs.TemplateProgram.prompt import epilogue_section as TemplateProgram_epilogue_section
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
    models = [
        "Qwen_Max",
        "Qwen_Max",
        "Qwen_Max",
    ]
    model_temperatures = [
        0.9,
        1.0,
        1.1,
    ]
    model_assess_window_size = 10
    model_assess_initial_score = 100.0
    model_sample_temperature = 45.0
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
        models = models,
        model_temperatures = model_temperatures,
        model_assess_window_size = model_assess_window_size,
        model_assess_initial_score = model_assess_initial_score,
        model_sample_temperature = model_sample_temperature,
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
    