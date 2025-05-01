from src.IdeaSearch.interface import IdeaSearch, cleanse_dataset 
from programs.TemplateProgram.evaluator.evaluator import evaluate as TemplateProgram_evaluate
from programs.TemplateProgram.prompt import prologue_section as TemplateProgram_prologue_section
from programs.TemplateProgram.prompt import epilogue_section as TemplateProgram_epilogue_section

# i love zhangboyuan

def IdeaSearch_interface()-> None:
    
    # Things you 【must】 modify
    program_name = "TemplateProgram"
    prologue_section = TemplateProgram_prologue_section
    epilogue_section = TemplateProgram_epilogue_section
    evaluate_func = TemplateProgram_evaluate
    
    # Algorithm parameters you may change
    samplers_num = 5
    sample_temperature = 50.0
    evaluators_num = samplers_num
    examples_num = 4
    generate_num = 2
    models = [
        "Qwen_Max",
        "Qwen_Max",
        "Qwen_Max",
        "Deepseek_V3",
        "Deepseek_V3",
        "Deepseek_V3",
    ]
    model_temperatures = [
        0.9,
        1.0,
        1.1,
        0.9,
        1.0,
        1.1,
    ]
    model_assess_window_size = 20
    model_assess_initial_score = 100.0
    model_assess_average_order = 1.0
    model_sample_temperature = 60.0
    initialization_cleanse_threshold = 1.0
    delete_when_initial_cleanse = True
    evaluator_handle_threshold = 0.0
    similarity_threshold = 0.1
    similarity_distance_func = None
    assess_func = None,
    assess_interval = None,
    assess_result_path = None,
    mutation_func = None,
    mutation_interval = None,
    crossover_func = None,
    crossover_interval = None,
    idea_uid_length = 4
    
    # Max interaction num
    max_interaction_num = 10
    
    # Paths
    diary_path = "src/diary.txt"
    database_path = f"programs/{program_name}/database/"
    api_keys_path = "src/API4LLMs/api_keys.json"
    
    # Start IdeaSearch
    # prompt = prologue section + examples section + epilogue section
    IdeaSearch(
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
        model_assess_average_order = model_assess_average_order,
        model_sample_temperature = model_sample_temperature,
        epilogue_section = epilogue_section,
        max_interaction_num = max_interaction_num,
        evaluate_func = evaluate_func,
        assess_func = assess_func,
        assess_interval = assess_interval,
        assess_result_path = assess_result_path,
        mutation_func = mutation_func,
        mutation_interval = mutation_interval,
        crossover_func = crossover_func,
        crossover_interval = crossover_interval,
        diary_path = diary_path,
        api_keys_path = api_keys_path,
        initialization_cleanse_threshold = initialization_cleanse_threshold,
        delete_when_initial_cleanse = delete_when_initial_cleanse,
        evaluator_handle_threshold = evaluator_handle_threshold,
        similarity_threshold = similarity_threshold,
        similarity_distance_func = similarity_distance_func, 
        database_path = database_path,
        idea_uid_length = idea_uid_length,
    )


if __name__ == "__main__":
    
    IdeaSearch_interface()
    
    # you can use helper function cleanse_dataset
    # cleanse_dataset(
    #     database_path = "programs/TemplateProgram/dataset/",
    #     evaluate_func = TemplateProgram_evaluate,
    #     cleanse_threshold = 1.0,
    # )
    
    pass