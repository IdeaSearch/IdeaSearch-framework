from src.IdeaSearch.interface import IdeaSearch, cleanse_dataset 
from programs.TemplateProgram.user_code.prompt import prologue_section as TemplateProgram_prologue_section
from programs.TemplateProgram.user_code.prompt import epilogue_section as TemplateProgram_epilogue_section
from programs.TemplateProgram.user_code.evaluation import evaluate as TemplateProgram_evaluate
from programs.TemplateProgram.user_code.mutation import mutate as TemplateProgram_mutate
from programs.TemplateProgram.user_code.crossover import crossover as TemplateProgram_crossover


def IdeaSearch_interface()-> None:
    
    # Things you 【must】 modify
    program_name = "TemplateProgram"
    prologue_section = TemplateProgram_prologue_section
    epilogue_section = TemplateProgram_epilogue_section
    evaluate_func = TemplateProgram_evaluate
    
    # Algorithm parameters you may change
    samplers_num = 3
    sample_temperature = 40.0
    evaluators_num = samplers_num
    examples_num = 3
    generate_num = 1
    models = [
        "Deepseek_V3",
        "Deepseek_V3",
        "Deepseek_V3",
    ]
    model_temperatures = [
        0.7,
        1.0,
        1.3,
    ]
    model_assess_window_size = 20
    model_assess_initial_score = 100.0
    model_assess_average_order = 1.0
    model_sample_temperature = 60.0
    initialization_cleanse_threshold = 0.0
    delete_when_initial_cleanse = True
    evaluator_hand_over_threshold = 0.0
    similarity_threshold = 0.1
    similarity_distance_func = None
    assess_func = None
    assess_interval = None
    assess_result_path = None
    mutation_func = TemplateProgram_mutate
    mutation_interval = 3
    mutation_num = 3
    mutation_temperature = 2 * sample_temperature
    crossover_func = TemplateProgram_crossover
    crossover_interval = 6
    corssover_num = 9
    crossover_temperature = 2 * sample_temperature
    idea_uid_length = 4
    record_prompt_in_diary = True
    similarity_sys_info_thresholds = [
        3,
    ]
    similarity_sys_info_prompts = [
        "还可以再来点！",
        "已经太多啦！",
    ]
    
    # Max interaction num
    max_interaction_num = 20
    
    # Paths
    database_path = f"programs/{program_name}/database/"
    api_keys_path = "src/API4LLMs/api_keys.json"
    diary_path = None # use default diary path: database_path + "log/diary.txt"
    
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
        mutation_num = mutation_num,
        mutation_temperature = mutation_temperature,
        crossover_func = crossover_func,
        crossover_interval = crossover_interval,
        crossover_num = corssover_num,
        crossover_temperature = crossover_temperature,
        diary_path = diary_path,
        api_keys_path = api_keys_path,
        initialization_cleanse_threshold = initialization_cleanse_threshold,
        delete_when_initial_cleanse = delete_when_initial_cleanse,
        evaluator_hand_over_threshold = evaluator_hand_over_threshold,
        similarity_threshold = similarity_threshold,
        similarity_distance_func = similarity_distance_func, 
        similarity_sys_info_thresholds = similarity_sys_info_thresholds,
        similarity_sys_info_prompts = similarity_sys_info_prompts,
        database_path = database_path,
        idea_uid_length = idea_uid_length,
        record_prompt_in_diary = record_prompt_in_diary,
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