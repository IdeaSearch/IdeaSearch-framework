from src.IdeaSearch.interface import IdeaSearch
from programs.TemplateProgram.user_code.prompt import system_prompt as TemplateProgram_system_prompt
from programs.TemplateProgram.user_code.prompt import prologue_section as TemplateProgram_prologue_section
from programs.TemplateProgram.user_code.prompt import epilogue_section as TemplateProgram_epilogue_section
from programs.TemplateProgram.user_code.evaluation import evaluate as TemplateProgram_evaluate
from programs.TemplateProgram.user_code.mutation import mutate as TemplateProgram_mutate
from programs.TemplateProgram.user_code.crossover import crossover as TemplateProgram_crossover
from programs.TemplateProgram.user_code.assessment import assess as TemplateProgram_assess


def IdeaSearch_interface()-> None:
    
    # Things you 【must】 modify
    program_name = "TemplateProgram"
    prologue_section = TemplateProgram_prologue_section
    epilogue_section = TemplateProgram_epilogue_section
    evaluate_func = TemplateProgram_evaluate
    
    # Algorithm parameters you may change
    score_range = (0.0, 100.0)
    samplers_num = 3
    sample_temperature = 40.0
    evaluators_num = samplers_num
    examples_num = 2
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
    model_assess_save_result = [
        False,
        True,
    ][1]
    model_assess_result_data_path = None  # use default: database_path + "data/model_scores.npz"
    model_assess_result_pic_path = None  # use default: database_path + "pic/model_scores.png"
    model_sample_temperature = 60.0
    initialization_cleanse_threshold = 0.0
    delete_when_initial_cleanse = True
    hand_over_threshold = 0.0
    similarity_threshold = 5.0
    similarity_distance_func = None
    assess_func = [
        None,
        TemplateProgram_assess,
    ][1]
    assess_interval = 1
    assess_baseline = 60.0
    assess_result_data_path = None # use default: database_path + "data/database_assessment.npz"
    assess_result_pic_path = None # use default: database_path + "pic/database_assessment.png"
    mutation_func = [
        None,
        TemplateProgram_mutate,
    ][0]
    mutation_interval = 3
    mutation_num = 3
    mutation_temperature = 2 * sample_temperature
    crossover_func = [
        None,
        TemplateProgram_crossover,
    ][0]
    crossover_interval = 6
    crossover_num = 9
    crossover_temperature = 2 * sample_temperature
    idea_uid_length = 4
    record_prompt_in_diary = [
        False,
        True,
    ][1]
    similarity_sys_info_thresholds = [
        5,
        10,
    ]
    similarity_sys_info_prompts = [
        "还可以再来点类似的，但要略有不同！",
        "已经有点多了，能不能在参考这个例子之余，稍微往别处想想，做做创新？",
        "太多了！请你之后回答时换一个和这个例子截然不同的思路吧！"
    ]
    system_prompt = TemplateProgram_system_prompt
    initialization_skip_evaluation = [
        False,
        True,
    ][0]
    
    # Max interaction num
    max_interaction_num = 80
    
    # Paths
    database_path = f"programs/{program_name}/database/"
    api_keys_path = "src/API4LLMs/api_keys.json"
    local_models_path = [
        "src/API4LLMs/local_models.json",
        None,
    ][1]
    diary_path = None # use default: database_path + "log/diary.txt"
    
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
        model_assess_save_result = model_assess_save_result,
        model_assess_result_data_path = model_assess_result_data_path,
        model_assess_result_pic_path = model_assess_result_pic_path,
        model_sample_temperature = model_sample_temperature,
        epilogue_section = epilogue_section,
        max_interaction_num = max_interaction_num,
        evaluate_func = evaluate_func,
        score_range = score_range,
        system_prompt = system_prompt, 
        assess_func = assess_func,
        assess_interval = assess_interval,
        assess_baseline = assess_baseline,
        assess_result_data_path = assess_result_data_path,
        assess_result_pic_path = assess_result_pic_path,
        mutation_func = mutation_func,
        mutation_interval = mutation_interval,
        mutation_num = mutation_num,
        mutation_temperature = mutation_temperature,
        crossover_func = crossover_func,
        crossover_interval = crossover_interval,
        crossover_num = crossover_num,
        crossover_temperature = crossover_temperature,
        diary_path = diary_path,
        api_keys_path = api_keys_path,
        local_models_path = local_models_path,
        initialization_cleanse_threshold = initialization_cleanse_threshold,
        initialization_skip_evaluation = initialization_skip_evaluation,
        delete_when_initial_cleanse = delete_when_initial_cleanse,
        hand_over_threshold = hand_over_threshold,
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
    
    pass