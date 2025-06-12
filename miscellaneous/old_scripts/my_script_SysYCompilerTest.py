from src.IdeaSearch.interface import IdeaSearch
from programs.SysYCompilerTest.user_code.prompt import system_prompt as SysYCompilerTest_system_prompt
from programs.SysYCompilerTest.user_code.prompt import prologue_section as SysYCompilerTest_prologue_section
from programs.SysYCompilerTest.user_code.prompt import epilogue_section as SysYCompilerTest_epilogue_section
from programs.SysYCompilerTest.user_code.evaluation import evaluate as SysYCompilerTest_evaluate
from programs.SysYCompilerTest.user_code.mutation import mutate as SysYCompilerTest_mutate
from programs.SysYCompilerTest.user_code.crossover import crossover as SysYCompilerTest_crossover
from programs.SysYCompilerTest.user_code.assessment import assess as SysYCompilerTest_assess


def IdeaSearch_interface()-> None:
    
    # Things you 【must】 modify
    program_name = "SysYCompilerTest"
    prologue_section = SysYCompilerTest_prologue_section
    epilogue_section = SysYCompilerTest_epilogue_section
    evaluate_func = SysYCompilerTest_evaluate
    
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
    model_assess_save_result = [
        True,
        False,
    ][0]
    model_assess_result_data_path = None
    model_sample_temperature = 60.0
    initialization_cleanse_threshold = 0.0
    delete_when_initial_cleanse = True
    hand_over_threshold = 1.0
    similarity_threshold = -0.1
    similarity_distance_func = None
    assess_func = [
        None,
        SysYCompilerTest_assess,
    ][1]
    assess_interval = None
    assess_result_data_path = None
    mutation_func = [
        None,
        SysYCompilerTest_mutate,
    ][0]
    mutation_interval = 8
    mutation_num = 2
    mutation_temperature = 2 * sample_temperature
    crossover_func = [
        None,
        SysYCompilerTest_crossover,
    ][0]
    crossover_interval = 15
    corssover_num = 5
    crossover_temperature = 2 * sample_temperature
    idea_uid_length = 6
    record_prompt_in_diary = False
    similarity_sys_info_thresholds = [
        5,
        10,
    ]
    similarity_sys_info_prompts = [
        "还可以再来点类似的，但要略有不同！",
        "已经有点多了，能不能在参考这个例子之余，稍微往别处想想，做做创新？",
        "太多了！请你之后回答时换一个和这个例子截然不同的思路吧！"
    ]
    system_prompt = SysYCompilerTest_system_prompt
    initialization_skip_evaluation = [
        False,
        True,
    ][0]
    evaluate_func_accept_evaluator_id = [
        False,
        True,
    ][1]
    
    # Max interaction num
    max_interaction_num = 10
    
    # Paths
    database_path = f"programs/{program_name}/database/"
    api_keys_path = "src/API4LLMs/api_keys.json"
    local_models_path = None
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
        model_assess_save_result = model_assess_save_result,
        model_assess_result_data_path = model_assess_result_data_path,
        model_sample_temperature = model_sample_temperature,
        epilogue_section = epilogue_section,
        max_interaction_num = max_interaction_num,
        evaluate_func = evaluate_func,
        system_prompt = system_prompt,
        assess_func = assess_func,
        assess_interval = assess_interval,
        assess_result_data_path = assess_result_data_path,
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
        evaluate_func_accept_evaluator_id = evaluate_func_accept_evaluator_id,
    )


if __name__ == "__main__":
    
    # IdeaSearch_interface()
    
    # you can use helper function cleanse_dataset
    # cleanse_dataset(
    #     database_path = "programs/SysYCompilerTest/dataset/",
    #     evaluate_func = SysYCompilerTest_evaluate,
    #     cleanse_threshold = 1.0,
    # )
    
    
    
    pass