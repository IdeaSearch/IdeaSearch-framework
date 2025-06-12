from src.IdeaSearch.interface import IdeaSearch
from programs.Combinatorics.user_code.problem_name import problem_name as Combinatorics_problem_name
from programs.Combinatorics.user_code.prompt import system_prompt as Combinatorics_system_prompt
from programs.Combinatorics.user_code.prompt import prologue_section as Combinatorics_prologue_section
from programs.Combinatorics.user_code.prompt import epilogue_section as Combinatorics_epilogue_section
from programs.Combinatorics.user_code.evaluation import evaluate as Combinatorics_evaluate
from programs.Combinatorics.user_code.assessment import assess as Combinatorics_assess
from programs.Combinatorics.user_code.mutation import mutate as Combinatorics_mutate
from programs.Combinatorics.user_code.crossover import crossover as Combinatorics_crossover


def IdeaSearch_interface()-> None:
    
    # Things you 【must】 modify
    program_name = "Combinatorics"
    problem_name = Combinatorics_problem_name
    prologue_section = Combinatorics_prologue_section
    epilogue_section = Combinatorics_epilogue_section
    evaluate_func = Combinatorics_evaluate
    
    # Algorithm parameters you may change
    samplers_num = 3
    sample_temperature = 30.0
    evaluators_num = samplers_num
    examples_num = 3
    generate_num = 1
    models = [
        "Deepseek_V3",
        "Deepseek_V3",
        "Deepseek_V3",
        "Deepseek_V3",
        "Deepseek_V3",
        "Deepseek_V3",
        "Deepseek_V3",
    ]
    model_temperatures = [
        0.2,
        0.4,
        0.7,
        1.0,
        1.1,
        1.2,
        1.3,
    ]
    model_assess_window_size = 20
    model_assess_initial_score = 100.0
    model_assess_average_order = 1.0
    model_assess_save_result = [
        False,
        True,
    ][1]
    model_assess_result_data_path = None
    model_sample_temperature = 60.0
    initialization_cleanse_threshold = 1.0
    delete_when_initial_cleanse = True
    hand_over_threshold = 1.0
    similarity_threshold = 0.1
    similarity_distance_func = None
    assess_func = [
        None, 
        Combinatorics_assess,
    ][1]
    assess_interval = 1
    assess_baseline = 80.0
    assess_result_data_path = None # use default: database_path + "data/database_assessment.npz"
    assess_result_pic_path = None # use default: database_path + "pic/database_assessment.png"
    mutation_func = [
        None,
        Combinatorics_mutate,
    ][1]
    mutation_interval = 1
    mutation_num = 10
    mutation_temperature = 40.0
    crossover_func = [
        None,
        Combinatorics_crossover,
    ][1]
    crossover_interval = 2
    corssover_num = 10
    crossover_temperature = 40.0
    idea_uid_length = 6
    record_prompt_in_diary = [
        False,
        True,
    ][0]
    similarity_sys_info_thresholds = [
        5,
        10,
    ]
    similarity_sys_info_prompts = [
        "还可以再来点类似的，但要略有不同！",
        "已经有点多了，能不能在参考这个例子之余，稍微往别处想想，做做创新？",
        "太多了！请你之后回答时换一个和这个例子截然不同的思路吧！"
    ]
    system_prompt = Combinatorics_system_prompt
    initialization_skip_evaluation = [
        False,
        True,
    ][1]
    evaluate_func_accept_evaluator_id = [
        False,
        True,
    ][0]
    
    # Max interaction num
    max_interaction_num = 20
    
    # Paths
    database_path = f"programs/{program_name}/database_{problem_name}/"
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
        assess_baseline = assess_baseline,
        assess_result_data_path = assess_result_data_path,
        assess_result_pic_path = assess_result_pic_path,
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
    
    IdeaSearch_interface()