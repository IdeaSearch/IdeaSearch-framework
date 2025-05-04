import concurrent.futures
from threading import Lock
from typing import Callable, Optional, List
from pathlib import Path

from src.utils import guarantee_path_exist, clear_file_content, append_to_file
from src.IdeaSearch.database import Idea, Database
from src.IdeaSearch.sampler import Sampler
from src.IdeaSearch.evaluator import Evaluator
from src.API4LLMs.model_manager import init_model_manager
from src.API4LLMs.model_manager import shutdown_model_manager


__all__ = [
    "IdeaSearch",
]


def IdeaSearch(
    program_name: str,
    prologue_section: str,
    epilogue_section: str,
    database_path: str,
    models: list[str],
    model_temperatures: list[float],
    max_interaction_num: int,
    evaluate_func: Callable[[str], tuple[float, str]],
    *, # 必填参数和选填参数的分界线
    score_range: tuple[float, float] = (0.0, 100.0),
    system_prompt: Optional[str] = None,
    diary_path: Optional[str] = None,
    api_keys_path: Optional[str] = None,
    local_models_path: Optional[str] = None,
    samplers_num: int = 5,
    evaluators_num: int = 5,
    sample_temperature: float = 50.0,
    examples_num: int = 3,
    generate_num: int = 5,
    assess_func: Optional[Callable[[list[str], list[float], list[str]], float]] = None,
    assess_interval: Optional[int] = None,
    assess_result_data_path: Optional[str] = None,
    assess_result_pic_path: Optional[str] = None,
    mutation_func: Optional[Callable[[str], str]] = None,
    mutation_interval: Optional[int] = None,
    mutation_num: Optional[int] = None,
    mutation_temperature: Optional[float] = None,
    crossover_func: Optional[Callable[[str, str], str]] = None,
    crossover_interval: Optional[int] = None,
    crossover_num: Optional[int] = None,
    crossover_temperature: Optional[float] = None,
    model_assess_window_size: int = 20,
    model_assess_initial_score: float = 100.0,
    model_assess_average_order: float = 1.0,
    model_assess_save_result: bool = True,
    model_assess_result_data_path: Optional[str] = None,
    model_sample_temperature: float = 50.0,
    evaluator_hand_over_threshold: float = 0.0,
    similarity_threshold: float = -1.0,
    similarity_distance_func: Optional[Callable[[str, str], float]] = None,
    similarity_sys_info_thresholds: Optional[list[int]] = None,
    similarity_sys_info_prompts: Optional[list[str]] = None,
    initialization_skip_evaluation: bool = True,
    initialization_cleanse_threshold: float = -1.0,
    delete_when_initial_cleanse: bool = False,
    idea_uid_length: int = 4,
    record_prompt_in_diary: bool = True,
) -> None:
    
    """
    czy remark (0501): doc string 暂未和项目的最新情况同步，有待维护！  
    启动并运行一个 IdeaSearch 搜索过程。

    该函数会创建一个线程安全的 Database 实例，并初始化指定数量的 Sampler 和 Evaluator 实例，
    使用线程池并发运行所有 Sampler，直到数据库达到最大交互次数为止。

    使用此函数的基本功能时，用户需自行创建 prologue_section、epilogue_section 以及 evaluate_func。
    其中 evaluate_func 是一个函数，接收一个字符串作为输入，返回一个二元组 (score, comment)，其中 score 为浮点数评分，
    comment 是可选的附加信息。我们建议 score 范围为 0.0 到 100.0。

    Args:
        program_name (str): 当前程序或实验的名称。
        prologue_section (str): 用于提示模型采样的前导文本片段。
        epilogue_section (str): 用于提示模型采样的结尾文本片段。
        database_path (str): 数据库路径，在此路径下存放 .idea 文件和自动生成的 score_sheet.json。
        diary_path (str): IdeaSearch 的日志文件路径。
        api_keys_path (str): 包含用于访问 LLM 的 API key 的 JSON 文件路径。
        models (list[str]): 用于生成 idea 的大语言模型名称列表，应在 api_keys_path 对应文件中定义。
        model_temperatures (list[float]): 与 models 对应的温度值列表，其长度应与 models 相同。
        max_interaction_num (int): 数据库允许的最大评估交互次数，超过此值即终止。
        evaluate_func (Callable[[str], tuple[float, str]]): 用于评估候选 idea 的函数，供 Evaluator 使用，返回 (score, comment)。
        
        samplers_num (int): 要创建的 Sampler 实例数量。
        evaluators_num (int): 要创建的 Evaluator 实例数量。
        sample_temperature (float): 采样 idea 时的温度参数，影响基于分数与相似度的 softmax 概率。
        examples_num (int): 每个 Prompt 展示给模型的示例 idea 数量。
        generate_num (int): 每个 Sampler 每轮生成的候选 idea 数量。
        assess_func (Optional[Callable[[list[str]], float]]): 评估一组 idea 的整体评分函数，可选。
        assess_interval (Optional[int]): 每隔多少轮评估一次所有 idea 的整体质量（调用 assess_func）。
        assess_result_data_path (Optional[str]): 存储每轮整体评估得分的文件路径。
        mutation_func (Optional[Callable[[str], str]]): 对 idea 进行突变的函数，可选。
        mutation_interval (Optional[int]): 每隔多少轮调用一次 mutation_func。
        crossover_func (Optional[Callable[[str, str], str]]): 对两个 idea 进行交叉的函数，可选。
        crossover_interval (Optional[int]): 每隔多少轮调用一次 crossover_func。
        model_assess_window_size (int): 模型评估窗口大小，决定各模型得分的滑动平均计算范围。
        model_assess_initial_score (float): 模型的初始分数，用于冷启动探索，建议设为较高值以鼓励尝试。
        model_assess_average_order (float): 评估模型时 p-范数的 p 值，影响对最大/最小值的敏感程度。
        model_sample_temperature (float): 模型采样温度参数，控制模型选择的概率分布平滑程度。
        evaluator_hand_over_threshold (float): Evaluator 允许 idea 进入数据库的最低分数门槛。
        similarity_threshold (float): 判定 idea 是否相似的阈值。默认为 -1.0，即仅完全一致时视为相似。
        similarity_distance_func (Optional[Callable[[str, str], float]]): 用于衡量 idea 相似度的距离函数，默认为分数差的绝对值。
        initialization_cleanse_threshold (float): 数据库初始化时清除低质量 idea 的分数阈值。
        delete_when_initial_cleanse (bool): 若为 True，在初始化清洗时会删除低分文件；否则仅跳过处理。
        idea_uid_length (int): 每个 idea 存储为 idea_{uid}.idea，uid 的字符长度由此参数决定。

    Returns:
        None
    """
    
    # czy remark (0501): entrance check 暂未和项目的最新情况同步，有待维护！  
    IdeaSearch_entrance_check(
        program_name,
        prologue_section,
        epilogue_section,
        database_path,
        diary_path,
        api_keys_path,
        models,
        model_temperatures,
        max_interaction_num,
        evaluate_func,
        samplers_num,
        evaluators_num,
        sample_temperature,
        examples_num,
        generate_num,
        assess_func,
        assess_interval,
        assess_result_data_path,
        mutation_func,
        mutation_interval,
        crossover_func,
        crossover_interval,
        model_assess_window_size,
        model_assess_initial_score,
        model_assess_average_order,
        model_sample_temperature,
        evaluator_hand_over_threshold,
        similarity_threshold,
        similarity_distance_func,
        initialization_cleanse_threshold,
        delete_when_initial_cleanse,
        idea_uid_length
    )
    
    if diary_path is None:
        diary_path = database_path + "log/diary.txt"
        
    if system_prompt is None:
        system_prompt = "You're a helpful assistant."
        
    if assess_func is not None:
        if assess_result_data_path is None:
            assess_result_data_path = database_path + "data/database_assessment.npz"
        if assess_result_pic_path is None:
            assess_result_pic_path = database_path + "data/database_assessment.png"
            
    if model_assess_save_result:
        if model_assess_result_data_path is None:
            model_assess_result_data_path = database_path + "data/model_scores.npy"
    
    def default_similarity_distance_func(idea1, idea2):
        return abs(evaluate_func(idea1)[0] - evaluate_func(idea2)[0])
    
    if similarity_distance_func is None:
        similarity_distance_func = default_similarity_distance_func
        
    init_model_manager(
        api_keys_path = api_keys_path,
        local_models_path = local_models_path,
    )
    
    guarantee_path_exist(diary_path)
    clear_file_content(diary_path)
    
    append_to_file(
        file_path = diary_path,
        content_str = f"【系统】 现在开始{program_name}的IdeaSearch！",
    )
    
    console_lock = Lock()

    database = Database(
        program_name = program_name,
        max_interaction_num = max_interaction_num,
        examples_num = examples_num,
        evaluate_func = evaluate_func,
        score_range = score_range,
        assess_func = assess_func,
        assess_interval = assess_interval,
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
        similarity_distance_func = similarity_distance_func,
        default_similarity_distance_func = default_similarity_distance_func,
        sample_temperature = sample_temperature,
        console_lock = console_lock,
        diary_path = diary_path,
        database_path = database_path,
        initialization_skip_evaluation = initialization_skip_evaluation,
        initialization_cleanse_threshold = initialization_cleanse_threshold,
        delete_when_initial_cleanse = delete_when_initial_cleanse,
        models = models,
        model_temperatures = model_temperatures,
        model_assess_window_size = model_assess_window_size,
        model_assess_initial_score = model_assess_initial_score,
        model_assess_average_order = model_assess_average_order,
        model_assess_save_result = model_assess_save_result,
        model_assess_result_data_path = model_assess_result_data_path,
        model_sample_temperature = model_sample_temperature,
        similarity_threshold = similarity_threshold,
        similarity_sys_info_thresholds = similarity_sys_info_thresholds,
        similarity_sys_info_prompts = similarity_sys_info_prompts,
        idea_uid_length = idea_uid_length,
    )
    evaluators = [
        Evaluator(
            evaluator_id = i + 1,
            database = database,
            evaluate_func = evaluate_func,
            console_lock = console_lock,
            diary_path = diary_path,
            evaluator_hand_over_threshold = evaluator_hand_over_threshold,
        ) 
        for i in range(evaluators_num)
    ]
    samplers = [
        Sampler(
            sampler_id = i + 1,
            prologue_section = prologue_section,
            epilogue_section = epilogue_section,
            evaluators = evaluators,
            generate_num = generate_num,
            database = database,
            console_lock = console_lock,
            diary_path = diary_path,
            system_prompt = system_prompt,
            record_prompt_in_diary = record_prompt_in_diary,
        )
        for i in range(samplers_num)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=samplers_num) as executor:
        futures = {executor.submit(sampler.run): sampler for sampler in samplers}
        for future in concurrent.futures.as_completed(futures):
            sampler = futures[future]
            try:
                _ = future.result() 
            except Exception as e:
                append_to_file(
                    file_path = diary_path,
                    content_str = f"【系统】 {sampler.id}号采样器在运行过程中出现错误：\n{e}\nIdeaSearch意外终止！",
                )
                exit()
                
    shutdown_model_manager()

    append_to_file(
        file_path = diary_path,
        content_str = f"【系统】 已达到最大互动次数， {program_name} 的 IdeaSearch 结束！",
    )
            
       
# czy remark (0501): entrance check 暂未和项目的最新情况同步，有待维护！     
def IdeaSearch_entrance_check(
    program_name,
    prologue_section,
    epilogue_section,
    database_path,
    diary_path,
    api_keys_path,
    models,
    model_temperatures,
    max_interaction_num,
    evaluate_func,
    samplers_num,
    evaluators_num,
    sample_temperature,
    examples_num,
    generate_num,
    assess_func,
    assess_interval,
    assess_result_data_path,
    mutation_func,
    mutation_interval,
    crossover_func,
    crossover_interval,
    model_assess_window_size,
    model_assess_initial_score,
    model_assess_average_order,
    model_sample_temperature,
    evaluator_hand_over_threshold,
    similarity_threshold,
    similarity_distance_func,
    initialization_cleanse_threshold,
    delete_when_initial_cleanse,
    idea_uid_length,
):
    # 基础字符串类型检查
    for name, val in {
        "program_name": program_name,
        "prologue_section": prologue_section,
        "epilogue_section": epilogue_section,
        "database_path": database_path,
        "api_keys_path": api_keys_path,
    }.items():
        if not isinstance(val, str):
            raise TypeError(f"【IdeaSearch参数类型错误】 `{name}` 应该是 str 类型，但接收到 {type(val).__name__}")

    if diary_path is not None:
        if not isinstance(diary_path, str):
            raise TypeError(f"【IdeaSearch参数类型错误】 `{diary_path}` 应该是 None 或 str 类型，但接收到 {type(diary_path).__name__}")
    
    # 列表类型检查
    if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
        raise TypeError("【IdeaSearch参数类型错误】 `models` 应该是 str 类型的列表")
    if not isinstance(model_temperatures, list) or not all(isinstance(t, (int, float)) for t in model_temperatures):
        raise TypeError("【IdeaSearch参数类型错误】 `model_temperatures` 应该是 float 类型的列表")

    # 整数、浮点数、函数等类型检查
    if not isinstance(max_interaction_num, int):
        raise TypeError("【IdeaSearch参数类型错误】 `max_interaction_num` 应该是 int 类型")
    if not callable(evaluate_func):
        raise TypeError("【IdeaSearch参数类型错误】 `evaluate_func` 应该是 callable 函数")
    for name, val in {
        "samplers_num": samplers_num,
        "evaluators_num": evaluators_num,
        "examples_num": examples_num,
        "generate_num": generate_num,
        "model_assess_window_size": model_assess_window_size,
        "idea_uid_length": idea_uid_length,
    }.items():
        if not isinstance(val, int):
            raise TypeError(f"【IdeaSearch参数类型错误】 `{name}` 应该是 int 类型，但接收到 {type(val).__name__}")
    for name, val in {
        "sample_temperature": sample_temperature,
        "model_assess_initial_score": model_assess_initial_score,
        "model_assess_average_order": model_assess_average_order,
        "model_sample_temperature": model_sample_temperature,
        "evaluator_hand_over_threshold": evaluator_hand_over_threshold,
        "similarity_threshold": similarity_threshold,
        "initialization_cleanse_threshold": initialization_cleanse_threshold,
    }.items():
        if not isinstance(val, (int, float)):
            raise TypeError(f"【IdeaSearch参数类型错误】 `{name}` 应该是 float 类型，但接收到 {type(val).__name__}")

    if not isinstance(delete_when_initial_cleanse, bool):
        raise TypeError("【IdeaSearch参数类型错误】 `delete_when_initial_cleanse` 应该是 bool 类型")

    # 函数及其配套参数的合法性检查
    if assess_func is not None:
        if not callable(assess_func):
            raise TypeError(f"【IdeaSearch参数类型错误】 `assess_func` 应该是 callable 函数")
        if assess_interval is None or not isinstance(assess_interval, int) or assess_interval <= 0:
            raise ValueError("【IdeaSearch参数值错误】 `assess_func` 设置时必须提供正整数的 assess_interval")
        if assess_result_data_path is not None and not isinstance(assess_result_data_path, str):
            raise ValueError("【IdeaSearch参数值错误】 `assess_func` 设置时 assess_result_data_path 应为 None 或一字符串！")
    
    if mutation_func is not None:
        if not callable(mutation_func):
            raise TypeError("【IdeaSearch参数类型错误】 `mutation_func` 应该是 callable 函数")
        if mutation_interval is None or not isinstance(mutation_interval, int) or mutation_interval <= 0:
            raise ValueError("【IdeaSearch参数值错误】 `mutation_func` 设置时必须提供正整数的 mutation_interval")

    if crossover_func is not None:
        if not callable(crossover_func):
            raise TypeError("【IdeaSearch参数类型错误】 `crossover_func` 应该是 callable 函数")
        if crossover_interval is None or not isinstance(crossover_interval, int) or crossover_interval <= 0:
            raise ValueError("【IdeaSearch参数值错误】 `crossover_func` 设置时必须提供正整数的 crossover_interval")
        
    if similarity_distance_func is not None and not callable(similarity_distance_func):
        raise TypeError("【IdeaSearch参数类型错误】 `similarity_distance_func` 应该是 Callable[[str, str], float] 或 None")