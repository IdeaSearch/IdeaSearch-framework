import concurrent.futures
from threading import Lock
from typing import Tuple
from typing import Callable
from typing import Optional
# from typing import List
# from pathlib import Path
from src.utils import guarantee_path_exist
from src.utils import clear_file_content
from src.utils import append_to_file
from src.API4LLMs.model_manager import init_model_manager
from src.API4LLMs.model_manager import shutdown_model_manager
# from src.IdeaSearch.database import Idea
from src.IdeaSearch.database import Database
from src.IdeaSearch.sampler import Sampler
from src.IdeaSearch.evaluator import Evaluator


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
    evaluate_func: Callable[[str], Tuple[float, Optional[str]]],
    
    *,  # 选填参数分界线
    
    # 基础设置
    score_range: Tuple[float, float] = (0.0, 100.0),
    hand_over_threshold: float = 0.0,
    system_prompt: Optional[str] = None,
    diary_path: Optional[str] = None,
    api_keys_path: Optional[str] = None,
    local_models_path: Optional[str] = None,

    # 并发配置
    samplers_num: int = 5,
    evaluators_num: int = 5,

    # 样本生成
    examples_num: int = 3,
    generate_num: int = 5,
    sample_temperature: float = 50.0,
    model_sample_temperature: float = 50.0,

    # 数据库评估
    assess_func: Optional[Callable[[list[str], list[float], list[Optional[str]]], float]] = None,
    assess_interval: Optional[int] = None,
    assess_baseline: Optional[float] = None,
    assess_result_data_path: Optional[str] = None,
    assess_result_pic_path: Optional[str] = None,

    # 模型评估
    model_assess_window_size: int = 20,
    model_assess_initial_score: float = 100.0,
    model_assess_average_order: float = 1.0,
    model_assess_save_result: bool = True,
    model_assess_result_data_path: Optional[str] = None,
    model_assess_result_pic_path: Optional[str] = None,

    # 单体突变
    mutation_func: Optional[Callable[[str], str]] = None,
    mutation_interval: Optional[int] = None,
    mutation_num: Optional[int] = None,
    mutation_temperature: Optional[float] = None,

    # 交叉变异
    crossover_func: Optional[Callable[[str, str], str]] = None,
    crossover_interval: Optional[int] = None,
    crossover_num: Optional[int] = None,
    crossover_temperature: Optional[float] = None,

    # 相似度判断
    similarity_threshold: float = -1.0,
    similarity_distance_func: Optional[Callable[[str, str], float]] = None,
    similarity_sys_info_thresholds: Optional[list[int]] = None,
    similarity_sys_info_prompts: Optional[list[str]] = None,

    # 初始化阶段清洗
    initialization_skip_evaluation: bool = True,
    initialization_cleanse_threshold: float = -1.0,
    delete_when_initial_cleanse: bool = False,

    # 其他配置
    idea_uid_length: int = 4,
    record_prompt_in_diary: bool = True,
    filter_func: Optional[Callable[[str], str]] = None,
) -> None:
    
    """
    启动并运行一个 IdeaSearch 搜索过程。

    该函数会创建一个线程安全的 Database 实例，并初始化指定数量的 Sampler 和 Evaluator 实例，
    使用线程池并发运行所有 Sampler，直到数据库达到最大交互次数为止。期间可选地进行突变、交叉、相似性筛查、
    整体评估与模型评分，以支持不同实验需求。

    使用此函数的基本功能时，用户需提供 prologue_section、epilogue_section 和 evaluate_func。
    其中 evaluate_func 是一个函数，接收字符串 idea 作为输入，返回一个二元组 (score, info)，score 为浮点数评分，
    info 是可选的模型反馈信息。建议评分范围在 0.0 到 100.0 之间。

    Args:
        program_name (str): 当前项目的名称。
        prologue_section (str): 用于提示模型采样的前导文本片段。
        epilogue_section (str): 用于提示模型采样的结尾文本片段。
        database_path (str): 数据库路径，其下 ideas/ 路径用于存放诸 .idea 文件和 score_sheet.json。
        models (list[str]): 参与生成 idea 的模型名称列表。
        model_temperatures (list[float]): 各模型的采样温度，与 models 等长。
        max_interaction_num (int): 最大交互轮数，超过此值后系统自动终止。
        evaluate_func (Callable[[str], Tuple[float, str]]): 对单个 idea 进行评分的函数。

    Keyword Args:
        # 基础设置
        score_range (Tuple[float, float]): 评分区间范围，用于归一化和显示。
        hand_over_threshold (float): idea 进入数据库的最低评分要求。
        system_prompt (Optional[str]): 系统提示词。
        diary_path (Optional[str]): 日志文件路径。
        api_keys_path (Optional[str]): API key 配置文件路径。
        local_models_path (Optional[str]): 本地模型路径（如使用本地推理）。

        # 并发配置
        samplers_num (int): Sampler 实例数量。
        evaluators_num (int): Evaluator 实例数量。

        # 样本生成
        examples_num (int): 每轮展示给模型的历史 idea 数量。
        generate_num (int): 每轮每个 Sampler 生成的 idea 数量。
        sample_temperature (float): 控制 idea 选择的 softmax 温度。
        model_sample_temperature (float): 控制模型选择的 softmax 温度。

        # 数据库评估
        assess_func (Optional[Callable[[list[str], list[float], list[str]], float]]): 全体 idea 的综合评估函数。
        assess_interval (Optional[int]): 每隔多少轮进行一次 assess_func 评估。
        assess_baseline (Optional[int]): 数据库评估的基线，会在图像中显示。
        assess_result_data_path (Optional[str]): 存储评估得分的路径（.npz）。
        assess_result_pic_path (Optional[str]): 存储评估图像的路径（.png）。

        # 模型评估
        model_assess_window_size (int): 模型滑动平均评估窗口大小。
        model_assess_initial_score (float): 模型初始得分。
        model_assess_average_order (float): 模型评分滑动平均的 p 范数。
        model_assess_save_result (bool): 是否保存模型评估结果。
        model_assess_result_data_path (Optional[str]): 模型评估结果数据保存路径（.npz）。
        model_assess_result_pic_path (Optional[str]): 模型评估图像保存路径（.png）。

        # 单体突变
        mutation_func (Optional[Callable[[str], str]]): idea 的突变函数。
        mutation_interval (Optional[int]): 每隔多少轮进行一次突变操作。
        mutation_num (Optional[int]): 每轮进行的突变数量。
        mutation_temperature (Optional[float]): 控制突变候选选择的 softmax 温度。

        # 交叉变异
        crossover_func (Optional[Callable[[str, str], str]]): idea 的交叉函数。
        crossover_interval (Optional[int]): 每隔多少轮进行一次交叉操作。
        crossover_num (Optional[int]): 每轮交叉生成的 idea 数量。
        crossover_temperature (Optional[float]): 控制交叉候选选择的 softmax 温度。

        # 相似度判断
        similarity_threshold (float): idea 相似性的距离阈值，-1 表示仅完全一致为相似。
        similarity_distance_func (Optional[Callable[[str, str], float]]): idea 的相似度计算函数，默认实现为分数差的绝对值。
        similarity_sys_info_thresholds (Optional[list[int]]): 有关相似度的系统提示（“重复情况说明”）的诸阈值，为 None 则不开启此系统提示。
        similarity_sys_info_prompts (Optional[list[str]]): 与 thresholds 对应的系统提示内容。

        # 初始化阶段清洗
        initialization_skip_evaluation (bool): 是否跳过初始化阶段的评估（尝试从 score_sheet.json 中迅捷加载）。
        initialization_cleanse_threshold (float): 初始清洗的最低评分阈值。
        delete_when_initial_cleanse (bool): 清洗时是否直接删除低分 idea。

        # 其他配置
        idea_uid_length (int): idea 文件名中 uid 的长度。
        record_prompt_in_diary (bool): 是否将每轮的 Prompt 记录到日志中（建议初创子项目时打开此选项，后续关闭）。
        filter_func (Optional[Callable[[str], str]]): 采样拼prompt前可以先过一遍自行实现的filter_func（以去掉代码头、多余文字）。

    Returns:
        None
    """
    
    IdeaSearch_entrance_check(
        program_name,
        prologue_section,
        epilogue_section,
        database_path,
        models,
        model_temperatures,
        max_interaction_num,
        evaluate_func,
        score_range,
        hand_over_threshold,
        system_prompt,
        diary_path,
        api_keys_path,
        local_models_path,
        samplers_num,
        evaluators_num,
        examples_num,
        generate_num,
        sample_temperature,
        model_sample_temperature,
        assess_func,
        assess_interval,
        assess_baseline,
        assess_result_data_path,
        assess_result_pic_path,
        model_assess_window_size,
        model_assess_initial_score,
        model_assess_average_order,
        model_assess_save_result,
        model_assess_result_data_path,
        model_assess_result_pic_path,
        mutation_func,
        mutation_interval,
        mutation_num,
        mutation_temperature,
        crossover_func,
        crossover_interval,
        crossover_num,
        crossover_temperature,
        similarity_threshold,
        similarity_distance_func,
        similarity_sys_info_thresholds,
        similarity_sys_info_prompts,
        initialization_skip_evaluation,
        initialization_cleanse_threshold,
        delete_when_initial_cleanse,
        idea_uid_length,
        record_prompt_in_diary,
        filter_func,
    )
    
    if diary_path is None:
        diary_path = database_path + "log/diary.txt"
        
    if system_prompt is None:
        system_prompt = "You're a helpful assistant."
        
    if assess_func is not None:
        if assess_result_data_path is None:
            assess_result_data_path = database_path + "data/database_assessment.npz"
        if assess_result_pic_path is None:
            assess_result_pic_path = database_path + "pic/database_assessment.png"
            
    if model_assess_save_result:
        if model_assess_result_data_path is None:
            model_assess_result_data_path = database_path + "data/model_scores.npz"
        if model_assess_result_pic_path is None:
            model_assess_result_pic_path = database_path + "pic/model_scores.png"
    
    def default_similarity_distance_func(idea1, idea2):
        return abs(evaluate_func(idea1)[0] - evaluate_func(idea2)[0])
    
    score_range_expand_ratio = 0.1
    score_range_delta = (score_range[1] - score_range[0]) * score_range_expand_ratio
    score_range = (score_range[0] - score_range_delta, score_range[1] + score_range_delta)
    
    if similarity_distance_func is None:
        similarity_distance_func = default_similarity_distance_func
        
    init_model_manager(
        api_keys_path = api_keys_path,
        local_models_path = local_models_path,
    )
    
    guarantee_path_exist(diary_path)
    clear_file_content(diary_path)
    
    console_lock = Lock()
    with console_lock:
        append_to_file(
            file_path = diary_path,
            content_str = f"【系统】 现在开始{program_name}的IdeaSearch！",
        )

    database = Database(
        program_name = program_name,
        database_path = database_path,
        models = models,
        model_temperatures = model_temperatures,
        max_interaction_num = max_interaction_num,
        evaluate_func = evaluate_func,
        score_range = score_range,
        hand_over_threshold = hand_over_threshold,
        diary_path = diary_path,
        examples_num = examples_num,
        sample_temperature = sample_temperature,
        model_sample_temperature = model_sample_temperature,
        assess_func = assess_func,
        assess_interval = assess_interval,
        assess_baseline = assess_baseline,
        assess_result_data_path = assess_result_data_path,
        assess_result_pic_path = assess_result_pic_path,
        model_assess_window_size = model_assess_window_size,
        model_assess_initial_score = model_assess_initial_score,
        model_assess_average_order = model_assess_average_order,
        model_assess_save_result = model_assess_save_result,
        model_assess_result_data_path = model_assess_result_data_path,
        model_assess_result_pic_path = model_assess_result_pic_path,
        mutation_func = mutation_func,
        mutation_interval = mutation_interval,
        mutation_num = mutation_num,
        mutation_temperature = mutation_temperature,
        crossover_func = crossover_func,
        crossover_interval = crossover_interval,
        crossover_num = crossover_num,
        crossover_temperature = crossover_temperature,
        similarity_threshold = similarity_threshold,
        similarity_distance_func = similarity_distance_func,
        default_similarity_distance_func = default_similarity_distance_func,
        similarity_sys_info_thresholds = similarity_sys_info_thresholds,
        similarity_sys_info_prompts = similarity_sys_info_prompts,
        initialization_skip_evaluation = initialization_skip_evaluation,
        initialization_cleanse_threshold = initialization_cleanse_threshold,
        delete_when_initial_cleanse = delete_when_initial_cleanse,
        idea_uid_length = idea_uid_length,
        console_lock = console_lock,
    )

    evaluators = [
        Evaluator(
            evaluator_id = i + 1,
            database = database,
            evaluate_func = evaluate_func,
            hand_over_threshold = hand_over_threshold,
            console_lock = console_lock,
            diary_path = diary_path,
        )
        for i in range(evaluators_num)
    ]

    samplers = [
        Sampler(
            sampler_id = i + 1,
            system_prompt = system_prompt,
            prologue_section = prologue_section,
            epilogue_section = epilogue_section,
            database = database,
            evaluators = evaluators,
            generate_num = generate_num,
            console_lock = console_lock,
            diary_path = diary_path,
            record_prompt_in_diary = record_prompt_in_diary,
            filter_func = filter_func,
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
    
    with console_lock:
        append_to_file(
            file_path = diary_path,
            content_str = f"【系统】 已达到最大互动次数， {program_name} 的 IdeaSearch 结束！",
        )
            
       
def IdeaSearch_entrance_check(
    program_name: str,
    prologue_section: str,
    epilogue_section: str,
    database_path: str,
    models: list[str],
    model_temperatures: list[float],
    max_interaction_num: int,
    evaluate_func: Callable[[str], Tuple[float, Optional[str]]],
    score_range: Tuple[float, float],
    hand_over_threshold: float,
    system_prompt: Optional[str],
    diary_path: Optional[str],
    api_keys_path: Optional[str],
    local_models_path: Optional[str],
    samplers_num: int,
    evaluators_num: int,
    examples_num: int,
    generate_num: int,
    sample_temperature: float,
    model_sample_temperature: float,
    assess_func: Optional[Callable[[list[str], list[float], list[Optional[str]]], float]],
    assess_interval: Optional[int],
    assess_baseline: Optional[float],
    assess_result_data_path: Optional[str],
    assess_result_pic_path: Optional[str],
    model_assess_window_size: int,
    model_assess_initial_score: float,
    model_assess_average_order: float,
    model_assess_save_result: bool,
    model_assess_result_data_path: Optional[str],
    model_assess_result_pic_path: Optional[str],
    mutation_func: Optional[Callable[[str], str]],
    mutation_interval: Optional[int],
    mutation_num: Optional[int],
    mutation_temperature: Optional[float],
    crossover_func: Optional[Callable[[str, str], str]],
    crossover_interval: Optional[int],
    crossover_num: Optional[int],
    crossover_temperature: Optional[float],
    similarity_threshold: float,
    similarity_distance_func: Optional[Callable[[str, str], float]],
    similarity_sys_info_thresholds: Optional[list[int]],
    similarity_sys_info_prompts: Optional[list[str]],
    initialization_skip_evaluation: bool,
    initialization_cleanse_threshold: float,
    delete_when_initial_cleanse: bool,
    idea_uid_length: int,
    record_prompt_in_diary: bool,
    filter_func: Optional[Callable[[str], str]] = None,
) -> None:

    for name, val in {
        "program_name": program_name,
        "prologue_section": prologue_section,
        "epilogue_section": epilogue_section,
        "database_path": database_path,
    }.items():
        if not isinstance(val, str):
            raise TypeError(f"【IdeaSearch参数类型错误】 `{name}` 应为 str，但接收到 {type(val).__name__}")

    for optional_str in [
        diary_path, 
        system_prompt,
        api_keys_path,
        local_models_path, 
        model_assess_result_data_path, 
        model_assess_result_pic_path, 
        assess_result_data_path,
        assess_result_pic_path,
    ]:
        if optional_str is not None and not isinstance(optional_str, str):
            raise TypeError(f"【IdeaSearch参数类型错误】 可选路径类参数应为 str 或 None，但接收到 {type(optional_str).__name__}")
        
    if api_keys_path is None and local_models_path is None:
        raise ValueError("【IdeaSearch参数值错误】 `api_keys_path` 和 `local_models_path` 至少一者应不为 None ！")

    if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
        raise TypeError("【IdeaSearch参数类型错误】 `models` 应为 str 类型的列表")
    if not isinstance(model_temperatures, list) or not all(isinstance(t, (int, float)) for t in model_temperatures):
        raise TypeError("【IdeaSearch参数类型错误】 `model_temperatures` 应为 float 类型的列表")

    int_params = {
        "max_interaction_num": max_interaction_num,
        "samplers_num": samplers_num,
        "evaluators_num": evaluators_num,
        "examples_num": examples_num,
        "generate_num": generate_num,
        "model_assess_window_size": model_assess_window_size,
        "idea_uid_length": idea_uid_length,
    }
    for name, val in int_params.items():
        if not isinstance(val, int):
            raise TypeError(f"【IdeaSearch参数类型错误】 `{name}` 应为 int 类型，但接收到 {type(val).__name__}")

    float_params = {
        "sample_temperature": sample_temperature,
        "model_sample_temperature": model_sample_temperature,
        "model_assess_initial_score": model_assess_initial_score,
        "model_assess_average_order": model_assess_average_order,
        "hand_over_threshold": hand_over_threshold,
        "similarity_threshold": similarity_threshold,
        "initialization_cleanse_threshold": initialization_cleanse_threshold,
    }
    for name, val in float_params.items():
        if not isinstance(val, (int, float)):
            raise TypeError(f"【IdeaSearch参数类型错误】 `{name}` 应为 float 类型，但接收到 {type(val).__name__}")

    if not isinstance(delete_when_initial_cleanse, bool):
        raise TypeError("【IdeaSearch参数类型错误】 `delete_when_initial_cleanse` 应为 bool 类型")
    if not isinstance(initialization_skip_evaluation, bool):
        raise TypeError("【IdeaSearch参数类型错误】 `initialization_skip_evaluation` 应为 bool 类型")
    if not isinstance(record_prompt_in_diary, bool):
        raise TypeError("【IdeaSearch参数类型错误】 `record_prompt_in_diary` 应为 bool 类型")
    if not isinstance(model_assess_save_result, bool):
        raise TypeError("【IdeaSearch参数类型错误】 `model_assess_save_result` 应为 bool 类型")

    if not callable(evaluate_func):
        raise TypeError("【IdeaSearch参数类型错误】 `evaluate_func` 应为 callable")

    if assess_func is not None:
        if not callable(assess_func):
            raise TypeError("【IdeaSearch参数类型错误】 `assess_func` 应为 callable")
        if not (isinstance(assess_interval, int) and assess_interval > 0):
            raise ValueError("【IdeaSearch参数值错误】 `assess_interval` 应为正整数")
        if assess_baseline is not None and not isinstance(assess_baseline, (int, float)):
            raise TypeError("【IdeaSearch参数类型错误】 `assess_baseline` 应为 None 或 float")
        if assess_result_data_path is not None and not isinstance(assess_result_data_path, str):
            raise TypeError("【IdeaSearch参数类型错误】 `assess_result_data_path` 应为 str 或 None")
        if assess_result_pic_path is not None and not isinstance(assess_result_pic_path, str):
            raise TypeError("【IdeaSearch参数类型错误】 `assess_result_pic_path` 应为 str 或 None")

    if mutation_func is not None:
        if not callable(mutation_func):
            raise TypeError("【IdeaSearch参数类型错误】 `mutation_func` 应为 callable")
        if not (isinstance(mutation_interval, int) and mutation_interval > 0):
            raise ValueError("【IdeaSearch参数值错误】 `mutation_interval` 应为正整数")
        if not (isinstance(mutation_num, int) and mutation_num > 0):
            raise ValueError("【IdeaSearch参数值错误】 `mutation_num` 应为正整数")
        if not isinstance(mutation_temperature, (int, float)):
            raise ValueError("【IdeaSearch参数值错误】 `mutation_temperature` 应为 float 类型")

    if crossover_func is not None:
        if not callable(crossover_func):
            raise TypeError("【IdeaSearch参数类型错误】 `crossover_func` 应为 callable")
        if not (isinstance(crossover_interval, int) and crossover_interval > 0):
            raise ValueError("【IdeaSearch参数值错误】 `crossover_interval` 应为正整数")
        if not (isinstance(crossover_num, int) and crossover_num > 0):
            raise ValueError("【IdeaSearch参数值错误】 `crossover_num` 应为正整数")
        if not isinstance(crossover_temperature, (int, float)):
            raise ValueError("【IdeaSearch参数值错误】 `crossover_temperature` 应为 float 类型")

    if similarity_distance_func is not None and not callable(similarity_distance_func):
        raise TypeError("【IdeaSearch参数类型错误】 `similarity_distance_func` 应为 callable 或 None")

    if similarity_sys_info_thresholds is not None:
        if not isinstance(similarity_sys_info_thresholds, list) or not all(isinstance(i, int) for i in similarity_sys_info_thresholds):
            raise TypeError("【IdeaSearch参数类型错误】`similarity_sys_info_thresholds` 应为 int 类型的列表")

        if similarity_sys_info_prompts is None:
            raise ValueError("【IdeaSearch参数错误】当指定 `similarity_sys_info_thresholds` 时，必须同时提供 `similarity_sys_info_prompts`")

        if not isinstance(similarity_sys_info_prompts, list) or not all(isinstance(s, str) for s in similarity_sys_info_prompts):
            raise TypeError("【IdeaSearch参数类型错误】`similarity_sys_info_prompts` 应为 str 类型的列表")

        if len(similarity_sys_info_prompts) != len(similarity_sys_info_thresholds) + 1:
            raise ValueError("【IdeaSearch参数错误】`similarity_sys_info_prompts` 的长度应比 `similarity_sys_info_thresholds` 多 1")

    if score_range is not None:
        if (not isinstance(score_range, Tuple) or len(score_range) != 2
            or not all(isinstance(x, (int, float)) for x in score_range)):
            raise TypeError("【IdeaSearch参数类型错误】 `score_range` 应为二元 float 元组")
        
    if filter_func is not None and not callable(filter_func):
        raise TypeError("【IdeaSearch参数类型错误】`filter_func` 应为 None 或 str -> str 的函数")
    
    return None