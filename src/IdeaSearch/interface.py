import concurrent.futures
from threading import Lock
from typing import Callable, Optional
from pathlib import Path

from src.utils import guarantee_path_exist, clear_file_content, append_to_file
from src.IdeaSearch.database import Idea, Database
from src.IdeaSearch.sampler import Sampler
from src.IdeaSearch.evaluator import Evaluator
from src.API4LLMs.model_manager import init_model_manager


__all__ = [
    "IdeaSearch",
    "cleanse_dataset",
]


def IdeaSearch(
    program_name: str,
    prologue_section: str,
    epilogue_section: str,
    database_path: str,
    diary_path: str,
    api_keys_path: str,
    models: list[str],
    model_temperatures: list[float],
    max_interaction_num: int,
    evaluate_func: Callable[[str], tuple[float, str]],
    *, # 必填参数和选填参数的分界线
    samplers_num: int = 5,
    evaluators_num: int = 5,
    sample_temperature: float = 50.0,
    examples_num: int = 3,
    generate_num: int = 5,
    model_assess_window_size: int = 20,
    model_assess_initial_score: float = 100.0,
    model_assess_average_order: float = 1.0,
    model_sample_temperature: float = 50.0,
    evaluator_handle_threshold: float = 0.0,
    similarity_threshold: float = -1.0,
    similarity_distance_func: Optional[Callable[[str, str], float]] = None,
    initialization_cleanse_threshold: float = -1.0,
    delete_when_initial_cleanse: bool = False,
    idea_uid_length: int = 4,
) -> None:
    
    """
    启动并运行一个 IdeaSearch 搜索过程。

    该函数会创建一个线程安全的 Database 实例，并初始化指定数量的 Sampler 和 Evaluator 实例，
    使用线程池并发运行所有 Sampler，直到数据库达到最大交互次数为止。
    
    为使用该函数的基本功能，用户需自行创建 prologue section 、 epilogue section 和 evaluate_func 并传入，其中
    evaluate_func是一个接收字符串、返回tuple[float, str | None]类型的分数、评语元组（评语可选）的函数。
    我们建议分数在0.0至100.0间变化。

    Args:
        program_name (str): 当前程序或实验的名称。
        prologue_section (str): 用于提示模型采样的前导文本片段。
        epilogue_section (str): 用于提示模型采样的结尾文本片段。
        database_path (str): 数据库路径，在此路径下存放.idea文件和自动生成的score_sheet.json。
        diary_path (str): IdeaSearch 的日志文件路径。
        api_keys_path (str): 用于向LLM提问的json文件路径，文件中应包含自己的api keys。
        models (list[str]): 用于生成 idea 的大语言模型名称列表，应在 api_keys_path 下的文件中定义。
        model_temperatures (list[float]): 与 models 对应的温度值列表，其长度应与 models 相同。
        max_interaction_num (int): 数据库允许的最大评估交互次数，超过即终止。
        evaluate_func (Callable[[str], tuple[float, str]]): 用于评估候选程序的函数，供 Evaluator 使用，返回 score 和 info。

        samplers_num (int): 要创建的采样器（Sampler）数量。
        evaluators_num (int): 要创建的评估器（Evaluator）数量。
        sample_temperature (float): 系统采样idea时的温度，诸 idea 被采样的概率正比于 exp(-score / sample_temperature) / N(idea)，其中 N(idea) 是 database 所有 ideas 中与 idea 相似的元素个数。
        examples_num (int): 每个 prompt 给大语言模型看的例子数量。
        generate_num (int): 每个 Sampler 每轮生成的候选 idea 数量。
        model_assess_window_size (int): 模型评估窗口大小，每个模型的实时得分是该模型在前 model_assess_window_size 轮生成的idea得分的平均值。
        model_assess_initial_score (float): 模型评估初始分数，用于冷启动时的模型分数基准，建议设为100.0甚至更大，以鼓励系统采样模型时的探索。
        model_assess_average_order (float): 模型评估p范数平均的阶数p，默认为1.0，p越大越强调最大值的影响，p越小越强调最小值的影响。
        model_sample_temperature (float): 系统采样模型时的温度，诸模型被采样的概率正比于 exp(-score / model_sample_temperature)。
        evaluator_handle_threshold (float): Evaluator 将 idea 递交给数据库的分数阈值，低于此阈值的 idea 会被舍弃。
        similarity_threshold (float): 用于判断两个 idea 是否相似的阈值， idea1 与 idea2 相似当且仅当 idea1 就是 idea2 或  similarity_distance_func(idea1, idea2) < similarity_threshold ；默认为 -1.0 ，相当于只认为相同的 idea 相似。
        similarity_distance_func (Callable[[str, str], float] | None): 衡量两个 idea 相似度的函数，默认为 None （会被赋值为返回 |score(idea1) - score(idea2)| 的函数），但也可以自行编写并传入。
        initialization_cleanse_threshold (float): 数据库初始化时的清除阈值分数，低于此阈值的 idea 将会被清除/忽略。
        delete_when_initial_cleanse (bool): 决定数据库初始化时对低于分数阈值的 idea 的行为：True 则删除文件；False 则仅仅忽视不见。
        idea_uid_length (int):  LLM 生成的 idea 会被存储至 database_path 下的 f"idea_{idea_uid}.idea" 文件中， idea_uid_length 决定 idea_uid 的长度，一般取为 4 。

    Returns:
        None
    """
    
    if not isinstance(program_name, str):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `program_name` 应该是 str 类型，"
            f"但接收到 {type(program_name).__name__}"
        )

    if not isinstance(prologue_section, str):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `prologue_section` 应该是 str 类型，"
            f"但接收到 {type(prologue_section).__name__}"
        )

    if not isinstance(epilogue_section, str):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `epilogue_section` 应该是 str 类型，"
            f"但接收到 {type(epilogue_section).__name__}"
        )

    if not isinstance(database_path, str):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `database_path` 应该是 str 类型，"
            f"但接收到 {type(database_path).__name__}"
        )

    if not isinstance(diary_path, str):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `diary_path` 应该是 str 类型，"
            f"但接收到 {type(diary_path).__name__}"
        )

    if not isinstance(api_keys_path, str):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `api_keys_path` 应该是 str 类型，"
            f"但接收到 {type(api_keys_path).__name__}"
        )

    if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `models` 应该是 str 类型的列表，"
            f"但接收到 {type(models).__name__}"
        )

    if not isinstance(model_temperatures, list) or not all(isinstance(t, (int, float)) for t in model_temperatures):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `model_temperatures` 应该是 float 类型的列表，"
            f"但接收到 {type(model_temperatures).__name__}"
        )

    if not isinstance(max_interaction_num, int):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `max_interaction_num` 应该是 int 类型，"
            f"但接收到 {type(max_interaction_num).__name__}"
        )

    if not callable(evaluate_func):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `evaluate_func` 应该是 callable 函数，"
            f"但接收到 {type(evaluate_func).__name__}"
        )

    if not isinstance(samplers_num, int):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `samplers_num` 应该是 int 类型，"
            f"但接收到 {type(samplers_num).__name__}"
        )

    if not isinstance(evaluators_num, int):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `evaluators_num` 应该是 int 类型，"
            f"但接收到 {type(evaluators_num).__name__}"
        )

    if not isinstance(sample_temperature, (int, float)):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `sample_temperature` 应该是 float 类型，"
            f"但接收到 {type(sample_temperature).__name__}"
        )

    if not isinstance(examples_num, int):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `examples_num` 应该是 int 类型，"
            f"但接收到 {type(examples_num).__name__}"
        )

    if not isinstance(generate_num, int):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `generate_num` 应该是 int 类型，"
            f"但接收到 {type(generate_num).__name__}"
        )

    if not isinstance(model_assess_window_size, int):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `model_assess_window_size` 应该是 int 类型，"
            f"但接收到 {type(model_assess_window_size).__name__}"
        )

    if not isinstance(model_assess_initial_score, (int, float)):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `model_assess_initial_score` 应该是 float 类型，"
            f"但接收到 {type(model_assess_initial_score).__name__}"
        )
        
    if not isinstance(model_assess_average_order, (int, float)):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `model_assess_average_order` 应该是 float 类型，"
            f"但接收到 {type(model_assess_average_order).__name__}"
        )

    if not isinstance(model_sample_temperature, (int, float)):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `model_sample_temperature` 应该是 float 类型，"
            f"但接收到 {type(model_sample_temperature).__name__}"
        )

    if not isinstance(evaluator_handle_threshold, (int, float)):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `evaluator_handle_threshold` 应该是 float 类型，"
            f"但接收到 {type(evaluator_handle_threshold).__name__}"
        )

    if not isinstance(similarity_threshold, (int, float)):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `similarity_threshold` 应该是 float 类型，"
            f"但接收到 {type(similarity_threshold).__name__}"
        )

    if similarity_distance_func is not None and not callable(similarity_distance_func):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `similarity_distance_func` 应该是 Callable[[str, str], float] 或 None，"
            f"但接收到 {type(similarity_distance_func).__name__}"
        )

    if not isinstance(initialization_cleanse_threshold, (int, float)):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `initialization_cleanse_threshold` 应该是 float 类型，"
            f"但接收到 {type(initialization_cleanse_threshold).__name__}"
        )

    if not isinstance(delete_when_initial_cleanse, bool):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `delete_when_initial_cleanse` 应该是 bool 类型，"
            f"但接收到 {type(delete_when_initial_cleanse).__name__}"
        )
        
    if not isinstance(idea_uid_length, int):
        raise TypeError(
            "【IdeaSearch参数类型错误】 `idea_uid_length` 应该是 int 类型，"
            f"但接收到 {type(idea_uid_length).__name__}"
        )
    
    def default_similarity_distance_func(idea1, idea2):
        return abs(evaluate_func(idea1)[0] - evaluate_func(idea2)[0])
    
    if similarity_distance_func is None:
        similarity_distance_func = default_similarity_distance_func
        
    init_model_manager(
        api_keys_path = api_keys_path,
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
        similarity_distance_func = similarity_distance_func,
        default_similarity_distance_func = default_similarity_distance_func,
        sample_temperature = sample_temperature,
        console_lock = console_lock,
        diary_path = diary_path,
        database_path = database_path,
        initialization_cleanse_threshold = initialization_cleanse_threshold,
        delete_when_initial_cleanse = delete_when_initial_cleanse,
        models = models,
        model_temperatures = model_temperatures,
        model_assess_window_size = model_assess_window_size,
        model_assess_initial_score = model_assess_initial_score,
        model_assess_average_order = model_assess_average_order,
        model_sample_temperature = model_sample_temperature,
        similarity_threshold = similarity_threshold,
        idea_uid_length = idea_uid_length,
    )
    evaluators = [
        Evaluator(
            evaluator_id = i + 1,
            database = database,
            evaluate_func = evaluate_func,
            console_lock = console_lock,
            diary_path = diary_path,
            evaluator_handle_threshold = evaluator_handle_threshold,
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

    append_to_file(
        file_path = diary_path,
        content_str = f"【系统】 已达到最大互动次数，{program_name}的IdeaSearch结束！",
    )
    

def cleanse_dataset(
    database_path: str,
    evaluate_func: Callable[[str], tuple[float, str]],
    cleanse_threshold: float,
):
    
    for path in Path(database_path).rglob('*.idea'):
            
        idea = Idea(
            path = path, 
            evaluate_func = evaluate_func,
        )
        
        if idea.score < cleanse_threshold:
            path.unlink()
            print(f"文件{path}得分未达到{cleanse_threshold:.2f}，已删除。")