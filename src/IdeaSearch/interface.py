from src.utils import guarantee_path_exist, clear_file_content, append_to_file
from src.IdeaSearch.database import Database
from src.IdeaSearch.sampler import Sampler
from src.IdeaSearch.evaluator import Evaluator
import concurrent.futures
from threading import Lock
from typing import Callable


def IdeaSearchInterface(
    program_name: str,
    samplers_num: int,
    sample_temperature: float,
    evaluators_num: int,
    prologue_section: str,
    models: list[str],
    model_temperatures: list[float],
    model_assess_window_size: int,
    model_assess_initial_score: float,
    model_sample_temperature: float,
    examples_num: int,
    generate_num: int,
    epilogue_section: str,
    max_interaction_num: int,
    evaluate_func: Callable[[str], tuple[float, str]],
    diary_path: str,
    initialization_cleanse_threshold: float,
    delete_when_initial_cleanse: bool,
    evaluator_handle_threshold: float,
) -> None:
    """
    启动并运行一个 IdeaSearch 搜索过程。

    该函数会创建一个线程安全的 Database 实例，并初始化指定数量的 Sampler 和 Evaluator 实例，
    使用线程池并发运行所有 Sampler，直到数据库达到最大交互次数为止。

    Args:
        program_name (str): 当前程序或实验的名称。
        samplers_num (int): 要创建的采样器（Sampler）数量。
        sample_temperature (float): 采样温度，诸idea被采样的概率正比于 exp (- score / sample_temperature)。
        evaluators_num (int): 要创建的评估器（Evaluator）数量。
        prologue_section (str): 用于提示模型采样的前导文本片段。
        model (str): 大语言模型的名字，应在API4LLMs/api_keys.json中出现。
        model_temperature (float): 大语言模型的温度。
        examples_num (int): 每个prompt给大语言模型看的例子数量。
        generate_num (int): 每个 Sampler 每轮生成的候选程序数量。
        epilogue_section (str): 用于提示模型采样的结尾文本片段。
        max_interaction_num (int): 数据库允许的最大评估交互次数，超过即终止。
        evaluate_func (Callable): 用于评估候选程序的函数，供 Evaluator 使用，返回score和info。
        diary_path (str): IdeaSearch的日志文件路径。
        initialization_cleanse_threshold (float): 数据库初始化时的清除阈值分数，低于此阈值的idea将会被清除/忽略
        delete_when_initial_cleanse (bool): 决定数据库初始化时对低于分数阈值的idea的行为：True则删除文件；False则仅仅忽视不见
        evaluator_handle_threshold (float): Evaluator将idea递交给数据库的分数阈值，低于此分数阈值的idea会被舍弃。

    Returns:
        None
    """
    
    guarantee_path_exist(diary_path)
    clear_file_content(diary_path)
    
    append_to_file(
        file_path = diary_path,
        content_str = f"现在开始{program_name}的IdeaSearch！",
    )
    
    console_lock = Lock()

    database = Database(
        program_name = program_name,
        max_interaction_num = max_interaction_num,
        examples_num = examples_num,
        evaluate_func = evaluate_func,
        sample_temperature = sample_temperature,
        console_lock = console_lock,
        diary_path = diary_path,
        initialization_cleanse_threshold = initialization_cleanse_threshold,
        delete_when_initial_cleanse = delete_when_initial_cleanse,
        models = models,
        model_temperatures = model_temperatures,
        model_assess_window_size = model_assess_window_size,
        model_assess_initial_score = model_assess_initial_score,
        model_sample_temperature = model_sample_temperature,
    )
    evaluators = [
        Evaluator(
            evaluator_id = i,
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
            sampler_id = i,
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
                    content_str = f"【{sampler.id}号采样器】 运行过程中出现错误：\n{e}\nIdeaSearch意外终止！",
                )
                exit()

    append_to_file(
        file_path = diary_path,
        content_str = f"已达到最大采样次数，{program_name}的IdeaSearch结束！",
    )