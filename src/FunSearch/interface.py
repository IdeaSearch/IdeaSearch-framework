from src.FunSearch.database import Database
from src.FunSearch.sampler import Sampler
from src.FunSearch.evaluator import Evaluator
import concurrent.futures
from threading import Lock
from typing import Callable


def FunSearchInterface(
    program_name: str,
    samplers_num: int,
    sample_temperature: float,
    evaluators_num: int,
    prologue_section: str,
    model: str,
    model_temperature: float,
    examples_num: int,
    generate_num: int,
    epilogue_section: str,
    max_interaction_num: int,
    evaluate_func: Callable[[str], tuple[float, str]],
) -> None:
    """
    启动并运行一个 FunSearch 搜索过程。

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

    Returns:
        None
    """
    
    print(f"现在开始{program_name}的FunSearch！")
    
    console_lock = Lock()

    database = Database(
        program_name = program_name,
        max_interaction_num = max_interaction_num,
        examples_num = examples_num,
        evaluate_func = evaluate_func,
        sample_temperature = sample_temperature,
        console_lock = console_lock,
    )
    evaluators = [
        Evaluator(
            evaluator_id = i,
            database = database,
            evaluate_func = evaluate_func,
            console_lock = console_lock,
        ) 
        for i in range(evaluators_num)
    ]
    samplers = [
        Sampler(
            sampler_id = i,
            model = model,
            prologue_section = prologue_section,
            epilogue_section = epilogue_section,
            evaluators = evaluators,
            generate_num = generate_num,
            database = database,
            model_temperature = model_temperature,
            console_lock = console_lock,
        )
        for i in range(samplers_num)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=samplers_num) as executor:
        futures = [executor.submit(sampler.run) for sampler in samplers]
        concurrent.futures.wait(futures)

    print(f"已达到最大采样次数，{program_name}的FunSearch结束！")