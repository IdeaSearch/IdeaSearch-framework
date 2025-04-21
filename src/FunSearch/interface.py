from src.FunSearch.database import Database
from src.FunSearch.sampler import Sampler
from src.FunSearch.evaluator import Evaluator
import concurrent.futures


def FunSearchInterface(
    program_name,
    samplers_num,
    evaluators_num,
    prologue_section,
    model,
    examples_num,
    generate_num,
    epilogue_section,
    max_interaction_num,
    evaluate_func,
):

    database = Database(
        program_name = program_name,
        max_interaction_num = max_interaction_num,
        examples_num = examples_num,
    )
    evaluators = [
        Evaluator(
            evaluator_id = i,
            database = database,
            evaluate_func = evaluate_func,
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
            database = database
        )
        for i in range(samplers_num)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=samplers_num) as executor:
        futures = [executor.submit(sampler.run) for sampler in samplers]
        concurrent.futures.wait(futures)

    print(f"已达到最大采样次数，{program_name}的FunSearch结束！")