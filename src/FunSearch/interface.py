from src.FunSearch.database import Database
from src.FunSearch.sampler import Sampler
from src.FunSearch.evaluator import Evaluator
import concurrent.futures


def FunSearchInterface(
    program_name,
    num_samplers,
    num_evaluators,
    max_interaction_num,
):

    database = Database(
        program_name = program_name,
        max_interaction_num = max_interaction_num,
    )
    evaluators = [
        Evaluator(
            evaluator_id = i,
            database = database,
        ) 
        for i in range(num_evaluators)
    ]
    samplers = [
        Sampler(
            sampler_id = i, 
            evaluators = evaluators,
            database = database
        ) 
        for i in range(num_samplers)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_samplers) as executor:
        futures = [executor.submit(sampler.run) for sampler in samplers]
        concurrent.futures.wait(futures)

    print("System terminated due to threshold reached.")