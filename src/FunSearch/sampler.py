import time
import random
from threading import Lock
from src.API4LLMs.get_answer import get_answer
from concurrent.futures import ThreadPoolExecutor, as_completed, wait


class Sampler:
    def __init__(
        self, 
        sampler_id, 
        model,
        prologue_section,
        epilogue_section,
        evaluators,
        generate_num,
        database,
    ):
        self.id = sampler_id + 1
        self.database = database
        self.program_name = database.program_name
        self.model = model
        self.prologue_section = prologue_section
        self.epilogue_section = epilogue_section
        self.generate_num = generate_num
        self.evaluators = evaluators

    def run(self):
        
        while self.database.get_status() == "Running":
            
            examples = self.database.get_examples()
            if examples is None: 
                break
            
            # TODO:
            examples_section = ""
            
            prompt = self.prologue_section + examples_section + self.epilogue_section
                
            llm_answers = [None] * self.generate_num
            with ThreadPoolExecutor() as executor:

                future_to_index = {
                    executor.submit(get_answer, self.model, prompt): i
                    for i in range(self.generate_num)
                }

                for future in as_completed(future_to_index):
                    i = future_to_index[future]
                    try:
                        llm_answers[i] = future.result()
                    except Exception as e:
                        print(f"尝试获取{self.model}的回答时发生错误: {e}")
            
            # 寻找空闲 evaluator
            evaluator = self._get_idle_evaluator()
            if evaluator:
                evaluator.evaluate(llm_answers)
                evaluator.release()
            else:
                print(f"{self.id}号采样器没有找到空闲的评估器，此次采样失败...")


    def _get_idle_evaluator(self):
        for evaluator in self.evaluators:
            if evaluator.try_acquire():
                return evaluator
        return None
