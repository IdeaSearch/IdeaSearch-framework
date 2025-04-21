import time
import random
from threading import Lock
from src.API4LLMs.get_answer import get_answer
from src.FunSearch.evaluator import Evaluator
from src.FunSearch.database import Database
from concurrent.futures import ThreadPoolExecutor, as_completed, wait


class Sampler:
    def __init__(
        self, 
        sampler_id, 
        model,
        prologue_section,
        epilogue_section,
        evaluators : Evaluator,
        generate_num,
        database : Database,
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
        
        print(f"【{self.id}号采样器】 已开始工作！")
        
        while self.database.get_status() == "Running":
            
            examples = self.database.get_examples()
            if examples is None: 
                print(f"【{self.id}号采样器】 工作结束。")
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
                        print(f"【{self.id}号采样器】 尝试获取{self.model}的回答时发生错误: {e}")
            
            # 寻找空闲 evaluator
            evaluator = self._get_idle_evaluator()
            if evaluator:
                evaluator.evaluate(llm_answers)
                print(f"【{self.id}号采样器】 已释放{evaluator.id}号评估器。")
                evaluator.release()
            else:
                print(f"【{self.id}号采样器】 没有找到空闲的评估器，此次采样失败...")
                
        print(f"【{self.id}号采样器】 工作结束。")


    def _get_idle_evaluator(self) -> Evaluator:
        for evaluator in self.evaluators:
            if evaluator.try_acquire():
                print(f"【{self.id}号采样器】 已找到{evaluator.id}号评估器对{self.model}的回答进行评估。")
                return evaluator
        return None
