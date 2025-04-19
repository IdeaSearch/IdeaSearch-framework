import time
import random


class Sampler:
    def __init__(
        self, 
        sampler_id, 
        evaluators,
        database,
    ):
        self.id = sampler_id + 1
        self.database = database
        self.program_name = database.program_name
        self.evaluators = evaluators

    def run(self):
        
        while self.database.get_status() == "Running":
            
            key = f"sample_{self.id}_{random.randint(0,1000)}"
            data = self.database.get_data(key)
            if data is None: 
                break
            time.sleep(random.uniform(0.1, 0.3))  # 模拟处理
            result = f"{data}_processed_by_{self.id}"
            
            # 寻找空闲 evaluator
            evaluator = self._get_idle_evaluator()
            if evaluator:
                evaluated = evaluator.evaluate(result)
                evaluator.release()
                self.database.receive_result(evaluated)
            else:
                print(f"[Sampler {self.id}] No evaluator available, retrying...")
                time.sleep(0.1)


    def _get_idle_evaluator(self):
        for evaluator in self.evaluators:
            if evaluator.try_acquire():
                return evaluator
        return None
