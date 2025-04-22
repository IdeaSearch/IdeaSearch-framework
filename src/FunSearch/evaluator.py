from threading import Lock
from src.FunSearch.database import Database


class Evaluator:
    
    def __init__(
        self, 
        evaluator_id,
        database : Database,
        evaluate_func,
        console_lock : Lock,
    ):
        
        self.id = evaluator_id + 1
        self.database = database
        self.program_name = database.program_name
        self.evaluate_func = evaluate_func
        self.console_lock = console_lock
        self.lock = Lock()
        self.status = "Vacant"
        

    def try_acquire(self):
        acquired = self.lock.acquire(blocking=False)
        if acquired:
            if self.status == "Vacant":
                self.status = "Busy"
                return True
            self.lock.release()
        return False

    def evaluate(self, generated_ideas : list[str]) -> None:
        
        accepted_ideas = []
        
        for idea in generated_ideas:
            score, info = self.evaluate_func(idea)
            if score >= 60.00 or True:
                accepted_ideas.append((idea, score, info))
                
        with self.console_lock:
            print(f"【{self.id}号评估器】 已将{len(accepted_ideas)}/{len(generated_ideas)}个满足条件的idea递交给数据库！")
                
        self.database.receive_result(accepted_ideas, self.id)         
            
        with self.console_lock:
            print(f"【{self.id}号评估器】 已完成一轮评估。")
    
    def release(self):
        with self.console_lock:
            assert self.status == "Busy", f"【{self.id}号评估器】 发生异常！"
        self.status = "Vacant"
        self.lock.release()
