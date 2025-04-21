from threading import Lock
from src.FunSearch.database import Database


class Evaluator:
    
    def __init__(
        self, 
        evaluator_id,
        database : Database,
        evaluate_func,
    ):
        
        self.id = evaluator_id + 1
        self.database = database
        self.program_name = database.program_name
        self.evaluate_func = evaluate_func
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

    def evaluate(self, llm_answers):
        print(f"【{self.id}号评估器】 已完成一轮评估。")
    
    def release(self):
        assert self.status == "Busy"
        self.status = "Vacant"
        self.lock.release()
