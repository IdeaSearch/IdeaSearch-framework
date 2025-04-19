from threading import Lock


class Evaluator:
    
    def __init__(
        self, 
        evaluator_id,
        database,
    ):
        
        self.id = evaluator_id + 1
        self.database = database
        self.program_name = database.program_name
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

    def evaluate(self, result):
        print(f"[Evaluator {self.id}] Evaluating result: {result}")
        # 模拟处理
        evaluated = f"{result}_evaluated_by_{self.id}"
        return evaluated
    
    def release(self):
        assert self.status == "Busy"
        self.status = "Vacant"
        self.lock.release()
