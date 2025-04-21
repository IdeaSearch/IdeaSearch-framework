from threading import Lock


class Evaluator:
    
    def __init__(
        self, 
        evaluator_id,
        database,
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
        print(
            f"[Evaluator {self.id}]\n"
            f"Evaluating result:\n"
        )
        for index, llm_answer in enumerate(llm_answers):
            print(f"{index+1}：{llm_answer}")
    
    def release(self):
        assert self.status == "Busy"
        self.status = "Vacant"
        self.lock.release()
