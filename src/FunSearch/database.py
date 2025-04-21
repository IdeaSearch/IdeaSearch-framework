from threading import Lock


class Database:

    def __init__(
        self, 
        program_name, 
        max_interaction_num,
        examples_num,
    ):
        
        self.program_name = program_name
        self.path = f"programs/{program_name}/database/"
        
        self.interaction_count = 0
        self.max_interaction_num = max_interaction_num
        self.examples_num = examples_num
        
        self.lock = Lock()
        self.status = "Running"

    def get_examples(self):
        
        with self.lock:
            
            if self.status == "Terminated":
                return None
            self.interaction_count += 1
            self._check_threshold()
            examples = [11, 22, 33]
            return examples
            
            
    def _check_threshold(self):
        if self.interaction_count >= self.max_interaction_num:
            print("[DB] Threshold reached, initiating shutdown.")
            self.status = "Terminated"

    def save_data(self, key, value):
        with self.lock:
            self.data_store[key] = value
            print(f"[DB] Saved data: {key} = {value}")

    def receive_result(self, result):
        with self.lock:
            print(f"[DB] Received evaluated result: {result}")

    def get_status(self):
        with self.lock:
            return self.status
