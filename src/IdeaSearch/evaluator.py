from threading import Lock
from math import isnan
from src.IdeaSearch.database import Database
from src.utils import append_to_file


class Evaluator:
    
    def __init__(
        self, 
        evaluator_id,
        database : Database,
        evaluate_func,
        console_lock : Lock,
        diary_path: str,
        evaluator_handle_threshold: float,
    ):
        
        self.id = evaluator_id
        self.database = database
        self.program_name = database.program_name
        self.evaluate_func = evaluate_func
        self.console_lock = console_lock
        self.lock = Lock()
        self.status = "Vacant"
        self.diary_path = diary_path
        self.evaluator_handle_threshold = evaluator_handle_threshold
        

    def try_acquire(self):
        acquired = self.lock.acquire(blocking=False)
        if acquired:
            if self.status == "Vacant":
                self.status = "Busy"
                return True
            self.lock.release()
        return False

    def evaluate(
        self, 
        generated_ideas: list[str],
        model: str,
        model_temperature: float,
    )-> None:
        
        accepted_ideas = []
        
        for idea in generated_ideas:
            
            try:
                score, info = self.evaluate_func(idea)
                
                if not isinstance(score, float):
                    with self.console_lock:
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = (
                                f"【{self.id}号评估器】 调用{self.program_name}的评估函数时发生错误："
                                f"返回结果中的score应为一浮点数，不应为一个{type(score)}类型的对象！"
                            ),
                        )
                    return
                
                if isnan(score):
                    with self.console_lock:
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = (
                                f"【{self.id}号评估器】 调用{self.program_name}的评估函数时发生错误："
                                f"返回结果中的score不应为NaN！"
                            ),
                        )
                    return
                
                if info is not None:
                    if not isinstance(info, str):
                        with self.console_lock:
                            append_to_file(
                                file_path = self.diary_path,
                                content_str = (
                                    f"【{self.id}号评估器】 调用{self.program_name}的评估函数时发生错误："
                                    f"返回结果中的info应为None或一字符串，不应为一个{type(info)}类型的对象！"
                                ),
                            )
                        return
                
            except Exception as e:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【{self.id}号评估器】 调用{self.program_name}的评估函数时发生错误：\n{e}\n评估终止！"
                        ),
                    )  
                return
            
            if score >= self.evaluator_handle_threshold:
                accepted_ideas.append((idea, score, info))
                
        with self.console_lock:
            append_to_file(
                file_path = self.diary_path,
                content_str = f"【{self.id}号评估器】 已将{len(accepted_ideas)}/{len(generated_ideas)}个满足条件的idea递交给数据库！",
            )
                
        self.database.receive_result(accepted_ideas, self.id, model, model_temperature)         
    
    def release(self):
        
        with self.console_lock:
            if self.status != "Busy":
                append_to_file(
                    file_path = self.diary_path,
                    content_str = f"【{self.id}号评估器】 发生异常，状态应为Busy，实为{self.status}！",
                )
                exit()

        self.status = "Vacant"
        self.lock.release()
