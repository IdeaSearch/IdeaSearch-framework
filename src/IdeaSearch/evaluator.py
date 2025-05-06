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
        hand_over_threshold: float,
    ):
        
        self.id = evaluator_id
        self.database = database
        self.program_name = database.program_name
        self.evaluate_func = evaluate_func
        self.console_lock = console_lock
        self.lock = Lock()
        self.status = "Vacant"
        self.diary_path = diary_path
        self.hand_over_threshold = hand_over_threshold
        

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
        score_result = []
        
        for idea in generated_ideas:
            
            try:
                score, info = self.evaluate_func(idea)
                
                if not isinstance(score, float):
                    with self.console_lock:
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = (
                                f"【{self.id}号评估器】 调用 {self.program_name} 的评估函数时发生错误："
                                f"返回结果中的 score 应为一浮点数，不应为一个 {type(score)} 类型的对象！"
                            ),
                        )
                    return
                
                if isnan(score):
                    with self.console_lock:
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = (
                                f"【{self.id}号评估器】 调用 {self.program_name} 的评估函数时发生错误："
                                f"返回结果中的 score 不应为 NaN ！"
                            ),
                        )
                    return
                
                if info is not None:
                    if not isinstance(info, str):
                        with self.console_lock:
                            append_to_file(
                                file_path = self.diary_path,
                                content_str = (
                                    f"【{self.id}号评估器】 调用 {self.program_name} 的评估函数时发生错误："
                                    f"返回结果中的 info 应为 None 或一字符串，不应为一个 {type(info)} 类型的对象！"
                                ),
                            )
                        return
                
            except Exception as e:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【{self.id}号评估器】 调用 {self.program_name} 的评估函数时发生错误：\n{e}\n评估终止！"
                        ),
                    )  
                return
            
            score_result.append(score)
            
            if score >= self.hand_over_threshold:
                accepted_ideas.append((idea, score, info))
                
        self.database.update_model_score(score_result, model, model_temperature)  
        
        if accepted_ideas:
            with self.console_lock:
                append_to_file(
                    file_path = self.diary_path,
                    content_str = f"【{self.id}号评估器】 已将 {len(accepted_ideas)}/{len(generated_ideas)} 个满足条件的 idea 递交给数据库！",
                )
                
            self.database.receive_result(accepted_ideas, self.id)
            
        else:
            with self.console_lock:
                append_to_file(
                    file_path = self.diary_path,
                    content_str = f"【{self.id}号评估器】 评估结束，此轮采样没有生成可递交给数据库的满足条件的 idea ！",
                )
                    
    
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
