from .utils import *
if TYPE_CHECKING: from .ideasearcher import IdeaSearcher
if TYPE_CHECKING: from .island import Island


# 国际化设置
_LOCALE_DIR = Path(__file__).parent / "locales"
_DOMAIN = "ideasearch"
gettext.bindtextdomain(_DOMAIN, _LOCALE_DIR)
gettext.textdomain(_DOMAIN)


__all__ = [
    "Evaluator",
]


class Evaluator:
    
    # ----------------------------- 评估器初始化 -----------------------------
    
    def __init__(
        self, 
        ideasearcher: "IdeaSearcher",
        evaluator_id: int,
        island: "Island",
        console_lock : Lock,
    ):
        
        self.id = evaluator_id
        self.island = island
        self.ideasearcher = ideasearcher
        self._console_lock = console_lock
        self._lock = Lock()
        self.status = "Vacant"
        
        # 获取国际化函数
        self._ = ideasearcher._
        
    # ----------------------------- 外部调用动作 ----------------------------- 

    def try_acquire(self):
        acquired = self._lock.acquire(blocking=False)
        if acquired:
            if self.status == "Vacant":
                self.status = "Busy"
                return True
            self._lock.release()
        return False

    def evaluate(
        self,
        generated_raw_responses: List[Optional[str]],
        generated_ideas: List[Optional[str]],
        model: str,
        model_temperature: float,
        example_idea_paths: Optional[List[str]],
        example_idea_scores: Optional[List[float]],
        level: Optional[int],
    )-> None:
        
        hand_over_threshold = self.ideasearcher.get_hand_over_threshold()
        evaluate_func = self.ideasearcher.get_evaluate_func()
        diary_path = self.ideasearcher.get_diary_path()
        program_name = self.ideasearcher.get_program_name()
        assert diary_path is not None
        assert evaluate_func is not None
        
        accepted_ideas = []
        score_result = []
        
        for raw_response, idea in zip(generated_raw_responses, generated_ideas):
            
            assert isinstance(raw_response, str)
            assert isinstance(idea, str)
            
            try:
                score, info = evaluate_func(idea)
                
                score = float(score)
                if info is not None: info = str(info)
                
                if not isinstance(score, float):
                    with self._console_lock:
                        append_to_file(
                            file_path = diary_path,
                            content = self._("【%d号岛屿的%d号评估器】 调用 %s 的评估函数时发生错误：返回结果中的 score 应为一浮点数，不应为一个 %s 类型的对象！") % (self.island.id, self.id, program_name, type(score)),
                        )
                    return
                
                if isnan(score):
                    with self._console_lock:
                        append_to_file(
                            file_path = diary_path,
                            content = self._("【%d号岛屿的%d号评估器】 调用 %s 的评估函数时发生错误：返回结果中的 score 不应为 NaN ！") % (self.island.id, self.id, program_name),
                        )
                    return
                
                if info is not None:
                    if not isinstance(info, str):
                        with self._console_lock:
                            append_to_file(
                                file_path = diary_path,
                                content = self._("【%d号岛屿的%d号评估器】 调用 %s 的评估函数时发生错误：返回结果中的 info 应为 None 或一字符串，不应为一个 %s 类型的对象！") % (self.island.id, self.id, program_name, type(info)),
                            )
                        return
                
            except Exception as e:
                with self._console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content = self._("【%d号岛屿的%d号评估器】 调用 %s 的评估函数时发生错误：\n%s\n评估终止！") % (self.island.id, self.id, program_name, e),
                    )  
                return
            
            score_result.append(score)
            
            if score >= hand_over_threshold:
                accepted_ideas.append((raw_response, idea, score, info))
                
        self.ideasearcher.update_model_score(score_result, model, model_temperature)  
        
        if accepted_ideas:
            with self._console_lock:
                append_to_file(
                    file_path = diary_path,
                    content = self._("【%d号岛屿的%d号评估器】 已将 %d/%d 个满足条件的 idea 递交给岛屿！") % (self.island.id, self.id, len(accepted_ideas), len(generated_ideas)),
                )
            
            if example_idea_paths is not None and example_idea_scores is not None and level is not None:
                
                example_idea_string = "，".join(
                    f"{basename(path)}({score:.2f})" 
                    for path, score in zip(example_idea_paths, example_idea_scores)
                )
                source = f"由 {model}(T={model_temperature:.2f}) 阅读 {example_idea_string} 后生成"
            
                self.island.receive_result(accepted_ideas, self.id, source, level)
            
            else:
                source = f"由 {model}(T={model_temperature:.2f}) 生成；由于使用自定义的generate prompt函数，框架无法自动推演level，认为是0"
                self.island.receive_result(accepted_ideas, self.id, source, 0)
        
        else:
            
            with self._console_lock:
                
                append_to_file(
                    file_path = diary_path,
                    content = self._("【%d号岛屿的%d号评估器】 评估结束，此轮采样没有生成可递交给岛屿的满足条件的 idea ！") % (self.island.id, self.id),
                )
                
            self.island.receive_result(accepted_ideas, self.id, "", 0)
                    
    
    def release(self):
        
        diary_path = self.ideasearcher.get_diary_path()
        assert diary_path is not None
        
        with self._console_lock:
            if self.status != "Busy":
                append_to_file(
                    file_path = diary_path,
                    content = self._("【%d号岛屿的%d号评估器】 发生异常，状态应为Busy，实为%s！") % (self.island.id, self.id, self.status),
                )
                exit()

        self.status = "Vacant"
        self._lock.release()
