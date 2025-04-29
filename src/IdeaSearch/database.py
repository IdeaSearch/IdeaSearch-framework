from threading import Lock
from pathlib import Path
import random
import os
import string
import json
from numpy import exp
import numpy as np
from src.utils import append_to_file

class Idea:
    
    def __init__(self, path, evaluate_func, content = None, score = None, info = None):
        self.path = str(path)
        if evaluate_func is not None:
            with open(path, 'r', encoding = "UTF-8") as file:
                self.content = file.read()
            self.score, self.info = evaluate_func(self.content) 
        else:
            self.content = content
            self.score = score
            self.info = info     


class Database:

    def __init__(
        self, 
        program_name, 
        max_interaction_num,
        examples_num,
        evaluate_func,
        sample_temperature : float,
        console_lock,
        diary_path: str,
        initialization_cleanse_threshold: float,
        delete_when_initial_cleanse: bool,
        models: list[str],
        model_temperatures: list[float],
        model_assess_window_size: int,
        model_assess_initial_score: float,
        model_sample_temperature: float,
    ):
        
        self.program_name = program_name
        self.sample_temperature = sample_temperature
        self.console_lock = console_lock
        self.path = f"programs/{program_name}/database/"
        self.diary_path = diary_path
        self.models = models
        self.model_temperatures = model_temperatures
        self.model_sample_temperature = model_sample_temperature
        self.model_recent_scores = []
        for _ in range(len(models)):
            self.model_recent_scores.append(
                np.full((model_assess_window_size,), model_assess_initial_score)
            )
        self.model_scores = [
            np.mean(self.model_recent_scores[i]) 
            for i in range(len(self.model_recent_scores))
        ]
        
        self.ideas = []
        for path in Path(self.path).rglob('*.idea'):
            
            idea = Idea(
                path = path, 
                evaluate_func = evaluate_func,
            )
            
            if idea.score < initialization_cleanse_threshold:
                if delete_when_initial_cleanse:
                    path.unlink()
                    with self.console_lock:
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = f"【数据库】 初始文件{path}得分未达到{initialization_cleanse_threshold}，已删除。",
                        )
                else:
                    with self.console_lock:
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = f"【数据库】 初始文件{path}得分未达到{initialization_cleanse_threshold}，已忽略。",
                        )
            else:
                self.ideas.append(idea)
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = f"【数据库】 初始文件{path}已评分并加入数据库。",
                    )
                    
        self._sync_score_sheet()
        
        self.interaction_count = 0
        self.max_interaction_num = max_interaction_num
        self.examples_num = examples_num
        
        self.lock = Lock()
        self.status = "Running"
        
        

    def get_examples(self) -> list[Idea]:
        
        with self.lock:
            
            if self.status == "Terminated":
                return None
            self.interaction_count += 1
            self._check_threshold()
            
            if len(self.ideas) == 0:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = "【数据库】 发生异常：ideas列表为空！",
                    )
                exit()
            
            # 使用 score / sample_temperature 的 softmax 作为采样权重（数值稳定写法）
            scores = np.array([idea.score for idea in self.ideas])
            scores = scores / self.sample_temperature
            shifted_scores = scores - np.max(scores)
            exp_scores = np.exp(shifted_scores)
            weights = exp_scores / np.sum(exp_scores)

            return random.choices(
                self.ideas,
                weights=weights,
                k=min(len(self.ideas), self.examples_num)
            )
            
    def get_model(self) -> tuple[str, float]:
        
        with self.lock:
            
            self._show_model_scores()
            
            probabilities = np.array(self.model_scores) / self.model_sample_temperature
            max_value = np.max(probabilities)
            probabilities = np.exp(probabilities - max_value)
            probabilities /= np.sum(probabilities)
            
            np.random.seed()
            selected_index = np.random.choice(len(self.models), p=probabilities)
            
            selected_model_name = self.models[selected_index]
            selected_model_temperature = self.model_temperatures[selected_index]
            
            return selected_model_name, selected_model_temperature
        
    def _show_model_scores(self)-> None:
        
        with self.console_lock:
            
            append_to_file(
                file_path = self.diary_path,
                content_str = "【数据库】 各模型目前得分情况如下：",
            )
            
            for index, model in enumerate(self.models):
                
                model_temperature = self.model_temperatures[index]
                
                append_to_file(
                    file_path = self.diary_path,
                    content_str = (
                        f"  {index+1}. {model}(T={model_temperature:.2f}): {self.model_scores[index]}"
                    ),
                )
                
            
            
    def _sync_score_sheet(self):
        
        score_sheet = {
            idea.path: {
                "score": idea.score,
                "info": idea.info if idea.info is not None else "",
            }
            for idea in self.ideas
        }

        with open(self.path + "score_sheet.json", 'w', encoding='utf-8') as json_file:
            json.dump(score_sheet, json_file, ensure_ascii=False, indent=4)
        
        with self.console_lock:   
            append_to_file(
                file_path = self.diary_path,
                content_str = f"【数据库】  {self.program_name}的score sheet已更新！",
            )
            
    def sync_score_sheet(self):
        with self.lock:
            self._sync_score_sheet()
            
    def _check_threshold(self):
        if self.interaction_count >= self.max_interaction_num:
            with self.console_lock:
                append_to_file(
                    file_path = self.diary_path,
                    content_str = "【数据库】 采样次数已分发完毕，IdeaSearch将在各采样器完成手头任务后结束。",
                )
            self.status = "Terminated"

    def receive_result(
        self, 
        result: list[tuple[Idea, float, str]], 
        evaluator_id: int,
        model: str,
        model_temperature: float,
    ):
        
        def generate_random_string(length=4):
            return ''.join(random.choices(string.ascii_lowercase, k=length))
        
        with self.lock:
            
            for idea_content, score, info in result:
                
                uid = generate_random_string()
                path = os.path.join("programs", self.program_name, "database", f"idea_{uid}.idea")
                while path in [idea.path for idea in self.ideas]:
                    uid = generate_random_string()
                    path = os.path.join("programs", self.program_name, "database", f"idea_{uid}.idea")
                
                with open(path, 'w', encoding='utf-8') as file:
                    file.write(idea_content)
                    
                self.ideas.append(Idea(
                    path=path,
                    evaluate_func=None,
                    content=idea_content,
                    score=score,
                    info=info,
                ))
                
            with self.console_lock:    
                append_to_file(
                    file_path = self.diary_path,
                    content_str = f"【数据库】 {evaluator_id}号评估器递交的{len(result)}个新文件已评分并加入数据库。",
                )
            
            self._sync_score_sheet()
            
            index = 0
            
            while index < len(self.models):
                
                if self.models[index] == model and self.model_temperatures[index] == model_temperature:
                    self.model_recent_scores[index][:-1] = self.model_recent_scores[index][1:]
                    self.model_recent_scores[index][-1] = np.mean(np.array([
                        result_tuple[1] for result_tuple in result
                    ]))
                    self.model_scores[index] = np.mean(self.model_recent_scores[index])
                    with self.console_lock:    
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = f"【数据库】 模型{model}(T={model_temperature:.2f})的分数已被更新为{self.model_scores[index]}！",
                        )
                    return
                
                index += 1
                
            with self.console_lock:    
                append_to_file(
                    file_path = self.diary_path,
                    content_str = f"【数据库】 出现错误！未知的模型名称及温度：{model}(T={model_temperature:.2f})！",
                )
                
            exit()  
                

    def get_status(self):
        with self.lock:
            return self.status
