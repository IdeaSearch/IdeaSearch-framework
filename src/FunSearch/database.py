from threading import Lock
from pathlib import Path
import random
import os
import string
import json
from numpy import exp
import numpy as np

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
        console_lock,
    ):
        
        self.program_name = program_name
        self.console_lock = console_lock
        self.path = f"programs/{program_name}/database/"
        
        self.ideas = []
        for path in Path(self.path).rglob('*.idea'):
            
            idea = Idea(
                path = path, 
                evaluate_func = evaluate_func,
            )
            
            if idea.score == 0:
                path.unlink()
                with self.console_lock:
                    print(f"【数据库】 初始文件{path}得分为零，已删除。")
            else:
                self.ideas.append(idea)
                with self.console_lock:
                    print(f"【数据库】 初始文件{path}已评分并加入数据库。")
                    
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
                    print("【数据库】 发生异常：ideas列表为空！")
                exit()
            
            # 使用 score 的 softmax 作为采样权重，数值稳定写法
            scores = np.array([idea.score for idea in self.ideas])
            shifted_scores = scores - np.max(scores)
            exp_scores = np.exp(shifted_scores)
            weights = exp_scores / np.sum(exp_scores)

            return random.choices(
                self.ideas,
                weights=weights,
                k=min(len(self.ideas), self.examples_num)
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
            print(f"【数据库】  {self.program_name}的score sheet已更新！")
            
    def sync_score_sheet(self):
        with self.lock:
            self._sync_score_sheet()
            
    def _check_threshold(self):
        if self.interaction_count >= self.max_interaction_num:
            with self.console_lock:
                print("【数据库】 采样次数已分发完毕，FunSearch将在各采样器完成手头任务后结束。")
            self.status = "Terminated"

    def save_data(self, key, value):
        with self.lock:
            self.data_store[key] = value
            with self.console_lock:
                print(f"[DB] Saved data: {key} = {value}")

    def receive_result(self, result : list[tuple[Idea, float, str]], evaluator_id : int):
        
        def generate_random_string(length=4):
            return ''.join(random.choices(string.ascii_lowercase, k=length))
        
        with self.lock:
            
            for idea_content, score, info in result:
                
                uid = generate_random_string()
                path = "programs\\" + f"{self.program_name}" + "\\database\\" + f"idea_{uid}.idea"
                while path in [idea.path for idea in self.ideas]:
                    uid = generate_random_string()
                    path = "programs\\" + f"{self.program_name}" + "\\database\\" + f"idea_{uid}.idea"
                
                with open(path, 'w', encoding='utf-8') as file:
                    file.write(idea_content)
                    
                self.ideas.append(Idea(
                    path = path,
                    evaluate_func = None,
                    content = idea_content,
                    score = score,
                    info = info,
                ))
                
            print(f"【数据库】 {evaluator_id}号评估器递交的{len(result)}个新文件已评分并加入数据库。")
            self._sync_score_sheet()
                
                

    def get_status(self):
        with self.lock:
            return self.status
