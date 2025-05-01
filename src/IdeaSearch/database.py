from threading import Lock
from pathlib import Path
import random
import os
import string
import json
from typing import Callable
import numpy as np
from src.utils import append_to_file, guarantee_path_exist

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
        assess_func,
        assess_interval,
        assess_result_path,
        mutation_func,
        mutation_interval,
        crossover_func,
        crossover_interval,
        similarity_distance_func: Callable[[str, str], float],
        default_similarity_distance_func: Callable[[str, str], float],
        sample_temperature: float,
        console_lock,
        diary_path: str,
        database_path: str,
        initialization_cleanse_threshold: float,
        delete_when_initial_cleanse: bool,
        models: list[str],
        model_temperatures: list[float],
        model_assess_window_size: int,
        model_assess_initial_score: float,
        model_assess_average_order: float,
        model_sample_temperature: float,
        similarity_threshold: float,
        idea_uid_length: int,
    )-> None:
        
        self.program_name = program_name
        self.sample_temperature = sample_temperature
        self.console_lock = console_lock
        self.path = database_path
        guarantee_path_exist(self.path + "score_sheet.json")
        self.diary_path = diary_path
        self.similarity_threshold = similarity_threshold
        self.similarity_distance_func = similarity_distance_func
        
        if assess_func is not None:
            self.assess_on = True
            self.assess_func = assess_func
            self.assess_interval = assess_interval
            self.assess_result_path = assess_result_path
        else:
            self.assess_on = False
        
        if mutation_func is not None:
            self.mutation_on = True
            self.mutation_func = mutation_func
            self.mutation_interval = mutation_interval
        else:
            self.mutation_on = False
            
        if crossover_func is not None:
            self.crossover_on = True
            self.crossover_func = crossover_func
            self.crossover_interval = crossover_interval
        else:
            self.crossover_on = False
        
        self.default_similarity_distance_func = default_similarity_distance_func
        self.idea_uid_length = idea_uid_length
        self.models = models
        self.model_temperatures = model_temperatures
        self.model_sample_temperature = model_sample_temperature
        self.model_recent_scores = []
        for _ in range(len(models)):
            self.model_recent_scores.append(
                np.full((model_assess_window_size,), model_assess_initial_score)
            )
        self.model_assess_average_order = model_assess_average_order
        p = self.model_assess_average_order
        self.model_scores = [
            (np.mean(np.abs(self.model_recent_scores[i]) ** p)) ** (1 / p)
            for i in range(len(self.model_recent_scores))
        ]
        
        self.ideas = []
        # 处理文件路径，确保使用正确的分隔符
        path_to_search = Path(self.path).resolve()

        for path in path_to_search.rglob('*.idea'):
            # 确保路径在 macOS 上的兼容性
            if os.path.isfile(path):
                idea = Idea(
                    path=path,
                    evaluate_func=evaluate_func,
                )
                
                if idea.score < initialization_cleanse_threshold:
                    if delete_when_initial_cleanse:
                        path.unlink()
                        with self.console_lock:
                            append_to_file(
                                file_path=self.diary_path,
                                content_str=f"【数据库】 初始文件{path}得分未达到{initialization_cleanse_threshold}，已删除。",
                            )
                    else:
                        with self.console_lock:
                            append_to_file(
                                file_path=self.diary_path,
                                content_str=f"【数据库】 初始文件{path}得分未达到{initialization_cleanse_threshold}，已忽略。",
                            )
                else:
                    self.ideas.append(idea)
                    with self.console_lock:
                        append_to_file(
                            file_path=self.diary_path,
                            content_str=f"【数据库】 初始文件{path}已评分并加入数据库。",
                        )
                    
        self._sync_score_sheet()
        self._sync_similar_num_list()
        
        self.interaction_count = 0
        self.max_interaction_num = max_interaction_num
        self.examples_num = examples_num
        
        self.lock = Lock()
        self.status = "Running"
        

    # 低效版本
    # def _sync_similar_num_list(self):
        
    #     self.idea_similar_nums = []

    #     for i, idea_i in enumerate(self.ideas):
    #         similar_count = 0
    #         for j, idea_j in enumerate(self.ideas):
    #             if i == j or idea_i.content == idea_j.content:
    #                 similar_count += 1
    #             if self.similarity_distance_func == self.default_similarity_distance_func:
    #                 score_diff = abs(idea_i.score - idea_j.score)
    #             else:
    #                 score_diff = self.similarity_distance_func(idea_i.content, idea_j.content)
    #             if score_diff <= self.similarity_threshold:
    #                 similar_count += 1
    #         self.idea_similar_nums.append(similar_count)
            
    #     with self.console_lock:
    #         append_to_file(
    #             file_path = self.diary_path,
    #             content_str = "【数据库】 成功将idea_similar_nums与ideas同步！",
    #         )
      
    # GPT完成的高效版本（仅在default_similarity_distance_func情形下优化）     
    def _sync_similar_num_list(self):
        epsilon = self.similarity_threshold
        default_sim = self.similarity_distance_func == self.default_similarity_distance_func

        self.idea_similar_nums = [0] * len(self.ideas)

        if default_sim:
            # 使用滑窗优化，只适用于默认 similarity_distance_func
            indexed_ideas = list(enumerate(self.ideas))
            indexed_ideas.sort(key=lambda x: x[1].score)

            n = len(indexed_ideas)
            left = 0
            for right in range(n):
                # 左边界移动到满足 score 差值小于 epsilon 的位置
                while indexed_ideas[right][1].score - indexed_ideas[left][1].score > epsilon:
                    left += 1
                count = 0
                for k in range(left, right + 1):
                    i = indexed_ideas[right][0]
                    j = indexed_ideas[k][0]
                    # 包含内容完全相同的情况
                    if i == j or self.ideas[i].content == self.ideas[j].content:
                        count += 1
                    elif abs(self.ideas[i].score - self.ideas[j].score) <= epsilon:
                        count += 1
                self.idea_similar_nums[indexed_ideas[right][0]] = count
        else:
            # 非默认相似度函数，使用两重循环
            for i, idea_i in enumerate(self.ideas):
                similar_count = 0
                for j, idea_j in enumerate(self.ideas):
                    if i == j or idea_i.content == idea_j.content:
                        similar_count += 1
                    else:
                        score_diff = self.similarity_distance_func(idea_i.content, idea_j.content)
                        if score_diff <= epsilon:
                            similar_count += 1
                self.idea_similar_nums[i] = similar_count

        with self.console_lock:
            append_to_file(
                file_path=self.diary_path,
                content_str="【数据库】 成功将idea_similar_nums与ideas同步！",
            )
            
            
    def _assess(self)-> None:
        with self.console_lock:
            append_to_file(
                file_path = self.diary_path,
                content_str = "【数据库】 现在开始评估库中idea的整体质量！",
            )
    
    
    def _mutate(self)-> None:
        with self.console_lock:
            append_to_file(
                file_path = self.diary_path,
                content_str = "【数据库】 现在开始进行单体突变！",
            )
    
    
    def _crossover(self)-> None:
        with self.console_lock:
            append_to_file(
                file_path = self.diary_path,
                content_str = "【数据库】 现在开始进行交叉变异！",
            )
        

    def get_examples(self) -> list[Idea]:
        
        with self.lock:
            
            if self.status == "Terminated":
                return None
            self.interaction_count += 1
            
            if self.assess_on:
                if self.interaction_count % self.assess_interval == 0:
                    self._assess()
            
            if self.mutation_on:
                if self.interaction_count % self.mutation_interval == 0:
                    self._mutate()
            
            if self.crossover_on:
                if self.interaction_count % self.crossover_interval == 0:
                    self._crossover()
            
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
            exp_scores /= np.array(self.idea_similar_nums)
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
                        f"  {index+1}. {model}(T={model_temperature:.2f}): {self.model_scores[index]:.2f}"
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
    ):
        with self.lock:
        
            def generate_random_string(length=self.idea_uid_length):
                return ''.join(random.choices(string.ascii_lowercase, k=length))
            
            for idea_content, score, info in result:
                
                idea_uid = generate_random_string()
                path = os.path.join("programs", self.program_name, "database", f"idea_{idea_uid}.idea")
                while path in [idea.path for idea in self.ideas]:
                    idea_uid = generate_random_string()
                    path = os.path.join("programs", self.program_name, "database", f"idea_{idea_uid}.idea")
                
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
            self._sync_similar_num_list()
            
            
    def update_model_score(
        self,
        score_result: list[float], 
        model: str,
        model_temperature: float,
    )-> None:
        
        with self.lock:
            
            index = 0
            
            while index < len(self.models):
                
                if self.models[index] == model and self.model_temperatures[index] == model_temperature:
                    self.model_recent_scores[index][:-1] = self.model_recent_scores[index][1:]
                    p = self.model_assess_average_order
                    scores_array = np.array(score_result)
                    self.model_recent_scores[index][-1] = (np.mean(np.abs(scores_array) ** p)) ** (1 / p)
                    self.model_scores[index] = (np.mean(np.abs(self.model_recent_scores[index]) ** p)) ** (1 / p)
                    with self.console_lock:    
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = (
                                f"【数据库】 模型{model}(T={model_temperature:.2f})此轮得分为{self.model_recent_scores[index][-1]:.2f}，"
                                f"其总得分已被更新为{self.model_scores[index]:.2f}！"
                            ),
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
