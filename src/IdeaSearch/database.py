from threading import Lock
from pathlib import Path
import random
import os
from os.path import basename
import string
import json
from typing import Callable, Optional
import numpy as np
from src.utils import append_to_file, guarantee_path_exist

class Idea:
    
    def __init__(
        self, 
        path,
        evaluate_func, 
        content = None, 
        score = None, 
        info = None
    ):
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

    # ----------------------------- 数据库初始化 ----------------------------- 

    def __init__(
        self, 
        program_name, 
        max_interaction_num,
        examples_num,
        evaluate_func,
        assess_func,
        assess_interval,
        assess_result_path,
        mutation_func: Optional[Callable[[str], str]],
        mutation_interval: Optional[int],
        mutation_num: Optional[int],
        mutation_temperature: Optional[float],
        crossover_func: Optional[Callable[[str, str], str]],
        crossover_interval: Optional[int],
        crossover_num: Optional[int],
        crossover_temperature: Optional[float],
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
        
        # 记录必要字段
        self.program_name = program_name
        self.sample_temperature = sample_temperature
        self.console_lock = console_lock
        self.path = database_path + "ideas/"
        self.diary_path = diary_path
        self.similarity_threshold = similarity_threshold
        self.similarity_distance_func = similarity_distance_func
        self.default_similarity_distance_func = default_similarity_distance_func
        self.idea_uid_length = idea_uid_length
        self.models = models
        self.model_temperatures = model_temperatures
        self.model_sample_temperature = model_sample_temperature
        self.model_assess_average_order = model_assess_average_order
        self.max_interaction_num = max_interaction_num
        self.examples_num = examples_num
        self.evaluate_func = evaluate_func
        
        # 确保score_sheet.json文件存在
        guarantee_path_exist(self.path + "score_sheet.json")
        
        # 处理评估
        if assess_func is not None:
            self.assess_on = True
            self.assess_func = assess_func
            self.assess_interval = assess_interval
            self.assess_result_path = assess_result_path
        else:
            self.assess_on = False
        
        # 处理单体突变
        if mutation_func is not None:
            self.mutation_on = True
            self.mutation_func = mutation_func
            self.mutation_interval = mutation_interval
            self.mutation_num = mutation_num
            self.mutation_temperature = mutation_temperature
        else:
            self.mutation_on = False
            
        # 处理交叉变异
        if crossover_func is not None:
            self.crossover_on = True
            self.crossover_func = crossover_func
            self.crossover_interval = crossover_interval
            self.crossover_num = crossover_num
            self.crossover_temperature = crossover_temperature
        else:
            self.crossover_on = False
        
        # 初始化模型得分记录列表与模型得分列表
        self.model_recent_scores = []
        self.model_scores = []
        for _ in range(len(models)):
            self.model_recent_scores.append(
                np.full((model_assess_window_size,), model_assess_initial_score)
            )
            self.model_scores.append(model_assess_initial_score)
        
        # 初始化ideas列表（疑似存在mac OS系统的兼容性问题）
        self.ideas: list[Idea] = []
        path_to_search = Path(self.path).resolve()
        for path in path_to_search.rglob('*.idea'):
            
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
                                content_str=f"【数据库】 初始文件 {basename(path)} 得分未达到{initialization_cleanse_threshold:.2f}，已删除。",
                            )
                            
                    else:
                        with self.console_lock:
                            append_to_file(
                                file_path=self.diary_path,
                                content_str=f"【数据库】 初始文件 {basename(path)} 得分未达到{initialization_cleanse_threshold:.2f}，已忽略。",
                            )
                            
                else:
                    self.ideas.append(idea)
                    with self.console_lock:
                        append_to_file(
                            file_path=self.diary_path,
                            content_str=f"【数据库】 初始文件 {basename(path)} 已评分并加入数据库。",
                        )
         
        # 同步 score sheet 与 idea_similar_nums 列表   
        self._sync_score_sheet()
        self._sync_similar_num_list()
        
        # 初始化互动时间戳、自锁、状态、随机数生成器
        self.interaction_count = 0
        self.lock = Lock()
        self.status = "Running"
        self.random_generator = np.random.default_rng()
    
    # ----------------------------- 外部调用动作 ----------------------------- 
    
    def get_status(self):
        with self.lock:
            return self.status
        
    
    def get_examples(
        self
    ) -> list[Idea]:
        
        with self.lock:
            
            if self.status == "Terminated":
                return None
            self.interaction_count += 1
            with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【数据库】 当前交互次数为： {self.interaction_count} ，"
                            f"还剩 {self.max_interaction_num-self.interaction_count} 次！"
                        ),
                    )
            
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
                        content_str = "【数据库】 发生异常： ideas 列表为空！",
                    )
                exit()
                
            probabilities = np.array([idea.score for idea in self.ideas])
            probabilities /= self.sample_temperature
            max_value = np.max(probabilities)
            probabilities = np.exp(probabilities - max_value)
            probabilities /= np.array(self.idea_similar_nums)
            probabilities /= np.sum(probabilities)
            
            selected_indices = self.random_generator.choice(
                a = len(self.ideas),
                size = min(len(self.ideas), self.examples_num),
                replace = False, # 不允许重复选择同一个元素
                p = probabilities,
            )

            return [self.ideas[int(i)] for i in selected_indices]
        
        
    def get_model(self) -> tuple[str, float]:
        
        with self.lock:
            
            self._show_model_scores()
            
            probabilities = np.array(self.model_scores) / self.model_sample_temperature
            max_value = np.max(probabilities)
            probabilities = np.exp(probabilities - max_value)
            probabilities /= np.sum(probabilities)
            
            selected_index = self.random_generator.choice(
                a = len(self.models), 
                p = probabilities,
            )
            
            selected_model_name = self.models[selected_index]
            selected_model_temperature = self.model_temperatures[selected_index]
            
            return selected_model_name, selected_model_temperature
        
        
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
                                f"【数据库】 模型 {model}(T={model_temperature:.2f}) 此轮得分为 {self.model_recent_scores[index][-1]:.2f} ，"
                                f"其总得分已被更新为 {self.model_scores[index]:.2f} ！"
                            ),
                        )
                    return
                
                index += 1
                
            with self.console_lock:    
                append_to_file(
                    file_path = self.diary_path,
                    content_str = f"【数据库】 出现错误！未知的模型名称及温度： {model}(T={model_temperature:.2f}) ！",
                )
                
            exit()  
        
        
    def receive_result(
        self, 
        result: list[tuple[Idea, float, str]], 
        evaluator_id: int,
    )-> None:
        with self.lock:
    
            for idea_content, score, info in result:
                
                self._store_idea(
                    idea = idea_content,
                    score = score,
                    info = info,
                )
                
            with self.console_lock:    
                append_to_file(
                    file_path = self.diary_path,
                    content_str = f"【数据库】 {evaluator_id} 号评估器递交的 {len(result)} 个新文件已评分并加入数据库。",
                )
            
            self._sync_score_sheet()
            self._sync_similar_num_list()
            
    # ----------------------------- 内部调用动作 -----------------------------           
            
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
                content_str = f"【数据库】  {self.program_name} 的 score sheet 已更新！",
            )


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
                content_str="【数据库】 成功将 idea_similar_nums 列表与 ideas 列表同步！",
            )
    # 低效版本（备份用）
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
    
    
    def _check_threshold(self):
        if self.interaction_count >= self.max_interaction_num:
            with self.console_lock:
                append_to_file(
                    file_path = self.diary_path,
                    content_str = "【数据库】 采样次数已分发完毕，IdeaSearch将在各采样器完成手头任务后结束。",
                )
            self.status = "Terminated"
    
    
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
            
        for index in range(self.mutation_num):
            
            probabilities = np.array([idea.score for idea in self.ideas]) / self.mutation_temperature
            max_value = np.max(probabilities)
            probabilities = np.exp(probabilities - max_value)
            probabilities /= np.array(self.idea_similar_nums)
            probabilities /= np.sum(probabilities)
            
            selected_index = self.random_generator.choice(
                a = len(self.ideas), 
                p = probabilities
            )
            selected_idea = self.ideas[selected_index]
            
            try:
                mutated_idea = self.mutation_func(selected_idea.content)
            except Exception as error:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【数据库】 第{index+1}次单体突变在运行 mutation_func 时发生了错误：\n"
                            f"{error}\n此轮单体突变意外终止！"
                        ),
                    )
                return
            
            path = self._store_idea(
                idea = mutated_idea,
                evaluate_func = self.evaluate_func,
            )
            
            if path is not None:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【数据库】 第{index+1}次单体突变："
                            f" {basename(selected_idea.path)} 突变为 {basename(path)} "
                        ),
                    )
                self._sync_score_sheet()
                self._sync_similar_num_list()
            else:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【数据库】 第{index+1}次单体突变发生了错误：\n"
                            f"{self._store_idea_error_message}\n此轮单体突变意外终止！"
                        ),
                    )
                return
                
        with self.console_lock:
            append_to_file(
                file_path = self.diary_path,
                content_str = "【数据库】 此轮单体突变已结束。",
            )
    
    
    def _crossover(self) -> None:
    
        with self.console_lock:
            append_to_file(
                file_path = self.diary_path,
                content_str = "【数据库】 现在开始进行交叉变异！",
            )

        for index in range(self.crossover_num):
            
            probabilities = np.array([idea.score for idea in self.ideas]) / self.crossover_temperature
            max_value = np.max(probabilities)
            probabilities = np.exp(probabilities - max_value)
            probabilities /= np.sum(probabilities)
            
            parent_indices = self.random_generator.choice(
                a = len(self.ideas), 
                size = 2, 
                replace = False, # 不允许重复选择同一个元素
                p = probabilities
            )
            parent_1 = self.ideas[parent_indices[0]]
            parent_2 = self.ideas[parent_indices[1]]

            try:
                crossed_content = self.crossover_func(
                    parent_1.content, parent_2.content
                )
            except Exception as error:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【数据库】 第{index+1}次交叉变异在运行 crossover_func 时发生了错误：\n"
                            f"{error}\n此轮交叉变异意外终止！"
                        ),
                    )
                return

            path = self._store_idea(
                idea = crossed_content,
                evaluate_func = self.evaluate_func,
            )

            if path is not None:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【数据库】 第{index+1}次交叉变异："
                            f"{basename(parent_1.path)} × {basename(parent_2.path)} 交叉为 {path} "
                        ),
                    )
                self._sync_score_sheet()
                self._sync_similar_num_list()
            else:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【数据库】 第{index+1}次交叉变异发生了错误：\n"
                            f"{self._store_idea_error_message}\n此轮交叉变异意外终止！"
                        ),
                    )
                return

        with self.console_lock:
            append_to_file(
                file_path = self.diary_path,
                content_str = "【数据库】 此轮交叉变异已结束。",
            )
        
        
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
    
    
    def _store_idea(
        self, 
        idea: str,
        evaluate_func: Optional[Callable[[str, str], float]] = None,
        score: Optional[float] = None,
        info: Optional[str] = None,
    )-> Optional[str]:
        
        """
        将一个新的 idea 内容保存为文件，并添加到内部 idea 列表中。
        为了避免重复运行 evaluate_func 带来的时间开销，允许调用者在以下两个情形间选择：
        
        1. evaluate_func is None （已在外部评估 idea ）
        这时应该传入 score 和 info
        
        2. evaluate_func is not None （尚未评估 idea ）
        这时可以不传入 score 和 info

        Args:
            idea (str): 需要存储的 idea 内容（字符串格式）。
            evaluate_func (Optional[Callable[[str, str], float]]): 用于评价 idea 的函数。
            score (Optional[float]): 预设的 idea 得分。
            info (Optional[str]): 与 idea 相关的附加信息。

        Returns:
            Optional[str]: 成功则返回该 idea 对应的文件路径；若出错则返回 None。
        """
        
        try:
            path = self._get_new_idea_path()
            
            with open(path, 'w', encoding='utf-8') as file:
                file.write(idea)
                
            self.ideas.append(Idea(
                path = path,
                evaluate_func = evaluate_func,
                content = idea,
                score = score,
                info = info,
            ))
                
            return path
        
        except Exception as error:
            self._store_idea_error_message = error
            return None
            
            
    def _get_new_idea_path(self)-> str:
        
        def generate_random_string(length=self.idea_uid_length):
            return ''.join(random.choices(string.ascii_lowercase, k=length))
        
        idea_uid = generate_random_string()
        path = os.path.join(f"{self.path}", f"idea_{idea_uid}.idea")
        current_idea_paths = [idea.path for idea in self.ideas]
        while path in current_idea_paths:
            idea_uid = generate_random_string()
            path = os.path.join(f"{self.path}", f"idea_{idea_uid}.idea")
            
        return path
            
        

    
            
            
    
                

    
