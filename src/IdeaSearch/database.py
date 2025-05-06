import os
import json
import random
import bisect
import string
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import isnan
from threading import Lock
from pathlib import Path
from typing import Callable
from typing import Optional
from os.path import basename
from src.utils import append_to_file
from src.utils import guarantee_path_exist


__all__ = [
    "Idea",
    "Database",
]


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
        program_name: str, 
        max_interaction_num: int,
        examples_num: int,
        evaluate_func: Callable[[str], tuple[float, str]],
        score_range: tuple[float, float],
        assess_func: Optional[Callable[[list[str], list[float], list[str]], float]],
        assess_interval: Optional[int],
        assess_result_data_path: Optional[str],
        assess_result_pic_path: Optional[str],
        hand_over_threshold: float,
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
        console_lock: Lock,
        diary_path: str,
        database_path: str,
        initialization_skip_evaluation: bool,
        initialization_cleanse_threshold: float,
        delete_when_initial_cleanse: bool,
        models: list[str],
        model_temperatures: list[float],
        model_assess_window_size: int,
        model_assess_initial_score: float,
        model_assess_average_order: float,
        model_assess_save_result: bool,
        model_assess_result_data_path: Optional[str],
        model_assess_result_pic_path: Optional[str],
        model_sample_temperature: float,
        similarity_threshold: float,
        similarity_sys_info_thresholds: Optional[list[int]],
        similarity_sys_info_prompts: Optional[list[str]],
        idea_uid_length: int,
    )-> None:

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
        self.model_assess_save_result = model_assess_save_result
        self.model_assess_result_data_path = model_assess_result_data_path
        self.model_assess_result_pic_path = model_assess_result_pic_path
        self.max_interaction_num = max_interaction_num
        self.handover_threshold = hand_over_threshold
        self.examples_num = examples_num
        self.evaluate_func = evaluate_func
        self.score_range = score_range
        
        if initialization_skip_evaluation:
            try:
                with open(self.path + "score_sheet.json", "r", encoding="UTF-8") as file:
                    score_sheet_backup = json.load(file)
                with self.console_lock:
                    append_to_file(
                        file_path=self.diary_path,
                        content_str=f"【数据库】 已从 {self.path + 'score_sheet.json'} 成功读取用于迅捷加载的 score_sheet.json 文件！",
                    )
            except Exception as error:
                score_sheet_backup = {}
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【数据库】 未从 {self.path + 'score_sheet.json'} 成功读取用于迅捷加载的 score_sheet.json 文件，报错：\n"
                            f"{error}\n"
                            "请检查该行为是否符合预期！"
                        ),
                    )
        
        guarantee_path_exist(self.path + "score_sheet.json")

        if mutation_func is not None:
            self.mutation_on = True
            self.mutation_func = mutation_func
            self.mutation_interval = mutation_interval
            self.mutation_num = mutation_num
            self.mutation_temperature = mutation_temperature
        else:
            self.mutation_on = False
        
        if crossover_func is not None:
            self.crossover_on = True
            self.crossover_func = crossover_func
            self.crossover_interval = crossover_interval
            self.crossover_num = crossover_num
            self.crossover_temperature = crossover_temperature
        else:
            self.crossover_on = False
        
        if similarity_sys_info_thresholds is not None:
            self.similarity_sys_info_on = True
            self.similarity_sys_info_thresholds = similarity_sys_info_thresholds
            self.similarity_sys_info_prompts = similarity_sys_info_prompts
        else:
            self.similarity_sys_info_on = False
        
        # 初始化ideas列表（疑似存在mac OS系统的兼容性问题）
        self.ideas: list[Idea] = []
        path_to_search = Path(self.path).resolve()
        for path in path_to_search.rglob('*.idea'):
            
            if os.path.isfile(path):
                
                if initialization_skip_evaluation:
                    if basename(path) in score_sheet_backup.keys():
                        
                        with open(path, 'r', encoding = "UTF-8") as file:
                            content = file.read()
                            
                        score = score_sheet_backup[basename(path)]["score"]
                            
                        info = score_sheet_backup[basename(path)]["info"]
                        if info == "": info = None
                            
                        idea = Idea(
                            path = path,
                            evaluate_func = None,
                            content = content,
                            score = score,
                            info = info,
                        )
                        
                        if info is not None:
                            with self.console_lock:
                                append_to_file(
                                    file_path=self.diary_path,
                                    content_str=f"【数据库】 已从 score_sheet.json 中迅捷加载初始文件 {basename(path)} 的得分与评语！",
                                )
                        else:
                            with self.console_lock:
                                append_to_file(
                                    file_path=self.diary_path,
                                    content_str=f"【数据库】 已从 score_sheet.json 中迅捷加载初始文件 {basename(path)} 的得分！",
                                )
                        
                    else:
                        with self.console_lock:
                            append_to_file(
                                file_path=self.diary_path,
                                content_str=f"【数据库】 没有在 score_sheet.json 中找到初始文件 {basename(path)} ，迅捷加载失败！",
                            )
                            
                        idea = Idea(
                            path = path,
                            evaluate_func = evaluate_func,
                        )
                    
                else:
                    idea = Idea(
                        path = path,
                        evaluate_func = evaluate_func,
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
                        
        ideas: list[str] = [current_idea.content for current_idea in self.ideas]
        scores: list[float] = [current_idea.score for current_idea in self.ideas]
        infos: list[Optional[str]] = [current_idea.info for current_idea in self.ideas]   
                     
        if assess_func is not None:
            self.assess_on = True
            self.assess_func = assess_func
            self.assess_interval = assess_interval
            self.assess_result_data_path = assess_result_data_path
            self.assess_result_pic_path = assess_result_pic_path
            self.assess_result_ndarray = np.zeros((1 + (max_interaction_num // assess_interval),))
            self.assess_result_ndarray_index = 1
            self.assess_result_ndarray_x_axis = np.linspace(
                start = 0, 
                stop = max_interaction_num, 
                num = 1 + (max_interaction_num // assess_interval), 
                endpoint = True
            )
            guarantee_path_exist(assess_result_data_path)
            guarantee_path_exist(assess_result_pic_path)
            
            get_database_initial_score_success = False
            try:
                database_initial_score = self.assess_func(
                    ideas,
                    scores,
                    infos,
                )
                get_database_initial_score_success = True
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = f"【数据库】 初始 ideas 的整体质量得分为：{database_initial_score:.2f}！",
                    )
                    
            except Exception as error:
                database_initial_score = 0
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【数据库】 评估库中初始 ideas 的整体质量时遇到错误：\n"
                            f"{error}"
                        ),
                    )
                    
            self.assess_result_ndarray[0] = database_initial_score
            self._sync_database_assess_result(
                is_initialization = True,
                get_database_score_success = get_database_initial_score_success,
            )
            
        else:
            self.assess_on = False
            
        self.model_recent_scores = []
        self.model_scores = []
        for _ in range(len(models)):
            self.model_recent_scores.append(
                np.full((model_assess_window_size,), model_assess_initial_score)
            )
            self.model_scores.append(model_assess_initial_score)
            
        if self.model_assess_save_result:
            self.scores_of_models = np.zeros((1+self.max_interaction_num, len(self.models)))
            self.scores_of_models_index = 0
            self.scores_of_models_x_axis = np.linspace(
                start = 0, 
                stop = max_interaction_num, 
                num = 1 + max_interaction_num, 
                endpoint = True
            )
            self._sync_model_score_result()
          
        self._sync_score_sheet()
        self._sync_similar_num_list()
        
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
                            f"【数据库】 已分发交互次数为： {self.interaction_count} ，"
                            f"还剩 {self.max_interaction_num-self.interaction_count} 次！"
                        ),
                    )
            
            if self.assess_on:
                if self.interaction_count % self.assess_interval == 0:
                    self._assess_database()
            
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
            
            selected_examples = []
            for i in selected_indices:
                selected_index = int(i)
                example_idea = self.ideas[selected_index]
                
                if self.similarity_sys_info_on:
                    similar_num = self.idea_similar_nums[selected_index]
                    similarity_prompt = get_label(
                        x = similar_num,
                        thresholds = self.similarity_sys_info_thresholds,
                        labels = self.similarity_sys_info_prompts
                    )
                else:
                    similar_num = None
                    similarity_prompt = None
                    
                selected_examples.append((
                    example_idea.content,
                    example_idea.score,
                    example_idea.info,
                    similar_num,
                    similarity_prompt,
                ))

            return selected_examples
        
        
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
                    self._sync_model_score_result()
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
            basename(idea.path): {
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
            
    
    def _sync_similar_num_list(self):
        
        self.idea_similar_nums = []

        for i, idea_i in enumerate(self.ideas):
            similar_count = 0
            for j, idea_j in enumerate(self.ideas):
                if i == j or idea_i.content == idea_j.content:
                    similar_count += 1
                if self.similarity_distance_func == self.default_similarity_distance_func:
                    score_diff = abs(idea_i.score - idea_j.score)
                else:
                    score_diff = self.similarity_distance_func(idea_i.content, idea_j.content)
                if score_diff <= self.similarity_threshold:
                    similar_count += 1
            self.idea_similar_nums.append(similar_count)
            
        with self.console_lock:
            append_to_file(
                file_path = self.diary_path,
                content_str = "【数据库】 成功将idea_similar_nums与ideas同步！",
            )
    
    
    def _check_threshold(self):
        if self.interaction_count >= self.max_interaction_num:
            with self.console_lock:
                append_to_file(
                    file_path = self.diary_path,
                    content_str = "【数据库】 采样次数已分发完毕，IdeaSearch将在各采样器完成手头任务后结束。",
                )
            self.status = "Terminated"
    
    
    def _assess_database(self)-> None:
        
        with self.console_lock:
            append_to_file(
                file_path = self.diary_path,
                content_str = "【数据库】 现在开始评估库中 ideas 的整体质量！",
            )
            
        ideas: list[str] = []
        scores: list[float] = []
        infos: list[Optional[str]] = []
        for idea in self.ideas:
            ideas.append(idea.content)
            scores.append(idea.score)
            infos.append(idea.info)
            
        get_database_score_success = False
        try:
            database_score = self.assess_func(
                ideas,
                scores,
                infos,
            )
            get_database_score_success = True
            with self.console_lock:
                append_to_file(
                    file_path = self.diary_path,
                    content_str = f"【数据库】 当前库中 ideas 的整体质量得分为：{database_score:.2f}！",
                )
                
        except Exception as error:
            database_score = 0
            with self.console_lock:
                append_to_file(
                    file_path = self.diary_path,
                    content_str = (
                        f"【数据库】 评估库中 ideas 的整体质量时遇到错误：\n"
                        f"{error}"
                    ),
                )
                
        self.assess_result_ndarray[self.assess_result_ndarray_index] = database_score
        self.assess_result_ndarray_index += 1
        
        self._sync_database_assess_result(
            is_initialization = False,
            get_database_score_success = get_database_score_success,
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
                if not isinstance(mutated_idea, str):
                    with self.console_lock:
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = (
                                f"【数据库】 调用 {self.program_name} 的单体突变函数时发生错误："
                                f"返回结果中的 mutated_idea 应为一字符串，不应为一个 {type(mutated_idea)} 类型的对象！"
                                "\n此轮单体突变意外终止！"
                            ),
                        )
                    return
            except Exception as error:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【数据库】 第 {index+1} 次单体突变在运行 mutation_func 时发生了错误：\n"
                            f"{error}\n此轮单体突变意外终止！"
                        ),
                    )
                return
            
            try:
                score, info = self.evaluate_func(mutated_idea)
                
                if not isinstance(score, float):
                    with self.console_lock:
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = (
                                f"【数据库】 调用 {self.program_name} 的评估函数时发生错误："
                                f"返回结果中的 score 应为一浮点数，不应为一个 {type(score)} 类型的对象！"
                                "\n此轮单体突变意外终止！"
                            ),
                        )
                    return
                
                if isnan(score):
                    with self.console_lock:
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = (
                                f"【数据库】 调用 {self.program_name} 的评估函数时发生错误："
                                f"返回结果中的 score 不应为 NaN ！"
                                "\n此轮单体突变意外终止！"
                            ),
                        )
                    return
                
                if info is not None:
                    if not isinstance(info, str):
                        with self.console_lock:
                            append_to_file(
                                file_path = self.diary_path,
                                content_str = (
                                    f"【数据库】 调用 {self.program_name} 的评估函数时发生错误："
                                    f"返回结果中的 info 应为 None 或一字符串，不应为一个 {type(info)} 类型的对象！"
                                    "\n此轮单体突变意外终止！"
                                ),
                            )
                        return
                
            except Exception as error:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【数据库】 调用 {self.program_name} 的评估函数时发生错误：\n{error}"
                            "\n此轮单体突变意外终止！"
                        ),
                    )  
                return
            
            if score >= self.handover_threshold:
            
                path = self._store_idea(
                    idea = mutated_idea,
                    score = score,
                    info = info, 
                )
                
                if path is not None:
                    with self.console_lock:
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = (
                                f"【数据库】 第 {index+1} 次单体突变："
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
                                f"【数据库】 第 {index+1} 次单体突变发生了错误：\n"
                                f"{self._store_idea_error_message}\n此轮单体突变意外终止！"
                            ),
                        )
                    return
                
            else:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【数据库】 第 {index+1} 次单体突变结果未达到入库分数阈值"
                            f"（{self.handover_threshold:.2f}分），已删除！"
                        ),
                    )
                
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
                crossover_idea = self.crossover_func(
                    parent_1.content, parent_2.content
                )
                if not isinstance(crossover_idea, str):
                    with self.console_lock:
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = (
                                f"【数据库】 调用 {self.program_name} 的交叉变异函数时发生错误："
                                f"返回结果中的 crossover_idea 应为一字符串，不应为一个 {type(crossover_idea)} 类型的对象！"
                                "\n此轮交叉变异意外终止！"
                            ),
                        )
                    return
            except Exception as error:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【数据库】 第 {index+1} 次交叉变异在运行 crossover_func 时发生了错误：\n"
                            f"{error}\n此轮交叉变异意外终止！"
                        ),
                    )
                return
            
            try:
                score, info = self.evaluate_func(crossover_idea)
                
                if not isinstance(score, float):
                    with self.console_lock:
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = (
                                f"【数据库】 调用 {self.program_name} 的评估函数时发生错误："
                                f"返回结果中的 score 应为一浮点数，不应为一个 {type(score)} 类型的对象！"
                                "\n此轮交叉变异意外终止！"
                            ),
                        )
                    return
                
                if isnan(score):
                    with self.console_lock:
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = (
                                f"【数据库】 调用 {self.program_name} 的评估函数时发生错误："
                                f"返回结果中的 score 不应为 NaN ！"
                                "\n此轮交叉变异意外终止！"
                            ),
                        )
                    return
                
                if info is not None:
                    if not isinstance(info, str):
                        with self.console_lock:
                            append_to_file(
                                file_path = self.diary_path,
                                content_str = (
                                    f"【数据库】 调用 {self.program_name} 的评估函数时发生错误："
                                    f"返回结果中的 info 应为 None 或一字符串，不应为一个 {type(info)} 类型的对象！"
                                    "\n此轮交叉变异意外终止！"
                                ),
                            )
                        return
                
            except Exception as error:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【数据库】 调用 {self.program_name} 的评估函数时发生错误：\n{error}"
                            "\n此轮交叉变异意外终止！"
                        ),
                    )  
                return

            if score >= self.handover_threshold:
                
                path = self._store_idea(
                    idea = crossover_idea,
                    score = score,
                    info = info,
                )

                if path is not None:
                    with self.console_lock:
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = (
                                f"【数据库】 第 {index+1} 次交叉变异："
                                f"{basename(parent_1.path)} × {basename(parent_2.path)} 交叉为 {basename(path)} "
                            ),
                        )
                    self._sync_score_sheet()
                    self._sync_similar_num_list()
                else:
                    with self.console_lock:
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = (
                                f"【数据库】 第 {index+1} 次交叉变异发生了错误：\n"
                                f"{self._store_idea_error_message}\n此轮交叉变异意外终止！"
                            ),
                        )
                    return
                
            else:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【数据库】 第 {index+1} 次交叉变异结果未达到入库分数阈值"
                            f"（{self.handover_threshold:.2f}分），已删除！"
                        ),
                    )

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
    
    
    def _sync_database_assess_result(
        self,
        is_initialization: bool,
        get_database_score_success: bool,
    )-> None:
        
        np.savez_compressed(
            file = self.assess_result_data_path, 
            interaction_num = self.assess_result_ndarray_x_axis,
            database_scores = self.assess_result_ndarray,
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.assess_result_ndarray_x_axis, 
            self.assess_result_ndarray, 
            label='Database Score', 
            color='dodgerblue', 
            marker='o'
        )
        plt.title("Database Assessment")
        plt.xlabel("Interaction No.")
        plt.ylabel("Database Score")
        plt.ylim(self.score_range)
        plt.grid(True)
        plt.legend()
        plt.savefig(self.assess_result_pic_path)
        plt.close()
        
        if get_database_score_success:
            if is_initialization:
                append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【数据库】 初始质量评估结束，"
                            f" {basename(self.assess_result_data_path)} 与 {basename(self.assess_result_pic_path)} 已更新！"
                        ),
                    )
            else:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【数据库】 此轮质量评估结束，"
                            f" {basename(self.assess_result_data_path)} 与 {basename(self.assess_result_pic_path)} 已更新！"
                        ),
                    )
                    
                    
    def _sync_model_score_result(self):
        
        self.scores_of_models[self.scores_of_models_index] = self.model_scores
        self.scores_of_models_index += 1
        
        scores_of_models = self.scores_of_models.T
        
        scores_of_models_dict = {}
        for model_name, model_temperature, model_scores in zip(self.models, self.model_temperatures, scores_of_models):
            scores_of_models_dict[f"{model_name}(T={model_temperature:.2f})"] = model_scores
        
        np.savez_compressed(
            file=self.model_assess_result_data_path,
            interaction_num=self.scores_of_models_x_axis,
            **scores_of_models_dict
        )

        plt.figure(figsize=(10, 6))
        for model_label, model_scores in scores_of_models_dict.items():
            plt.plot(
                self.scores_of_models_x_axis,
                model_scores,
                label=model_label,
                marker='o'
            )
        plt.title("Model Scores")
        plt.xlabel("Interaction No.")
        plt.ylabel("Model Score")
        plt.ylim(self.score_range)
        plt.grid(True)
        plt.legend()
        plt.savefig(self.model_assess_result_pic_path)
        plt.close()
        
        with self.console_lock:
            append_to_file(
                file_path=self.diary_path,
                content_str=(
                    f"【数据库】 "
                    f" {basename(self.model_assess_result_data_path)} 与 {basename(self.model_assess_result_pic_path)} 已更新！"
                ),
            )

def get_label(
    x: int, 
    thresholds: list[int], 
    labels: list[str]
) -> str:
    if not thresholds:
        raise ValueError("thresholds 列表不能为空")

    if len(labels) != len(thresholds) + 1:
        raise ValueError(
            f"labels 列表长度应比 thresholds 长 1，"
            f"但实际为 labels={len(labels)}, thresholds={len(thresholds)}"
        )
    
    index = bisect.bisect_right(thresholds, x)
    return labels[index]
            