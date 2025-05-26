import os
import json
import random
import bisect
import string
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import perf_counter
from math import isnan
from threading import Lock
from pathlib import Path
from typing import Tuple
from typing import Callable
from typing import Optional
from os.path import basename
from src.utils import append_to_file
from src.utils import guarantee_path_exist


__all__ = [
    "Idea",
    "Island",
]


class Idea:
    
    def __init__(
        self, 
        path: str,
        evaluate_func: Optional[Callable[[str], Tuple[float, Optional[str]]]], 
        content: Optional[str] = None, 
        score: Optional[float] = None, 
        info: Optional[str] = None,
        source: Optional[str] = None,
    ):
        self.path = str(path)
        self.source = source
        if evaluate_func is not None:
            with open(path, 'r', encoding = "UTF-8") as file:
                self.content = file.read()
            self.score, self.info = evaluate_func(self.content)
        else:
            self.content = content
            self.score = score
            self.info = info


class Island:

    # ----------------------------- 岛屿初始化 ----------------------------- 

    def __init__(
        self,
        ideasearcher,
        island_id: int,
        default_similarity_distance_func: Callable[[str, str], float],
        console_lock: Lock,
    )-> None:

        self.ideasearcher = ideasearcher
        self.default_similarity_distance_func = default_similarity_distance_func
        self.console_lock = console_lock
        self.island_id = island_id
        self.interaction_count = 0
        self.interaction_num = 0
        self.lock = Lock()
        self.status = "Running"
        self.random_generator = np.random.default_rng()
        
        database_path = self.ideasearcher.get_database_path()
        assert database_path is not None
        self.path = database_path + f"ideas/island{self.island_id}/"
        guarantee_path_exist(self.path)
        self.diary_path = self.ideasearcher.get_diary_path()
        
        initial_idea_path = database_path + f"ideas/initial_ideas/"
        
        
        score_sheet_backup: Optional[dict] = None
        
        initialization_skip_evaluation = self.ideasearcher.get_initialization_skip_evaluation()
        
        if initialization_skip_evaluation:
            try:
                with open(initial_idea_path + "score_sheet.json", "r", encoding="UTF-8") as file:
                    score_sheet_backup = json.load(file)
                with self.console_lock:
                    append_to_file(
                        file_path=self.diary_path,
                        content_str=f"【{self.island_id}号岛屿】 已从 {initial_idea_path + 'score_sheet.json'} 成功读取用于迅捷加载的 score_sheet.json 文件！",
                    )
            except Exception as error:
                score_sheet_backup = {}
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【{self.island_id}号岛屿】 未从 {initial_idea_path + 'score_sheet.json'} 成功读取用于迅捷加载的 score_sheet.json 文件，报错：\n"
                            f"{error}\n"
                            "请检查该行为是否符合预期！"
                        ),
                    )
        
        guarantee_path_exist(self.path + "score_sheet.json")

        mutation_func = self.ideasearcher.get_mutation_func()
        if mutation_func is not None:
            self.mutation_on = True
            self.mutation_func = mutation_func
            self.mutation_interval = self.ideasearcher.get_mutation_interval()
            self.mutation_num = self.ideasearcher.get_mutation_num()
            self.mutation_temperature = self.ideasearcher.get_mutation_temperature()
        else:
            self.mutation_on = False
        
        crossover_func = self.ideasearcher.get_crossover_func()
        if crossover_func is not None:
            self.crossover_on = True
            self.crossover_func = crossover_func
            self.crossover_interval = self.ideasearcher.get_crossover_interval()
            self.crossover_num = self.ideasearcher.get_crossover_num()
            self.crossover_temperature = self.ideasearcher.get_crossover_temperature()
        else:
            self.crossover_on = False
        
        similarity_sys_info_thresholds = self.ideasearcher.get_similarity_sys_info_thresholds()
        if similarity_sys_info_thresholds is not None:
            self.similarity_sys_info_on = True
            self.similarity_sys_info_thresholds = similarity_sys_info_thresholds
            self.similarity_sys_info_prompts = self.ideasearcher.get_similarity_sys_info_prompts()
        else:
            self.similarity_sys_info_on = False
        
        # 初始化ideas列表（疑似存在mac OSIdeaSearcher的兼容性问题）
        evaluate_func = self.ideasearcher.get_evaluate_func()
        initialization_cleanse_threshold = self.ideasearcher.get_initialization_cleanse_threshold()
        delete_when_initial_cleanse = self.ideasearcher.get_delete_when_initial_cleanse()
        self.ideas: list[Idea] = []
        path_to_search = Path(initial_idea_path).resolve()
        source = "初始化时读入"
        for path in path_to_search.rglob('*.idea'):
            
            if os.path.isfile(path):
                
                idea: Optional[Idea] = None
                
                if initialization_skip_evaluation:
                    
                    assert score_sheet_backup is not None
                    
                    if basename(path) in score_sheet_backup:
                        
                        with open(path, 'r', encoding = "UTF-8") as file:
                            content = file.read()
                            
                        score = score_sheet_backup[basename(path)]["score"]
                            
                        info = score_sheet_backup[basename(path)]["info"]
                        if info == "": info = None
                            
                        idea = Idea(
                            path = str(path),
                            evaluate_func = None,
                            content = content,
                            score = score,
                            info = info,
                            source = source, 
                        )
                        
                        if info is not None:
                            with self.console_lock:
                                append_to_file(
                                    file_path=self.diary_path,
                                    content_str=f"【{self.island_id}号岛屿】 已从 score_sheet.json 中迅捷加载初始文件 {basename(path)} 的评分与评语！",
                                )
                        else:
                            with self.console_lock:
                                append_to_file(
                                    file_path=self.diary_path,
                                    content_str=f"【{self.island_id}号岛屿】 已从 score_sheet.json 中迅捷加载初始文件 {basename(path)} 的评分！",
                                )
                        
                    else:
                        with self.console_lock:
                            append_to_file(
                                file_path=self.diary_path,
                                content_str=f"【{self.island_id}号岛屿】 没有在 score_sheet.json 中找到初始文件 {basename(path)} ，迅捷加载失败！",
                            )
                            
                        idea = Idea(
                            path = str(path),
                            evaluate_func = evaluate_func,
                            source = source,
                        )
                    
                else:
                    idea = Idea(
                        path = str(path),
                        evaluate_func = evaluate_func,
                        source = source,
                    )
                
                assert idea.score is not None
                
                if idea.score < initialization_cleanse_threshold:
                    
                    if delete_when_initial_cleanse:
                        path.unlink()
                        with self.console_lock:
                            append_to_file(
                                file_path=self.diary_path,
                                content_str=f"【{self.island_id}号岛屿】 初始文件 {basename(path)} 评分未达到{initialization_cleanse_threshold:.2f}，已删除。",
                            )
                            
                    else:
                        with self.console_lock:
                            append_to_file(
                                file_path=self.diary_path,
                                content_str=f"【{self.island_id}号岛屿】 初始文件 {basename(path)} 评分未达到{initialization_cleanse_threshold:.2f}，已忽略。",
                            )
                            
                else:
                    self.ideas.append(idea)
                    with self.console_lock:
                        append_to_file(
                            file_path=self.diary_path,
                            content_str=f"【{self.island_id}号岛屿】 初始文件 {basename(path)} 已评分并加入{self.island_id}号岛屿。",
                        )
                        
        ideas: list[str] = []
        scores: list[float] = []
        infos: list[Optional[str]] = []
                        
        for current_idea in self.ideas:
            
            assert current_idea.content is not None
            assert current_idea.score is not None
            
            ideas.append(current_idea.content)
            scores.append(current_idea.score)
            infos.append(current_idea.info)
            
        assess_func = self.ideasearcher.get_assess_func()
        if assess_func is not None:
            
            self.assess_interaction_count = 0
            
            assess_interval = self.ideasearcher.get_assess_interval()
            assert assess_interval is not None
            
            assess_result_data_path = self.ideasearcher.get_assess_result_data_path()
            assess_result_pic_path = self.ideasearcher.get_assess_result_pic_path()
            
            self.assess_on = True
            self.assess_func = assess_func
            self.assess_interval = assess_interval
            self.assess_baseline = self.ideasearcher.get_assess_baseline()
            self.assess_result_data_path = assess_result_data_path
            self.assess_result_pic_path = assess_result_pic_path
            self.assess_result_ndarray = np.zeros((1 + (self.interaction_num // assess_interval),))
            self.assess_result_ndarray_length = 1
            self.assess_result_ndarray_x_axis = np.linspace(
                start = 0, 
                stop = self.interaction_num, 
                num = 1 + (self.interaction_num // assess_interval), 
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
                        content_str = f"【{self.island_id}号岛屿】 初始 ideas 的整体质量评分为：{database_initial_score:.2f}！",
                    )
                    
            except Exception as error:
                database_initial_score = 0
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【{self.island_id}号岛屿】 评估库中初始 ideas 的整体质量时遇到错误：\n"
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
        models = self.ideasearcher.get_models()
        model_assess_window_size = self.ideasearcher.get_model_assess_window_size()
        model_assess_initial_score = self.ideasearcher.get_model_assess_initial_score()
        model_assess_save_result = self.ideasearcher.get_model_assess_save_result()
        assert models is not None
        for _ in range(len(models)):
            self.model_recent_scores.append(
                np.full((model_assess_window_size,), model_assess_initial_score)
            )
            self.model_scores.append(model_assess_initial_score)
            
        if model_assess_save_result:
            self.scores_of_models = np.zeros((1+self.interaction_num, len(models)))
            self.scores_of_models_length = 0
            self.scores_of_models_x_axis = np.linspace(
                start = 0, 
                stop = self.interaction_num, 
                num = 1 + self.interaction_num, 
                endpoint = True
            )
            self._sync_model_score_result()
          
        self._sync_score_sheet()
        self._sync_similar_num_list()
        
        
    def link_samplers(
        self,
        samplers
    )-> None:
        
        self.samplers = samplers
        
        
    def fuel(
        self,
        additional_interaction_num: int,
    ):
        
        with self.lock:
            
            if additional_interaction_num <= 0:
                raise RuntimeError(f"【{self.island_id}号岛屿】 fuel 动作的参数 `additional_interaction_num` 应为一正整数，不应为{additional_interaction_num}！")
            
            self.status = "Running"
            
            models = self.ideasearcher.get_models()

            self.interaction_num += additional_interaction_num
            
            if self.assess_on:
                
                assess_interval = self.ideasearcher.get_assess_interval()
                
                new_assess_result_ndarray = np.zeros((1 + (self.interaction_num // assess_interval),))
                new_assess_result_ndarray[:len(self.assess_result_ndarray)] = self.assess_result_ndarray
                self.assess_result_ndarray = new_assess_result_ndarray
                self.assess_result_ndarray_x_axis = np.linspace(
                    start = 0, 
                    stop = self.interaction_num, 
                    num = 1 + (self.interaction_num // assess_interval), 
                    endpoint = True
                )
                self._sync_database_assess_result(
                    is_initialization = False,
                    get_database_score_success = True,
                )
                
            model_assess_save_result = self.ideasearcher.get_model_assess_save_result()
            
            if model_assess_save_result:
                new_scores_of_models = np.zeros((1+self.interaction_num, len(models)))
                new_scores_of_models[:len(self.scores_of_models)] = self.scores_of_models
                self.scores_of_models = new_scores_of_models
                self.scores_of_models_length = 0
                self.scores_of_models_x_axis = np.linspace(
                    start = 0, 
                    stop = self.interaction_num, 
                    num = 1 + self.interaction_num, 
                    endpoint = True
                )
                self._sync_model_score_result()     
    
    # ----------------------------- 外部调用动作 ----------------------------- 
    
    def get_status(self):
        with self.lock:
            return self.status
        
    
    def get_examples(
        self
    )-> Optional[list[Tuple]]:
        
        with self.lock:
            
            if self.status == "Terminated":
                return None
            self.interaction_count += 1
            with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【{self.island_id}号岛屿】 已分发交互次数为： {self.interaction_count} ，"
                            f"还剩 {self.interaction_num-self.interaction_count} 次！"
                        ),
                    )
            
            if self.mutation_on:
                
                assert self.mutation_interval is not None
                
                if self.interaction_count % self.mutation_interval == 0:
                    self._mutate()
            
            if self.crossover_on:
                
                assert self.crossover_interval is not None
                
                if self.interaction_count % self.crossover_interval == 0:
                    self._crossover()
            
            self._check_threshold()
            
            if len(self.ideas) == 0:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = "【{self.island_id}号岛屿】 发生异常： ideas 列表为空！",
                    )
                exit()
                
            probabilities = np.array([idea.score for idea in self.ideas])
            probabilities /= self.ideasearcher.get_sample_temperature()
            max_value = np.max(probabilities)
            probabilities = np.exp(probabilities - max_value)
            probabilities /= np.array(self.idea_similar_nums)
            probabilities /= np.sum(probabilities)
            
            selected_indices = self.random_generator.choice(
                a = len(self.ideas),
                size = min(len(self.ideas), self.ideasearcher.get_examples_num()),
                replace = False, # 不允许重复选择同一个元素
                p = probabilities,
            )
            
            selected_examples = []
            for i in selected_indices:
                selected_index = int(i)
                example_idea = self.ideas[selected_index]
                
                if self.similarity_sys_info_on:
                    
                    assert self.similarity_sys_info_prompts is not None
                    
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
                    example_idea.path,
                ))

            return selected_examples
        
        
    def get_model(self) -> Tuple[str, float]:
        
        with self.lock:
            
            self._show_model_scores()
            
            models = self.ideasearcher.get_models()
            model_temperatures = self.ideasearcher.get_model_temperatures()
            assert models is not None
            assert model_temperatures is not None
            
            probabilities = np.array(self.model_scores) / self.ideasearcher.get_model_sample_temperature()
            max_value = np.max(probabilities)
            probabilities = np.exp(probabilities - max_value)
            probabilities /= np.sum(probabilities)
            
            selected_index = self.random_generator.choice(
                a = len(models), 
                p = probabilities,
            )
            
            selected_model_name = models[selected_index]
            selected_model_temperature = model_temperatures[selected_index]
            
            return selected_model_name, selected_model_temperature
        
        
    def update_model_score(
        self,
        score_result: list[float], 
        model: str,
        model_temperature: float,
    )-> None:
        
        with self.lock:
            
            if self.assess_on:
                
                self.assess_interaction_count += 1
                
                assert self.assess_interval is not None
                
                if self.assess_interaction_count % self.assess_interval == 0:
                    self._assess_database()
            
            index = 0
            
            models = self.ideasearcher.get_models()
            model_temperatures = self.ideasearcher.get_model_temperatures()
            p = self.ideasearcher.get_model_assess_average_order()
            model_assess_save_result = self.ideasearcher.get_model_assess_save_result()
            assert models is not None
            assert model_temperatures is not None
            
            
            while index < len(models):
                
                if models[index] == model and model_temperatures[index] == model_temperature:
                    self.model_recent_scores[index][:-1] = self.model_recent_scores[index][1:]
                    scores_array = np.array(score_result)
                    if p != np.inf:
                        self.model_recent_scores[index][-1] = (np.mean(np.abs(scores_array) ** p)) ** (1 / p)
                        self.model_scores[index] = (np.mean(np.abs(self.model_recent_scores[index]) ** p)) ** (1 / p)
                    else:
                        self.model_recent_scores[index][-1] = np.max(scores_array)
                        self.model_scores[index] = np.max(self.model_recent_scores[index])
                    with self.console_lock:    
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = (
                                f"【{self.island_id}号岛屿】 模型 {model}(T={model_temperature:.2f}) 此轮评分为 {self.model_recent_scores[index][-1]:.2f} ，"
                                f"其总评分已被更新为 {self.model_scores[index]:.2f} ！"
                            ),
                        )
                    if model_assess_save_result:
                        self._sync_model_score_result()
                    return
                
                index += 1
                
            with self.console_lock:    
                append_to_file(
                    file_path = self.diary_path,
                    content_str = f"【{self.island_id}号岛屿】 出现错误！未知的模型名称及温度： {model}(T={model_temperature:.2f}) ！",
                )
                
            exit()  
        
        
    def receive_result(
        self, 
        result: list[Tuple[str, float, str]], 
        evaluator_id: int,
        source: str,
    )-> None:
        
        with self.lock:
    
            for idea_content, score, info in result:
                
                self._store_idea(
                    idea = idea_content,
                    score = score,
                    info = info,
                    source = source,
                )
                
            with self.console_lock:    
                append_to_file(
                    file_path = self.diary_path,
                    content_str = f"【{self.island_id}号岛屿】 {evaluator_id} 号评估器递交的 {len(result)} 个新文件已评分并加入{self.island_id}号岛屿。",
                )
            
            self._sync_score_sheet()
            self._sync_similar_num_list()
            
            
    # ----------------------------- 内部调用动作 -----------------------------           
            
    def _sync_score_sheet(self):
        
        program_name = self.ideasearcher.get_program_name()
        
        start_time = perf_counter()
        
        score_sheet = {
            basename(idea.path): {
                "score": idea.score,
                "info": idea.info if idea.info is not None else "",
                "source": idea.source if idea.source is not None else "未知",
            }
            for idea in self.ideas
        }

        with open(self.path + "score_sheet.json", 'w', encoding='utf-8') as json_file:
            json.dump(score_sheet, json_file, ensure_ascii=False, indent=4)
            
        end_time = perf_counter()
        total_time = end_time - start_time
        
        with self.console_lock:   
            append_to_file(
                file_path = self.diary_path,
                content_str = f"【{self.island_id}号岛屿】  {program_name} 的 score sheet 已更新，用时{total_time:.2f}秒！",
            )
            
    
    def _sync_similar_num_list(self):
        
        start_time = perf_counter()
        
        similarity_distance_func = self.ideasearcher.get_similarity_distance_func()
        similarity_threshold = self.ideasearcher.get_similarity_threshold()
        assert similarity_distance_func is not None
        
        self.idea_similar_nums = []
        
        if similarity_distance_func == self.default_similarity_distance_func:

            scores = np.array([idea.score for idea in self.ideas])
            diff_matrix = np.abs(scores[:, None] - scores[None, :])

            for i, idea_i in enumerate(self.ideas):
                score_similar_indices = set(np.where(diff_matrix[i] <= similarity_threshold)[0])
                content_equal_indices = set(
                    j for j, idea_j in enumerate(self.ideas)
                    if idea_j.content == idea_i.content
                )
                total_similar_indices = score_similar_indices | content_equal_indices

                self.idea_similar_nums.append(len(total_similar_indices))
            
        else:
            for i, idea_i in enumerate(self.ideas):
                
                similar_count = 0
                
                for j, idea_j in enumerate(self.ideas):
                    
                    if i == j or idea_i.content == idea_j.content:
                        similar_count += 1
                        continue
                    
                    assert idea_i.content is not None
                    assert idea_j.content is not None
                        
                    score_diff = similarity_distance_func(idea_i.content, idea_j.content) 
                    if score_diff <= similarity_threshold: similar_count += 1
                        
                self.idea_similar_nums.append(similar_count)
            
        end_time = perf_counter()
        total_time = end_time - start_time
            
        with self.console_lock:
            append_to_file(
                file_path = self.diary_path,
                content_str = f"【{self.island_id}号岛屿】 成功将idea_similar_nums与ideas同步，用时{total_time:.2f}秒！",
            )
    
    
    def _check_threshold(self):
        if self.interaction_count >= self.interaction_num:
            with self.console_lock:
                append_to_file(
                    file_path = self.diary_path,
                    content_str = f"【{self.island_id}号岛屿】 采样次数已分发完毕，IdeaSearch将在各采样器完成手头任务后结束。",
                )
            self.status = "Terminated"
    
    
    def _assess_database(self)-> None:
        
        start_time = perf_counter()
        
        with self.console_lock:
            append_to_file(
                file_path = self.diary_path,
                content_str = f"【{self.island_id}号岛屿】 现在开始评估岛内 ideas 的整体质量！",
            )
            
        ideas: list[str] = []
        scores: list[float] = []
        infos: list[Optional[str]] = []
        for idea in self.ideas:
            
            assert idea.content is not None
            assert idea.score is not None
            
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
            
            end_time = perf_counter()
            total_time = end_time - start_time
            
            with self.console_lock:
                append_to_file(
                    file_path = self.diary_path,
                    content_str = f"【{self.island_id}号岛屿】 当前岛屿 ideas 的整体质量评分为：{database_score:.2f}！评估用时：{total_time:.2f}秒。",
                )
                
        except Exception as error:
            database_score = 0
            with self.console_lock:
                append_to_file(
                    file_path = self.diary_path,
                    content_str = (
                        f"【{self.island_id}号岛屿】 评估岛内 ideas 的整体质量时遇到错误：\n"
                        f"{error}"
                    ),
                )
                
        self.assess_result_ndarray[self.assess_result_ndarray_length] = database_score
        self.assess_result_ndarray_length += 1
        
        self._sync_database_assess_result(
            is_initialization = False,
            get_database_score_success = get_database_score_success,
        )

    
    def _mutate(self)-> None:
        
        with self.console_lock:
            append_to_file(
                file_path = self.diary_path,
                content_str = "【{self.island_id}号岛屿】 现在开始进行单体突变！",
            )
            
        assert self.mutation_num is not None
        
        program_name = self.ideasearcher.get_program_name()
        evaluate_func = self.ideasearcher.get_evaluate_func()
        handover_threshold = self.ideasearcher.get_hand_over_threshold()
        assert evaluate_func is not None
        
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
                assert selected_idea.content is not None
                mutated_idea = self.mutation_func(selected_idea.content)
                if not isinstance(mutated_idea, str):
                    with self.console_lock:
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = (
                                f"【{self.island_id}号岛屿】 调用 {program_name} 的单体突变函数时发生错误："
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
                            f"【{self.island_id}号岛屿】 第 {index+1} 次单体突变在运行 mutation_func 时发生了错误：\n"
                            f"{error}\n此轮单体突变意外终止！"
                        ),
                    )
                return
            
            try:
                score, info = evaluate_func(mutated_idea)
                
                if not isinstance(score, float):
                    with self.console_lock:
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = (
                                f"【{self.island_id}号岛屿】 调用 {program_name} 的评估函数时发生错误："
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
                                f"【{self.island_id}号岛屿】 调用 {program_name} 的评估函数时发生错误："
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
                                    f"【{self.island_id}号岛屿】 调用 {program_name} 的评估函数时发生错误："
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
                            f"【{self.island_id}号岛屿】 调用 {program_name} 的评估函数时发生错误：\n{error}"
                            "\n此轮单体突变意外终止！"
                        ),
                    )  
                return
            
            source = f"由 {basename(selected_idea.path)}({selected_idea.score:.2f}) 突变而来"
            
            if score >= handover_threshold:
            
                path = self._store_idea(
                    idea = mutated_idea,
                    score = score,
                    info = info, 
                    source = source,
                )
                
                if path is not None:
                    with self.console_lock:
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = (
                                f"【{self.island_id}号岛屿】 第 {index+1} 次单体突变："
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
                                f"【{self.island_id}号岛屿】 第 {index+1} 次单体突变发生了错误：\n"
                                f"{self._store_idea_error_message}\n此轮单体突变意外终止！"
                            ),
                        )
                    return
                
            else:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【{self.island_id}号岛屿】 第 {index+1} 次单体突变结果未达到入库分数阈值"
                            f"（{handover_threshold:.2f}分），已删除！"
                        ),
                    )
                
        with self.console_lock:
            append_to_file(
                file_path = self.diary_path,
                content_str = "【{self.island_id}号岛屿】 此轮单体突变已结束。",
            )
    
    
    def _crossover(self) -> None:
    
        with self.console_lock:
            append_to_file(
                file_path = self.diary_path,
                content_str = "【{self.island_id}号岛屿】 现在开始进行交叉变异！",
            )
            
        assert self.crossover_num is not None
        
        program_name = self.ideasearcher.get_program_name()
        evaluate_func = self.ideasearcher.get_evaluate_func()
        handover_threshold = self.ideasearcher.get_hand_over_threshold()
        assert evaluate_func is not None

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
                                f"【{self.island_id}号岛屿】 调用 {program_name} 的交叉变异函数时发生错误："
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
                            f"【{self.island_id}号岛屿】 第 {index+1} 次交叉变异在运行 crossover_func 时发生了错误：\n"
                            f"{error}"
                        ),
                    )
                continue
            
            try:
                score, info = evaluate_func(crossover_idea)
                
                if not isinstance(score, float):
                    with self.console_lock:
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = (
                                f"【{self.island_id}号岛屿】 调用 {program_name} 的评估函数时发生错误："
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
                                f"【{self.island_id}号岛屿】 调用 {program_name} 的评估函数时发生错误："
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
                                    f"【{self.island_id}号岛屿】 调用 {program_name} 的评估函数时发生错误："
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
                            f"【{self.island_id}号岛屿】 调用 {program_name} 的评估函数时发生错误：\n{error}"
                            "\n此轮交叉变异意外终止！"
                        ),
                    )  
                return
            
            source = f"由 {basename(parent_1.path)}({parent_1.score:.2f}) 和 {basename(parent_2.path)}({parent_2.score:.2f}) 交叉而来"

            if score >= handover_threshold:
                
                path = self._store_idea(
                    idea = crossover_idea,
                    score = score,
                    info = info,
                    source = source,
                )

                if path is not None:
                    with self.console_lock:
                        append_to_file(
                            file_path = self.diary_path,
                            content_str = (
                                f"【{self.island_id}号岛屿】 第 {index+1} 次交叉变异："
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
                                f"【{self.island_id}号岛屿】 第 {index+1} 次交叉变异发生了错误：\n"
                                f"{self._store_idea_error_message}\n此轮交叉变异意外终止！"
                            ),
                        )
                    return
                
            else:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【{self.island_id}号岛屿】 第 {index+1} 次交叉变异结果未达到入库分数阈值"
                            f"（{handover_threshold:.2f}分），已删除！"
                        ),
                    )

        with self.console_lock:
            append_to_file(
                file_path = self.diary_path,
                content_str = "【{self.island_id}号岛屿】 此轮交叉变异已结束。",
            )
        
        
    def _show_model_scores(self)-> None:
        
        with self.console_lock:
            
            models = self.ideasearcher.get_models()
            model_temperatures = self.ideasearcher.get_model_temperatures()
            assert models is not None
            assert model_temperatures is not None
            
            append_to_file(
                file_path = self.diary_path,
                content_str = f"【{self.island_id}号岛屿】 各模型目前评分情况如下：",
            )
            
            for index, model in enumerate(models):
                
                model_temperature = model_temperatures[index]
                
                append_to_file(
                    file_path = self.diary_path,
                    content_str = (
                        f"  {index+1}. {model}(T={model_temperature:.2f}): {self.model_scores[index]:.2f}"
                    ),
                )
    
    
    def _store_idea(
        self, 
        idea: str,
        evaluate_func: Optional[Callable[[str], Tuple[float, Optional[str]]]] = None,
        score: Optional[float] = None,
        info: Optional[str] = None,
        source: Optional[str] = None,
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
            score (Optional[float]): 预设的 idea 评分。
            info (Optional[str]): 与 idea 相关的附加信息。
            source (Optional[str]): idea 的来源。

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
                source = source,
            ))
                
            return path
        
        except Exception as error:
            self._store_idea_error_message = error
            return None
            
            
    def _get_new_idea_path(self)-> str:
        
        def generate_random_string(length=self.ideasearcher.get_idea_uid_length()):
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
        
        if self.interaction_num == 0: return
        
        assert self.assess_result_data_path is not None
        assert self.assess_result_pic_path is not None
        
        score_range = self.ideasearcher.get_score_range()
        
        np.savez_compressed(
            file = self.assess_result_data_path, 
            interaction_num = self.assess_result_ndarray_x_axis,
            database_scores = self.assess_result_ndarray,
        )
        
        point_num = len(self.assess_result_ndarray_x_axis)
        auto_markersize = self._get_auto_markersize(point_num)
        
        x_axis_range = (0, self.interaction_num)
        x_axis_range_expand_ratio = 0.08
        x_axis_range_delta = (x_axis_range[1] - x_axis_range[0]) * x_axis_range_expand_ratio
        x_axis_range = (x_axis_range[0] - x_axis_range_delta, x_axis_range[1] + x_axis_range_delta)
        
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.assess_result_ndarray_x_axis[:self.assess_result_ndarray_length], 
            self.assess_result_ndarray[:self.assess_result_ndarray_length], 
            label='Database Score', 
            color='dodgerblue', 
            marker='o',
            markersize = auto_markersize,
        )
        if self.assess_baseline is not None:
            plt.axhline(
                y=self.assess_baseline,
                color='red',
                linestyle='--',
                label='Baseline',
            )
        plt.title("Database Assessment")
        plt.xlabel("Total Interaction No.")
        plt.ylabel("Database Score")
        plt.xlim(x_axis_range)
        plt.ylim(score_range)
        plt.grid(True)
        plt.legend()
        plt.savefig(self.assess_result_pic_path)
        plt.close()
        
        if get_database_score_success:
            if is_initialization:
                append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【{self.island_id}号岛屿】 初始质量评估结束，"
                            f" {basename(self.assess_result_data_path)} 与 {basename(self.assess_result_pic_path)} 已更新！"
                        ),
                    )
            else:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【{self.island_id}号岛屿】 此轮质量评估结束，"
                            f" {basename(self.assess_result_data_path)} 与 {basename(self.assess_result_pic_path)} 已更新！"
                        ),
                    )
                    
                    
    def _sync_model_score_result(self):
        
        if self.interaction_num == 0: return
        
        model_assess_result_data_path = self.ideasearcher.get_model_assess_result_data_path()
        model_assess_result_pic_path = self.ideasearcher.get_model_assess_result_pic_path()
        models = self.ideasearcher.get_models()
        model_temperatures = self.ideasearcher.get_model_temperatures()
        score_range = self.ideasearcher.get_score_range()
        
        assert model_assess_result_data_path is not None
        assert model_assess_result_pic_path is not None
        assert models is not None
        assert model_temperatures is not None
        
        self.scores_of_models[self.scores_of_models_length] = self.model_scores
        self.scores_of_models_length += 1
        
        scores_of_models = self.scores_of_models.T
        
        scores_of_models_dict = {}
        for model_name, model_temperature, model_scores in zip(models, model_temperatures, scores_of_models):
            scores_of_models_dict[f"{model_name}(T={model_temperature:.2f})"] = model_scores
        
        np.savez_compressed(
            file=model_assess_result_data_path,
            interaction_num=self.scores_of_models_x_axis,
            **scores_of_models_dict
        )
        
        point_num = len(self.scores_of_models_x_axis)
        auto_markersize = self._get_auto_markersize(point_num)
        
        x_axis_range = (0, self.interaction_num)
        x_axis_range_expand_ratio = 0.08
        x_axis_range_delta = (x_axis_range[1] - x_axis_range[0]) * x_axis_range_expand_ratio
        x_axis_range = (x_axis_range[0] - x_axis_range_delta, x_axis_range[1] + x_axis_range_delta)

        plt.figure(figsize=(10, 6))
        for model_label, model_scores in scores_of_models_dict.items():
            plt.plot(
                self.scores_of_models_x_axis[:self.scores_of_models_length],
                model_scores[:self.scores_of_models_length],
                label=model_label,
                marker='o',
                markersize = auto_markersize,
            )
        plt.title("Model Scores")
        plt.xlabel("Interaction No.")
        plt.ylabel("Model Score")
        plt.xlim(x_axis_range)
        plt.ylim(score_range)
        plt.grid(True)
        plt.legend()
        plt.savefig(model_assess_result_pic_path)
        plt.close()
        
        with self.console_lock:
            append_to_file(
                file_path=self.diary_path,
                content_str=(
                    f"【{self.island_id}号岛屿】 "
                    f" {basename(model_assess_result_data_path)} 与 {basename(model_assess_result_pic_path)} 已更新！"
                ),
            )
            
            
    def _get_auto_markersize(
        self, 
        point_num: int
    )-> int:
        
        if point_num <= 20:
            auto_markersize = 8
        elif point_num <= 50:
            auto_markersize = 6
        elif point_num <= 100:
            auto_markersize = 4
        else:
            auto_markersize = 2
            
        return auto_markersize


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
            