import concurrent.futures
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
from typing import List
from typing import Dict
from os.path import basename
from src.utils import append_to_file
from src.utils import guarantee_path_exist
from src.API4LLMs.model_manager import ModelManager
from src.API4LLMs.get_answer import get_answer_online
from src.API4LLMs.get_answer import get_answer_local
from src.IdeaSearch.sampler import Sampler
from src.IdeaSearch.evaluator import Evaluator
from src.IdeaSearch.island import Island


class IdeaSearcher:
    
    # ----------------------------- IdeaSearhcer 初始化 ----------------------------- 
    
    def __init__(
        self
    ) -> None:
    
        self._program_name: Optional[str] = None
        self._prologue_section: Optional[str] = None
        self._epilogue_section: Optional[str] = None
        self._database_path: Optional[str] = None
        self._models: Optional[List[str]] = None
        self._model_temperatures: Optional[List[float]] = None
        self._evaluate_func: Optional[Callable[[str], Tuple[float, Optional[str]]]] = None
        self._score_range: Tuple[float, float] = (0.0, 100.0)
        self._hand_over_threshold: float = 0.0
        self._system_prompt: Optional[str] = None
        self._diary_path: Optional[str] = None
        self._api_keys_path: Optional[str] = None
        self._local_models_path: Optional[str] = None
        self._samplers_num: int = 3
        self._evaluators_num: int = 3
        self._examples_num: int = 3
        self._generate_num: int = 3
        self._sample_temperature: float = 50.0
        self._model_sample_temperature: float = 50.0
        self._assess_func: Optional[Callable[[List[str], List[float], List[Optional[str]]], float]] = None
        self._assess_interval: Optional[int] = None
        self._assess_baseline: Optional[float] = None
        self._assess_result_data_path: Optional[str] = None
        self._assess_result_pic_path: Optional[str] = None
        self._model_assess_window_size: int = 20
        self._model_assess_initial_score: float = 100.0
        self._model_assess_average_order: float = 1.0
        self._model_assess_save_result: bool = True
        self._model_assess_result_data_path: Optional[str] = None
        self._model_assess_result_pic_path: Optional[str] = None
        self._mutation_func: Optional[Callable[[str], str]] = None
        self._mutation_interval: Optional[int] = None
        self._mutation_num: Optional[int] = None
        self._mutation_temperature: Optional[float] = None
        self._crossover_func: Optional[Callable[[str, str], str]] = None
        self._crossover_interval: Optional[int] = None
        self._crossover_num: Optional[int] = None
        self._crossover_temperature: Optional[float] = None
        self._similarity_threshold: float = -1.0
        self._similarity_distance_func: Optional[Callable[[str, str], float]] = None
        self._similarity_sys_info_thresholds: Optional[List[int]] = None
        self._similarity_sys_info_prompts: Optional[List[str]] = None
        self._initialization_skip_evaluation: bool = True
        self._initialization_cleanse_threshold: float = -1.0
        self._delete_when_initial_cleanse: bool = False
        self._idea_uid_length: int = 6
        self._record_prompt_in_diary: bool = False
        self._filter_func: Optional[Callable[[str], str]] = None
        self._generation_bonus: float = 0.0


        def evaluate_func(
            idea: str,
        )-> Tuple[float, Optional[str]]:
            return 0.0, None
    
        def default_similarity_distance_func(idea1, idea2):
            return abs(evaluate_func(idea1)[0] - evaluate_func(idea2)[0])
    
        self._lock: Lock = Lock()
        self._console_lock: Lock = Lock()
        self._model_manager: ModelManager = ModelManager()
        self._islands: Dict[int, Island] = {}
        self._next_island_id: int = 1
        self._default_similarity_distance_func = default_similarity_distance_func

    # ----------------------------- 外部调用动作 ----------------------------- 
    
    def load_models(
        self
    )-> None:
    
        with self._lock:
    
            if self._api_keys_path is None and self._local_models_path is None:
                raise ValueError(
                    "【IdeaSearcher】 加载模型时发生错误："
                    " api keys path 和 local models path 中至少应有一个不为 None ！"
                )
                
            if self._api_keys_path is not None:
                self._model_manager.load_api_keys(self._api_keys_path)
                
            if self._local_models_path is not None:
                self._model_manager.load_local_models(self._local_models_path)
                
                
    def add_island(
        self,
    )-> int:
        
        with self._lock:
        
            missing_param = self._check_runnability()
            if missing_param is not None:
                raise RuntimeError(f"【IdeaSearcher】 参数`{missing_param}`未传入，在当前设置下无法进行 add_island 动作！")
        
            evaluators_num = self._evaluators_num
            samplers_num = self._samplers_num
            
            island_id = self._next_island_id
            self._next_island_id += 1
        
            island = Island(
                ideasearcher = self,
                island_id = island_id,
                default_similarity_distance_func = self._default_similarity_distance_func,
                console_lock = self._console_lock,
            )
            
            evaluators = [
                Evaluator(
                    ideasearcher = self,
                    evaluator_id = i + 1,
                    island = island,
                    console_lock = self._console_lock,
                )
                for i in range(evaluators_num)
            ]

            samplers = [
                Sampler(
                    ideasearcher = self,
                    sampler_id = i + 1,
                    island = island,
                    evaluators = evaluators,
                    console_lock = self._console_lock,
                )
                for i in range(samplers_num)
            ]
            
            island.link_samplers(samplers)
            
            self._islands[island_id] = island
            
            return island_id
            
    
    def run(
        self,
        additional_interaction_num: int,
    )-> None:

        with self._lock:
        
            missing_param = self._check_runnability()
            if missing_param is not None:
                raise RuntimeError(f"【IdeaSearcher】 参数`{missing_param}`未传入，在当前设置下无法进行 run 动作！")
        
            max_workers_num = 0
            for island_id in self._islands:
                island = self._islands[island_id]
                island.fuel(additional_interaction_num)
                max_workers_num += len(island.samplers)
                
            append_to_file(
                file_path = self._diary_path,
                content_str = f"【IdeaSearcher】 {self._program_name} 的 IdeaSearch 正在运行，此次运行每个岛屿会演化 {additional_interaction_num} 个 epoch ！"
            )
            
            with concurrent.futures.ThreadPoolExecutor(
                max_workers = max_workers_num
            ) as executor:
            
                futures = {executor.submit(sampler.run): (island_id, sampler.id)
                    for island_id in self._islands
                    for sampler in self._islands[island_id].samplers
                }
                for future in concurrent.futures.as_completed(futures):
                    island_id, sampler_id = futures[future]
                    try:
                        _ = future.result() 
                    except Exception as e:
                        append_to_file(
                            file_path = self._diary_path,
                            content_str = f"【IdeaSearcher】 {island_id}号岛屿的{sampler_id}号采样器在运行过程中出现错误：\n{e}\nIdeaSearch意外终止！",
                        )
                        exit()
                        
                        
    def repopulate_islands(
        self,
    )-> None:
    
        with self._lock:
        
            with self._console_lock:
                append_to_file(
                    file_path = self._diary_path,
                    content_str = f"【IdeaSearcher】 现在 ideas 开始在岛屿间重分布"
                )
            
            island_ids = self._islands.keys()
            
            island_ids = sorted(
                island_ids,
                key = lambda id: self._islands[id]._best_score,
                reverse = True,
            )
            
            N = len(island_ids)
            M = N // 2
            
            for index in range(M):
            
                island_to_colonize = self._islands[island_ids[index]]
                
                self._islands[island_ids[-index]].ideas = [
                    island_to_colonize._best_idea
                ]
                self._islands[island_ids[-index]].idea_similar_nums = [1]
                
            with self._console_lock:
                append_to_file(
                    file_path = self._diary_path,
                    content_str = f"【IdeaSearcher】 此次 ideas 在岛屿间的重分布已完成"
                )
                
                
    def shutdown_models(
        self,
    )-> None:

        with self._lock:
        
            self._model_manager.shutdown()
    
    
    def set_program_name(
        self,
        value: str,
    ) -> None:

        if not isinstance(value, str):
            raise TypeError(f"【IdeaSearcher】 参数`program_name`类型应为str，实为{str(type(value))}")

        with self._lock:
            self._program_name = value


    def set_prologue_section(
        self,
        value: str,
    ) -> None:

        if not isinstance(value, str):
            raise TypeError(f"【IdeaSearcher】 参数`prologue_section`类型应为str，实为{str(type(value))}")

        with self._lock:
            self._prologue_section = value


    def set_epilogue_section(
        self,
        value: str,
    ) -> None:

        if not isinstance(value, str):
            raise TypeError(f"【IdeaSearcher】 参数`epilogue_section`类型应为str，实为{str(type(value))}")

        with self._lock:
            self._epilogue_section = value


    def set_database_path(
        self,
        value: str,
    ) -> None:

        if not isinstance(value, str):
            raise TypeError(f"【IdeaSearcher】 参数`database_path`类型应为str，实为{str(type(value))}")

        with self._lock:
            self._database_path = value


    def set_models(
        self,
        value: List[str],
    ) -> None:

        if not hasattr(value, "__iter__") and not isinstance(value, str):
            raise TypeError(f"【IdeaSearcher】 参数`models`类型应为List[str]，实为{str(type(value))}")

        with self._lock:
            self._models = value


    def set_model_temperatures(
        self,
        value: List[float],
    ) -> None:

        if not hasattr(value, "__iter__") and not isinstance(value, str):
            raise TypeError(f"【IdeaSearcher】 参数`model_temperatures`类型应为List[float]，实为{str(type(value))}")

        with self._lock:
            self._model_temperatures = value


    def set_evaluate_func(
        self,
        value: Callable[[str], Tuple[float, Optional[str]]],
    ) -> None:

        if not callable(value):
            raise TypeError(f"【IdeaSearcher】 参数`evaluate_func`类型应为Callable[[str], Tuple[float, Optional[str]]]，实为{str(type(value))}")

        with self._lock:
            self._evaluate_func = value


    def set_score_range(
        self,
        value: Tuple[float, float],
    ) -> None:

        if not isinstance(value, tuple):
            raise TypeError(f"【IdeaSearcher】 参数`score_range`类型应为Tuple[float, float]，实为{str(type(value))}")

        with self._lock:
            self._score_range = value


    def set_hand_over_threshold(
        self,
        value: float,
    ) -> None:

        if not isinstance(value, float):
            raise TypeError(f"【IdeaSearcher】 参数`hand_over_threshold`类型应为float，实为{str(type(value))}")

        with self._lock:
            self._hand_over_threshold = value


    def set_system_prompt(
        self,
        value: Optional[str],
    ) -> None:

        if not (value is None or isinstance(value, str)):
            raise TypeError(f"【IdeaSearcher】 参数`system_prompt`类型应为Optional[str]，实为{str(type(value))}")

        with self._lock:
            self._system_prompt = value


    def set_diary_path(
        self,
        value: Optional[str],
    ) -> None:

        if not (value is None or isinstance(value, str)):
            raise TypeError(f"【IdeaSearcher】 参数`diary_path`类型应为Optional[str]，实为{str(type(value))}")

        with self._lock:
            self._diary_path = value


    def set_api_keys_path(
        self,
        value: Optional[str],
    ) -> None:

        if not (value is None or isinstance(value, str)):
            raise TypeError(f"【IdeaSearcher】 参数`api_keys_path`类型应为Optional[str]，实为{str(type(value))}")

        with self._lock:
            self._api_keys_path = value


    def set_local_models_path(
        self,
        value: Optional[str],
    ) -> None:

        if not (value is None or isinstance(value, str)):
            raise TypeError(f"【IdeaSearcher】 参数`local_models_path`类型应为Optional[str]，实为{str(type(value))}")

        with self._lock:
            self._local_models_path = value


    def set_samplers_num(
        self,
        value: int,
    ) -> None:

        if not isinstance(value, int):
            raise TypeError(f"【IdeaSearcher】 参数`samplers_num`类型应为int，实为{str(type(value))}")

        with self._lock:
            self._samplers_num = value


    def set_evaluators_num(
        self,
        value: int,
    ) -> None:

        if not isinstance(value, int):
            raise TypeError(f"【IdeaSearcher】 参数`evaluators_num`类型应为int，实为{str(type(value))}")

        with self._lock:
            self._evaluators_num = value


    def set_examples_num(
        self,
        value: int,
    ) -> None:

        if not isinstance(value, int):
            raise TypeError(f"【IdeaSearcher】 参数`examples_num`类型应为int，实为{str(type(value))}")

        with self._lock:
            self._examples_num = value


    def set_generate_num(
        self,
        value: int,
    ) -> None:

        if not isinstance(value, int):
            raise TypeError(f"【IdeaSearcher】 参数`generate_num`类型应为int，实为{str(type(value))}")

        with self._lock:
            self._generate_num = value


    def set_sample_temperature(
        self,
        value: float,
    ) -> None:

        if not isinstance(value, float):
            raise TypeError(f"【IdeaSearcher】 参数`sample_temperature`类型应为float，实为{str(type(value))}")

        with self._lock:
            self._sample_temperature = value


    def set_model_sample_temperature(
        self,
        value: float,
    ) -> None:

        if not isinstance(value, float):
            raise TypeError(f"【IdeaSearcher】 参数`model_sample_temperature`类型应为float，实为{str(type(value))}")

        with self._lock:
            self._model_sample_temperature = value


    def set_assess_func(
        self,
        value: Optional[Callable[[List[str], List[float], List[Optional[str]]], float]],
    ) -> None:

        if not (value is None or callable(value)):
            raise TypeError(f"【IdeaSearcher】 参数`assess_func`类型应为Optional[Callable[[List[str], List[float], List[Optional[str]]], float]]，实为{str(type(value))}")

        with self._lock:
            self._assess_func = value


    def set_assess_interval(
        self,
        value: Optional[int],
    ) -> None:

        if not (value is None or isinstance(value, int)):
            raise TypeError(f"【IdeaSearcher】 参数`assess_interval`类型应为Optional[int]，实为{str(type(value))}")

        with self._lock:
            self._assess_interval = value


    def set_assess_baseline(
        self,
        value: Optional[float],
    ) -> None:

        if not (value is None or isinstance(value, float)):
            raise TypeError(f"【IdeaSearcher】 参数`assess_baseline`类型应为Optional[float]，实为{str(type(value))}")

        with self._lock:
            self._assess_baseline = value


    def set_assess_result_data_path(
        self,
        value: Optional[str],
    ) -> None:

        if not (value is None or isinstance(value, str)):
            raise TypeError(f"【IdeaSearcher】 参数`assess_result_data_path`类型应为Optional[str]，实为{str(type(value))}")

        with self._lock:
            self._assess_result_data_path = value


    def set_assess_result_pic_path(
        self,
        value: Optional[str],
    ) -> None:

        if not (value is None or isinstance(value, str)):
            raise TypeError(f"【IdeaSearcher】 参数`assess_result_pic_path`类型应为Optional[str]，实为{str(type(value))}")

        with self._lock:
            self._assess_result_pic_path = value


    def set_model_assess_window_size(
        self,
        value: int,
    ) -> None:

        if not isinstance(value, int):
            raise TypeError(f"【IdeaSearcher】 参数`model_assess_window_size`类型应为int，实为{str(type(value))}")

        with self._lock:
            self._model_assess_window_size = value


    def set_model_assess_initial_score(
        self,
        value: float,
    ) -> None:

        if not isinstance(value, float):
            raise TypeError(f"【IdeaSearcher】 参数`model_assess_initial_score`类型应为float，实为{str(type(value))}")

        with self._lock:
            self._model_assess_initial_score = value


    def set_model_assess_average_order(
        self,
        value: float,
    ) -> None:

        if not isinstance(value, float):
            raise TypeError(f"【IdeaSearcher】 参数`model_assess_average_order`类型应为float，实为{str(type(value))}")

        with self._lock:
            self._model_assess_average_order = value


    def set_model_assess_save_result(
        self,
        value: bool,
    ) -> None:

        if not isinstance(value, bool):
            raise TypeError(f"【IdeaSearcher】 参数`model_assess_save_result`类型应为bool，实为{str(type(value))}")

        with self._lock:
            self._model_assess_save_result = value


    def set_model_assess_result_data_path(
        self,
        value: Optional[str],
    ) -> None:

        if not (value is None or isinstance(value, str)):
            raise TypeError(f"【IdeaSearcher】 参数`model_assess_result_data_path`类型应为Optional[str]，实为{str(type(value))}")

        with self._lock:
            self._model_assess_result_data_path = value


    def set_model_assess_result_pic_path(
        self,
        value: Optional[str],
    ) -> None:

        if not (value is None or isinstance(value, str)):
            raise TypeError(f"【IdeaSearcher】 参数`model_assess_result_pic_path`类型应为Optional[str]，实为{str(type(value))}")

        with self._lock:
            self._model_assess_result_pic_path = value


    def set_mutation_func(
        self,
        value: Optional[Callable[[str], str]],
    ) -> None:

        if not (value is None or callable(value)):
            raise TypeError(f"【IdeaSearcher】 参数`mutation_func`类型应为Optional[Callable[[str], str]]，实为{str(type(value))}")

        with self._lock:
            self._mutation_func = value


    def set_mutation_interval(
        self,
        value: Optional[int],
    ) -> None:

        if not (value is None or isinstance(value, int)):
            raise TypeError(f"【IdeaSearcher】 参数`mutation_interval`类型应为Optional[int]，实为{str(type(value))}")

        with self._lock:
            self._mutation_interval = value


    def set_mutation_num(
        self,
        value: Optional[int],
    ) -> None:

        if not (value is None or isinstance(value, int)):
            raise TypeError(f"【IdeaSearcher】 参数`mutation_num`类型应为Optional[int]，实为{str(type(value))}")

        with self._lock:
            self._mutation_num = value


    def set_mutation_temperature(
        self,
        value: Optional[float],
    ) -> None:

        if not (value is None or isinstance(value, float)):
            raise TypeError(f"【IdeaSearcher】 参数`mutation_temperature`类型应为Optional[float]，实为{str(type(value))}")

        with self._lock:
            self._mutation_temperature = value


    def set_crossover_func(
        self,
        value: Optional[Callable[[str, str], str]],
    ) -> None:

        if not (value is None or callable(value)):
            raise TypeError(f"【IdeaSearcher】 参数`crossover_func`类型应为Optional[Callable[[str, str], str]]，实为{str(type(value))}")

        with self._lock:
            self._crossover_func = value


    def set_crossover_interval(
        self,
        value: Optional[int],
    ) -> None:

        if not (value is None or isinstance(value, int)):
            raise TypeError(f"【IdeaSearcher】 参数`crossover_interval`类型应为Optional[int]，实为{str(type(value))}")

        with self._lock:
            self._crossover_interval = value


    def set_crossover_num(
        self,
        value: Optional[int],
    ) -> None:

        if not (value is None or isinstance(value, int)):
            raise TypeError(f"【IdeaSearcher】 参数`crossover_num`类型应为Optional[int]，实为{str(type(value))}")

        with self._lock:
            self._crossover_num = value


    def set_crossover_temperature(
        self,
        value: Optional[float],
    ) -> None:

        if not (value is None or isinstance(value, float)):
            raise TypeError(f"【IdeaSearcher】 参数`crossover_temperature`类型应为Optional[float]，实为{str(type(value))}")

        with self._lock:
            self._crossover_temperature = value


    def set_similarity_threshold(
        self,
        value: float,
    ) -> None:

        if not isinstance(value, float):
            raise TypeError(f"【IdeaSearcher】 参数`similarity_threshold`类型应为float，实为{str(type(value))}")

        with self._lock:
            self._similarity_threshold = value


    def set_similarity_distance_func(
        self,
        value: Optional[Callable[[str, str], float]],
    ) -> None:

        if not (value is None or callable(value)):
            raise TypeError(f"【IdeaSearcher】 参数`similarity_distance_func`类型应为Optional[Callable[[str, str], float]]，实为{str(type(value))}")

        with self._lock:
            self._similarity_distance_func = value


    def set_similarity_sys_info_thresholds(
        self,
        value: Optional[List[int]],
    ) -> None:

        if not (value is None or (hasattr(value, "__iter__") and not isinstance(value, str))):
            raise TypeError(f"【IdeaSearcher】 参数`similarity_sys_info_thresholds`类型应为Optional[List[int]]，实为{str(type(value))}")

        with self._lock:
            self._similarity_sys_info_thresholds = value


    def set_similarity_sys_info_prompts(
        self,
        value: Optional[List[str]],
    ) -> None:

        if not (value is None or (hasattr(value, "__iter__") and not isinstance(value, str))):
            raise TypeError(f"【IdeaSearcher】 参数`similarity_sys_info_prompts`类型应为Optional[List[str]]，实为{str(type(value))}")

        with self._lock:
            self._similarity_sys_info_prompts = value


    def set_initialization_skip_evaluation(
        self,
        value: bool,
    ) -> None:

        if not isinstance(value, bool):
            raise TypeError(f"【IdeaSearcher】 参数`initialization_skip_evaluation`类型应为bool，实为{str(type(value))}")

        with self._lock:
            self._initialization_skip_evaluation = value


    def set_initialization_cleanse_threshold(
        self,
        value: float,
    ) -> None:

        if not isinstance(value, float):
            raise TypeError(f"【IdeaSearcher】 参数`initialization_cleanse_threshold`类型应为float，实为{str(type(value))}")

        with self._lock:
            self._initialization_cleanse_threshold = value


    def set_delete_when_initial_cleanse(
        self,
        value: bool,
    ) -> None:

        if not isinstance(value, bool):
            raise TypeError(f"【IdeaSearcher】 参数`delete_when_initial_cleanse`类型应为bool，实为{str(type(value))}")

        with self._lock:
            self._delete_when_initial_cleanse = value


    def set_idea_uid_length(
        self,
        value: int,
    ) -> None:

        if not isinstance(value, int):
            raise TypeError(f"【IdeaSearcher】 参数`idea_uid_length`类型应为int，实为{str(type(value))}")

        with self._lock:
            self._idea_uid_length = value


    def set_record_prompt_in_diary(
        self,
        value: bool,
    ) -> None:

        if not isinstance(value, bool):
            raise TypeError(f"【IdeaSearcher】 参数`record_prompt_in_diary`类型应为bool，实为{str(type(value))}")

        with self._lock:
            self._record_prompt_in_diary = value


    def set_filter_func(
        self,
        value: Optional[Callable[[str], str]],
    ) -> None:

        if not (value is None or callable(value)):
            raise TypeError(f"【IdeaSearcher】 参数`filter_func`类型应为Optional[Callable[[str], str]]，实为{str(type(value))}")

        with self._lock:
            self._filter_func = value


    def set_generation_bonus(
        self,
        value: float,
    ) -> None:

        if not isinstance(value, float):
            raise TypeError(f"【IdeaSearcher】 参数`generation_bonus`类型应为float，实为{str(type(value))}")

        with self._lock:
            self._generation_bonus = value


    def get_program_name(
        self,
    )-> Optional[str]:
        
            return self._program_name


    def get_prologue_section(
        self,
    )-> Optional[str]:
        
            return self._prologue_section


    def get_epilogue_section(
        self,
    )-> Optional[str]:
        
            return self._epilogue_section


    def get_database_path(
        self,
    )-> Optional[str]:
        
            return self._database_path


    def get_models(
        self,
    )-> Optional[List[str]]:
        
            return self._models


    def get_model_temperatures(
        self,
    )-> Optional[List[float]]:
        
            return self._model_temperatures


    def get_evaluate_func(
        self,
    )-> Optional[Callable[[str], Tuple[float, Optional[str]]]]:
        
            return self._evaluate_func


    def get_score_range(
        self,
    )-> Tuple[float, float]:
        
            return self._score_range


    def get_hand_over_threshold(
        self,
    )-> float:
        
            return self._hand_over_threshold


    def get_system_prompt(
        self,
    )-> Optional[str]:
        
            return self._system_prompt


    def get_diary_path(
        self,
    )-> Optional[str]:
        
            return self._diary_path


    def get_api_keys_path(
        self,
    )-> Optional[str]:
        
            return self._api_keys_path


    def get_local_models_path(
        self,
    )-> Optional[str]:
        
            return self._local_models_path


    def get_samplers_num(
        self,
    )-> int:
        
            return self._samplers_num


    def get_evaluators_num(
        self,
    )-> int:
        
            return self._evaluators_num


    def get_examples_num(
        self,
    )-> int:
        
            return self._examples_num


    def get_generate_num(
        self,
    )-> int:
        
            return self._generate_num


    def get_sample_temperature(
        self,
    )-> float:
        
            return self._sample_temperature


    def get_model_sample_temperature(
        self,
    )-> float:
        
            return self._model_sample_temperature


    def get_assess_func(
        self,
    )-> Optional[Callable[[List[str], List[float], List[Optional[str]]], float]]:
        
            return self._assess_func


    def get_assess_interval(
        self,
    )-> Optional[int]:
        
            return self._assess_interval


    def get_assess_baseline(
        self,
    )-> Optional[float]:
        
            return self._assess_baseline


    def get_assess_result_data_path(
        self,
    )-> Optional[str]:
        
            return self._assess_result_data_path


    def get_assess_result_pic_path(
        self,
    )-> Optional[str]:
        
            return self._assess_result_pic_path


    def get_model_assess_window_size(
        self,
    )-> int:
        
            return self._model_assess_window_size


    def get_model_assess_initial_score(
        self,
    )-> float:
        
            return self._model_assess_initial_score


    def get_model_assess_average_order(
        self,
    )-> float:
        
            return self._model_assess_average_order


    def get_model_assess_save_result(
        self,
    )-> bool:
        
            return self._model_assess_save_result


    def get_model_assess_result_data_path(
        self,
    )-> Optional[str]:
        
            return self._model_assess_result_data_path


    def get_model_assess_result_pic_path(
        self,
    )-> Optional[str]:
        
            return self._model_assess_result_pic_path


    def get_mutation_func(
        self,
    )-> Optional[Callable[[str], str]]:
        
            return self._mutation_func


    def get_mutation_interval(
        self,
    )-> Optional[int]:
        
            return self._mutation_interval


    def get_mutation_num(
        self,
    )-> Optional[int]:
        
            return self._mutation_num


    def get_mutation_temperature(
        self,
    )-> Optional[float]:
        
            return self._mutation_temperature


    def get_crossover_func(
        self,
    )-> Optional[Callable[[str, str], str]]:
        
            return self._crossover_func


    def get_crossover_interval(
        self,
    )-> Optional[int]:
        
            return self._crossover_interval


    def get_crossover_num(
        self,
    )-> Optional[int]:
        
            return self._crossover_num


    def get_crossover_temperature(
        self,
    )-> Optional[float]:
        
            return self._crossover_temperature


    def get_similarity_threshold(
        self,
    )-> float:
        
            return self._similarity_threshold


    def get_similarity_distance_func(
        self,
    )-> Optional[Callable[[str, str], float]]:
        
            return self._similarity_distance_func


    def get_similarity_sys_info_thresholds(
        self,
    )-> Optional[List[int]]:
        
            return self._similarity_sys_info_thresholds


    def get_similarity_sys_info_prompts(
        self,
    )-> Optional[List[str]]:
        
            return self._similarity_sys_info_prompts


    def get_initialization_skip_evaluation(
        self,
    )-> bool:
        
            return self._initialization_skip_evaluation


    def get_initialization_cleanse_threshold(
        self,
    )-> float:
        
            return self._initialization_cleanse_threshold


    def get_delete_when_initial_cleanse(
        self,
    )-> bool:
        
            return self._delete_when_initial_cleanse


    def get_idea_uid_length(
        self,
    )-> int:
        
            return self._idea_uid_length


    def get_record_prompt_in_diary(
        self,
    )-> bool:
        
            return self._record_prompt_in_diary


    def get_filter_func(
        self,
    )-> Optional[Callable[[str], str]]:
        
            return self._filter_func


    def get_generation_bonus(
        self,
    )-> float:
        
            return self._generation_bonus

    # ----------------------------- 内部调用动作 ----------------------------- 
    
    def _is_online_model(
        self,
        model_name: str,
    )-> bool:
    
        return self._model_manager.is_online_model(model_name)
        
    def _get_online_model_instance(
        self,
        model_name: str,
    )-> Tuple[str, str, str]:
        return self._model_manager.get_online_model_instance(model_name)


    def _get_local_model_instance(
        self,
        model_name: str,
    )-> int:
        return self._model_manager.get_local_model_instance(model_name)

    
    def _get_answer(
        self,
        model_name : str, 
        model_temperature : Optional[float],
        system_prompt: str,
        prompt : str,
    ):
        
        if self._is_online_model(model_name):
            api_key, base_url, model = self._get_online_model_instance(model_name)
            
            return get_answer_online(
                api_key = api_key,
                base_url = base_url,
                model = model,
                temperature = model_temperature,
                system_prompt = system_prompt,
                prompt = prompt,
            )
        
        else:
            port = self._get_local_model_instance(model_name)
            
            return get_answer_local(
                port = port,
                temperature = model_temperature,
                system_prompt = system_prompt,
                prompt = prompt,
            )
            
            
    def _check_runnability(
        self,
    )-> Optional[str]:
        
        missing_param = None
        
        if self._program_name is None:
            missing_param = "program_name"
            
        if self._prologue_section is None:
            missing_param = "prologue_section"
            
        if self._epilogue_section is None:
            missing_param = "epilogue_section"
            
        if self._database_path is None:
            missing_param = "database_path"
            
        if self._models is None:
            missing_param = "models"
            
        if self._model_temperatures is None:
            missing_param = "model_temperatures"
            
        if self._evaluate_func is None:
            missing_param = "evaluate_func"
               
        if self._assess_func is not None:
            if self._assess_interval is None:
                missing_param = "assess_interval"
        
        if self._mutation_func is not None:
            if self._mutation_interval is None:
                missing_param = "mutation_interval"
            if self._mutation_num is None:
                missing_param = "mutation_num"
            if self._mutation_temperature is None:
                missing_param = "mutation_temperature"
                
        if self._crossover_func is not None:
            if self._crossover_interval is None:
                missing_param = "crossover_interval"
            if self._crossover_num is None:
                missing_param = "crossover_num"
            if self._crossover_temperature is None:
                missing_param = "crossover_temperature"
                
        if missing_param is not None: return missing_param
        assert self._database_path is not None
        
        if self._similarity_distance_func is None:
            self._similarity_distance_func = self._default_similarity_distance_func
        
        if self._diary_path is None:
            self._diary_path = self._database_path + "log/diary.txt"
            
        if self._system_prompt is None:
            self._system_prompt = "You're a helpful assistant."
            
        if self._assess_func is not None:
            if self._assess_result_data_path is None:
                self._assess_result_data_path = self._database_path + "data/database_assessment.npz"
            if self._assess_result_pic_path is None:
                self._assess_result_pic_path = self._database_path + "pic/database_assessment.png"
                
        if self._model_assess_save_result:
            if self._model_assess_result_data_path is None:
                self._model_assess_result_data_path = self._database_path + "data/model_scores.npz"
            if self._model_assess_result_pic_path is None:
                self._model_assess_result_pic_path = self._database_path + "pic/model_scores.png"
                
        return None
