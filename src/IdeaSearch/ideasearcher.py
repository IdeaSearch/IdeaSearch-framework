import concurrent.futures
import os
import json
import math
import gettext
import random
import shutil
import string
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import perf_counter
from threading import Lock
from typing import Tuple
from typing import Callable
from typing import Optional
from typing import List
from typing import Dict
from pathlib import Path
from os.path import basename
from os.path import sep as seperator
from pywheels.file_tools import append_to_file
from pywheels.file_tools import guarantee_file_exist
from pywheels.file_tools import clear_file
from pywheels.llm_tools import ModelManager
from .utils import get_auto_markersize
from .utils import default_assess_func
from .utils import make_boltzmann_choice
from .sampler import Sampler
from .evaluator import Evaluator
from .island import Island


# 国际化设置
_LOCALE_DIR = Path(__file__).parent / "locales"
_DOMAIN = "ideasearch"
gettext.bindtextdomain(_DOMAIN, _LOCALE_DIR)
gettext.textdomain(_DOMAIN)


class IdeaSearcher:
    
    # ----------------------------- IdeaSearhcer 初始化 ----------------------------- 

    def __init__(
        self
    ) -> None:
    
        # 国际化设置
        self._language: str = 'zh_CN'
        self._translation = gettext.translation(_DOMAIN, _LOCALE_DIR, languages=[self._language], fallback=True)
        self._ = self._translation.gettext
    
        self._program_name: Optional[str] = None
        self._prologue_section: Optional[str] = None
        self._epilogue_section: Optional[str] = None
        self._database_path: Optional[str] = None
        self._models: Optional[List[str]] = None
        self._model_temperatures: Optional[List[float]] = None
        self._evaluate_func: Optional[Callable[[str], Tuple[float, Optional[str]]]] = None
        self._api_keys_path: Optional[str] = None
        self._score_range: Tuple[float, float] = (0.0, 100.0)
        self._hand_over_threshold: float = 0.0
        self._system_prompt: Optional[str] = None
        self._diary_path: Optional[str] = None
        self._samplers_num: int = 3
        self._evaluators_num: int = 3
        self._examples_num: int = 3
        self._generate_num: int = 1
        self._sample_temperature: float = 50.0
        self._model_sample_temperature: float = 50.0
        self._assess_func: Optional[Callable[[List[str], List[float], List[Optional[str]]], float]] = default_assess_func
        self._assess_interval: Optional[int] = 1
        self._assess_baseline: Optional[float] = 60.0
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
        self._load_idea_skip_evaluation: bool = True
        self._initialization_cleanse_threshold: float = -1.0
        self._delete_when_initial_cleanse: bool = False
        self._idea_uid_length: int = 6
        self._record_prompt_in_diary: bool = False
        self._filter_func: Optional[Callable[[str], str]] = None
        self._generation_bonus: float = 0.0
        self._backup_path: Optional[str] = None
        self._backup_on: bool = True
        self._generate_prompt_func: Optional[Callable[[List[str], List[float], List[Optional[str]]], str]] = None
        self._explicit_prompt_structure: bool = False

        self._lock: Lock = Lock()
        self._user_lock: Lock = Lock()
        self._console_lock: Lock = Lock()

        def evaluate_func_example(
            idea: str,
        )-> Tuple[float, Optional[str]]:
            return 0.0, None
    
        # This will not be really executed, just its address used. 
        def default_similarity_distance_func(idea1, idea2):
            return abs(evaluate_func_example(idea1)[0] - evaluate_func_example(idea2)[0])
            
        self._default_similarity_distance_func = default_similarity_distance_func

        self._random_generator = np.random.default_rng()
        self._model_manager: ModelManager = ModelManager()
        
        self._next_island_id: int = 1
        self._islands: Dict[int, Island] = {}
        
        self._database_assessment_config_loaded = False
        self._model_score_config_loaded = False
        
        self._total_interaction_num = 0
        self._first_time_run = True
        self._first_time_add_island = True
        self._assigned_idea_uids = set()
        self._recorded_ideas = []
        self._recorded_idea_names = set()
        self._added_initial_idea_no: int = 1
        self._models_loaded_from_api_keys_json: bool = False


    def __dir__(self):
        # 返回类的所有属性和方法
        return [
            attr for attr in super().__dir__() 
            if not attr.startswith('_') and not callable(getattr(self, attr))
        ] + [
            'run', 'load_models', 'shutdown_models', 'get_best_score', 
            'get_best_idea', 'add_island', 'delete_island', 
            'repopulate_islands', 'get_idea_uid', 'record_ideas_in_backup',
            'assess_database', 'get_model'
        ]

    # ----------------------------- 核心功能 ----------------------------- 
    
    # ⭐️ Important
    def run(
        self,
        additional_interaction_num: int,
    )-> None:
    
        """
        Run the IdeaSearch process for each island, extending their evolution by the given number of epochs.
        This method performs internal initialization, launches all samplers, and handles logging and error management.
        """

        with self._user_lock:
        
            missing_param = self._check_runnability()
            if missing_param is not None:
                raise RuntimeError(self._("【IdeaSearcher】 参数`%s`未传入，在当前设置下无法进行 run 动作！") % missing_param)
                
            if not self._models_loaded_from_api_keys_json:
                self._load_models()
                self._models_loaded_from_api_keys_json = True
                
            diary_path = self._diary_path
            database_path = self._database_path
            program_name = self._program_name
            assert diary_path is not None
            assert database_path is not None
            assert program_name is not None
                
            append_to_file(
                file_path = diary_path,
                content = self._("【IdeaSearcher】 %s 的 IdeaSearch 正在运行，此次运行每个岛屿会演化 %d 个 epoch ！") % (program_name, additional_interaction_num)
            )
                
            self._total_interaction_num += len(self._islands) * additional_interaction_num
            
            for island_id in self._islands:
                island = self._islands[island_id]
                island.fuel(additional_interaction_num)
                
            if self._first_time_run:
                self._load_database_assessment_config()
                self._load_model_score_config()        
                self._first_time_run = False
            else:
                if self._assess_on:
                    self._expand_database_assessment_range()
                if self._model_assess_save_result:
                    self._expand_model_score_range()
                
            max_workers_num = 0
            for island_id in self._islands:
                island = self._islands[island_id]
                max_workers_num += len(island.samplers)
            
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
                            file_path = diary_path,
                            content = self._("【IdeaSearcher】 %d号岛屿的%d号采样器在运行过程中出现错误：\n%s\nIdeaSearch意外终止！") % (island_id, sampler_id, e),
                        )
                        exit()


    def _check_runnability(
        self,
    )-> Optional[str]:
        
        missing_param = None
        
        if self._database_path is None:
            missing_param = "database_path"
        
        if self._program_name is None:
            missing_param = "program_name"
            
        if self._prologue_section is None and self._generate_prompt_func is None:
            missing_param = "prologue_section"

        if self._epilogue_section is None and self._generate_prompt_func is None:
            missing_param = "epilogue_section"
            
        if self._evaluate_func is None:
            missing_param = "evaluate_func"
           
        if self._models is None:
            missing_param = "models"

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
        
        database_path = self._database_path
        models = self._models
        assert database_path is not None
        assert models is not None
        
        default_model_temperature = 0.9
        if self._model_temperatures is None:
            self._model_temperatures = [default_model_temperature] * len(models)
        
        if self._similarity_distance_func is None:
            self._similarity_distance_func = self._default_similarity_distance_func
        
        if self._diary_path is None:
            self._diary_path = f"{database_path}{seperator}log{seperator}diary.txt"
       
        if self._system_prompt is None:
            self._system_prompt = "You're a helpful assistant."
    
        if self._assess_func is not None:
            if self._assess_result_data_path is None:
                self._assess_result_data_path = f"{database_path}{seperator}data{seperator}database_assessment.npz"
            if self._assess_result_pic_path is None:
                self._assess_result_pic_path = f"{database_path}{seperator}pic{seperator}database_assessment.png"
                
        if self._model_assess_save_result:
            if self._model_assess_result_data_path is None:
                self._model_assess_result_data_path = f"{database_path}{seperator}data{seperator}model_scores.npz"
            if self._model_assess_result_pic_path is None:
                self._model_assess_result_pic_path = f"{database_path}{seperator}pic{seperator}model_scores.png"
                
        if self._backup_path is None:
            self._backup_path = f"{database_path}{seperator}ideas{seperator}backup"
                
        return None

    # ----------------------------- API4LLMs 相关 ----------------------------- 

    def _load_models(
        self
    )-> None:
    
        """
        Load API keys for all models from the specified configuration file.
        Parameter  `api_keys_path` must be set before calling this method; otherwise, a ValueError will be raised.
        """
    
        if self._api_keys_path is None:
            raise ValueError(
                self._("【IdeaSearcher】 加载模型时发生错误： api keys path 不应为 None ！")
            )
            
        self._model_manager.load_api_keys(self._api_keys_path)
  

    def _get_answer(
        self,
        model_name : str, 
        model_temperature : Optional[float],
        system_prompt: str,
        prompt : str,
    ):
        
        return self._model_manager.get_answer(
            model_name = model_name,
            prompt = prompt,
            model_temperature = model_temperature,
            system_prompt = system_prompt,
        )

    # ----------------------------- Ideas 管理相关 ----------------------------- 
    
    # ⭐️ Important
    def get_best_score(
        self,
    )-> float:
    
        """
        Return the highest score among all ideas across islands.
        Raises a RuntimeError if any required parameter is missing or if there are no ideas available.
        """
    
        with self._user_lock:
        
            missing_param = self._check_runnability()
            if missing_param is not None:
                raise RuntimeError(self._("【IdeaSearcher】 参数`%s`未传入，在当前设置下无法进行 get_best_score 动作！") % missing_param)
            
            scores: list[float] = []
            
            for island_id in self._islands:
                island = self._islands[island_id]
                for idea in island.ideas:
                    assert idea.score is not None
                    scores.append(idea.score)
                    
            if not scores: raise RuntimeError(self._("【IdeaSearcher】 目前各岛屿均无 ideas ，无法进行 get_best_score 动作！"))
                
            return max(scores)


    # ⭐️ Important
    def get_best_idea(
        self,
    )-> str:
    
        """
        Return the content of the idea with the highest score across all islands.
        Raises a RuntimeError if any required parameter is missing or if there are no ideas available.
        """
    
        with self._user_lock:
        
            missing_param = self._check_runnability()
            if missing_param is not None:
                raise RuntimeError(self._("【IdeaSearcher】 参数`%s`未传入，在当前设置下无法进行 get_best_idea 动作！") % missing_param)
        
            scores: list[float] = []
            ideas: list[str] = []
            
            for island_id in self._islands:
                island = self._islands[island_id]
                for idea in island.ideas:
                    assert idea.score is not None
                    assert idea.content is not None
                    scores.append(idea.score)
                    ideas.append(idea.content)
                    
            if not scores: raise RuntimeError(self._("【IdeaSearcher】 目前各岛屿均无 ideas ，无法进行 get_best_idea 动作！"))
                
            return ideas[scores.index(max(scores))]


    # ⭐️ Important
    def _add_initial_ideas(
        self,
        initial_ideas: List[str],
    )-> None:
    
        database_path = self._database_path
        if database_path is None:
            raise RuntimeError(self._(
                "【IdeaSearcher】 添加初始 ideas 失败：应先设置数据库路径！"
            ))
            
        initial_ideas_path = f"{database_path}{seperator}ideas{seperator}initial_ideas"
        guarantee_file_exist(
            file_path = initial_ideas_path,
            is_directory = True,
        )
        
        for initial_idea in initial_ideas:
        
            with open(
                file = f"{initial_ideas_path}{seperator}added_initial_idea{self._added_initial_idea_no}.idea",
                mode = "w",
                encoding = "UTF-8",
            ) as file:
            
                file.write(initial_idea)
                
            self._added_initial_idea_no += 1
    
    
    def add_initial_ideas(
        self,
        initial_ideas: List[str],
    ):
        with self._user_lock: self._add_initial_ideas(initial_ideas)

    
    def get_idea_uid(
        self,
    )-> str:
    
        with self._lock:
        
            idea_uid_length = self._idea_uid_length
            
            idea_uid = ''.join(random.choices(
                population = string.ascii_lowercase, 
                k = idea_uid_length,
            ))
            
            while idea_uid in self._assigned_idea_uids:
                idea_uid = ''.join(random.choices(
                    population = string.ascii_lowercase, 
                    k = idea_uid_length,
                ))
                
            self._assigned_idea_uids.add(idea_uid)
            
            return idea_uid


    def record_ideas_in_backup(
        self,
        ideas_to_record,
    ):
    
        with self._lock:
        
            database_path = self._database_path
            backup_path = self._backup_path
            backup_on = self._backup_on
            assert database_path is not None
            assert backup_path is not None
            
            if not backup_on: return
            
            guarantee_file_exist(f"{backup_path}{seperator}score_sheet_backup.json")
        
            for idea in ideas_to_record:
                
                if basename(idea.path) not in self._recorded_idea_names:
                    
                    self._recorded_ideas.append(idea)
                    self._recorded_idea_names.add(basename(idea.path))
                
                    with open(
                        file = f"{backup_path}{seperator}{basename(idea.path)}",
                        mode = "w",
                        encoding = "UTF-8",
                    ) as file:

                        file.write(idea.content)
                        
            score_sheet = {
                basename(idea.path): {
                    "score": idea.score,
                    "info": idea.info if idea.info is not None else "",
                    "source": idea.source,
                    "level": idea.level,
                    "created_at": idea.created_at,
                }
                for idea in self._recorded_ideas
            }

            with open(
                file = f"{backup_path}{seperator}score_sheet_backup.json", 
                mode = "w", 
                encoding = "UTF-8",
            ) as file:
                
                json.dump(
                    obj = score_sheet, 
                    fp = file, 
                    ensure_ascii = False,
                    indent = 4
                )

    # ----------------------------- 岛屿相关 ----------------------------- 

    # ⭐️ Important
    def add_island(
        self,
    )-> int:
    
        """
        Add a new island to the IdeaSearcher system and return its island_id.
        If this is the first island added, the method will also perform necessary initialization,
        such as clearing diary logs, removing old idea directories, and resetting backup folders.
        Raises RuntimeError if essential parameters are missing.
        """
        
        with self._user_lock:
        
            missing_param = self._check_runnability()
            if missing_param is not None:
                raise RuntimeError(self._("【IdeaSearcher】 参数`%s`未传入，在当前设置下无法进行 add_island 动作！") % missing_param)
                
            diary_path = self._diary_path
            database_path = self._database_path
            backup_path = self._backup_path
            backup_on = self._backup_on
            assert diary_path is not None
            assert database_path is not None
            assert backup_path is not None
                
            if self._first_time_add_island:
            
                clear_file(diary_path)
                
                if backup_on:
                    guarantee_file_exist(f"{backup_path}{seperator}score_sheet_backup.json")
                    shutil.rmtree(f"{backup_path}")
                    guarantee_file_exist(f"{backup_path}{seperator}score_sheet_backup.json")
                    
                for item in os.listdir(f"{database_path}{seperator}ideas"):
                    full_path = os.path.join(f"{database_path}{seperator}ideas", item)
                    if os.path.isdir(full_path) and item.startswith('island'):
                        shutil.rmtree(full_path)
                self._first_time_add_island = False
        
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
            
            island.load_ideas_from("initial_ideas")
            island.link_samplers(samplers)
            
            self._islands[island_id] = island
            
            return island_id

           
    # ⭐️ Important 
    def delete_island(
        self,
        island_id: int,
    )-> int:

        """
        Delete the island with the given island_id from the IdeaSearcher system.
        Returns 1 if deletion is successful, or 0 if the island_id does not exist.
        """
    
        with self._user_lock:
            
            if island_id in self._islands:
                del self._islands[island_id]
                return 1
                
            else:
                return 0

    
    # ⭐️ Important
    def repopulate_islands(
        self,
    )-> None:
    
        """
        Redistribute ideas among islands by colonization from top-performing islands to lower-ranked ones.

        This method sorts all islands by their _best_score in descending order, then copies the best ideas from the top half 
        of islands to the bottom half to promote idea sharing and improve overall search performance.

        Logs the start and completion of the redistribution process to the diary file.

        It helps to prevent local optima stagnation by enabling migration of high-quality ideas across islands.
        """
    
        diary_path = self._diary_path
        assert diary_path is not None
        
        with self._user_lock:
        
            with self._console_lock:
                append_to_file(
                    file_path = diary_path,
                    content = self._("【IdeaSearcher】 现在 ideas 开始在岛屿间重分布")
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
                assert island_to_colonize._best_idea is not None
                
                self._islands[island_ids[-index]].accept_colonization(
                    [island_to_colonize._best_idea]
                )
                
            with self._console_lock:
                append_to_file(
                    file_path = diary_path,
                    content = self._("【IdeaSearcher】 此次 ideas 在岛屿间的重分布已完成")
                )

    # ----------------------------- Model Score 相关 ----------------------------- 
    
    def _load_model_score_config(
        self,
    )-> None:
        
        models = self._models
        model_assess_save_result = self._model_assess_save_result
        model_assess_window_size = self._model_assess_window_size
        model_assess_initial_score = self._model_assess_initial_score
        assert models is not None
    
        self._model_recent_scores = []
        self._model_scores = []
        
        for _ in range(len(models)):
            self._model_recent_scores.append(
                np.full((model_assess_window_size,), model_assess_initial_score)
            )
            self._model_scores.append(model_assess_initial_score)
            
        if model_assess_save_result:
            self._scores_of_models = np.zeros((1+self._total_interaction_num, len(models)))
            self._scores_of_models_length = 0
            self._scores_of_models_x_axis = np.linspace(
                start = 0, 
                stop = self._total_interaction_num, 
                num = 1 + self._total_interaction_num, 
                endpoint = True
            )
            self._sync_model_score_result()


    def _expand_model_score_range(
        self,
    )-> None:
    
        models = self._models
        assert models is not None

        new_scores_of_models = np.zeros((1+self._total_interaction_num, len(models)))
        new_scores_of_models[:len(self._scores_of_models)] = self._scores_of_models
        self._scores_of_models = new_scores_of_models
        
        self._scores_of_models_x_axis = np.linspace(
            start = 0, 
            stop = self._total_interaction_num, 
            num = 1 + self._total_interaction_num, 
            endpoint = True
        )


    def update_model_score(
        self,
        score_result: list[float], 
        model: str,
        model_temperature: float,
    )-> None:
        
        with self._lock:
            
            diary_path = self._diary_path
            assert diary_path is not None
            
            index = 0
            
            models = self._models
            model_temperatures = self._model_temperatures
            p = self._model_assess_average_order
            model_assess_save_result = self._model_assess_save_result
            assert models is not None
            assert model_temperatures is not None
            
            
            while index < len(models):
                
                if models[index] == model and model_temperatures[index] == model_temperature:
                    self._model_recent_scores[index][:-1] = self._model_recent_scores[index][1:]
                    scores_array = np.array(score_result)
                    if p != np.inf:
                        self._model_recent_scores[index][-1] = (np.mean(np.abs(scores_array) ** p)) ** (1 / p)
                        self._model_scores[index] = (np.mean(np.abs(self._model_recent_scores[index]) ** p)) ** (1 / p)
                    else:
                        self._model_recent_scores[index][-1] = np.max(scores_array)
                        self._model_scores[index] = np.max(self._model_recent_scores[index])
                    with self._console_lock:    
                        append_to_file(
                            file_path = diary_path,
                            content = self._("【IdeaSearcher】 模型 %s(T=%.2f) 此轮评分为 %.2f ，其总评分已被更新为 %.2f ！") % (model, model_temperature, self._model_recent_scores[index][-1], self._model_scores[index]),
                        )
                    if model_assess_save_result:
                        self._sync_model_score_result()
                    return
                
                index += 1
                
            with self._console_lock:    
                append_to_file(
                    file_path = diary_path,
                    content = self._("【IdeaSearcher】 出现错误！未知的模型名称及温度： %s(T=%.2f) ！") % (model, model_temperature),
                )
                
            exit()


    def _sync_model_score_result(self):
    
        if self._total_interaction_num == 0: return
        
        diary_path = self._diary_path
        model_assess_result_data_path = self._model_assess_result_data_path
        model_assess_result_pic_path = self._model_assess_result_pic_path
        models = self._models
        model_temperatures = self._model_temperatures
        score_range = self._score_range
        
        assert diary_path is not None
        assert model_assess_result_data_path is not None
        assert model_assess_result_pic_path is not None
        assert models is not None
        assert model_temperatures is not None
        
        self._scores_of_models[self._scores_of_models_length] = self._model_scores
        self._scores_of_models_length += 1
        
        scores_of_models = self._scores_of_models.T
        
        scores_of_models_dict = {}
        for model_name, model_temperature, model_scores in zip(models, model_temperatures, scores_of_models):
            scores_of_models_dict[f"{model_name}(T={model_temperature:.2f})"] = model_scores
        
        np.savez_compressed(
            file = model_assess_result_data_path,
            interaction_num = self._scores_of_models_x_axis,
            **scores_of_models_dict
        )
        
        point_num = len(self._scores_of_models_x_axis)
        auto_markersize = get_auto_markersize(point_num)
        
        range_expand_ratio = 0.08
        x_axis_range = (0, self._total_interaction_num)
        x_axis_range_delta = (x_axis_range[1] - x_axis_range[0]) * range_expand_ratio
        x_axis_range = (
            int(math.floor(x_axis_range[0] - x_axis_range_delta)), 
            int(math.ceil(x_axis_range[1] + x_axis_range_delta))
        )
        score_range_delta = (score_range[1] - score_range[0]) * range_expand_ratio
        score_range = (score_range[0] - score_range_delta, score_range[1] + score_range_delta)

        plt.figure(figsize=(10, 6))
        for model_label, model_scores in scores_of_models_dict.items():
            plt.plot(
                self._scores_of_models_x_axis[:self._scores_of_models_length],
                model_scores[:self._scores_of_models_length],
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
        
        with self._console_lock:
            append_to_file(
                file_path=diary_path,
                content=(
                    f"【IdeaSearcher】 "
                    f" {basename(model_assess_result_data_path)} 与 {basename(model_assess_result_pic_path)} 已更新！"
                ),
            )

                        
    def get_model(
        self
    )-> Tuple[str, float]:
        
        with self._lock:
            
            self._show_model_scores()
            
            models = self._models
            model_temperatures = self._model_temperatures
            model_sample_temperature = self._model_sample_temperature
            assert models is not None
            assert model_temperatures is not None
            assert model_sample_temperature is not None
            
            selected_index = make_boltzmann_choice(
                energies = self._model_scores,
                temperature = model_sample_temperature,
            )
            assert isinstance(selected_index, int)
            
            selected_model_name = models[selected_index]
            selected_model_temperature = model_temperatures[selected_index]
            
            return selected_model_name, selected_model_temperature

       
    def _show_model_scores(
        self
    )-> None:
        
        diary_path = self._diary_path
        models = self._models
        model_temperatures = self._model_temperatures
        
        assert diary_path is not None
        assert models is not None
        assert model_temperatures is not None
            
        with self._console_lock:
            
            append_to_file(
                file_path = diary_path,
                content = self._("【IdeaSearcher】 各模型目前评分情况如下："),
            )
            for index, model in enumerate(models):
                
                model_temperature = model_temperatures[index]
                
                append_to_file(
                    file_path = diary_path,
                    content = (
                        f"  {index+1}. {model}(T={model_temperature:.2f}): {self._model_scores[index]:.2f}"
                    ),
                )

    # ----------------------------- Database Assessment 相关 ----------------------------- 
            
    def _load_database_assessment_config(
        self,
    )-> None:
    
        diary_path = self._diary_path
        assess_func = self._assess_func
        assess_interval = self._assess_interval
        assess_result_data_path = self._assess_result_data_path
        assess_result_pic_path = self._assess_result_pic_path
        
        assert diary_path is not None
        assert assess_result_data_path is not None
        assert assess_result_pic_path is not None
        
        if assess_func is not None:
        
            assert assess_interval is not None

            self._assess_on = True
            self._assess_interaction_count = 0
            
            self._assess_result_ndarray = np.zeros((1 + (self._total_interaction_num // assess_interval),))
            self._assess_result_ndarray_length = 1
            self._assess_result_ndarray_x_axis = np.linspace(
                start = 0, 
                stop = self._total_interaction_num, 
                num = 1 + (self._total_interaction_num // assess_interval), 
                endpoint = True
            )
            
            guarantee_file_exist(assess_result_data_path)
            guarantee_file_exist(assess_result_pic_path)
            
            ideas: list[str] = []
            scores: list[float] = []
            infos: list[Optional[str]] = []
                            
            for island_id in self._islands:
                island = self._islands[island_id]
                for current_idea in island.ideas:
                    
                    assert current_idea.content is not None
                    assert current_idea.score is not None
                    
                    ideas.append(current_idea.content)
                    scores.append(current_idea.score)
                    infos.append(current_idea.info)
                    
            get_database_initial_score_success = False
            
            try:
                database_initial_score = assess_func(
                    ideas,
                    scores,
                    infos,
                )
                get_database_initial_score_success = True
                with self._console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content = self._("【IdeaSearcher】 初始 ideas 的整体质量评分为：%.2f！") % database_initial_score,
                    )
                    
            except Exception as error:
                database_initial_score = 0.0
                with self._console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content = self._("【IdeaSearcher】 评估库中初始 ideas 的整体质量时遇到错误：\n%s") % error,
                    )
                    
            self._assess_result_ndarray[0] = database_initial_score
            self._sync_database_assessment_result(
                is_initialization = True,
                get_database_score_success = get_database_initial_score_success,
            )
            
        else:
            self._assess_on = False

            
    def _expand_database_assessment_range(
        self,
    )-> None:
    
        assess_interval = self._assess_interval
        assert assess_interval is not None
    
        new_assess_result_ndarray = np.zeros((1 + (self._total_interaction_num // assess_interval),))
        new_assess_result_ndarray[:len(self._assess_result_ndarray)] = self._assess_result_ndarray
        self._assess_result_ndarray = new_assess_result_ndarray
        
        self._assess_result_ndarray_x_axis = np.linspace(
            start = 0, 
            stop = self._total_interaction_num, 
            num = 1 + (self._total_interaction_num // assess_interval), 
            endpoint = True
        )  


    def assess_database(
        self,
    )-> None:
        
        with self._lock:
        
            if not self._assess_on: return
        
            diary_path = self._diary_path
            assess_func = self._assess_func
            assess_interval = self._assess_interval
            
            assert diary_path is not None
            assert assess_func is not None
            assert assess_interval is not None
        
            self._assess_interaction_count += 1
            if self._assess_interaction_count % assess_interval != 0: return

            start_time = perf_counter()
            
            with self._console_lock:
                append_to_file(
                    file_path = diary_path,
                    content = self._("【IdeaSearcher】 现在开始评估数据库中 ideas 的整体质量！"),
                )
                
            ideas: list[str] = []
            scores: list[float] = []
            infos: list[Optional[str]] = []
            
            for island_id in self._islands:
                island = self._islands[island_id]
                for idea in island.ideas:
                    
                    assert idea.content is not None
                    assert idea.score is not None
                    
                    ideas.append(idea.content)
                    scores.append(idea.score)
                    infos.append(idea.info)
                
            get_database_score_success = False
            try:
                database_score = assess_func(
                    ideas,
                    scores,
                    infos,
                )
                get_database_score_success = True
                
                end_time = perf_counter()
                total_time = end_time - start_time
                
                with self._console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content = self._("【IdeaSearcher】 数据库中 ideas 的整体质量评分为：%.2f！评估用时：%.2f秒。") % (database_score, total_time),
                    )
                    
            except Exception as error:
                database_score = 0
                with self._console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content = self._("【IdeaSearcher】 评估库中 ideas 的整体质量时遇到错误：\n%s") % error,
                    )
                    
            self._assess_result_ndarray[self._assess_result_ndarray_length] = database_score
            self._assess_result_ndarray_length += 1
            
            self._sync_database_assessment_result(
                is_initialization = False,
                get_database_score_success = get_database_score_success,
            )


    def _sync_database_assessment_result(
        self,
        is_initialization: bool,
        get_database_score_success: bool,
    )-> None:
    
        if self._total_interaction_num == 0: return
        
        diary_path = self._diary_path
        score_range = self._score_range
        assess_result_data_path = self._assess_result_data_path
        assess_result_pic_path = self._assess_result_pic_path
        assess_baseline = self._assess_baseline
        
        assert diary_path is not None
        assert assess_result_data_path is not None
        assert assess_result_pic_path is not None
        
        np.savez_compressed(
            file = assess_result_data_path, 
            interaction_num = self._assess_result_ndarray_x_axis,
            database_scores = self._assess_result_ndarray,
        )
        
        point_num = len(self._assess_result_ndarray_x_axis)
        auto_markersize = get_auto_markersize(point_num)
        
        range_expand_ratio = 0.08
        x_axis_range = (0, self._total_interaction_num)
        x_axis_range_delta = (x_axis_range[1] - x_axis_range[0]) * range_expand_ratio
        x_axis_range = (
            int(math.floor(x_axis_range[0] - x_axis_range_delta)), 
            int(math.ceil(x_axis_range[1] + x_axis_range_delta))
        )
        score_range_delta = (score_range[1] - score_range[0]) * range_expand_ratio
        score_range = (score_range[0] - score_range_delta, score_range[1] + score_range_delta)
        
        plt.figure(figsize=(10, 6))
        plt.plot(
            self._assess_result_ndarray_x_axis[:self._assess_result_ndarray_length], 
            self._assess_result_ndarray[:self._assess_result_ndarray_length], 
            label='Database Score', 
            color='dodgerblue', 
            marker='o',
            markersize = auto_markersize,
        )
        if assess_baseline is not None:
            plt.axhline(
                y = assess_baseline,
                color = "red",
                linestyle = "--",
                label = "Baseline",
            )
        plt.title("Database Assessment")
        plt.xlabel("Total Interaction No.")
        plt.ylabel("Database Score")
        plt.xlim(x_axis_range)
        plt.ylim(score_range)
        plt.grid(True)
        plt.legend()
        plt.savefig(assess_result_pic_path)
        plt.close()
        
        if get_database_score_success:
            if is_initialization:
                append_to_file(
                        file_path = diary_path,
                        content = self._("【IdeaSearcher】 初始质量评估结束， %s 与 %s 已更新！") % (basename(assess_result_data_path), basename(assess_result_pic_path)),
                    )
            else:
                with self._console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content = self._("【IdeaSearcher】 此轮质量评估结束， %s 与 %s 已更新！") % (basename(assess_result_data_path), basename(assess_result_pic_path)),
                    )

    # ----------------------------- Helper 拓展相关 ----------------------------- 
            
    def bind_helper(
        self,
        helper: object,
    )-> None:
    
        with self._lock:
        
            if self._database_path is None:
            
                raise RuntimeError(self._(
                    "【IdeaSearcher】 绑定 helper 失败：应先设置数据库路径！"
                ))
        
            if not hasattr(helper, "prologue_section"):
            
                raise ValueError(self._(
                    "【IdeaSearcher】 绑定 helper 失败：helper 缺失属性 `prologue_section` ！"
                ))
                
            if not hasattr(helper, "epilogue_section"):
            
                raise ValueError(self._(
                    "【IdeaSearcher】 绑定 helper 失败：helper 缺失属性 `epilogue_section` ！"
                ))
                
            if not hasattr(helper, "evaluate_func"):
            
                raise ValueError(self._(
                    "【IdeaSearcher】 绑定 helper 失败：helper 缺失属性 `evaluate_func` ！"
                ))
                
            self._prologue_section = helper.prologue_section # type: ignore
            self._epilogue_section = helper.epilogue_section # type: ignore
            self._evaluate_function = helper.evaluate_func # type: ignore
            if hasattr(helper, "initial_ideas"): self._add_initial_ideas(helper.initial_ideas) # type: ignore
            if hasattr(helper, "assess_func"): self._mutation_function = helper.assess_func # type: ignore
            if hasattr(helper, "mutation_func"): self._mutation_function = helper.mutation_func # type: ignore
            if hasattr(helper, "crossover_func"): self._crossover_function = helper.crossover_func # type: ignore

    # ----------------------------- Getters and Setters ----------------------------- 
    
    def set_language(
        self,
        value: str,
    ) -> None:
    
        """
        Set the parameter `language` to the given value, if it is of the type str.
        This parameter sets the language for the user interface and translations; currently, only 'zh_CN' and 'en' are supported.
        The shorthand 'zh' will be converted to 'zh_CN'.
        """

        if not isinstance(value, str):
            raise TypeError(self._("【IdeaSearcher】 参数`language`类型应为str，实为%s") % str(type(value)))

        with self._user_lock:
            if value == "zh":
                value = "zh_CN"
            if value == "zh_TW":
                raise ValueError(self._("【IdeaSearcher】 语言`zh_TW`不受支持，请使用`zh_CN`代替。"))
            self._language = value
            self._translation = gettext.translation(_DOMAIN, _LOCALE_DIR, languages=[self._language], fallback=True)
            self._ = self._translation.gettext

        
    # ⭐️ Important
    def set_program_name(
        self,
        value: str,
    )-> None:
    
        """
        ⭐️ Important
        Set the parameter `program_name` to the given value, if it is of the type str.
        This parameter is the name of the current project.
        This parameter is important and must be set manually by the user.
        """

        if not isinstance(value, str):
            raise TypeError(self._("【IdeaSearcher】 参数`program_name`类型应为str，实为%s") % str(type(value)))

        with self._user_lock:
            self._program_name = value


    # ⭐️ Important
    def set_prologue_section(
        self,
        value: str,
    )-> None:
    
        """
        ⭐️ Important
        Set the parameter `prologue_section` to the given value, if it is of the type str.
        This parameter holds the leading text snippet of every prompt.
        This parameter is important and must be set manually by the user.
        """

        if not isinstance(value, str):
            raise TypeError(self._("【IdeaSearcher】 参数`prologue_section`类型应为str，实为%s") % str(type(value)))

        with self._user_lock:
            self._prologue_section = value


    # ⭐️ Important
    def set_epilogue_section(
        self,
        value: str,
    )-> None:
    
        """
        ⭐️ Important
        Set the parameter `epilogue_section` to the given value, if it is of the type str.
        This parameter holds the ending text snippet of every prompt.
        This parameter is important and must be set manually by the user.
        """

        if not isinstance(value, str):
            raise TypeError(self._("【IdeaSearcher】 参数`epilogue_section`类型应为str，实为%s") % str(type(value)))

        with self._user_lock:
            self._epilogue_section = value


    # ⭐️ Important
    def set_database_path(
        self,
        value: str,
    )-> None:
    
        """
        ⭐️ Important
        Set the parameter `database_path` to the given value, if it is of the type str.
        The user-specified database path is the only location that the IdeaSearch package will modify on your file system once it starts running. When you begin, this path should already contain a subfolder named ideas/initial_ideas/, which itself should hold some .idea text files; these will be read in as the initial ideas for the system. Subsequently, the system will automatically generate folders like island1, island2, etc., under ideas/ to store ideas from different islands. Additionally, data/, pic/, and log/ folders will be automatically created at the same level as ideas/ to display system operation information.
        This parameter is important and must be set manually by the user.
        """

        if not isinstance(value, str):
            raise TypeError(self._("【IdeaSearcher】 参数`database_path`类型应为str，实为%s") % str(type(value)))

        with self._user_lock:
            self._database_path = value


    # ⭐️ Important
    def set_models(
        self,
        value: List[str],
    )-> None:
    
        """
        ⭐️ Important
        Set the parameter `models` to the given value, if it is of the type List[str].
        This parameter is a list of model names participating in idea generation.
        This parameter is important and must be set manually by the user.
        """

        if not hasattr(value, "__iter__") and not isinstance(value, str):
            raise TypeError(self._("【IdeaSearcher】 参数`models`类型应为List[str]，实为%s") % str(type(value)))

        with self._user_lock:
            self._models = value


    # ⭐️ Important
    def set_model_temperatures(
        self,
        value: List[float],
    )-> None:
    
        """
        ⭐️ Important
        Set the parameter `model_temperatures` to the given value, if it is of the type List[float].
        This parameter defines the sampling temperature for each model, with its length matching the models list.
        This parameter is important and must be set manually by the user.
        """

        if not hasattr(value, "__iter__") and not isinstance(value, str):
            raise TypeError(self._("【IdeaSearcher】 参数`model_temperatures`类型应为List[float]，实为%s") % str(type(value)))

        with self._user_lock:
            self._model_temperatures = value


    # ⭐️ Important
    def set_evaluate_func(
        self,
        value: Callable[[str], Tuple[float, Optional[str]]],
    )-> None:
    
        """
        ⭐️ Important
        Set the parameter `evaluate_func` to the given value, if it is of the type Callable[[str], Tuple[float, Optional[str]]].
        This parameter is a function designed to score a single idea; it should accept a string as input and output a tuple containing a float for the score and an optional string for additional information.
        This parameter is important and must be set manually by the user.
        """

        if not callable(value):
            raise TypeError(self._("【IdeaSearcher】 参数`evaluate_func`类型应为Callable[[str], Tuple[float, Optional[str]]]，实为%s") % str(type(value)))

        with self._user_lock:
            self._evaluate_func = value


    # ⭐️ Important
    def set_api_keys_path(
        self,
        value: str,
    )-> None:
    
        """
        ⭐️ Important
        Set the parameter `api_keys_path` to the given value, if it is of the type str.
        This parameter indicates the path to the API key configuration file.
        This parameter is important and must be set manually by the user.
        """

        if not isinstance(value, str):
            raise TypeError(self._("【IdeaSearcher】 参数`api_keys_path`类型应为str，实为%s") % str(type(value)))

        with self._user_lock:
            self._api_keys_path = value


    def set_score_range(
        self,
        value: Tuple[float, float],
    )-> None:
    
        """
        Set the parameter score_range to the given value, if it is of the type Tuple[float, float].
        This parameter defines the range of scores, which is used for normalization and display, and scores given by the `evaluate_func` should fall within this range.
        Its default value is (0.0, 100.0).
        """

        if not isinstance(value, tuple):
            raise TypeError(self._("【IdeaSearcher】 参数`score_range`类型应为Tuple[float, float]，实为%s") % str(type(value)))

        with self._user_lock:
            self._score_range = value


    def set_hand_over_threshold(
        self,
        value: float,
    )-> None:
    
        """
        Set the parameter hand_over_threshold to the given value, if it is of the type float.
        This parameter defines the minimum score an idea must meet to enter the island.
        Its default value is 0.0.
        """

        if not isinstance(value, float):
            raise TypeError(self._("【IdeaSearcher】 参数`hand_over_threshold`类型应为float，实为%s") % str(type(value)))

        with self._user_lock:
            self._hand_over_threshold = value


    def set_system_prompt(
        self,
        value: Optional[str],
    )-> None:
    
        """
        Set the parameter system_prompt to the given value, if it is of the type Optional[str].
        This parameter contains the system prompt for IdeaSearcher.
        Its default value is None.
        """

        if not (value is None or isinstance(value, str)):
            raise TypeError(self._("【IdeaSearcher】 参数`system_prompt`类型应为Optional[str]，实为%s") % str(type(value)))

        with self._user_lock:
            self._system_prompt = value


    def set_diary_path(
        self,
        value: Optional[str],
    )-> None:
    
        """
        Set the parameter diary_path to the given value, if it is of the type Optional[str].
        This parameter specifies the path to the log file; if set to `None` at runtime, it will automatically default to `database_path` + 'log/diary.txt'.
        Its default value is None.
        """

        if not (value is None or isinstance(value, str)):
            raise TypeError(self._("【IdeaSearcher】 参数`diary_path`类型应为Optional[str]，实为%s") % str(type(value)))

        with self._user_lock:
            self._diary_path = value


    def set_samplers_num(
        self,
        value: int,
    )-> None:
    
        """
        Set the parameter samplers_num to the given value, if it is of the type int.
        This parameter determines the number of Samplers equipped for each island.
        Its default value is 3.
        """

        if not isinstance(value, int):
            raise TypeError(self._("【IdeaSearcher】 参数`samplers_num`类型应为int，实为%s") % str(type(value)))

        with self._user_lock:
            self._samplers_num = value


    def set_evaluators_num(
        self,
        value: int,
    )-> None:
    
        """
        Set the parameter evaluators_num to the given value, if it is of the type int.
        This parameter determines the number of Evaluators equipped for each island.
        Its default value is 3.
        """

        if not isinstance(value, int):
            raise TypeError(self._("【IdeaSearcher】 参数`evaluators_num`类型应为int，实为%s") % str(type(value)))

        with self._user_lock:
            self._evaluators_num = value


    def set_examples_num(
        self,
        value: int,
    )-> None:
    
        """
        Set the parameter examples_num to the given value, if it is of the type int.
        This parameter specifies the number of historical ideas displayed to the model per round.
        Its default value is 3.
        """

        if not isinstance(value, int):
            raise TypeError(self._("【IdeaSearcher】 参数`examples_num`类型应为int，实为%s") % str(type(value)))

        with self._user_lock:
            self._examples_num = value


    def set_generate_num(
        self,
        value: int,
    )-> None:
    
        """
        Set the parameter generate_num to the given value, if it is of the type int.
        This parameter sets the number of ideas generated by each Sampler per round.
        Its default value is 1.
        """

        if not isinstance(value, int):
            raise TypeError(self._("【IdeaSearcher】 参数`generate_num`类型应为int，实为%s") % str(type(value)))

        with self._user_lock:
            self._generate_num = value


    def set_sample_temperature(
        self,
        value: float,
    )-> None:
    
        """
        Set the parameter sample_temperature to the given value, if it is of the type float.
        This parameter controls the softmax temperature for idea selection.
        Its default value is 50.0.
        """

        if not isinstance(value, float):
            raise TypeError(self._("【IdeaSearcher】 参数`sample_temperature`类型应为float，实为%s") % str(type(value)))

        with self._user_lock:
            self._sample_temperature = value


    def set_model_sample_temperature(
        self,
        value: float,
    )-> None:
    
        """
        Set the parameter model_sample_temperature to the given value, if it is of the type float.
        This parameter controls the softmax temperature for model selection.
        Its default value is 50.0.
        """

        if not isinstance(value, float):
            raise TypeError(self._("【IdeaSearcher】 参数`model_sample_temperature`类型应为float，实为%s") % str(type(value)))

        with self._user_lock:
            self._model_sample_temperature = value


    def set_assess_func(
        self,
        value: Optional[Callable[[List[str], List[float], List[Optional[str]]], float]],
    )-> None:
    
        """
        Set the parameter assess_func to the given value, if it is of the type Optional[Callable[[List[str], List[float], List[Optional[str]]], float]].
        This parameter defines the comprehensive evaluation function for all ideas.
        Its default value is default_assess_func.
        """

        if not (value is None or callable(value)):
            raise TypeError(self._("【IdeaSearcher】 参数`assess_func`类型应为Optional[Callable[[List[str], List[float], List[Optional[str]]], float]]，实为%s") % str(type(value)))

        with self._user_lock:
            self._assess_func = value


    def set_assess_interval(
        self,
        value: Optional[int],
    )-> None:
    
        """
        Set the parameter assess_interval to the given value, if it is of the type Optional[int].
        This parameter sets the interval in rounds between assess_func evaluations.
        Its default value is 1.
        """

        if not (value is None or isinstance(value, int)):
            raise TypeError(self._("【IdeaSearcher】 参数`assess_interval`类型应为Optional[int]，实为%s") % str(type(value)))

        with self._user_lock:
            self._assess_interval = value


    def set_assess_baseline(
        self,
        value: Optional[float],
    )-> None:
    
        """
        Set the parameter assess_baseline to the given value, if it is of the type Optional[float].
        This parameter establishes the baseline for island assessment, which is displayed in the graph.
        Its default value is 60.0.
        """

        if not (value is None or isinstance(value, float)):
            raise TypeError(self._("【IdeaSearcher】 参数`assess_baseline`类型应为Optional[float]，实为%s") % str(type(value)))

        with self._user_lock:
            self._assess_baseline = value


    def set_assess_result_data_path(
        self,
        value: Optional[str],
    )-> None:
    
        """
        Set the parameter assess_result_data_path to the given value, if it is of the type Optional[str].
        This parameter specifies the path to store assessment scores in an .npz file; if set to `None` at runtime, it will automatically default to `database_path` + 'data/database_assessment.npz'.
        Its default value is None.
        """

        if not (value is None or isinstance(value, str)):
            raise TypeError(self._("【IdeaSearcher】 参数`assess_result_data_path`类型应为Optional[str]，实为%s") % str(type(value)))

        with self._user_lock:
            self._assess_result_data_path = value


    def set_assess_result_pic_path(
        self,
        value: Optional[str],
    )-> None:
    
        """
        Set the parameter assess_result_pic_path to the given value, if it is of the type Optional[str].
        This parameter specifies the path to store assessment images as a .png file; if set to `None` at runtime, it will automatically default to `database_path` + 'pic/database_assessment.png'.
        Its default value is None.
        """

        if not (value is None or isinstance(value, str)):
            raise TypeError(self._("【IdeaSearcher】 参数`assess_result_pic_path`类型应为Optional[str]，实为%s") % str(type(value)))

        with self._user_lock:
            self._assess_result_pic_path = value


    def set_model_assess_window_size(
        self,
        value: int,
    )-> None:
    
        """
        Set the parameter model_assess_window_size to the given value, if it is of the type int.
        This parameter sets the window size for model moving average assessment.
        Its default value is 20.
        """

        if not isinstance(value, int):
            raise TypeError(self._("【IdeaSearcher】 参数`model_assess_window_size`类型应为int，实为%s") % str(type(value)))

        with self._user_lock:
            self._model_assess_window_size = value


    def set_model_assess_initial_score(
        self,
        value: float,
    )-> None:
    
        """
        Set the parameter model_assess_initial_score to the given value, if it is of the type float.
        This parameter defines the initial score for models and is set high in order to encourage exploration.
        Its default value is 100.0.
        """

        if not isinstance(value, float):
            raise TypeError(self._("【IdeaSearcher】 参数`model_assess_initial_score`类型应为float，实为%s") % str(type(value)))

        with self._user_lock:
            self._model_assess_initial_score = value


    def set_model_assess_average_order(
        self,
        value: float,
    )-> None:
    
        """
        Set the parameter model_assess_average_order to the given value, if it is of the type float.
        This parameter sets the p-norm for the model score moving average.
        Its default value is 1.0.
        """

        if not isinstance(value, float):
            raise TypeError(self._("【IdeaSearcher】 参数`model_assess_average_order`类型应为float，实为%s") % str(type(value)))

        with self._user_lock:
            self._model_assess_average_order = value


    def set_model_assess_save_result(
        self,
        value: bool,
    )-> None:
    
        """
        Set the parameter model_assess_save_result to the given value, if it is of the type bool.
        This parameter determines whether to save model assessment results.
        Its default value is True.
        """

        if not isinstance(value, bool):
            raise TypeError(self._("【IdeaSearcher】 参数`model_assess_save_result`类型应为bool，实为%s") % str(type(value)))

        with self._user_lock:
            self._model_assess_save_result = value


    def set_model_assess_result_data_path(
        self,
        value: Optional[str],
    )-> None:
    
        """
        Set the parameter model_assess_result_data_path to the given value, if it is of the type Optional[str].
        This parameter specifies the save path for model assessment result data in an .npz file; if set to `None` at runtime, it will automatically default to `database_path` + 'data/model_scores.npz'.
        Its default value is None.
        """

        if not (value is None or isinstance(value, str)):
            raise TypeError(self._("【IdeaSearcher】 参数`model_assess_result_data_path`类型应为Optional[str]，实为%s") % str(type(value)))

        with self._user_lock:
            self._model_assess_result_data_path = value


    def set_model_assess_result_pic_path(
        self,
        value: Optional[str],
    )-> None:
    
        """
        Set the parameter model_assess_result_pic_path to the given value, if it is of the type Optional[str].
        This parameter specifies the save path for model assessment images as a .png file; if set to `None` at runtime, it will automatically default to `database_path` + 'pic/model_scores.png'.
        Its default value is None.
        """

        if not (value is None or isinstance(value, str)):
            raise TypeError(self._("【IdeaSearcher】 参数`model_assess_result_pic_path`类型应为Optional[str]，实为%s") % str(type(value)))

        with self._user_lock:
            self._model_assess_result_pic_path = value


    def set_mutation_func(
        self,
        value: Optional[Callable[[str], str]],
    )-> None:
    
        """
        Set the parameter mutation_func to the given value, if it is of the type Optional[Callable[[str], str]].
        This parameter is the mutation function for ideas.
        Its default value is None.
        """

        if not (value is None or callable(value)):
            raise TypeError(self._("【IdeaSearcher】 参数`mutation_func`类型应为Optional[Callable[[str], str]]，实为%s") % str(type(value)))

        with self._user_lock:
            self._mutation_func = value


    def set_mutation_interval(
        self,
        value: Optional[int],
    )-> None:
    
        """
        Set the parameter mutation_interval to the given value, if it is of the type Optional[int].
        This parameter sets the interval in rounds between mutation operations.
        Its default value is None.
        """

        if not (value is None or isinstance(value, int)):
            raise TypeError(self._("【IdeaSearcher】 参数`mutation_interval`类型应为Optional[int]，实为%s") % str(type(value)))

        with self._user_lock:
            self._mutation_interval = value


    def set_mutation_num(
        self,
        value: Optional[int],
    )-> None:
    
        """
        Set the parameter mutation_num to the given value, if it is of the type Optional[int].
        This parameter determines the number of mutations performed per round.
        Its default value is None.
        """

        if not (value is None or isinstance(value, int)):
            raise TypeError(self._("【IdeaSearcher】 参数`mutation_num`类型应为Optional[int]，实为%s") % str(type(value)))

        with self._user_lock:
            self._mutation_num = value


    def set_mutation_temperature(
        self,
        value: Optional[float],
    )-> None:
    
        """
        Set the parameter mutation_temperature to the given value, if it is of the type Optional[float].
        This parameter controls the softmax temperature for mutation candidate selection.
        Its default value is None.
        """

        if not (value is None or isinstance(value, float)):
            raise TypeError(self._("【IdeaSearcher】 参数`mutation_temperature`类型应为Optional[float]，实为%s") % str(type(value)))

        with self._user_lock:
            self._mutation_temperature = value


    def set_crossover_func(
        self,
        value: Optional[Callable[[str, str], str]],
    )-> None:
    
        """
        Set the parameter crossover_func to the given value, if it is of the type Optional[Callable[[str, str], str]].
        This parameter is the crossover function for ideas.
        Its default value is None.
        """

        if not (value is None or callable(value)):
            raise TypeError(self._("【IdeaSearcher】 参数`crossover_func`类型应为Optional[Callable[[str, str], str]]，实为%s") % str(type(value)))

        with self._user_lock:
            self._crossover_func = value


    def set_crossover_interval(
        self,
        value: Optional[int],
    )-> None:
    
        """
        Set the parameter crossover_interval to the given value, if it is of the type Optional[int].
        This parameter sets the interval in rounds between crossover operations.
        Its default value is None.
        """

        if not (value is None or isinstance(value, int)):
            raise TypeError(self._("【IdeaSearcher】 参数`crossover_interval`类型应为Optional[int]，实为%s") % str(type(value)))

        with self._user_lock:
            self._crossover_interval = value


    def set_crossover_num(
        self,
        value: Optional[int],
    )-> None:
    
        """
        Set the parameter crossover_num to the given value, if it is of the type Optional[int].
        This parameter determines the number of ideas generated by crossover per round.
        Its default value is None.
        """

        if not (value is None or isinstance(value, int)):
            raise TypeError(self._("【IdeaSearcher】 参数`crossover_num`类型应为Optional[int]，实为%s") % str(type(value)))

        with self._user_lock:
            self._crossover_num = value


    def set_crossover_temperature(
        self,
        value: Optional[float],
    )-> None:
    
        """
        Set the parameter crossover_temperature to the given value, if it is of the type Optional[float].
        This parameter controls the softmax temperature for crossover candidate selection.
        Its default value is None.
        """

        if not (value is None or isinstance(value, float)):
            raise TypeError(self._("【IdeaSearcher】 参数`crossover_temperature`类型应为Optional[float]，实为%s") % str(type(value)))

        with self._user_lock:
            self._crossover_temperature = value


    def set_similarity_threshold(
        self,
        value: float,
    )-> None:
    
        """
        Set the parameter similarity_threshold to the given value, if it is of the type float.
        This parameter defines the distance threshold for idea similarity; a value of -1 means only exact matches are considered similar.
        Its default value is -1.0.
        """

        if not isinstance(value, float):
            raise TypeError(self._("【IdeaSearcher】 参数`similarity_threshold`类型应为float，实为%s") % str(type(value)))

        with self._user_lock:
            self._similarity_threshold = value


    def set_similarity_distance_func(
        self,
        value: Optional[Callable[[str, str], float]],
    )-> None:
    
        """
        Set the parameter similarity_distance_func to the given value, if it is of the type Optional[Callable[[str, str], float]].
        This parameter is the function used to calculate idea similarity, defaulting to the absolute difference in scores.
        Its default value is None.
        """

        if not (value is None or callable(value)):
            raise TypeError(self._("【IdeaSearcher】 参数`similarity_distance_func`类型应为Optional[Callable[[str, str], float]]，实为%s") % str(type(value)))

        with self._user_lock:
            self._similarity_distance_func = value


    def set_similarity_sys_info_thresholds(
        self,
        value: Optional[List[int]],
    )-> None:
    
        """
        Set the parameter similarity_sys_info_thresholds to the given value, if it is of the type Optional[List[int]].
        This parameter is a list of thresholds that trigger similarity system prompts.
        Its default value is None.
        """

        if not (value is None or (hasattr(value, "__iter__") and not isinstance(value, str))):
            raise TypeError(self._("【IdeaSearcher】 参数`similarity_sys_info_thresholds`类型应为Optional[List[int]]，实为%s") % str(type(value)))

        with self._user_lock:
            self._similarity_sys_info_thresholds = value


    def set_similarity_sys_info_prompts(
        self,
        value: Optional[List[str]],
    )-> None:
    
        """
        Set the parameter similarity_sys_info_prompts to the given value, if it is of the type Optional[List[str]].
        This parameter contains the system prompt content corresponding to the thresholds. The length of `similarity_sys_info_prompts` should be the length of `similarity_sys_info_thresholds` + 1.
        Its default value is None.
        """

        if not (value is None or (hasattr(value, "__iter__") and not isinstance(value, str))):
            raise TypeError(self._("【IdeaSearcher】 参数`similarity_sys_info_prompts`类型应为Optional[List[str]]，实为%s") % str(type(value)))

        with self._user_lock:
            self._similarity_sys_info_prompts = value


    def set_load_idea_skip_evaluation(
        self,
        value: bool,
    )-> None:
    
        """
        Set the parameter load_idea_skip_evaluation to the given value, if it is of the type bool.
        This parameter determines whether to attempt skipping evaluation by loading from score_sheet.json.
        Its default value is True.
        """

        if not isinstance(value, bool):
            raise TypeError(self._("【IdeaSearcher】 参数`load_idea_skip_evaluation`类型应为bool，实为%s") % str(type(value)))

        with self._user_lock:
            self._load_idea_skip_evaluation = value


    def set_initialization_cleanse_threshold(
        self,
        value: float,
    )-> None:
    
        """
        Set the parameter initialization_cleanse_threshold to the given value, if it is of the type float.
        This parameter sets the minimum score threshold for initial cleansing.
        Its default value is -1.0.
        """

        if not isinstance(value, float):
            raise TypeError(self._("【IdeaSearcher】 参数`initialization_cleanse_threshold`类型应为float，实为%s") % str(type(value)))

        with self._user_lock:
            self._initialization_cleanse_threshold = value


    def set_delete_when_initial_cleanse(
        self,
        value: bool,
    )-> None:
    
        """
        Set the parameter delete_when_initial_cleanse to the given value, if it is of the type bool.
        This parameter determines whether to directly delete low-scoring ideas during cleansing.
        Its default value is False.
        """

        if not isinstance(value, bool):
            raise TypeError(self._("【IdeaSearcher】 参数`delete_when_initial_cleanse`类型应为bool，实为%s") % str(type(value)))

        with self._user_lock:
            self._delete_when_initial_cleanse = value


    def set_idea_uid_length(
        self,
        value: int,
    )-> None:
    
        """
        Set the parameter idea_uid_length to the given value, if it is of the type int.
        This parameter specifies the length of the UID in idea filenames.
        Its default value is 6.
        """

        if not isinstance(value, int):
            raise TypeError(self._("【IdeaSearcher】 参数`idea_uid_length`类型应为int，实为%s") % str(type(value)))

        with self._user_lock:
            self._idea_uid_length = value


    def set_record_prompt_in_diary(
        self,
        value: bool,
    )-> None:
    
        """
        Set the parameter record_prompt_in_diary to the given value, if it is of the type bool.
        This parameter controls whether to record each round's prompt in the diary.
        Its default value is False.
        """

        if not isinstance(value, bool):
            raise TypeError(self._("【IdeaSearcher】 参数`record_prompt_in_diary`类型应为bool，实为%s") % str(type(value)))

        with self._user_lock:
            self._record_prompt_in_diary = value


    def set_filter_func(
        self,
        value: Optional[Callable[[str], str]],
    )-> None:
    
        """
        Set the parameter filter_func to the given value, if it is of the type Optional[Callable[[str], str]].
        This parameter is a function for preprocessing before sampling and prompt construction.
        Its default value is None.
        """

        if not (value is None or callable(value)):
            raise TypeError(self._("【IdeaSearcher】 参数`filter_func`类型应为Optional[Callable[[str], str]]，实为%s") % str(type(value)))

        with self._user_lock:
            self._filter_func = value


    def set_generation_bonus(
        self,
        value: float,
    )-> None:
    
        """
        Set the parameter generation_bonus to the given value, if it is of the type float.
        This parameter provides a bonus for ideas from more recent generations, which is then incorporated into their scores for softmax sampling.
        Its default value is 0.0.
        """

        if not isinstance(value, float):
            raise TypeError(self._("【IdeaSearcher】 参数`generation_bonus`类型应为float，实为%s") % str(type(value)))

        with self._user_lock:
            self._generation_bonus = value


    def set_backup_path(
        self,
        value: Optional[str],
    )-> None:
    
        """
        Set the parameter backup_path to the given value, if it is of the type Optional[str].
        This parameter specifies the path for backups; if set to `None` at runtime, it will automatically default to `database_path` + 'ideas/backup/.
        Its default value is None.
        """

        if not (value is None or isinstance(value, str)):
            raise TypeError(self._("【IdeaSearcher】 参数`backup_path`类型应为Optional[str]，实为%s") % str(type(value)))

        with self._user_lock:
            self._backup_path = value


    def set_backup_on(
        self,
        value: bool,
    )-> None:
    
        """
        Set the parameter backup_on to the given value, if it is of the type bool.
        This parameter indicates whether backups are enabled.
        Its default value is True.
        """

        if not isinstance(value, bool):
            raise TypeError(self._("【IdeaSearcher】 参数`backup_on`类型应为bool，实为%s") % str(type(value)))

        with self._user_lock:
            self._backup_on = value


    def set_generate_prompt_func(
        self,
        value: Optional[Callable[[List[str], List[float], List[Optional[str]]], str]],
    )-> None:
    
        """
        Set the parameter generate_prompt_func to the given value, if it is of the type Optional[Callable[[List[str], List[float], List[Optional[str]]], str]].
        This parameter allows users to customize the generation of prompts based on given lists of ideas, their scores, and optional infos.
        Its default value is None.
        """

        if not (value is None or callable(value)):
            raise TypeError(self._("【IdeaSearcher】 参数`generate_prompt_func`类型应为Optional[Callable[[List[str], List[float], List[Optional[str]]], str]]，实为%s") % str(type(value)))

        with self._user_lock:
            self._generate_prompt_func = value


    def set_explicit_prompt_structure(
        self,
        value: bool,
    )-> None:
    
        """
        Set the parameter explicit_prompt_structure to the given value, if it is of the type bool.
        If True, the prompt will include auto-generated structural information.
        Its default value is False.
        """

        if not isinstance(value, bool):
            raise TypeError(self._("【IdeaSearcher】 参数`explicit_prompt_structure`类型应为bool，实为%s") % str(type(value)))

        with self._user_lock:
            self._explicit_prompt_structure = value


    def get_language(
        self,
    )-> str:
    
        """
        Get the current value of the `language` parameter.
        This parameter determines the active language for translations and interface text.
        """
        
        return self._language


    def get_program_name(
        self,
    )-> Optional[str]:
        
        """
        Get the current value of the `program_name` parameter.
        This parameter is the name of the current project.
        """

        return self._program_name


    def get_prologue_section(
        self,
    )-> Optional[str]:
        
        """
        Get the current value of the `prologue_section` parameter.
        This parameter holds the leading text snippet of every prompt.
        """

        return self._prologue_section


    def get_epilogue_section(
        self,
    )-> Optional[str]:
        
        """
        Get the current value of the `epilogue_section` parameter.
        This parameter holds the ending text snippet of every prompt.
        """

        return self._epilogue_section


    def get_database_path(
        self,
    )-> Optional[str]:
        
        """
        Get the current value of the `database_path` parameter.
        The user-specified database path is the only location that the IdeaSearch package will modify on your file system once it starts running. When you begin, this path should already contain a subfolder named ideas/initial_ideas/, which itself should hold some .idea text files; these will be read in as the initial ideas for the system. Subsequently, the system will automatically generate folders like island1, island2, etc., under ideas/ to store ideas from different islands. Additionally, data/, pic/, and log/ folders will be automatically created at the same level as ideas/ to display system operation information.
        """

        return self._database_path


    def get_models(
        self,
    )-> Optional[List[str]]:
        
        """
        Get the current value of the `models` parameter.
        This parameter is a list of model names participating in idea generation.
        """

        return self._models


    def get_model_temperatures(
        self,
    )-> Optional[List[float]]:
        
        """
        Get the current value of the `model_temperatures` parameter.
        This parameter defines the sampling temperature for each model, with its length matching the models list.
        """

        return self._model_temperatures


    def get_evaluate_func(
        self,
    )-> Optional[Callable[[str], Tuple[float, Optional[str]]]]:
        
        """
        Get the current value of the `evaluate_func` parameter.
        This parameter is a function designed to score a single idea; it should accept a string as input and output a tuple containing a float for the score and an optional string for additional information.
        """

        return self._evaluate_func


    def get_api_keys_path(
        self,
    )-> Optional[str]:
        
        """
        Get the current value of the `api_keys_path` parameter.
        This parameter indicates the path to the API key configuration file.
        """

        return self._api_keys_path


    def get_score_range(
        self,
    )-> Tuple[float, float]:
        
        """
        Get the current value of the `score_range` parameter.
        This parameter defines the range of scores, which is used for normalization and display, and scores given by the `evaluate_func` should fall within this range.
        """

        return self._score_range


    def get_hand_over_threshold(
        self,
    )-> float:
        
        """
        Get the current value of the `hand_over_threshold` parameter.
        This parameter defines the minimum score an idea must meet to enter the island.
        """

        return self._hand_over_threshold


    def get_system_prompt(
        self,
    )-> Optional[str]:
        
        """
        Get the current value of the `system_prompt` parameter.
        This parameter contains the system prompt for IdeaSearcher.
        """

        return self._system_prompt


    def get_diary_path(
        self,
    )-> Optional[str]:
        
        """
        Get the current value of the `diary_path` parameter.
        This parameter specifies the path to the log file; if set to `None` at runtime, it will automatically default to `database_path` + 'log/diary.txt'.
        """

        return self._diary_path


    def get_samplers_num(
        self,
    )-> int:
        
        """
        Get the current value of the `samplers_num` parameter.
        This parameter determines the number of Samplers equipped for each island.
        """

        return self._samplers_num


    def get_evaluators_num(
        self,
    )-> int:
        
        """
        Get the current value of the `evaluators_num` parameter.
        This parameter determines the number of Evaluators equipped for each island.
        """

        return self._evaluators_num


    def get_examples_num(
        self,
    )-> int:
        
        """
        Get the current value of the `examples_num` parameter.
        This parameter specifies the number of historical ideas displayed to the model per round.
        """

        return self._examples_num


    def get_generate_num(
        self,
    )-> int:
        
        """
        Get the current value of the `generate_num` parameter.
        This parameter sets the number of ideas generated by each Sampler per round.
        """

        return self._generate_num


    def get_sample_temperature(
        self,
    )-> float:
        
        """
        Get the current value of the `sample_temperature` parameter.
        This parameter controls the softmax temperature for idea selection.
        """

        return self._sample_temperature


    def get_model_sample_temperature(
        self,
    )-> float:
        
        """
        Get the current value of the `model_sample_temperature` parameter.
        This parameter controls the softmax temperature for model selection.
        """

        return self._model_sample_temperature


    def get_assess_func(
        self,
    )-> Optional[Callable[[List[str], List[float], List[Optional[str]]], float]]:
        
        """
        Get the current value of the `assess_func` parameter.
        This parameter defines the comprehensive evaluation function for all ideas.
        """

        return self._assess_func


    def get_assess_interval(
        self,
    )-> Optional[int]:
        
        """
        Get the current value of the `assess_interval` parameter.
        This parameter sets the interval in rounds between assess_func evaluations.
        """

        return self._assess_interval


    def get_assess_baseline(
        self,
    )-> Optional[float]:
        
        """
        Get the current value of the `assess_baseline` parameter.
        This parameter establishes the baseline for island assessment, which is displayed in the graph.
        """

        return self._assess_baseline


    def get_assess_result_data_path(
        self,
    )-> Optional[str]:
        
        """
        Get the current value of the `assess_result_data_path` parameter.
        This parameter specifies the path to store assessment scores in an .npz file; if set to `None` at runtime, it will automatically default to `database_path` + 'data/database_assessment.npz'.
        """

        return self._assess_result_data_path


    def get_assess_result_pic_path(
        self,
    )-> Optional[str]:
        
        """
        Get the current value of the `assess_result_pic_path` parameter.
        This parameter specifies the path to store assessment images as a .png file; if set to `None` at runtime, it will automatically default to `database_path` + 'pic/database_assessment.png'.
        """

        return self._assess_result_pic_path


    def get_model_assess_window_size(
        self,
    )-> int:
        
        """
        Get the current value of the `model_assess_window_size` parameter.
        This parameter sets the window size for model moving average assessment.
        """

        return self._model_assess_window_size


    def get_model_assess_initial_score(
        self,
    )-> float:
        
        """
        Get the current value of the `model_assess_initial_score` parameter.
        This parameter defines the initial score for models and is set high in order to encourage exploration.
        """

        return self._model_assess_initial_score


    def get_model_assess_average_order(
        self,
    )-> float:
        
        """
        Get the current value of the `model_assess_average_order` parameter.
        This parameter sets the p-norm for the model score moving average.
        """

        return self._model_assess_average_order


    def get_model_assess_save_result(
        self,
    )-> bool:
        
        """
        Get the current value of the `model_assess_save_result` parameter.
        This parameter determines whether to save model assessment results.
        """

        return self._model_assess_save_result


    def get_model_assess_result_data_path(
        self,
    )-> Optional[str]:
        
        """
        Get the current value of the `model_assess_result_data_path` parameter.
        This parameter specifies the save path for model assessment result data in an .npz file; if set to `None` at runtime, it will automatically default to `database_path` + 'data/model_scores.npz'.
        """

        return self._model_assess_result_data_path


    def get_model_assess_result_pic_path(
        self,
    )-> Optional[str]:
        
        """
        Get the current value of the `model_assess_result_pic_path` parameter.
        This parameter specifies the save path for model assessment images as a .png file; if set to `None` at runtime, it will automatically default to `database_path` + 'pic/model_scores.png'.
        """

        return self._model_assess_result_pic_path


    def get_mutation_func(
        self,
    )-> Optional[Callable[[str], str]]:
        
        """
        Get the current value of the `mutation_func` parameter.
        This parameter is the mutation function for ideas.
        """

        return self._mutation_func


    def get_mutation_interval(
        self,
    )-> Optional[int]:
        
        """
        Get the current value of the `mutation_interval` parameter.
        This parameter sets the interval in rounds between mutation operations.
        """

        return self._mutation_interval


    def get_mutation_num(
        self,
    )-> Optional[int]:
        
        """
        Get the current value of the `mutation_num` parameter.
        This parameter determines the number of mutations performed per round.
        """

        return self._mutation_num


    def get_mutation_temperature(
        self,
    )-> Optional[float]:
        
        """
        Get the current value of the `mutation_temperature` parameter.
        This parameter controls the softmax temperature for mutation candidate selection.
        """

        return self._mutation_temperature


    def get_crossover_func(
        self,
    )-> Optional[Callable[[str, str], str]]:
        
        """
        Get the current value of the `crossover_func` parameter.
        This parameter is the crossover function for ideas.
        """

        return self._crossover_func


    def get_crossover_interval(
        self,
    )-> Optional[int]:
        
        """
        Get the current value of the `crossover_interval` parameter.
        This parameter sets the interval in rounds between crossover operations.
        """

        return self._crossover_interval


    def get_crossover_num(
        self,
    )-> Optional[int]:
        
        """
        Get the current value of the `crossover_num` parameter.
        This parameter determines the number of ideas generated by crossover per round.
        """

        return self._crossover_num


    def get_crossover_temperature(
        self,
    )-> Optional[float]:
        
        """
        Get the current value of the `crossover_temperature` parameter.
        This parameter controls the softmax temperature for crossover candidate selection.
        """

        return self._crossover_temperature


    def get_similarity_threshold(
        self,
    )-> float:
        
        """
        Get the current value of the `similarity_threshold` parameter.
        This parameter defines the distance threshold for idea similarity; a value of -1 means only exact matches are considered similar.
        """

        return self._similarity_threshold


    def get_similarity_distance_func(
        self,
    )-> Optional[Callable[[str, str], float]]:
        
        """
        Get the current value of the `similarity_distance_func` parameter.
        This parameter is the function used to calculate idea similarity, defaulting to the absolute difference in scores.
        """

        return self._similarity_distance_func


    def get_similarity_sys_info_thresholds(
        self,
    )-> Optional[List[int]]:
        
        """
        Get the current value of the `similarity_sys_info_thresholds` parameter.
        This parameter is a list of thresholds that trigger similarity system prompts.
        """

        return self._similarity_sys_info_thresholds


    def get_similarity_sys_info_prompts(
        self,
    )-> Optional[List[str]]:
        
        """
        Get the current value of the `similarity_sys_info_prompts` parameter.
        This parameter contains the system prompt content corresponding to the thresholds. The length of `similarity_sys_info_prompts` should be the length of `similarity_sys_info_thresholds` + 1.
        """

        return self._similarity_sys_info_prompts


    def get_load_idea_skip_evaluation(
        self,
    )-> bool:
        
        """
        Get the current value of the `load_idea_skip_evaluation` parameter.
        This parameter determines whether to attempt skipping evaluation by loading from score_sheet.json.
        """

        return self._load_idea_skip_evaluation


    def get_initialization_cleanse_threshold(
        self,
    )-> float:
        
        """
        Get the current value of the `initialization_cleanse_threshold` parameter.
        This parameter sets the minimum score threshold for initial cleansing.
        """

        return self._initialization_cleanse_threshold


    def get_delete_when_initial_cleanse(
        self,
    )-> bool:
        
        """
        Get the current value of the `delete_when_initial_cleanse` parameter.
        This parameter determines whether to directly delete low-scoring ideas during cleansing.
        """

        return self._delete_when_initial_cleanse


    def get_idea_uid_length(
        self,
    )-> int:
        
        """
        Get the current value of the `idea_uid_length` parameter.
        This parameter specifies the length of the UID in idea filenames.
        """

        return self._idea_uid_length


    def get_record_prompt_in_diary(
        self,
    )-> bool:
        
        """
        Get the current value of the `record_prompt_in_diary` parameter.
        This parameter controls whether to record each round's prompt in the diary.
        """

        return self._record_prompt_in_diary


    def get_filter_func(
        self,
    )-> Optional[Callable[[str], str]]:
        
        """
        Get the current value of the `filter_func` parameter.
        This parameter is a function for preprocessing before sampling and prompt construction.
        """

        return self._filter_func


    def get_generation_bonus(
        self,
    )-> float:
        
        """
        Get the current value of the `generation_bonus` parameter.
        This parameter provides a bonus for ideas from more recent generations, which is then incorporated into their scores for softmax sampling.
        """

        return self._generation_bonus


    def get_backup_path(
        self,
    )-> Optional[str]:
        
        """
        Get the current value of the `backup_path` parameter.
        This parameter specifies the path for backups; if set to `None` at runtime, it will automatically default to `database_path` + 'ideas/backup/.
        """

        return self._backup_path


    def get_backup_on(
        self,
    )-> bool:
        
        """
        Get the current value of the `backup_on` parameter.
        This parameter indicates whether backups are enabled.
        """

        return self._backup_on


    def get_generate_prompt_func(
        self,
    )-> Optional[Callable[[List[str], List[float], List[Optional[str]]], str]]:
        
        """
        Get the current value of the `generate_prompt_func` parameter.
        This parameter allows users to customize the generation of prompts based on given lists of ideas, their scores, and optional infos.
        """

        return self._generate_prompt_func


    def get_explicit_prompt_structure(
        self,
    )-> bool:
        
        """
        Get the current value of the `explicit_prompt_structure` parameter.
        If True, the prompt will include auto-generated structural information.
        """

        return self._explicit_prompt_structure

