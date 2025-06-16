import re
from os.path import sep as seperator


def main():
    
    ideasearcher_path = f"src{seperator}IdeaSearch{seperator}ideasearcher.py"
    
    params = [
        ("program_name", "str", "", "当前项目的名称。"),
        ("prologue_section", "str", "", "用于提示模型采样的前导文本片段。"),
        ("epilogue_section", "str", "", "用于提示模型采样的结尾文本片段。"),
        ("database_path", "str", "", "岛屿路径，其下 ideas/ 路径用于存放 .idea 文件和 score_sheet.json。"),
        ("models", "List[str]", "", "参与生成 idea 的模型名称列表。"),
        ("model_temperatures", "List[float]", "", "各模型的采样温度，与 models 等长。"),
        ("evaluate_func", "Callable[[str], Tuple[float, Optional[str]]]", "", "对单个 idea 进行评分的函数。"),
        
        ("score_range", "Tuple[float, float]", "(0.0, 100.0)", "评分区间范围，用于归一化和显示。"),
        ("hand_over_threshold", "float", "0.0", "idea 进入岛屿的最低评分要求。"),
        ("system_prompt", "Optional[str]", "None", "IdeaSearcher 的提示词。"),
        ("diary_path", "Optional[str]", "None", "日志文件路径。"),
        ("api_keys_path", "Optional[str]", "None", "API key 配置文件路径。"),
        # ("local_models_path", "Optional[str]", "None", "本地模型配置文件路径（如使用本地推理）。"),

        ("samplers_num", "int", "3", "每个岛屿配备的 Sampler 数量。"),
        ("evaluators_num", "int", "3", "每个岛屿配备的 Evaluator 数量。"),

        ("examples_num", "int", "3", "每轮展示给模型的历史 idea 数量。"),
        ("generate_num", "int", "1", "每轮每个 Sampler 生成的 idea 数量。"),
        ("sample_temperature", "float", "50.0", "控制 idea 选择的 softmax 温度。"),
        ("model_sample_temperature", "float", "50.0", "控制模型选择的 softmax 温度。"),

        ("assess_func", "Optional[Callable[[List[str], List[float], List[Optional[str]]], float]]", "default_assess_func", "全体 idea 的综合评估函数。"),
        ("assess_interval", "Optional[int]", "1", "每隔多少轮进行一次 assess_func 评估。"),
        ("assess_baseline", "Optional[float]", "60.0", "岛屿评估的基线，会在图像中显示。"),
        ("assess_result_data_path", "Optional[str]", "None", "存储评估得分的路径（.npz）。"),
        ("assess_result_pic_path", "Optional[str]", "None", "存储评估图像的路径（.png）。"),

        ("model_assess_window_size", "int", "20", "模型滑动平均评估窗口大小。"),
        ("model_assess_initial_score", "float", "100.0", "模型初始得分。"),
        ("model_assess_average_order", "float", "1.0", "模型评分滑动平均的 p 范数。"),
        ("model_assess_save_result", "bool", "True", "是否保存模型评估结果。"),
        ("model_assess_result_data_path", "Optional[str]", "None", "模型评估结果数据保存路径（.npz）。"),
        ("model_assess_result_pic_path", "Optional[str]", "None", "模型评估图像保存路径（.png）。"),

        ("mutation_func", "Optional[Callable[[str], str]]", "None", "idea 的突变函数。"),
        ("mutation_interval", "Optional[int]", "None", "每隔多少轮进行一次突变操作。"),
        ("mutation_num", "Optional[int]", "None", "每轮进行的突变数量。"),
        ("mutation_temperature", "Optional[float]", "None", "控制突变候选选择的 softmax 温度。"),

        ("crossover_func", "Optional[Callable[[str, str], str]]", "None", "idea 的交叉函数。"),
        ("crossover_interval", "Optional[int]", "None", "每隔多少轮进行一次交叉操作。"),
        ("crossover_num", "Optional[int]", "None", "每轮交叉生成的 idea 数量。"),
        ("crossover_temperature", "Optional[float]", "None", "控制交叉候选选择的 softmax 温度。"),

        ("similarity_threshold", "float", "-1.0", "idea 相似性的距离阈值，-1 表示仅完全一致为相似。"),
        ("similarity_distance_func", "Optional[Callable[[str, str], float]]", "None", "idea 的相似度计算函数，默认为分数差的绝对值。"),
        ("similarity_sys_info_thresholds", "Optional[List[int]]", "None", "控制相似度系统提示的触发阈值列表。"),
        ("similarity_sys_info_prompts", "Optional[List[str]]", "None", "与 thresholds 对应的系统提示内容。"),

        ("load_idea_skip_evaluation", "bool", "True", "是否尝试跳过评估（从 score_sheet.json 中加载）。"),
        ("initialization_cleanse_threshold", "float", "-1.0", "初始清洗的最低评分阈值。"),
        ("delete_when_initial_cleanse", "bool", "False", "清洗时是否直接删除低分 idea。"),

        ("idea_uid_length", "int", "6", "idea 文件名中 uid 的长度。"),
        ("record_prompt_in_diary", "bool", "False", "是否将每轮的 Prompt 记录到日志中。"),
        ("filter_func", "Optional[Callable[[str], str]]", "None", "采样拼 prompt 前进行预处理的函数。"),
        ("generation_bonus", "float", "0.0", ""),
        ("backup_path", "Optional[str]", "None", ""),
        ("backup_on", "bool", "True", ""),
        ("generate_prompt_func", "Optional[Callable[[List[str], List[float], List[Optional[str]]], str]]", "None", "")
    ]
    
    init_code = f"""    def __init__(
        self
    ) -> None:
    
        # 国际化设置
        self._language: str = 'zh_CN'
        self._translation = gettext.translation(_DOMAIN, _LOCALE_DIR, languages=[self._language], fallback=True)
        self._ = self._translation.gettext
    
"""
    set_code = ""
    get_code = ""
    for index, (param_name, param_type, param_default_value, _) in enumerate(params):
        
        if index:
            set_code += "\n\n"
            get_code += "\n\n"
        
        if param_default_value != "":
            inner_param_type = param_type
            init_code += f"        self._{param_name}: {param_type} = {param_default_value}\n"
        else:
            inner_param_type = f"Optional[{param_type}]"
            init_code += f"        self._{param_name}: {inner_param_type} = None\n"
            set_code += "    # ⭐️ Important\n"
            
        optional_match = re.match(r"Optional\[(.+)\]", param_type)
        
        if optional_match:
            is_optional = True
            inner_type = optional_match.group(1).strip()
        else:
            is_optional = False
            inner_type = param_type.strip()
            
        if inner_type.startswith("Callable"):
            if is_optional:
                check = "(value is None or callable(value))"
            else:
                check = "callable(value)"
        elif inner_type.startswith("List"):
            if is_optional:
                check = '(value is None or (hasattr(value, "__iter__") and not isinstance(value, str)))'
            else:
                check = 'hasattr(value, "__iter__") and not isinstance(value, str)'
        elif inner_type.startswith("Tuple"):
            if is_optional:
                check = "(value is None or isinstance(value, tuple))"
            else:
                check = "isinstance(value, tuple)"
        else:
            if is_optional:
                check = f"(value is None or isinstance(value, {inner_type}))"
            else:
                check = f"isinstance(value, {inner_type})"

        set_code += f"""    def set_{param_name}(
        self,
        value: {param_type},
    ) -> None:

        if not {check}:
            raise TypeError(self._("【IdeaSearcher】 参数`{param_name}`类型应为{param_type}，实为%s") % str(type(value)))

        with self._user_lock:
            self._{param_name} = value
"""
        
        get_code += f"""    def get_{param_name}(
        self,
    )-> {inner_param_type}:
        
        return self._{param_name}
"""

    init_code += f"""
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
        self._islands: Dict[int, Island] = {{}}
        
        self._database_assessment_config_loaded = False
        self._model_score_config_loaded = False
        
        self._total_interaction_num = 0
        self._first_time_run = True
        self._first_time_add_island = True
        self._assigned_idea_uids = set()
        self._recorded_ideas = []
        self._recorded_idea_names = set()
"""


    import_section = """import concurrent.futures
import os
import json
import math
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
from os.path import basename
from os.path import sep as seperator
from IdeaSearch.utils import append_to_file
from IdeaSearch.utils import guarantee_path_exist
from IdeaSearch.utils import get_auto_markersize
from IdeaSearch.utils import clear_file_content
from IdeaSearch.utils import default_assess_func
from IdeaSearch.utils import make_boltzmann_choice
import gettext
from pathlib import Path
from IdeaSearch.sampler import Sampler
from IdeaSearch.evaluator import Evaluator
from IdeaSearch.island import Island
from IdeaSearch.API4LLMs.model_manager import ModelManager
from IdeaSearch.API4LLMs.get_answer import get_answer_online

# 国际化设置
_LOCALE_DIR = Path(__file__).parent / "locales"
_DOMAIN = "ideasearch"
gettext.bindtextdomain(_DOMAIN, _LOCALE_DIR)
gettext.textdomain(_DOMAIN)
"""

    load_models = """    def load_models(
        self
    )-> None:
    
        with self._user_lock:
    
            if self._api_keys_path is None:
                raise ValueError(
                    self._("【IdeaSearcher】 加载模型时发生错误： api keys path 不应为 None ！")
                )
                
            self._model_manager.load_api_keys(self._api_keys_path)
"""

    add_island = f"""    def add_island(
        self,
    )-> int:
        
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
            
                clear_file_content(diary_path)
                
                if backup_on:
                    guarantee_path_exist(f"{{backup_path}}{{seperator}}score_sheet_backup.json")
                    shutil.rmtree(f"{{backup_path}}")
                    guarantee_path_exist(f"{{backup_path}}{{seperator}}score_sheet_backup.json")
                    
                for item in os.listdir(f"{{database_path}}{{seperator}}ideas"):
                    full_path = os.path.join(f"{{database_path}}{{seperator}}ideas", item)
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
"""

    delete_island = """    def delete_island(
        self,
        island_id: int,
    )-> int:
    
        with self._user_lock:
            
            if island_id in self._islands:
                del self._islands[island_id]
                return 1
                
            else:
                return 0
"""

    update_model_score = """    def update_model_score(
        self,
        score_result: list[float], 
        model: str,
        model_temperature: float,
    )-> None:
        
        with self._lock:
            
            diary_path = self._diary_path
            
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
                            content_str = self._("【IdeaSearcher】 模型 %s(T=%.2f) 此轮评分为 %.2f ，其总评分已被更新为 %.2f ！") % (model, model_temperature, self._model_recent_scores[index][-1], self._model_scores[index]),
                        )
                    if model_assess_save_result:
                        self._sync_model_score_result()
                    return
                
                index += 1
                
            with self._console_lock:    
                append_to_file(
                    file_path = diary_path,
                    content_str = self._("【IdeaSearcher】 出现错误！未知的模型名称及温度： %s(T=%.2f) ！") % (model, model_temperature),
                )
                
            exit()
"""

    run = f"""    def run(
        self,
        additional_interaction_num: int,
    )-> None:

        with self._user_lock:
        
            missing_param = self._check_runnability()
            if missing_param is not None:
                raise RuntimeError(self._("【IdeaSearcher】 参数`%s`未传入，在当前设置下无法进行 run 动作！") % missing_param)
                
            diary_path = self._diary_path
            database_path = self._database_path
            program_name = self._program_name
            assert diary_path is not None
            assert database_path is not None
            assert program_name is not None
                
            append_to_file(
                file_path = diary_path,
                content_str = self._("【IdeaSearcher】 %s 的 IdeaSearch 正在运行，此次运行每个岛屿会演化 %d 个 epoch ！") % (program_name, additional_interaction_num)
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
            
                futures = {{executor.submit(sampler.run): (island_id, sampler.id)
                    for island_id in self._islands
                    for sampler in self._islands[island_id].samplers
                }}
                for future in concurrent.futures.as_completed(futures):
                    island_id, sampler_id = futures[future]
                    try:
                        _ = future.result() 
                    except Exception as e:
                        append_to_file(
                            file_path = diary_path,
                            content_str = self._("【IdeaSearcher】 %d号岛屿的%d号采样器在运行过程中出现错误：\\n%s\\nIdeaSearch意外终止！") % (island_id, sampler_id, e),
                        )
                        exit()
"""

    get_model = """    def get_model(
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
"""

    repopulate_islands = """    def repopulate_islands(
        self,
    )-> None:
    
        with self._user_lock:
        
            with self._console_lock:
                append_to_file(
                    file_path = self._diary_path,
                    content_str = self._("【IdeaSearcher】 现在 ideas 开始在岛屿间重分布")
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
                    file_path = self._diary_path,
                    content_str = self._("【IdeaSearcher】 此次 ideas 在岛屿间的重分布已完成")
                )
"""

    shutdown_models = """    def shutdown_models(
        self,
    )-> None:

        with self._user_lock:
        
            self._model_manager.shutdown()
"""

    is_online_model = """    def _is_online_model(
        self,
        model_name: str,
    )-> bool:
    
        return self._model_manager.is_online_model(model_name)
"""

    get_online_model_instance = """    def _get_online_model_instance(
        self,
        model_name: str,
    )-> Tuple[str, str, str]:
        return self._model_manager.get_online_model_instance(model_name)
"""

    get_local_model_instance = """    def _get_local_model_instance(
        self,
        model_name: str,
    )-> int:
        return self._model_manager.get_local_model_instance(model_name)
"""

    get_answer = f"""    def _get_answer(
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
            raise RuntimeError(f"【IdeaSearcher】 get answer 过程报错：模型 {{model_name}} 未被记录！")
"""

    check_runnability = f"""    def _check_runnability(
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
        
        database_path = self._database_path
        models = self._models
        assert database_path is not None
        assert models is not None
        
        if self._model_temperatures is None:
            self._model_temperatures = [1.0] * len(models)
        
        if self._similarity_distance_func is None:
            self._similarity_distance_func = self._default_similarity_distance_func
        
        if self._diary_path is None:
            self._diary_path = f"{{database_path}}{{seperator}}log{{seperator}}diary.txt"
            
        if self._system_prompt is None:
            self._system_prompt = "You're a helpful assistant."
            
        if self._assess_func is not None:
            if self._assess_result_data_path is None:
                self._assess_result_data_path = f"{{database_path}}{{seperator}}data{{seperator}}database_assessment.npz"
            if self._assess_result_pic_path is None:
                self._assess_result_pic_path = f"{{database_path}}{{seperator}}pic{{seperator}}database_assessment.png"
                
        if self._model_assess_save_result:
            if self._model_assess_result_data_path is None:
                self._model_assess_result_data_path = f"{{database_path}}{{seperator}}data{{seperator}}model_scores.npz"
            if self._model_assess_result_pic_path is None:
                self._model_assess_result_pic_path = f"{{database_path}}{{seperator}}pic{{seperator}}model_scores.png"
                
        if self._backup_path is None:
            self._backup_path = f"{{database_path}}{{seperator}}ideas{{seperator}}backup"
                
        return None
"""

    load_model_score_config = f"""    def _load_model_score_config(
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
"""

    expand_model_score_range = f"""    def _expand_model_score_range(
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
"""

    show_model_scores = """    def _show_model_scores(
        self
    )-> None:
        
        diary_path = self._diary_path
        models = self._models
        model_temperatures = self._model_temperatures
        assert models is not None
        assert model_temperatures is not None
            
        with self._console_lock:
            
            append_to_file(
                file_path = diary_path,
                content_str = self._("【IdeaSearcher】 各模型目前评分情况如下："),
            )
            for index, model in enumerate(models):
                
                model_temperature = model_temperatures[index]
                
                append_to_file(
                    file_path = diary_path,
                    content_str = (
                        f"  {index+1}. {model}(T={model_temperature:.2f}): {self._model_scores[index]:.2f}"
                    ),
                )
"""

    sync_database_assessment_result = f"""    def _sync_database_assessment_result(
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
                        content_str = self._("【IdeaSearcher】 初始质量评估结束， %s 与 %s 已更新！") % (basename(assess_result_data_path), basename(assess_result_pic_path)),
                    )
            else:
                with self._console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content_str = self._("【IdeaSearcher】 此轮质量评估结束， %s 与 %s 已更新！") % (basename(assess_result_data_path), basename(assess_result_pic_path)),
                    )
"""

    sync_model_score_result = f"""    def _sync_model_score_result(self):
    
        if self._total_interaction_num == 0: return
        
        diary_path = self._diary_path
        model_assess_result_data_path = self._model_assess_result_data_path
        model_assess_result_pic_path = self._model_assess_result_pic_path
        models = self._models
        model_temperatures = self._model_temperatures
        score_range = self._score_range
        
        assert model_assess_result_data_path is not None
        assert model_assess_result_pic_path is not None
        assert models is not None
        assert model_temperatures is not None
        
        self._scores_of_models[self._scores_of_models_length] = self._model_scores
        self._scores_of_models_length += 1
        
        scores_of_models = self._scores_of_models.T
        
        scores_of_models_dict = {{}}
        for model_name, model_temperature, model_scores in zip(models, model_temperatures, scores_of_models):
            scores_of_models_dict[f"{{model_name}}(T={{model_temperature:.2f}})"] = model_scores
        
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
                content_str=(
                    f"【IdeaSearcher】 "
                    f" {{basename(model_assess_result_data_path)}} 与 {{basename(model_assess_result_pic_path)}} 已更新！"
                ),
            )
"""

    load_database_assessment_config = f"""    def _load_database_assessment_config(
        self,
    )-> None:
    
        diary_path = self._diary_path
        assess_func = self._assess_func
        assess_interval = self._assess_interval
        assess_result_data_path = self._assess_result_data_path
        assess_result_pic_path = self._assess_result_pic_path
        assert diary_path is not None
        
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
            
            guarantee_path_exist(assess_result_data_path)
            guarantee_path_exist(assess_result_pic_path)
            
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
                        content_str = self._("【IdeaSearcher】 初始 ideas 的整体质量评分为：%.2f！") % database_initial_score,
                    )
                    
            except Exception as error:
                database_initial_score = 0.0
                with self._console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content_str = self._("【IdeaSearcher】 评估库中初始 ideas 的整体质量时遇到错误：\\n%s") % error,
                    )
                    
            self._assess_result_ndarray[0] = database_initial_score
            self._sync_database_assessment_result(
                is_initialization = True,
                get_database_score_success = get_database_initial_score_success,
            )
            
        else:
            self._assess_on = False
"""

    expand_database_assessment_range = f"""    def _expand_database_assessment_range(
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
"""

    assess_database = f"""    def assess_database(
        self,
    )-> None:
        
        with self._lock:
        
            if not self._assess_on: return
        
            diary_path = self._diary_path
            assess_func = self._assess_func
            assess_interval = self._assess_interval
            assert assess_func is not None
            assert assess_interval is not None
        
            self._assess_interaction_count += 1
            if self._assess_interaction_count % assess_interval != 0: return

            start_time = perf_counter()
            
            with self._console_lock:
                append_to_file(
                    file_path = diary_path,
                    content_str = self._("【IdeaSearcher】 现在开始评估数据库中 ideas 的整体质量！"),
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
                        content_str = self._("【IdeaSearcher】 数据库中 ideas 的整体质量评分为：%.2f！评估用时：%.2f秒。") % (database_score, total_time),
                    )
                    
            except Exception as error:
                database_score = 0
                with self._console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content_str = self._("【IdeaSearcher】 评估库中 ideas 的整体质量时遇到错误：\\n%s") % error,
                    )
                    
            self._assess_result_ndarray[self._assess_result_ndarray_length] = database_score
            self._assess_result_ndarray_length += 1
            
            self._sync_database_assessment_result(
                is_initialization = False,
                get_database_score_success = get_database_score_success,
            )
"""

    get_idea_uid = f"""    def get_idea_uid(
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
"""

    get_best_score = f"""    def get_best_score(
        self,
    )-> float:
    
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
"""

    get_best_idea = f"""    def get_best_idea(
        self,
    )-> str:
    
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
"""

    record_ideas_in_backup = f"""    def record_ideas_in_backup(
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
            
            guarantee_path_exist(f"{{backup_path}}{{seperator}}score_sheet_backup.json")
        
            for idea in ideas_to_record:
                
                if basename(idea.path) not in self._recorded_idea_names:
                    
                    self._recorded_ideas.append(idea)
                    self._recorded_idea_names.add(basename(idea.path))
                
                    with open(
                        file = f"{{backup_path}}{{seperator}}{{basename(idea.path)}}",
                        mode = "w",
                        encoding = "UTF-8",
                    ) as file:

                        file.write(idea.content)
                        
            score_sheet = {{
                basename(idea.path): {{
                    "score": idea.score,
                    "info": idea.info if idea.info is not None else "",
                    "source": idea.source,
                    "level": idea.level,
                    "created_at": idea.created_at,
                }}
                for idea in self._recorded_ideas
            }}

            with open(
                file = f"{{backup_path}}{{seperator}}score_sheet_backup.json", 
                mode = "w", 
                encoding = "UTF-8",
            ) as file:
                
                json.dump(
                    obj = score_sheet, 
                    fp = file, 
                    ensure_ascii = False,
                    indent = 4
                )
"""

    set_language = """    def set_language(
        self,
        value: str,
    ) -> None:

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
"""

    get_language = """    def get_language(
        self,
    )-> str:
        
        return self._language
"""

    dir_code = """    def __dir__(self):
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
"""
    
    ideasearcher_code = f"""{import_section}

class IdeaSearcher:
    
    # ----------------------------- IdeaSearhcer 初始化 ----------------------------- 

{init_code}

{dir_code}
    # ----------------------------- 核心功能 ----------------------------- 
    
    # ⭐️ Important
{run}

{check_runnability}
    # ----------------------------- API4LLMs 相关 ----------------------------- 
   
    # ⭐️ Important
{load_models} 

    # ⭐️ Important
{shutdown_models}   

{is_online_model}
        
{get_online_model_instance}

{get_answer}
    # ----------------------------- Ideas 管理相关 ----------------------------- 
    
    # ⭐️ Important
{get_best_score}

    # ⭐️ Important
{get_best_idea}
    
{get_idea_uid}

{record_ideas_in_backup}
    # ----------------------------- 岛屿相关 ----------------------------- 

    # ⭐️ Important
{add_island}
           
    # ⭐️ Important 
{delete_island}
    
    # ⭐️ Important
{repopulate_islands}
    # ----------------------------- Model Score 相关 ----------------------------- 
    
{load_model_score_config}

{expand_model_score_range}

{update_model_score}

{sync_model_score_result}
                        
{get_model}
       
{show_model_scores}
    # ----------------------------- Database Assessment 相关 ----------------------------- 
            
{load_database_assessment_config}
            
{expand_database_assessment_range}

{assess_database}

{sync_database_assessment_result}
    # ----------------------------- Getters and Setters ----------------------------- 
    
{set_language}
        
{set_code}

{get_language}

{get_code}
"""
    
    with open(
        file = ideasearcher_path,
        mode = "w",
        encoding = "UTF-8",
    ) as file:
        
        file.write(ideasearcher_code)


if __name__ == "__main__":

    main()