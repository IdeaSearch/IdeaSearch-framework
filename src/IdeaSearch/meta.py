import re
from os.path import sep as seperator


def main():
    
    ideasearcher_path = f"src{seperator}IdeaSearch{seperator}ideasearcher.py"
    
    params = [
        # 必填参数
        ("program_name", "str", ""),
        ("prologue_section", "str", ""),
        ("epilogue_section", "str", ""),
        ("database_path", "str", ""),
        ("models", "List[str]", ""),
        ("model_temperatures", "List[float]", ""),
        ("interaction_num", "int", ""),
        ("evaluate_func", "Callable[[str], Tuple[float, Optional[str]]]", ""),

        # 可选参数（以下全部带默认值）
        ("score_range", "Tuple[float, float]", "(0.0, 100.0)"),
        ("hand_over_threshold", "float", "0.0"),
        ("system_prompt", "Optional[str]", "None"),
        ("diary_path", "Optional[str]", "None"),
        ("api_keys_path", "Optional[str]", "None"),
        ("local_models_path", "Optional[str]", "None"),

        ("samplers_num", "int", "5"),
        ("evaluators_num", "int", "5"),

        ("examples_num", "int", "3"),
        ("generate_num", "int", "5"),
        ("sample_temperature", "float", "50.0"),
        ("model_sample_temperature", "float", "50.0"),

        ("assess_func", "Optional[Callable[[List[str], List[float], List[Optional[str]]], float]]", "None"),
        ("assess_interval", "Optional[int]", "None"),
        ("assess_baseline", "Optional[float]", "None"),
        ("assess_result_data_path", "Optional[str]", "None"),
        ("assess_result_pic_path", "Optional[str]", "None"),

        ("model_assess_window_size", "int", "20"),
        ("model_assess_initial_score", "float", "100.0"),
        ("model_assess_average_order", "float", "1.0"),
        ("model_assess_save_result", "bool", "True"),
        ("model_assess_result_data_path", "Optional[str]", "None"),
        ("model_assess_result_pic_path", "Optional[str]", "None"),

        ("mutation_func", "Optional[Callable[[str], str]]", "None"),
        ("mutation_interval", "Optional[int]", "None"),
        ("mutation_num", "Optional[int]", "None"),
        ("mutation_temperature", "Optional[float]", "None"),

        ("crossover_func", "Optional[Callable[[str, str], str]]", "None"),
        ("crossover_interval", "Optional[int]", "None"),
        ("crossover_num", "Optional[int]", "None"),
        ("crossover_temperature", "Optional[float]", "None"),

        ("similarity_threshold", "float", "-1.0"),
        ("similarity_distance_func", "Optional[Callable[[str, str], float]]", "None"),
        ("similarity_sys_info_thresholds", "Optional[List[int]]", "None"),
        ("similarity_sys_info_prompts", "Optional[List[str]]", "None"),

        ("initialization_skip_evaluation", "bool", "True"),
        ("initialization_cleanse_threshold", "float", "-1.0"),
        ("delete_when_initial_cleanse", "bool", "False"),

        ("idea_uid_length", "int", "4"),
        ("record_prompt_in_diary", "bool", "True"),
        ("filter_func", "Optional[Callable[[str], str]]", "None"),
    ]
    
    init_code = ""
    set_code = ""
    get_code = ""
    for index, (param_name, param_type, param_default_value) in enumerate(params):
        
        if param_default_value != "":
            inner_param_type = param_type
            init_code += f"        self._{param_name}: {param_type} = {param_default_value}\n"
        else:
            inner_param_type = f"Optional[{param_type}]"
            init_code += f"        self._{param_name}: {inner_param_type} = None\n"
            
        if index:
            set_code += "\n\n"
            get_code += "\n\n"
            
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
                check = f"(value is None or isinstance(value, {param_type}))"
            else:
                check = f"isinstance(value, {param_type})"

        set_code += f"""    def set_{param_name}(
        self,
        value: {param_type},
    ) -> None:

        if not {check}:
            raise TypeError(f"【IdeaSearcher】 参数`{param_name}`类型应为{param_type}，实为{{str(type(value))}}")

        with self._lock:
            self._{param_name} = value
"""
        
        get_code += f"""    def get_{param_name}(
        self,
    )-> {inner_param_type}:
        
            return self._{param_name}
"""
    
    ideasearcher_code = f"""import os
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
from os.path import basename
from src.utils import append_to_file
from src.utils import guarantee_path_exist
from src.API4LLMs.model_manager import ModelManager
from src.API4LLMs.get_answer import get_answer_online
from src.API4LLMs.get_answer import get_answer_local


class IdeaSearcher:
    
    # ----------------------------- IdeaSearhcer 初始化 ----------------------------- 
    
    def __init__(
        self
    ) -> None:
    
        self._lock: Lock = Lock()
        self._model_manager: ModelManager = ModelManager()
        
{init_code}
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
                
                
    def shutdown_models(
        self,
    )-> None:

        with self._lock:
        
            self._model_manager.shutdown()
    
    
{set_code}

{get_code}
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
"""
    
    with open(
        file = ideasearcher_path,
        mode = "w",
        encoding = "UTF-8",
    ) as file:
        
        file.write(ideasearcher_code)


if __name__ == "__main__":
    
    main()