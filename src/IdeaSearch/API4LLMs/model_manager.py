import json
import os
from typing import Optional
from typing import Tuple
from threading import Lock
from time import sleep


__all__ = [
    "init_model_manager",
    "is_online_model",
    "get_online_model_instance",
    "shutdown_model_manager",
    "ModelManager",
]


class ModelManager:
    
    # ----------------------------- Model Manager 初始化 ----------------------------- 
    
    def __init__(self):
        
        self.lock = Lock()
        self._online_models = {}
        self._is_online_model = {}

    # ----------------------------- 外部动作 ----------------------------- 

    def load_api_keys(
        self, 
        api_keys_path: str,
    )-> None:
        
        with self.lock:
        
            if os.path.exists(api_keys_path) and os.path.isfile(api_keys_path):
                
                with open(api_keys_path, 'r') as file:
                    
                    api_keys_dict = json.load(file)
                    
                    for model_name in api_keys_dict:
                    
                        self._is_online_model[model_name] = True
                        self._online_models[model_name] = {
                            "instances": [
                                {
                                    "api_key": api_keys_dict[model_name][index]["api_key"],
                                    "base_url": api_keys_dict[model_name][index]["base_url"],
                                    "model": api_keys_dict[model_name][index]["model"],
                                }
                                for index in range(len(api_keys_dict[model_name]))
                            ],
                            "next_choice_index": 0,
                        }
                    
            else:
                raise ValueError(f" api keys 文件 {api_keys_path} 异常！")
            
            
    def is_online_model(
        self,
        model_name: str,
    )-> bool:
        with self.lock:
            if model_name in self._is_online_model:
                return self._is_online_model[model_name]
            else:
                raise ValueError(f" 模型 {model_name} 未被 model manager 记录！")
            
            
    def get_online_model_instance(
        self,
        model_name: str,
    )-> Tuple[str, str, str]:
        
        with self.lock:
            
            online_model = self._online_models[model_name]
            
            index_backup = online_model["next_choice_index"]
            self._online_models[model_name]["next_choice_index"] = (online_model["next_choice_index"]+1) % len(online_model["instances"])
            
            return (
                online_model["instances"][index_backup]["api_key"],
                online_model["instances"][index_backup]["base_url"],
                online_model["instances"][index_backup]["model"],
            )
            
        
    def shutdown(self):
        
        with self.lock:
            sleep(6)
             
                
model_manager = ModelManager()

# ----------------------------- APIs -----------------------------

def init_model_manager(
    api_keys_path: Optional[str] = None,
)-> None:
    
    if api_keys_path is None:
        raise ValueError(
            "初始化 model manager 时发生错误："
            " api keys path 不应为 None ！"
        )
        
    if api_keys_path is not None:
        model_manager.load_api_keys(api_keys_path)
        
        
def is_online_model(
    model_name: str,
)-> bool:
    return model_manager.is_online_model(model_name)


def get_online_model_instance(
    model_name: str,
)-> Tuple[str, str, str]:
    return model_manager.get_online_model_instance(model_name)


def shutdown_model_manager()-> None:
    model_manager.shutdown()