import json
from typing import Any


__all__ = [
    "load_from_json",
    "save_to_json",
    "serialize_json",
    "deserialize_json",
]


def load_from_json(
    json_path: str,
    encoding: str = "UTF-8",   
)-> Any:
    
    with open(
        file = json_path,
        mode = "r",
        encoding = encoding,
    ) as file_pointer:
        
        data = json.load(file_pointer)

    return data


def save_to_json(
    json_path: str,
    obj: Any,
    ensure_ascii: bool = False,
    encoding: str = "UTF-8", 
)-> None:
    
    with open(
        file = json_path,
        mode = "w",
        encoding = encoding,
    ) as file_pointer:
        
        json.dump(
            obj = obj, 
            fp = file_pointer,
            ensure_ascii = ensure_ascii,
        )


serialize_json = lambda json_string: json.dumps(json_string, ensure_ascii=False)


deserialize_json = json.loads