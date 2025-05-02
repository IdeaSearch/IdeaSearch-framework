import requests
from openai import OpenAI
from typing import Optional
from src.API4LLMs.model_manager import is_online_model
from src.API4LLMs.model_manager import get_online_model_instance
from src.API4LLMs.model_manager import get_local_model_instance


__all__ = [
    "get_answer",
]


def get_answer(
    model_name : str, 
    model_temperature : Optional[str],
    system_prompt: str,
    prompt : str,
):
    
    if is_online_model(model_name):
        api_key, base_url, model = get_online_model_instance(model_name)
        
        return get_answer_online(
            api_key = api_key,
            base_url = base_url,
            model = model,
            temperature = model_temperature,
            system_prompt = system_prompt,
            prompt = prompt,
        )
    
    else:
        port = get_local_model_instance(model_name)
        
        return get_answer_local(
            port = port,
            temperature = model_temperature,
            system_prompt = system_prompt,
            prompt = prompt,
        )


def get_answer_online(
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    system_prompt: str,
    prompt: str,
)-> str:

    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)
        
    if temperature is not None:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature = temperature,
            stream = False
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream = False
        )

    return response.choices[0].message.content


def get_answer_local(
    port: int,
    temperature: float,
    system_prompt: str,
    prompt: str,
) -> str:
    
    url = f"http://127.0.0.1:{port}/generate"
    
    headers = {"Content-Type": "application/json"}
    data = {
        "temperature": temperature,
        "system_prompt": system_prompt,
        "prompt": prompt,
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            return response.json().get("generated_text", "")
        else:
            print(f"【Model Manager】 错误: {response.json().get('error', '未知错误')}")
            return ""
    except Exception as e:
        print(f"【Model Manager】 请求失败: {str(e)}")
        return ""
