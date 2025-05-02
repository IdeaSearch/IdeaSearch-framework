from openai import OpenAI
from typing import Optional

from src.API4LLMs.model_manager import is_online_model
from src.API4LLMs.model_manager import get_online_model_instance
from src.API4LLMs.model_manager import get_local_model_instance


__all__ = [
    "get_answer",
]


local_model_max_new_token = None


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
)-> str:
    pass


if __name__ == "__main__":
    pass
