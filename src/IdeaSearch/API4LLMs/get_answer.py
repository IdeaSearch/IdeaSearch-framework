from openai import OpenAI
from typing import Optional


__all__ = [
    "get_answer_online",
]


def get_answer_online(
    api_key: str,
    base_url: str,
    model: str,
    temperature: Optional[float],
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
        
    if isinstance(response, str):
        return response
    
    else:
        response_content = response.choices[0].message.content
        
        if response_content is not None:
            return response_content
        
        else:
            return ""