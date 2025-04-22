from src.API4LLMs.model_manager import model_manager
from openai import OpenAI

def get_answer(model_name, question):
    
    api_key = model_manager.models[model_name]["api_key"]
    base_url = model_manager.models[model_name]["base_url"]
    model = model_manager.models[model_name]["model"]
    
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)
        
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": question},
        ],
        stream=False
    )
    # print(response)
    return response.choices[0].message.content

if __name__ == "__main__":
    pass
