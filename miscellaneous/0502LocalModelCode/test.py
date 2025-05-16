import requests

def get_answer(prompt, max_length=100, temperature=0.7):
    url = "http://localhost:5000/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "max_length": max_length,
        "temperature": temperature
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            return response.json()["generated_text"]
        else:
            print(f"Error: {response.json().get('error', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"Request failed: {str(e)}")
        return None

# 示例调用
if __name__ == "__main__":
    prompt = "Hello, how are you?"
    generated_text = get_answer(prompt, max_length=150, temperature=0.0)
    if generated_text:
        print("Generated Text:")
        print(generated_text)