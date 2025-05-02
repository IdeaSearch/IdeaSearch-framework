from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from safetensors.torch import load_file

app = Flask(__name__)

port_1 = 5000
port_2 = 5001
port_3 = 5002
port_4 = 5003
model_path_1 = "/data/sonny/.cache/modelscope/hub/models/Qwen/Qwen3-32B"
model_path_2 = "/data/sonny/.cache/modelscope/hub/models/Qwen/QwQ-32B"
model_path_3 = "/data/sonny/.cache/modelscope/hub/models/deepseek-ai/deepseek-coder-33b-instruct"
model_path_4 = "/data/sonny/.cache/modelscope/hub/models/Qwen/DeepSeek-R1-Distill-Qwen-32B"

port = port_1
max_length_0 = 2048
device = 'cuda:1'

if port == port_1:
    model_path = model_path_1
elif port == port_2:
    model_path = model_path_2
elif port == port_3:
    model_path = model_path_3
elif port == port_4:
    model_path = model_path_4

# 初始化模型和分词器
def init_model():
    # 指定模型的本地路径
    
    # 从模型路径加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map=device, 
        trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # 将模型移动到设备
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

model, tokenizer, device = init_model()

@app.route('/generate', methods=['POST'])
def generate():
    max_length = max_length_0
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        #max_length = data.get('max_length', 100)
        temperature = data.get('temperature', 0.7)
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # 生成文本
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if temperature == 0.0:
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_length, 
                do_sample=False,
            )    
        else:
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_length, 
                do_sample=True, 
                temperature=temperature,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({
            "generated_text": generated_text,
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=False)