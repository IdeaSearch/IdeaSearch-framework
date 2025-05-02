from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from safetensors.torch import load_file

app = Flask(__name__)

# 初始化模型和分词器
def init_model():
    # 指定模型的本地路径
    model_path = "/data/sonny/.cache/modelscope/hub/models/Qwen/Qwen3-32B"
    
    # 从模型路径加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map='cuda', 
        trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # 将模型移动到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

model, tokenizer, device = init_model()

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        max_length = data.get('max_length', 100)
        temperature = data.get('temperature', 0.7)
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # 生成文本
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if temperature == 0.0:
            outputs = model.generate(
                **inputs, 
                max_length=max_length, 
                do_sample=False,
            )    
        else:
            outputs = model.generate(
                **inputs, 
                max_length=max_length, 
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
    app.run(host='0.0.0.0', port=5000, debug=False)