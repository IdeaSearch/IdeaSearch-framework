# 导入 Flask 和用于 HTTP 请求处理的模块
from flask import Flask, request, jsonify

# 导入 HuggingFace 的 Transformers 库
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from safetensors.torch import load_file

# 创建一个 Flask 应用实例，Flask 是一个轻量级 Web 框架
app = Flask(__name__)

# 初始化模型和分词器的函数
def init_model():
    # 指定模型的本地路径（可以替换为你自己下载好的模型路径）
    model_path = "/data/sonny/.cache/modelscope/hub/models/Qwen/Qwen3-32B"
    
    # 从本地路径加载分词器（Tokenizer）
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=False, 
        trust_remote_code=True
    )
    
    # 从本地路径加载模型，并设置设备与精度
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map='cuda', 
        trust_remote_code=True
    )
    
    # 设置模型的填充 token（某些模型没有默认 pad_token）
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # 判断是否有 GPU，有则用 GPU，否则用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 设置为推理模式（不会更新梯度）
    
    return model, tokenizer, device

# 初始化模型
model, tokenizer, device = init_model()

# 定义一个路由 '/generate'，处理 POST 请求，用于生成文本
@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()  # 从请求中获取 JSON 数据
        prompt = data.get('prompt', '')  # 获取 prompt 字符串
        max_length = data.get('max_length', 100)  # 最长生成长度
        temperature = data.get('temperature', 0.7)  # 控制随机性的参数
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # 将输入编码为张量，并转移到设备上
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 根据 temperature 控制采样行为（temperature=0 相当于 greedy）
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
        
        # 解码生成的张量为字符串
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 返回 JSON 响应
        return jsonify({
            "generated_text": generated_text,
            "status": "success"
        })
    
    except Exception as e:
        # 如果出错，返回错误信息
        return jsonify({"error": str(e)}), 500

# 定义一个健康检查接口，用于测试服务是否正常运行
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

# 程序入口：启动 Flask 应用
if __name__ == '__main__':
    # 让 Flask 在 0.0.0.0 上监听所有请求（可远程访问），端口为 5000，关闭调试模式
    app.run(host='0.0.0.0', port=5000, debug=False)
