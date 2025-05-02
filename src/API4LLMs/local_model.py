import socket
import torch
import requests
import threading
from flask import Flask
from flask import request
from flask import jsonify
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


__all__ = [
    "launch_model_inference_port",
    "shutdown_model_inference_port",
]


local_model_max_new_token = 2048


def launch_model_inference_port(port: int, model_path: str) -> int:
    
    if port == 0:
        port = find_free_port()
    
    app = Flask(__name__)

    def init_model():
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='cuda:1',
            trust_remote_code=True
        )
        model.config.pad_token_id = tokenizer.eos_token_id
        model.to('cuda:1')
        model.eval()
        return model, tokenizer
    
    model, tokenizer = init_model()

    @app.route('/generate', methods=['POST'])
    def generate():

        try:
            data = request.get_json()
            prompt = data.get('prompt', '')
            temperature = data.get('temperature', 0.7)
            
            if not prompt:
                return jsonify({"error": "提示信息是必须的"}), 400

            inputs = tokenizer(prompt, return_tensors="pt").to('cuda:1')
            if temperature == 0.0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=local_model_max_new_token,
                    do_sample=False,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=local_model_max_new_token,
                    do_sample=True,
                    temperature=temperature,
                )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return jsonify({
                "generated_text": generated_text,
                "status": "成功"
            })
        
        except Exception as e:
            return jsonify({"error": f"发生错误: {str(e)}"}), 500

    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "健康"})
    
    @app.route('/shutdown', methods=['POST'])
    def shutdown():
        request.environ.get('werkzeug.server.shutdown')()
        return jsonify({"status": "服务器正在关闭"}), 200

    def run_app():
        app.run(host='0.0.0.0', port=port, debug=False)

    thread = threading.Thread(target=run_app)
    thread.start()

    return port


def shutdown_model_inference_port(port: int):
    shutdown_url = f'http://127.0.0.1:{port}/shutdown'
    try:
        response = requests.post(shutdown_url)
        if response.status_code == 200:
            print(f"【Model Manager】 端口 {port} 上的服务器正在关闭。")
        else:
            print(f"【Model Manager】 无法关闭端口 {port} 上的服务器。")
    except Exception as e:
        print(f"【Model Manager】 关闭服务器时发生错误: {e}")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', 0))
        s.listen(1)
        address = s.getsockname()
        return address[1]
