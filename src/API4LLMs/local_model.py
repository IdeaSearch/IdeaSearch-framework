import socket
import torch
import requests
import threading
from flask import Flask
from flask import request
from flask import jsonify
from threading import Lock
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


__all__ = [
    "launch_model_inference_port",
    "shutdown_model_inference_port",
]


local_model_max_new_token = 2048
local_model_port_to_lock = {}
local_model_port_to_cuda_device_no = {}


def launch_model_inference_port(port: int, model_path: str) -> int:
    
    if port == 0:
        port = find_free_port()
    
    app = Flask(__name__)

    def init_model():
        cuda_device_no = get_free_cuda_device()
        local_model_port_to_lock[port] = cuda_device_no
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=f'cuda:{cuda_device_no}',
            trust_remote_code=True
        )
        model.config.pad_token_id = tokenizer.eos_token_id
        model.to(f'cuda:{cuda_device_no}')
        model.eval()
        return model, tokenizer
    
    model, tokenizer = init_model()

    @app.route('/generate', methods=['POST'])
    def generate():

        try:
            data = request.get_json()
            current_port = data.get('port', 0)
            temperature = data.get('temperature', 0.7)
            system_prompt = data.get('system_prompt', '')
            prompt = data.get('prompt', '')
            cuda_device_no = local_model_port_to_cuda_device_no[port]
            
            if not prompt:
                return jsonify({"error": "提示信息是必须的"}), 400
            
            with local_model_port_to_lock[current_port]:

                inputs = tokenizer(system_prompt + prompt, return_tensors="pt").to(f'cuda:{cuda_device_no}')
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
    
    local_model_port_to_lock[port] = Lock()

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
    
    
# 在这个函数中，“空闲” 的定义是内存占用最少的设备
def get_free_cuda_device():
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        raise RuntimeError("没有可用的CUDA设备")

    free_device = None
    min_memory = float('inf')

    for i in range(num_devices):
        allocated_memory = torch.cuda.memory_allocated(i)
        if allocated_memory < min_memory:
            min_memory = allocated_memory
            free_device = i
    
    return free_device
