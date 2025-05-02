from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from werkzeug.serving import make_server
import torch
import threading

app = Flask(__name__)

# 初始化模型和分词器
def init_model():
    model_path = "/data/sonny/.cache/modelscope/hub/models/Qwen/Qwen3-32B"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='cuda',
        trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.eos_token_id

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

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if temperature == 0.0:
            outputs = model.generate(**inputs, max_length=max_length, do_sample=False)
        else:
            outputs = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=temperature)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({
            "generated_text": generated_text,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 使用线程启动 Flask，并获取真实端口
class ServerThread(threading.Thread):
    def __init__(self, app, host='0.0.0.0', port=0):
        threading.Thread.__init__(self)
        self.server = make_server(host, port, app)
        self.port = self.server.socket.getsockname()[1]
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        print(f"⚡ Flask running on port {self.port}")
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()

if __name__ == '__main__':
    server_thread = ServerThread(app)
    server_thread.start()

    # 你可以在程序中使用这个端口
    print(f"✅ Flask 实际监听端口为: {server_thread.port}")
