import torch

# config.py
class BaseConfig:
    # 模型配置
    MODEL_PATH = "/data/konformal/nn/model/DeepSeek-R1-Distill-Qwen-32B"
    DEVICE = "cuda:1"
    DTYPE = torch.bfloat16
    
    # 搜索参数
    MAX_ATTEMPTS = 3
    EVAL_TIMEOUT = 240  # 秒
    CODE_SAVE_PERIOD = 1
    N_DECIMAL = 1  # 分数取整位数
    
    # 温度调度
    INIT_TEMP = 5
    TEMP_DECAY = 1.0
    ROUND_LIMIT = 307
    
    # 岛屿参数
    NUM_ISLANDS = 10
    ISLAND_CLEAN_INTERVAL = 50
    SHOT_NUM = 3

    # 新增岛屿清理参数
    KILL_RATIO = 0.5          # 每次清理替换50%的岛屿
    MAX_ISLAND_CAPACITY = 50  # 每个岛屿最大代码容量
    STAGNATION_LIMIT = 10     # 停滞周期限制
    TEMP_INCREASE_RATE = 0.05 # 温度增长速率
    MIN_TEMP = 0.1            # 最低温度
    MAX_TEMP = 100.0          # 最高温度

class ProblemConfig(BaseConfig):
    # 问题特定配置
    TARGET_FUNCTION = "create_ansatz(n)"
    EVALUATOR_SCRIPT = "evaluator.py"
    EVALUATOR_SCRIPT_ISLAND = "island.py"
    TEMPLATE_FILES = ["templet1.txt", "templet2.txt", "templet3.txt"]
    INITIAL_SCORES = [-195, -195, -195]

# model_loader.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ModelLoader:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None

    def load_base_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.MODEL_PATH, 
            use_fast=False,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_PATH,
            torch_dtype=self.config.DTYPE,
            device_map=self.config.DEVICE,
            trust_remote_code=True
        )
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def load_finetuned(self, adapter_path):
        if self.model is None:
            self.load_base_model()
        # 加载微调权重逻辑
        # ...

# code_generator.py
import re
import ast
from transformers import logging

logging.set_verbosity_error()

class CodeGenerator:
    def __init__(self, model_loader, config):
        model_loader.load_base_model()
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        self.config = config
        
    def _extract_function(self, code):
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == self.config.TARGET_FUNCTION.split('(')[0]:
                    return ast.unparse(node)
            return f"def {self.config.TARGET_FUNCTION}:\n\tqc = QuantumCircuit(n)\n\treturn qc"
        except:
            return ''
    
    def _extract_import(self, code):
        """从代码中提取所有import语句并进行格式化"""
        try:
            tree = ast.parse(code)
            imports = []
            
            # 遍历AST节点
            for node in ast.walk(tree):
                # 处理普通import
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imp_str = f"import {alias.name}"
                        if alias.asname:
                            imp_str += f" as {alias.asname}"
                        imports.append(imp_str)
                
                # 处理from...import
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    import_str = f"from {module} import "
                    
                    # 处理不同导入形式
                    import_list = []
                    for n in node.names:
                        if n.name == "*":
                            import_str = f"from {module} import *"
                            import_list = [import_str]
                            break
                        item = n.name
                        if n.asname:
                            item += f" as {n.asname}"
                        import_list.append(item)
                    
                    if import_list and import_list[0] != "*":
                        imports.append(import_str + ", ".join(import_list))

            # 去重并保持顺序
            seen = set()
            unique_imports = []
            for imp in imports:
                if imp not in seen:
                    seen.add(imp)
                    unique_imports.append(imp)
            
            return "\n".join(unique_imports)
        
        except (SyntaxError, AttributeError):
            return ""
            
    def generate(self, prompt, temperature=0.9, top_p=0.9, max_length=2048):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.DEVICE)
        
        generated_ids = self.model.generate(
            inputs.input_ids,
            do_sample=True,
            max_length=max_length+len(prompt),
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=self.config.MAX_ATTEMPTS
        )
        
        candidates = []
        for gid in generated_ids:
            code = self.tokenizer.decode(gid, skip_special_tokens=True)
            #print(code[len(prompt):])
            if match := re.search(r"```python(.*?)```", code, re.DOTALL):
                code = match.group(1).strip()
            candidates.append(self._extract_import(code)+ '\n\n' + self._extract_function(code))
        
        return [c for c in candidates if c is not None]
    
# evaluator.py
import subprocess
import numpy as np

class CodeEvaluator:
    def __init__(self, config):
        self.config = config
        self.length_penalty = lambda l: min(2*(l//40)/10, 2)
        
    def evaluate(self, code):
        try:
            with open(self.config.EVALUATOR_SCRIPT_ISLAND, 'w') as f:
                f.write(code)
                
            result = subprocess.run(
                ['python', self.config.EVALUATOR_SCRIPT],
                capture_output=True,
                text=True,
                timeout=self.config.EVAL_TIMEOUT
            )

            #print(result)
            
            if score := re.search(r"输出结果：(-?\d+\.?\d*)", result.stdout):
                raw_score = float(score.group(1))
                #print(raw_score)
                penalty = self.length_penalty(len(code))
                return round(raw_score - penalty, self.config.N_DECIMAL)
            return -np.inf
        except:
            return -np.inf

# island_manager.py
import random
import numpy as np
from collections import defaultdict

class IslandManager:
    def __init__(self, config):
        self.config = config
        self.islands = self._initialize_islands()
        self.temperature = config.INIT_TEMP
        self.best_history = []
        self.stagnation_counter = 0

    def _initialize_islands(self):
        """使用模板文件初始化岛屿种群"""
        templates = []
        for file in self.config.TEMPLATE_FILES:
            with open(file) as f:
                templates.append(f.read())
        return [
            {score: code for score, code in zip(self.config.INITIAL_SCORES, templates)}
            for _ in range(self.config.NUM_ISLANDS)
        ]

    def select_examples(self, island_idx):
        """基于当前温度选择示例代码"""
        island = self.islands[island_idx]
        #print(island)
        if not island:
            return random.sample(self.config.INITIAL_SCORES, self.config.SHOT_NUM)
        
        scores = np.array(list(island.keys()))
        #print(scores)
        probs = np.exp(scores / self.temperature)
        probs /= probs.sum()  # 防止除以零
        
        selected = set()
        N=0
        while len(selected) < self.config.SHOT_NUM and N<20:
            idx = np.random.choice(len(scores), p=probs)
            selected.add(scores[idx])
            N+=1
        return [island[s] for s in selected]

    def update_island(self, island_idx, code, score):
        """更新指定岛屿的代码库"""
        self.islands[island_idx][score] = code
        # 保持每个岛屿的代码数量不超过容量限制
        if len(self.islands[island_idx]) > self.config.MAX_ISLAND_CAPACITY:
            min_score = min(self.islands[island_idx].keys())
            del self.islands[island_idx][min_score]

    def clean_islands(self):
        """执行岛屿清理和温度调整"""
        # 1. 收集各岛屿的精英代码
        elite_pool = defaultdict(list)
        for idx, island in enumerate(self.islands):
            if island:
                top_score = max(island.keys())
                elite_pool[top_score].append((idx, island[top_score]))
        
        # 2. 没有有效精英时保持现状
        if not elite_pool:
            return

        # 3. 排序精英池
        sorted_elites = sorted(elite_pool.items(), reverse=True)
        best_score = sorted_elites[0][0]
        
        # 4. 记录历史最佳
        if not self.best_history or best_score > self.best_history[-1]:
            self.best_history.append(best_score)
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        # 5. 替换表现差的岛屿
        replace_num = int(self.config.NUM_ISLANDS * self.config.KILL_RATIO)
        replace_candidates = []
        
        # 找到表现最差的岛屿索引
        island_scores = [(idx, max(island.keys())) for idx, island in enumerate(self.islands)]
        worst_islands = sorted(island_scores, key=lambda x: x[1])[:replace_num]
        
        # 用精英替换最差岛屿
        for widx, _ in worst_islands:
            # 随机选择一个精英模板
            elite_score, elite_group = random.choice(sorted_elites[:3])  # 取前三名的精英
            src_idx, elite_code = random.choice(elite_group)
            
            # 保留历史最佳代码的10%
            keep_prob = 0.1 if elite_score == best_score else 0.01
            new_island = {
                k: v for k, v in self.islands[widx].items() 
                if random.random() < keep_prob
            }
            new_island[elite_score] = elite_code
            self.islands[widx] = new_island

        # 6. 调整温度参数
        #if self.stagnation_counter > self.config.STAGNATION_LIMIT:
        #    self.temperature = max(
        #        self.temperature * self.config.TEMP_DECAY,
        #        self.config.MIN_TEMP
        #    )
        #    self.stagnation_counter = 0
        #else:
        #    self.temperature = min(
        #        self.temperature * (1 + self.config.TEMP_INCREASE_RATE),
        #        self.config.MAX_TEMP
        #    )