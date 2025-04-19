# main.py
from src.FunSearch.modules import ProblemConfig
from src.FunSearch.modules import ModelLoader
from src.FunSearch.modules import CodeGenerator
from src.FunSearch.modules import CodeEvaluator 
from src.FunSearch.modules import IslandManager
import numpy as np
import os 
import datetime

class FunSearch:
    def __init__(self):
        self.config = ProblemConfig()
        self.model_loader = ModelLoader(self.config)
        self.code_gen = CodeGenerator(self.model_loader, self.config)
        self.evaluator = CodeEvaluator(self.config)
        self.island_manager = IslandManager(self.config)
        
    def run(self, max_epochs=1000):
        best_score = -np.inf
        history = []
        
        for epoch in range(max_epochs):
            island_idx = epoch % self.config.NUM_ISLANDS
            
            # 生成提示
            examples = self.island_manager.select_examples(island_idx)
            prompt = self._build_prompt(examples)
            
            # 生成并评估代码
            candidates = self.code_gen.generate(prompt)
            for code in candidates:
                score = self.evaluator.evaluate(code)
                if score > best_score:
                    best_score = score
                    self._save_best(code, score)
                
                self.island_manager.update_island(island_idx, code, score)
            
            # 定期维护岛屿
            if epoch % self.config.ISLAND_CLEAN_INTERVAL == 0:
                self.island_manager.clean_islands()
                
    def _build_prompt(self, examples):
        """构建包含示例的代码生成提示"""
        problem_desc = (
            r"</think>"
            "<我们正在尝试通过生成一个ansatz并对其变分来寻找最适合制备模拟规范场初态的量子线路>\n\n"

            "需要实现的函数框架：\n"
            "from qiskit import QuantumCircuit, transpile\n"
            "from qiskit.circuit import Parameter,ParameterVector\n"
            f"def {self.config.TARGET_FUNCTION}:\n"
            "\t# 这里应该是ansatz实现\n"
        )

        existing_code = (
            "#def do_vqe():\n"
            "\tvqe = VQE(estimator=estimator,ansatz=qc_ansatz, optimizer=optimizer,callback=callback)\n"
            "\tresult = vqe.compute_minimum_eigenvalue(operator=H)\n"
            "\treturn result.eigenvalue\n"
            "# 其他相关函数结构...\n"
        )

        examples_section = "\n".join(
            [f"# 示例代码 {i+1}：\n{example}" 
             for i, example in enumerate(examples)]
        )

        requirements = (
            "\n# 生成要求：\n"
            "# 1. 必须正确定义生成函数\n"
            "# 2. 函数输出应具有物理合理性（纠缠，转动等）\n"
            "# 3. 避免直接复制示例，至少完成以下一项（泛化之前的代码，简化之前的代码，进化之前的代码，创建独特的实现）\n"
            "# 4. 确保代码包含必要的数学运算库导入\n"
            "# 5. 注意代码参数尽可能的简洁性和可拓展性\n"
            "# 6. 将最终代码包裹在Python代码块中\n"
            r"</answer>"
        )

        return f"{problem_desc}\n{existing_code}\n{examples_section}\n{requirements}"

    def _save_best(self, code, score):
        """保存最佳结果到文件"""
        save_dir = "best_results"
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/best_{timestamp}_{int(score)}.py"
        
        meta_info = (
            f"# Best solution @ {timestamp}\n"
            f"# Score: {score}\n"
            f"# Config: {self.config.__dict__}\n\n"
        )

        try:
            with open(filename, "w") as f:
                f.write(meta_info)
                f.write("# --- GENERATED CODE ---\n")
                f.write(code)
            
            # 同时更新latest文件
            latest_path = f"{save_dir}/latest_best.py"
            with open(latest_path, "w") as f:
                f.write(meta_info)
                f.write(code)
                
            print(f"√ 保存最佳结果到 {filename}")
        except Exception as e:
            print(f"× 保存失败: {str(e)}")

        # 同时记录到历史文件
        history_path = f"{save_dir}/search_history.md"
        with open(history_path, "a") as f:
            f.write(f"## {timestamp}\n")
            f.write(f"- **Score**: {score}\n")
            f.write(f"- **Code**: [查看代码]({filename})\n")
            f.write("\n```python\n")
            f.write(code[:500] + "\n...\n")  # 保存部分代码预览
            f.write("```\n\n")

if __name__ == "__main__":
    searcher = FunSearch()
    searcher.run()