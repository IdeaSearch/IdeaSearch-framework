import time
import random
from threading import Lock
from src.API4LLMs.get_answer import get_answer
from src.IdeaSearch.evaluator import Evaluator
from src.IdeaSearch.database import Database
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from src.utils import append_to_file


class Sampler:
    def __init__(
        self, 
        sampler_id, 
        model,
        prologue_section,
        epilogue_section,
        evaluators: Evaluator,
        generate_num,
        database: Database,
        model_temperature: float,
        console_lock: Lock,
        diary_path: str,
    ):
        self.id = sampler_id + 1
        self.database = database
        self.program_name = database.program_name
        self.model = model
        self.prologue_section = prologue_section
        self.epilogue_section = epilogue_section
        self.generate_num = generate_num
        self.evaluators = evaluators
        self.model_temperature = model_temperature
        self.console_lock = console_lock
        self.diary_path = diary_path

    def run(self):
        
        with self.console_lock:
            append_to_file(
                file_path = self.diary_path,
                content_str = f"【{self.id}号采样器】 已开始工作！",
            )
        
        while self.database.get_status() == "Running":
            
            with self.console_lock:
                append_to_file(
                    file_path = self.diary_path,
                    content_str = f"【{self.id}号采样器】 已开始新一轮采样！",
                )
            
            examples = self.database.get_examples()
            if examples is None: 
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = f"【{self.id}号采样器】 工作结束。",
                    )
                break
            else:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = f"【{self.id}号采样器】 已从数据库采样{len(examples)}个idea！",
                    )
            
            examples_section = ""
            for index, example in enumerate(examples):
                examples_section += f"[Example {index + 1}]\n"
                examples_section += f"Score: {example.score}\n"
                if example.info is not None:
                    examples_section += f"Info: {example.info}\n"
                examples_section += f"Content:\n"
                examples_section += f"{example.content}\n"
            
            prompt = self.prologue_section + examples_section + self.epilogue_section
            
            with self.console_lock:
                append_to_file(
                    file_path = self.diary_path,
                    content_str = (
                        f"【{self.id}号采样器】 已向{self.model}(T={self.model_temperature:.1f})"
                        f"发送prompt，正等待回答！",
                    )
                )
                
            generated_ideas = [None] * self.generate_num
            with ThreadPoolExecutor() as executor:

                future_to_index = {
                    executor.submit(get_answer, self.model, prompt, self.model_temperature): i
                    for i in range(self.generate_num)
                }

                for future in as_completed(future_to_index):
                    i = future_to_index[future]
                    try:
                        generated_ideas[i] = future.result()
                    except Exception as e:
                        with self.console_lock:
                            append_to_file(
                                file_path = self.diary_path,
                                content_str = f"【{self.id}号采样器】 尝试获取{self.model}的回答时发生错误: {e}",
                            )
                            
            with self.console_lock:
                append_to_file(
                    file_path = self.diary_path,
                    content_str = (
                        f"【{self.id}号采样器】 已收到来自{self.model}(T={self.model_temperature:.1f})"
                        f"的{self.generate_num}个回答！",
                    )
                )
            
            # 寻找空闲 evaluator
            evaluator = self._get_idle_evaluator()
            if evaluator:
                evaluator.evaluate(generated_ideas)
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = f"【{self.id}号采样器】 已释放{evaluator.id}号评估器。",
                    )
                evaluator.release()
            else:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = f"【{self.id}号采样器】 没有找到空闲的评估器，此次采样失败...",
                    )
        
        with self.console_lock:    
            append_to_file(
                file_path = self.diary_path,
                content_str = f"【{self.id}号采样器】 工作结束。",
            )


    def _get_idle_evaluator(self) -> Evaluator:
        for evaluator in self.evaluators:
            if evaluator.try_acquire():
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = f"【{self.id}号采样器】 已找到{evaluator.id}号评估器对{self.model}的回答进行评估。"
                    )
                return evaluator
        return None
