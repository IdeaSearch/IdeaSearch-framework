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
        prologue_section,
        epilogue_section,
        evaluators: Evaluator,
        generate_num,
        database: Database,
        console_lock: Lock,
        diary_path: str,
        record_prompt_in_diary: str,
    ):
        self.id = sampler_id
        self.database = database
        self.program_name = database.program_name
        self.prologue_section = prologue_section
        self.epilogue_section = epilogue_section
        self.generate_num = generate_num
        self.evaluators = evaluators
        self.console_lock = console_lock
        self.diary_path = diary_path
        self.record_prompt_in_diary = record_prompt_in_diary

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
                return
            else:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = f"【{self.id}号采样器】 已从数据库采样{len(examples)}个idea！",
                    )
            
            examples_section = f"举例部分（一共有{len(examples)}个例子）：\n"
            for index, example in enumerate(examples):
                examples_section += f"[第 {index + 1} 个例子]\n"
                examples_section += f"得分：{example.score:.2f}\n"
                if example.info is not None:
                    examples_section += f"说明：{example.info}\n"
                examples_section += f"内容：\n"
                examples_section += f"{example.content}\n"
            
            prompt = self.prologue_section + examples_section + self.epilogue_section
            
            with self.console_lock:
                append_to_file(
                    file_path = self.diary_path,
                    content_str = (
                        f"【{self.id}号采样器】 正在询问数据库使用何模型。。。"
                    ),
                )
            
            model, model_temperature = self.database.get_model()
            
            with self.console_lock:
                append_to_file(
                    file_path = self.diary_path,
                    content_str = (
                        f"【{self.id}号采样器】 根据各模型得分情况，依概率选择了{model}(T={model_temperature:.2f})！"
                    ),
                )
                
            if self.record_prompt_in_diary:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【{self.id}号采样器】 向{model}(T={model_temperature:.2f})发送的prompt是：\n"
                            f"{prompt}"
                        ),
                    )
            
            with self.console_lock:
                append_to_file(
                    file_path = self.diary_path,
                    content_str = (
                        f"【{self.id}号采样器】 已向{model}(T={model_temperature:.2f})"
                        f"发送prompt，正在等待回答！"
                    ),
                )
                
            generated_ideas = [None] * self.generate_num
            with ThreadPoolExecutor() as executor:

                future_to_index = {
                    executor.submit(get_answer, model, prompt, model_temperature): i
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
                                content_str = (
                                    f"【{self.id}号采样器】 尝试获取{model}(T={model_temperature:.2f})的回答时发生错误: \n{e}\n"
                                    "此轮采样失败。。。"
                                ),
                            )
                        continue
                            
            if any(idea is None for idea in generated_ideas):
                
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = f"【{self.id}号采样器】 因异常没有获得应生成的全部idea，此次采样失败。。。",
                    )
                    
                continue
                            
            with self.console_lock:
                append_to_file(
                    file_path = self.diary_path,
                    content_str = (
                        f"【{self.id}号采样器】 已收到来自{model}(T={model_temperature:.2f})"
                        f"的{self.generate_num}个回答！"
                    ),
                )
            
            # 寻找空闲 evaluator
            evaluator = self._get_idle_evaluator()
            if evaluator:
                evaluator.evaluate(generated_ideas, model, model_temperature)
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
                        content_str = f"【{self.id}号采样器】 没有找到空闲的评估器，此轮采样失败。。。",
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
                        content_str = f"【{self.id}号采样器】 已找到{evaluator.id}号评估器进行评估！",
                    )
                return evaluator
        return None
