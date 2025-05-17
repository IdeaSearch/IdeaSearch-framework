from threading import Lock
from typing import Optional
from typing import Callable
from typing import Tuple
from typing import List
from os.path import basename
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from src.utils import append_to_file
from src.API4LLMs.get_answer import get_answer
from src.IdeaSearch.evaluator import Evaluator
from src.IdeaSearch.database import Database


class Sampler:
    def __init__(
        self, 
        sampler_id: int, 
        system_prompt: str,
        prologue_section: str,
        epilogue_section: str,
        database: Database,
        evaluators: List[Evaluator],
        generate_num: int,
        console_lock: Lock,
        diary_path: str,
        record_prompt_in_diary: bool,
        filter_func: Optional[Callable[[str], str]],
    ):
        self.id = sampler_id
        self.database = database
        self.program_name = database.program_name
        self.system_prompt = system_prompt
        self.prologue_section = prologue_section
        self.epilogue_section = epilogue_section
        self.generate_num = generate_num
        self.evaluators = evaluators
        self.console_lock = console_lock
        self.diary_path = diary_path
        self.record_prompt_in_diary = record_prompt_in_diary
        self.filter_func = filter_func

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
                        content_str = f"【{self.id}号采样器】 已从数据库采样 {len(examples)} 个idea！",
                    )
            
            examples_section = f"举例部分（一共有{len(examples)}个例子）：\n"
            for index, example in enumerate(examples):
                idea, score, info, similar_num, similarity_prompt, path = example
                examples_section += f"[第 {index + 1} 个例子]\n"
                examples_section += f"内容：\n"
                
                if self.filter_func is not None:
                    try:
                        idea = self.filter_func(idea)
                    except Exception as error:
                        with self.console_lock:
                            append_to_file(
                                file_path = self.diary_path,
                                content_str = (
                                    f"【{self.id}号采样器】 "
                                    f"将 filter_func 作用于 {basename(path)} 时发生错误：\n"
                                    f"{error}\n延用原来的 idea ！"
                                ),
                            )

                examples_section += f'{idea}\n'
                examples_section += f"评分：{score:.2f}\n"
                if info is not None:
                    examples_section += f"评语：{info}\n"
                if similar_num is not None:
                    examples_section += (
                        f"重复情况说明：目前数据库里有{similar_num}个例子和这个例子相似\n"
                        f"{similarity_prompt}\n"
                    )
            
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
                        f"【{self.id}号采样器】 根据各模型得分情况，依概率选择了 {model}(T={model_temperature:.2f}) ！"
                    ),
                )
                
            if self.record_prompt_in_diary:
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【{self.id}号采样器】 向 {model}(T={model_temperature:.2f}) 发送的 system prompt 是：\n"
                            f"{self.system_prompt}"
                        ),
                    )
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = (
                            f"【{self.id}号采样器】 向 {model}(T={model_temperature:.2f}) 发送的 prompt 是：\n"
                            f"{prompt}"
                        ),
                    )
            
            with self.console_lock:
                append_to_file(
                    file_path = self.diary_path,
                    content_str = (
                        f"【{self.id}号采样器】 已向 {model}(T={model_temperature:.2f}) "
                        f"发送prompt，正在等待回答！"
                    ),
                )
                
            generated_ideas = [""] * self.generate_num
            with ThreadPoolExecutor() as executor:

                future_to_index = {
                    executor.submit(
                        get_answer, 
                        model, 
                        model_temperature, 
                        self.system_prompt, 
                        prompt
                    ): i
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
                                    f"【{self.id}号采样器】 尝试获取 {model}(T={model_temperature:.2f}) 的回答时发生错误: \n{e}\n"
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
                        f"【{self.id}号采样器】 已收到来自 {model}(T={model_temperature:.2f}) "
                        f"的 {self.generate_num} 个回答！"
                    ),
                )
            
            example_idea_paths = [current_idea[-1] for current_idea in examples]
            example_idea_scores = [current_idea[1] for current_idea in examples]
            
            evaluator = self._get_idle_evaluator()
            if evaluator:
                
                evaluator.evaluate(
                    generated_ideas = generated_ideas, 
                    model = model, 
                    model_temperature = model_temperature, 
                    example_idea_paths = example_idea_paths, 
                    example_idea_scores = example_idea_scores
                )
                
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


    def _get_idle_evaluator(
        self
    )-> Optional[Evaluator]:
        for evaluator in self.evaluators:
            if evaluator.try_acquire():
                with self.console_lock:
                    append_to_file(
                        file_path = self.diary_path,
                        content_str = f"【{self.id}号采样器】 已找到{evaluator.id}号评估器进行评估！",
                    )
                return evaluator
        return None
