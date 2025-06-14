from threading import Lock
from typing import Optional
from os.path import basename
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from src.IdeaSearch.utils import append_to_file


__all__ = [
    "Sampler",
]


class Sampler:
    def __init__(
        self, 
        ideasearcher,
        sampler_id: int, 
        island,
        evaluators,
        console_lock: Lock,
    ):
        
        self.id = sampler_id
        self.island = island
        self.ideasearcher = ideasearcher
        self.evaluators = evaluators
        self.console_lock = console_lock

    def run(self):
        
        diary_path = self.ideasearcher.get_diary_path()
        system_prompt = self.ideasearcher.get_system_prompt()
        prologue_section = self.ideasearcher.get_prologue_section()
        epilogue_section = self.ideasearcher.get_epilogue_section()
        generate_num = self.ideasearcher.get_generate_num()
        filter_func = self.ideasearcher.get_filter_func()
        record_prompt_in_diary = self.ideasearcher.get_record_prompt_in_diary()
        generate_prompt_func = self.ideasearcher.get_generate_prompt_func()
        
        assert system_prompt is not None
        assert prologue_section is not None
        assert epilogue_section is not None
        
        with self.console_lock:
            append_to_file(
                file_path = diary_path,
                content_str = f"【{self.island.id}号岛屿的{self.id}号采样器】 已开始工作！",
            )
        
        while self.island.get_status() == "Running":
            
            with self.console_lock:
                append_to_file(
                    file_path = diary_path,
                    content_str = f"【{self.island.id}号岛屿的{self.id}号采样器】 已开始新一轮采样！",
                )
                
            if generate_prompt_func is None:
            
                examples = self.island.get_examples()
                if examples is None: 
                    with self.console_lock:
                        append_to_file(
                            file_path = diary_path,
                            content_str = f"【{self.island.id}号岛屿的{self.id}号采样器】 工作结束。",
                        )
                    return
                else:
                    with self.console_lock:
                        append_to_file(
                            file_path = diary_path,
                            content_str = f"【{self.island.id}号岛屿的{self.id}号采样器】 已从{self.island.id}号岛屿采样 {len(examples)} 个idea！",
                        )
                        
                example_idea_paths = [current_idea[-2] for current_idea in examples]
                example_idea_scores = [current_idea[1] for current_idea in examples]
                example_idea_levels = [current_idea[-1] for current_idea in examples]
                level = max(example_idea_levels) + 1
                
                examples_section = f"举例部分（一共有{len(examples)}个例子）：\n"
                for index, example in enumerate(examples):
                    idea, score, info, similar_num, similarity_prompt, path, _ = example
                    examples_section += f"[第 {index + 1} 个例子]\n"
                    examples_section += f"内容：\n"
                    
                    if filter_func is not None:
                        try:
                            idea = filter_func(idea)
                        except Exception as error:
                            with self.console_lock:
                                append_to_file(
                                    file_path = diary_path,
                                    content_str = (
                                        f"【{self.island.id}号岛屿的{self.id}号采样器】 "
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
                            f"重复情况说明：{self.island.id}号岛屿里有{similar_num}个例子和这个例子相似\n"
                            f"{similarity_prompt}\n"
                        )
                
                prompt = prologue_section + examples_section + epilogue_section
                
            else:
                
                island_ideas = self.island.get_ideas_called_when_generate_prompt_func_set()
                if island_ideas is None: 
                    with self.console_lock:
                        append_to_file(
                            file_path = diary_path,
                            content_str = f"【{self.island.id}号岛屿的{self.id}号采样器】 工作结束。",
                        )
                    return
                else:
                    with self.console_lock:
                        append_to_file(
                            file_path = diary_path,
                            content_str = f"【{self.island.id}号岛屿的{self.id}号采样器】 已从{self.island.id}号岛屿获得所有 ideas ，预备调用自定义的 generate prompt 函数！",
                        )
                        
                example_idea_paths = None
                example_idea_scores = None
                example_idea_levels = None
                level = None
                
                ideas: list[str] = []
                scores: list[float] = []
                infos: list[Optional[str]] = []
                
                for idea in island_ideas:
                        
                    assert idea.content is not None
                    assert idea.score is not None
                    
                    ideas.append(idea.content)
                    scores.append(idea.score)
                    infos.append(idea.info)
                
                try:
                    prompt = generate_prompt_func(
                        ideas,
                        scores,
                        infos,
                    )
                    
                except Exception as error:
                    with self.console_lock:
                        append_to_file(
                            file_path = diary_path,
                            content_str = (
                                f"【{self.island.id}号岛屿的{self.id}号采样器】 "
                                f"使用自定义的 generate_prompt_func 时发生错误：\n"
                                f"{error}\nIdeaSearch终止！"
                            ),
                        )
                    return
            
            with self.console_lock:
                append_to_file(
                    file_path = diary_path,
                    content_str = (
                        f"【{self.island.id}号岛屿的{self.id}号采样器】 正在询问 IdeaSearcher 使用何模型。。。"
                    ),
                )
            
            model, model_temperature = self.ideasearcher.get_model()
            
            with self.console_lock:
                append_to_file(
                    file_path = diary_path,
                    content_str = (
                        f"【{self.island.id}号岛屿的{self.id}号采样器】 根据各模型得分情况，依概率选择了 {model}(T={model_temperature:.2f}) ！"
                    ),
                )
                
            if record_prompt_in_diary:
                with self.console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content_str = (
                            f"【{self.island.id}号岛屿的{self.id}号采样器】 向 {model}(T={model_temperature:.2f}) 发送的 system prompt 是：\n"
                            f"{system_prompt}"
                        ),
                    )
                    append_to_file(
                        file_path = diary_path,
                        content_str = (
                            f"【{self.island.id}号岛屿的{self.id}号采样器】 向 {model}(T={model_temperature:.2f}) 发送的 prompt 是：\n"
                            f"{prompt}"
                        ),
                    )
            
            with self.console_lock:
                append_to_file(
                    file_path = diary_path,
                    content_str = (
                        f"【{self.island.id}号岛屿的{self.id}号采样器】 已向 {model}(T={model_temperature:.2f}) "
                        f"发送prompt，正在等待回答！"
                    ),
                )
                
            generated_ideas = [""] * generate_num
            with ThreadPoolExecutor() as executor:

                future_to_index = {
                    executor.submit(
                        self.ideasearcher._get_answer, 
                        model, 
                        model_temperature, 
                        system_prompt, 
                        prompt
                    ): i
                    for i in range(generate_num)
                }

                for future in as_completed(future_to_index):
                    i = future_to_index[future]
                    try:
                        generated_ideas[i] = future.result()
                    except Exception as e:
                        with self.console_lock:
                            append_to_file(
                                file_path = diary_path,
                                content_str = (
                                    f"【{self.island.id}号岛屿的{self.id}号采样器】 尝试获取 {model}(T={model_temperature:.2f}) 的回答时发生错误: \n{e}\n"
                                    "此轮采样失败。。。"
                                ),
                            )
                        continue
                            
            if any(idea is None for idea in generated_ideas):
                
                with self.console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content_str = f"【{self.island.id}号岛屿的{self.id}号采样器】 因异常没有获得应生成的全部idea，此次采样失败。。。",
                    )
                    
                continue
                            
            with self.console_lock:
                append_to_file(
                    file_path = diary_path,
                    content_str = (
                        f"【{self.island.id}号岛屿的{self.id}号采样器】 已收到来自 {model}(T={model_temperature:.2f}) "
                        f"的 {generate_num} 个回答！"
                    ),
                )
            
            evaluator = self._get_idle_evaluator()
            if evaluator:
                
                evaluator.evaluate(
                    generated_ideas = generated_ideas, 
                    model = model, 
                    model_temperature = model_temperature, 
                    example_idea_paths = example_idea_paths, 
                    example_idea_scores = example_idea_scores,
                    level = level,
                )
                
                with self.console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content_str = f"【{self.island.id}号岛屿的{self.id}号采样器】 已释放{evaluator.id}号评估器。",
                    )
                evaluator.release()
            else:
                with self.console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content_str = f"【{self.island.id}号岛屿的{self.id}号采样器】 没有找到空闲的评估器，此轮采样失败。。。",
                    )
        
        with self.console_lock:    
            append_to_file(
                file_path = diary_path,
                content_str = f"【{self.island.id}号岛屿的{self.id}号采样器】 工作结束。",
            )


    def _get_idle_evaluator(
        self
    ):
        
        diary_path = self.ideasearcher.get_diary_path()
        
        for evaluator in self.evaluators:
            if evaluator.try_acquire():
                with self.console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content_str = f"【{self.island.id}号岛屿的{self.id}号采样器】 已找到{evaluator.id}号评估器进行评估！",
                    )
                return evaluator
        return None
