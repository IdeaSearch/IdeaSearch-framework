import gettext
from threading import Lock
from typing import Optional
from os.path import basename
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from pywheels.file_tools import append_to_file
from .typing import *


_LOCALE_DIR = Path(__file__).parent / "locales"
_DOMAIN = "ideasearch"
gettext.bindtextdomain(_DOMAIN, _LOCALE_DIR)
gettext.textdomain(_DOMAIN)


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
        
        # 获取国际化函数
        self._ = ideasearcher._

    def run(self):
        
        diary_path = self.ideasearcher.get_diary_path()
        system_prompt = self.ideasearcher.get_system_prompt()
        prologue_section = self.ideasearcher.get_prologue_section()
        epilogue_section = self.ideasearcher.get_epilogue_section()
        generate_num = self.ideasearcher.get_generate_num()
        filter_func = self.ideasearcher.get_filter_func()
        postprocess_func = self.ideasearcher.get_postprocess_func()
        record_prompt_in_diary = self.ideasearcher.get_record_prompt_in_diary()
        generate_prompt_func = self.ideasearcher.get_generate_prompt_func()
        explicit_prompt_structure = self.ideasearcher.get_explicit_prompt_structure()
        
        assert system_prompt is not None
        
        with self.console_lock:
            append_to_file(
                file_path = diary_path,
                content = self._("【%d号岛屿的%d号采样器】 已开始工作！") % (self.island.id, self.id),
            )
        
        while self.island.get_status() == "Running":
            
            with self.console_lock:
                append_to_file(
                    file_path = diary_path,
                    content = self._("【%d号岛屿的%d号采样器】 已开始新一轮采样！") % (self.island.id, self.id),
                )
                
            if generate_prompt_func is None:
            
                examples = self.island.get_examples()
                if examples is None: 
                    with self.console_lock:
                        append_to_file(
                            file_path = diary_path,
                            content= self._("【%d号岛屿的%d号采样器】 工作结束。") % (self.island.id, self.id)
                        )
                    return
                else:
                    with self.console_lock:
                        append_to_file(
                            file_path = diary_path,
                            content= self._(
                                "【%d号岛屿的%d号采样器】 已从%d号岛屿采样 %d 个idea！"
                            ) % (self.island.id, self.id, self.island.id, len(examples))
                        )
                        
                example_idea_paths = [current_idea[-2] for current_idea in examples]
                example_idea_scores = [current_idea[1] for current_idea in examples]
                example_idea_levels = [current_idea[-1] for current_idea in examples]
                level = max(example_idea_levels) + 1
                
                examples_section = f"举例部分（一共有{len(examples)}个例子）：\n" \
                    if explicit_prompt_structure else ""
                    
                for index, example in enumerate(examples):
                    
                    idea, score, info, similar_num, similarity_prompt, path, _ = example
                    
                    examples_section += f"[第 {index + 1} 个例子]\n" \
                        if explicit_prompt_structure else ""
                        
                    examples_section += f"内容：\n" \
                        if explicit_prompt_structure else ""
                    
                    if filter_func is not None:
                        try:
                            idea = filter_func(idea)
                        except Exception as error:
                            with self.console_lock:
                                append_to_file(
                                    file_path = diary_path,
                                    content= self._(
                                        "【%d号岛屿的%d号采样器】将 filter_func 作用于 %s 时发生错误：\n %s \n延用原来的 idea ！"
                                    ) % (
                                        self.island.id, 
                                        self.id, 
                                        basename(path), 
                                        error
                                    )
                                )

                    examples_section += f'{idea}\n'
                    
                    examples_section += f"Score：{score:.2f}\n" \
                        if explicit_prompt_structure or info is None else ""
                    
                    if info is not None:
                        examples_section += f"Info：{info}\n" \
                            if explicit_prompt_structure else f"{info}\n"
                        
                    if explicit_prompt_structure and similar_num is not None:
                        examples_section += (
                            f"Similarity Note: There are {similar_num} examples similar to this one.\n"
                            f"{similarity_prompt}\n"
                        )
                
                prompt = prologue_section + examples_section + epilogue_section
                
            else:
                
                island_ideas = self.island.get_ideas_called_when_generate_prompt_func_set()
                if island_ideas is None: 
                    with self.console_lock:
                        append_to_file(
                            file_path = diary_path,
                            content = self._("【%d号岛屿的%d号采样器】 工作结束。") % (self.island.id, self.id)
                        )
                    return
                else:
                    with self.console_lock:
                        append_to_file(
                            file_path = diary_path,
                            content= self._(
                                "【%d号岛屿的%d号采样器】 已从%d号岛屿采样 %d 个idea，预备调用自定义的 generate_prompt_func！"
                            ) % (self.island.id, self.id, self.island.id, len(island_ideas))
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
                            content= self._(
                                "【%d号岛屿的%d号采样器】使用自定义的 generate_prompt_func 时发生错误：\n %s\nIdeaSearch终止！"
                            ) % (self.island.id, self.id, error)
                        )
                    return
            
            with self.console_lock:
                append_to_file(
                    file_path = diary_path,
                    content = self._("【%d号岛屿的%d号采样器】 正在询问 IdeaSearcher 使用何模型。。。") % (self.island.id, self.id),
                )
            
            model, model_temperature = self.ideasearcher.get_model()
            
            with self.console_lock:
                append_to_file(
                    file_path = diary_path,
                    content = self._("【%d号岛屿的%d号采样器】 根据各模型得分情况，依概率选择了 %s(T=%.2f) ！") % (self.island.id, self.id, model, model_temperature),
                )
                
            if record_prompt_in_diary:
                with self.console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content = self._("【%d号岛屿的%d号采样器】 向 %s(T=%.2f) 发送的 system prompt 是：\n%s") % (self.island.id, self.id, model, model_temperature, system_prompt),
                    )
                    append_to_file(
                        file_path = diary_path,
                        content = self._("【%d号岛屿的%d号采样器】 向 %s(T=%.2f) 发送的 prompt 是：\n%s") % (self.island.id, self.id, model, model_temperature, prompt),
                    )
            
            with self.console_lock:
                append_to_file(
                    file_path = diary_path,
                    content = self._("【%d号岛屿的%d号采样器】 已向 %s(T=%.2f) 发送prompt，正在等待回答！") % (self.island.id, self.id, model, model_temperature),
                )
                
            def get_postprocessed_idea(
                model: str, 
                model_temperature: float,
                system_prompt: str,
                prompt: str,
            )-> Tuple[str, str]:
                
                raw_response = self.ideasearcher._get_answer(
                    model = model,
                    model_temperature = model_temperature,
                    system_prompt = system_prompt,
                    prompt = prompt,
                )
                
                if postprocess_func is not None:
                    idea = postprocess_func(raw_response)
                else:
                    idea = raw_response
                    raw_response = self._("Same as the idea.")
                
                return raw_response, idea
            
            generated_raw_responses = [None] * generate_num
            generated_ideas = [None] * generate_num
            with ThreadPoolExecutor() as executor:

                future_to_index = {
                    executor.submit(
                        get_postprocessed_idea, 
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
                        raw_response, idea = future.result()
                        generated_raw_responses[i] = raw_response
                        generated_ideas[i] = idea
                    except Exception as e:
                        with self.console_lock:
                            append_to_file(
                                file_path = diary_path,
                                content = self._("【%d号岛屿的%d号采样器】 尝试获取 %s(T=%.2f) 的回答并进行后处理时发生错误: \n%s\n此轮采样失败。。。") % (self.island.id, self.id, model, model_temperature, e),
                            )
                        break
                            
            if any(idea is None for idea in generated_ideas) or \
                any(raw_response is None for raw_response in generated_raw_responses):
                
                with self.console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content = self._("【%d号岛屿的%d号采样器】 因异常没有获得应生成的全部idea，此次采样失败。。。") % (self.island.id, self.id),
                    )
                    
                continue
                            
            with self.console_lock:
                append_to_file(
                    file_path = diary_path,
                    content = self._("【%d号岛屿的%d号采样器】 已收到来自 %s(T=%.2f) 的 %d 个回答！") % (self.island.id, self.id, model, model_temperature, generate_num),
                )
            
            evaluator = self._get_idle_evaluator()
            if evaluator:
                
                evaluator.evaluate(
                    generated_raw_responses = generated_raw_responses,
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
                        content = self._("【%d号岛屿的%d号采样器】 已释放%d号评估器。") % (self.island.id, self.id, evaluator.id),
                    )
                evaluator.release()
            else:
                with self.console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content = self._("【%d号岛屿的%d号采样器】 没有找到空闲的评估器，此轮采样失败。。。") % (self.island.id, self.id),
                    )
        
        with self.console_lock:    
            append_to_file(
                file_path = diary_path,
                content = self._("【%d号岛屿的%d号采样器】 工作结束。") % (self.island.id, self.id),
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
                        content = self._("【%d号岛屿的%d号采样器】 已找到%d号评估器进行评估！") % (self.island.id, self.id, evaluator.id),
                    )
                return evaluator
        return None
