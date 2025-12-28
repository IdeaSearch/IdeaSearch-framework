import re
from os.path import sep as seperator


def main():
    
    ideasearcher_path = f"src{seperator}IdeaSearch{seperator}ideasearcher.py"
    
    params = [

        # program name
        ("program_name", "str", "", "The name of the current project, used for identification in logs and outputs."),

        # interaction with file system
        ("database_path", "str", "", "The root path for the database. Prerequisite: Must contain `ideas/initial_ideas/` with initial `.idea` files. The system will automatically generate subdirectories for island-specific ideas (`ideas/island*/`), data (`data/`), visualizations (`pic/`), and logs (`log/`) under this path. This is the only location on the file system that IdeaSearch will modify."),
        ("diary_path", "Optional[str]", "None", "Path to the log file. If `None`, defaults to `{database_path}/log/diary.txt`."),
        ("backup_path", "Optional[str]", "None", "Path for storing backups. If `None`, defaults to `{database_path}/ideas/backup/`."),
        ("model_assess_result_data_path", "Optional[str]", "None", "Path to save model assessment scores as an `.npz` file. If `None`, defaults to `{database_path}/data/model_scores.npz`."),
        ("model_assess_result_pic_path", "Optional[str]", "None", "Path to save the model assessment visualization as a `.png` file. If `None`, defaults to `{database_path}/pic/model_scores.png`."),
        ("assess_result_data_path", "Optional[str]", "None", "Path to save overall database assessment scores as an `.npz` file. If `None`, defaults to `{database_path}/data/database_assessment.npz`."),
        ("assess_result_pic_path", "Optional[str]", "None", "Path to save the overall database assessment visualization as a `.png` file. If `None`, defaults to `{database_path}/pic/database_assessment.png`."),

        # about initialization
        ("load_idea_skip_evaluation", "bool", "True", "If `True`, attempts to skip re-evaluating initial ideas by loading their scores from a `score_sheet.json` file found in the same directory."),
        ("initialization_cleanse_threshold", "float", "-1.0", "The minimum score an idea must have to survive the initial cleansing phase. Ideas below this threshold will be deleted."),
        ("delete_when_initial_cleanse", "bool", "False", "If `True`, ideas scoring below `initialization_cleanse_threshold` are permanently deleted."),

        # about sampling
        ("samplers_num", "int", "3", "The number of Sampler threads to run in parallel for each island."),
        ("sample_temperature", "float", "50.0", "The softmax temperature for sampling ideas to be used as context in the prompt. Higher values increase randomness."),

        # about prompt: prompt = prologue + filtered ideas + epilogue, text-image intertwined format
        ("system_prompt", "Optional[str]", "None", "The system-level instruction for the large language model, setting the overall context and persona."),
        ("explicit_prompt_structure", "bool", "True", "If `True`, automatically includes structural headers in the prompt for better organization."),
        ("prologue_section", "str", "", "A user-defined string that appears at the beginning of every prompt, typically used for instructions or context."),
        ("filter_func", "Optional[Callable[[str], str]]", "None", "A custom function to preprocess idea content before it is sampled and included in a prompt."),
        ("examples_num", "int", "3", "The number of sampled historical ideas to include as examples in the prompt for each generation round."),
        ("include_info_in_prompt", "bool", "True", "If `True`, includes the supplementary `info` string (returned by `evaluate_func`) alongside the idea's content and score in the prompt."),
        ("epilogue_section", "str", "", "A user-defined string that appears at the end of every prompt, often used for formatting instructions or final commands."),
        ("images", "List[Any]", "[]", "A list of images to be passed to a Vision Language Model (VLM). Use placeholders in `prologue_section` or `epilogue_section` to position them."),
        ("image_placeholder", "str", '"<image>"', "The placeholder string (e.g., '<image>') used in prompt sections to indicate where an image from the `images` list should be inserted."),
        ("generate_prompt_func", "Optional[Callable[[List[str], List[float], List[Optional[str]]], str]]", "None", "A custom function that provides complete control over prompt generation, overriding the default structure (prologue, examples, epilogue). Note: This is an advanced feature and may be unstable."),

        # about models
        ("api_keys_path", "str", "", "The file path to the JSON configuration file containing API keys and model endpoint information."),
        ("models", "List[str]", "", "A list of model aliases (e.g., 'GPT4_o', 'Deepseek_V3') to be used for idea generation. These aliases must be a subset of the keys in the `api_keys_path` file."),
        ("model_temperatures", "List[float]", "", "A list of sampling temperatures for the LLMs. The list must have the same length and order as the `models` list."),
        ("model_sample_temperature", "float", "50.0", "The softmax temperature for selecting which model to use for the next generation. Higher values increase randomness in model choice."),
        ("top_p", "Optional[float]", "None", "The nucleus sampling parameter `top_p`, controlling the cumulative probability of token choices. Corresponds to the standard API parameter."),
        ("max_completion_tokens", "Optional[int]", "None", "The maximum number of tokens to generate in a completion. Corresponds to the standard API parameter."),

        # generate, postprocess and handover
        ("generate_num", "int", "1", "The number of new ideas each Sampler thread will attempt to generate in a single round."),
        ("postprocess_func", "Optional[Callable[[str], str]]", "None", "A custom function to clean or format the raw text generated by the LLM before it is saved as an idea file."),
        ("hand_over_threshold", "float", "0.0", "The minimum score a newly generated idea must achieve from the Evaluator to be accepted into the island's population."),

        # about evaluators
        ("evaluators_num", "int", "3", "The number of Evaluator threads to run in parallel for each island."),
        ("evaluate_func", "Callable[[str], Tuple[float, Optional[str]]]", "", "The core evaluation function. It takes an idea's content (string) and must return a tuple `(score: float, info: Optional[str])`."),
        ("score_range", "Tuple[float, float]", "(0.0, 100.0)", "A tuple `(min_score, max_score)` defining the expected output range of `evaluate_func`. Used for normalization and visualization."),

        # about database assessment
        ("assess_func", "Optional[Callable[[List[str], List[float], List[Optional[str]]], float]]", "default_assess_func", "A custom function to assess the overall state of the entire idea database, providing a holistic quality metric."),
        ("assess_interval", "Optional[int]", "1", "The frequency (in rounds) at which the `assess_func` is called to evaluate the entire database."),
        ("assess_baseline", "Optional[float]", "60.0", "A baseline score value to be drawn as a horizontal line on the database assessment graph for easy performance comparison."),

        # about model assessment
        ("model_assess_window_size", "int", "20", "The number of recent ideas generated by a model to consider when calculating its moving average performance score."),
        ("model_assess_initial_score", "float", "100.0", "The initial score assigned to each model. A high value encourages initial exploration of all available models."),
        ("model_assess_average_order", "float", "1.0", "The order `p` for the p-norm (generalized mean) used to calculate the moving average of model scores. `p=1` is arithmetic mean, higher values give more weight to high scores."),
        ("model_assess_save_result", "bool", "True", "If `True`, saves the model assessment data and visualization to the paths specified by `model_assess_result_*_path`."),

        # about mutation
        ("mutation_func", "Optional[Callable[[str], str]]", "None", "A custom function that takes one idea's content and returns a slightly modified version of it."),
        ("mutation_interval", "Optional[int]", "None", "The frequency (in rounds) at which the mutation operation is performed on the island's population."),
        ("mutation_num", "Optional[int]", "None", "The number of new ideas to be generated via mutation each time the operation is triggered."),
        ("mutation_temperature", "Optional[float]", "None", "The softmax temperature for selecting parent ideas for mutation. Higher values increase the chance of lower-scoring ideas being mutated."),

        # about crossover
        ("crossover_func", "Optional[Callable[[str, str], str]]", "None", "A custom function that takes two ideas' content and returns a new idea that combines elements of both parents."),
        ("crossover_interval", "Optional[int]", "None", "The frequency (in rounds) at which the crossover operation is performed."),
        ("crossover_num", "Optional[int]", "None", "The number of new ideas to be generated via crossover each time the operation is triggered."),
        ("crossover_temperature", "Optional[float]", "None", "The softmax temperature for selecting parent ideas for crossover. Higher values increase randomness in parent selection."),
        
        # about grouping and potential calculation
        ("identify_func", "Optional[Callable[[str, str], bool]]", "None", "A function that determines whether two ideas belong to the same group."),
        ("potential_weight", "float", "0.0", "Weight of group potential in boltzmann sampling energies."),
        ("potential_sample_threshold", "int", "1", "The minimum number of times a group must act as a source to be considered statistically valid for potential calculation. Filters out noise from rare groups."),
        ("potential_normalization_mode", "str", '"low_reject"', "The strategy for calculating the normalization factor N_0. Options: 'low_reject' (uses local transition counts, recommended) or 'high_reject' (uses a global constant)."),
        ("potential_regularization_alpha", "float", "1e-4", "The regularization strength to constrain the mean potential during optimization. Prevents numerical drift."),
        ("potential_confidence_ratio", "float", "0.5", "The ratio k in the potential confidence interval formula `mean_potential ± k * ln(N).`"),

        # miscellaneous
        ("idea_uid_length", "int", "6", "The character length of the Unique Identifier (UID) used in the filenames of `.idea` files."),
        ("record_prompt_in_diary", "bool", "False", "If `True`, the full prompt sent to the LLM in each generation round will be recorded in the log file."),
        ("backup_on", "bool", "True", "If `True`, enables the automatic backup of the `ideas` directory at the start of the `run` method."),
        ("shutdown_score", "float", "float('inf')", "If the best score across all islands reaches this value, the IdeaSearch process will terminate gracefully."),
    ]
    
    init_code = f"""    def __init__(
        self
    ) -> None:
    
        # 国际化设置
        self._language: str = 'zh_CN'
        self._translation = gettext.translation(_DOMAIN, _LOCALE_DIR, languages=[self._language], fallback=True)
        self._ = self._translation.gettext
    
"""
    set_code = ""
    get_code = ""
    for index, (param_name, param_type, param_default_value, param_description) in enumerate(params):
        
        if index:
            set_code += "\n\n"
            get_code += "\n\n"
        
        if param_default_value != "":
            inner_param_type = param_type
            init_code += f"        self._{param_name}: {param_type} = {param_default_value}\n"
        else:
            inner_param_type = f"Optional[{param_type}]"
            init_code += f"        self._{param_name}: {inner_param_type} = None\n"
            set_code += "    # ⭐️ Important\n"
            
        optional_match = re.match(r"Optional\[(.+)\]", param_type)
        
        if optional_match:
            is_optional = True
            inner_type = optional_match.group(1).strip()
        else:
            is_optional = False
            inner_type = param_type.strip()
            
        if inner_type.startswith("Callable"):
            if is_optional:
                check = "(value is None or callable(value))"
            else:
                check = "callable(value)"
        elif inner_type.startswith("List"):
            if is_optional:
                check = '(value is None or (hasattr(value, "__iter__") and not isinstance(value, str)))'
            else:
                check = 'hasattr(value, "__iter__") and not isinstance(value, str)'
        elif inner_type.startswith("Tuple"):
            if is_optional:
                check = "(value is None or isinstance(value, tuple))"
            else:
                check = "isinstance(value, tuple)"
        else:
            if is_optional:
                check = f"(value is None or isinstance(value, {inner_type}))"
            else:
                check = f"isinstance(value, {inner_type})"
                
        if param_default_value != "":
            set_code_doc_string = f'''        """
        Set the parameter {param_name} to the given value, if it is of the type {param_type}.
        {param_description}
        Its default value is {param_default_value}.
        """
'''

        else:  
            set_code_doc_string = f'''        """
        ⭐️ Important
        Set the parameter `{param_name}` to the given value, if it is of the type {param_type}.
        {param_description}
        This parameter is important and must be set manually by the user.
        """
'''

        get_code_doc_string = f'''        """
        Get the current value of the `{param_name}` parameter.
        {param_description}
        """
'''

        set_code += f'''    def set_{param_name}(
        self,
        value: {param_type},
    )-> None:
    
{set_code_doc_string}
        if not {check}:
            raise TypeError(self._("【IdeaSearcher】 参数`{param_name}`类型应为{param_type}，实为%s") % str(type(value)))

        with self._user_lock:
            self._{param_name} = value
'''
        
        get_code += f"""    def get_{param_name}(
        self,
    )-> {inner_param_type}:
        
{get_code_doc_string}
        return self._{param_name}
"""

    init_code += f"""
        self._lock: Lock = Lock()
        self._user_lock: Lock = Lock()
        self._console_lock: Lock = Lock()

        self._random_generator = np.random.default_rng()
        self._model_manager: ModelManager = ModelManager()
        
        self._next_island_id: int = 1
        self._islands: Dict[int, Island] = {{}}
        
        self._database_assessment_config_loaded = False
        self._model_score_config_loaded = False
        
        self._total_interaction_num = 0
        self._first_time_run = True
        self._first_time_add_island = True
        self._assigned_idea_uids: Set[str] = set()
        self._recorded_ideas = []
        self._recorded_idea_names: Set[str] = set()
        self._added_initial_idea_no = 1
        self._models_loaded_from_api_keys_json = False
        self._default_model_temperature = 0.9
        
        self._group_representatives: List[str] = []
        self._idea_to_group_index: Dict[str, int] = {{}}
        self._transition_counts: Dict[Tuple[int, int], int] = {{}}
        self._group_source_counts: Dict[int, int] = {{}}
        self._group_potentials: Dict[int, float] = {{}}
"""


    import_section = """from .sampler import Sampler
from .evaluator import Evaluator
from .island import Island
from .utils import *


# 国际化设置
_LOCALE_DIR = Path(__file__).parent / "locales"
_DOMAIN = "ideasearch"
gettext.bindtextdomain(_DOMAIN, _LOCALE_DIR)
gettext.textdomain(_DOMAIN)
"""

    load_models = """    def _load_models(
        self
    )-> None:
    
        \"""
        Load API keys for all models from the specified configuration file.
        Parameter  `api_keys_path` must be set before calling this method; otherwise, a ValueError will be raised.
        \"""
    
        if self._api_keys_path is None:
            raise ValueError(
                self._("【IdeaSearcher】 加载模型时发生错误： api keys path 不应为 None ！")
            )
            
        self._model_manager.load_api_keys(self._api_keys_path)
"""

    add_island = f"""    def add_island(
        self,
    )-> int:
    
        \"""
        Add a new island to the IdeaSearcher system and return its island_id.
        If this is the first island added, the method will also perform necessary initialization,
        such as clearing diary logs, removing old idea directories, and resetting backup folders.
        Raises RuntimeError if essential parameters are missing.
        \"""
        
        with self._user_lock:
        
            missing_param = self._check_runnability(exemptions=["models"])
            if missing_param is not None:
                raise RuntimeError(self._("【IdeaSearcher】 参数`%s`未传入，在当前设置下无法进行 add_island 动作！") % missing_param)
                
            diary_path = self._diary_path
            database_path = self._database_path
            backup_path = self._backup_path
            backup_on = self._backup_on
            assert diary_path is not None
            assert database_path is not None
            assert backup_path is not None
                
            if self._first_time_add_island:
            
                clear_file(diary_path)
                
                if backup_on:
                    guarantee_file_exist(f"{{backup_path}}{{seperator}}score_sheet_backup.json")
                    shutil.rmtree(f"{{backup_path}}")
                    guarantee_file_exist(f"{{backup_path}}{{seperator}}score_sheet_backup.json")
                    
                for item in os.listdir(f"{{database_path}}{{seperator}}ideas"):
                    full_path = os.path.join(f"{{database_path}}{{seperator}}ideas", item)
                    if os.path.isdir(full_path) and item.startswith('island'):
                        shutil.rmtree(full_path)
                self._first_time_add_island = False
        
            evaluators_num = self._evaluators_num
            samplers_num = self._samplers_num
            
            island_id = self._next_island_id
            self._next_island_id += 1
        
            island = Island(
                ideasearcher = self,
                island_id = island_id,
                console_lock = self._console_lock,
            )
            
            evaluators = [
                Evaluator(
                    ideasearcher = self,
                    evaluator_id = i + 1,
                    island = island,
                    console_lock = self._console_lock,
                )
                for i in range(evaluators_num)
            ]

            samplers = [
                Sampler(
                    ideasearcher = self,
                    sampler_id = i + 1,
                    island = island,
                    evaluators = evaluators,
                    console_lock = self._console_lock,
                )
                for i in range(samplers_num)
            ]
            
            island.load_ideas_from("initial_ideas")
            island.link_samplers(samplers)
            
            self._islands[island_id] = island
            
            return island_id
"""

    delete_island = """    def delete_island(
        self,
        island_id: int,
    )-> int:

        \"""
        Delete the island with the given island_id from the IdeaSearcher system.
        Returns 1 if deletion is successful, or 0 if the island_id does not exist.
        \"""
    
        with self._user_lock:
            
            if island_id in self._islands:
                del self._islands[island_id]
                return 1
                
            else:
                return 0
"""

    update_model_score = """    def update_model_score(
        self,
        score_result: list[float], 
        model: str,
        model_temperature: float,
    )-> None:
        
        with self._lock:
            
            diary_path = self._diary_path
            assert diary_path is not None
            
            index = 0
            
            models = self._models
            model_temperatures = self._model_temperatures
            p = self._model_assess_average_order
            model_assess_save_result = self._model_assess_save_result
            assert models is not None
            assert model_temperatures is not None
            
            
            while index < len(models):
                
                if models[index] == model and model_temperatures[index] == model_temperature:
                    self._model_recent_scores[index][:-1] = self._model_recent_scores[index][1:]
                    scores_array = np.array(score_result)
                    if p != np.inf:
                        self._model_recent_scores[index][-1] = (np.mean(np.abs(scores_array) ** p)) ** (1 / p)
                        self._model_scores[index] = (np.mean(np.abs(self._model_recent_scores[index]) ** p)) ** (1 / p)
                    else:
                        self._model_recent_scores[index][-1] = np.max(scores_array)
                        self._model_scores[index] = np.max(self._model_recent_scores[index])
                    with self._console_lock:    
                        append_to_file(
                            file_path = diary_path,
                            content = self._("【IdeaSearcher】 模型 %s(T=%.2f) 此轮评分为 %.2f ，其总评分已被更新为 %.2f ！") % (model, model_temperature, self._model_recent_scores[index][-1], self._model_scores[index]),
                        )
                    if model_assess_save_result:
                        self._sync_model_score_result()
                    return
                
                index += 1
                
            with self._console_lock:    
                append_to_file(
                    file_path = diary_path,
                    content = self._("【IdeaSearcher】 出现错误！未知的模型名称及温度： %s(T=%.2f) ！") % (model, model_temperature),
                )
                
            exit()
"""

    run = f"""    def run(
        self,
        additional_interaction_num: int,
    )-> None:
    
        \"""
        Run the IdeaSearch process for each island, extending their evolution by the given number of epochs.
        This method performs internal initialization, launches all samplers, and handles logging and error management.
        \"""

        with self._user_lock:
        
            missing_param = self._check_runnability()
            if missing_param is not None:
                raise RuntimeError(self._("【IdeaSearcher】 参数`%s`未传入，在当前设置下无法进行 run 动作！") % missing_param)
                
            if not self._models_loaded_from_api_keys_json:
                self._load_models()
                self._models_loaded_from_api_keys_json = True
                
            diary_path = self._diary_path
            database_path = self._database_path
            program_name = self._program_name
            models = self._models
            assert diary_path is not None
            assert database_path is not None
            assert program_name is not None
            assert models is not None
            
            append_to_file(
                file_path = diary_path,
                content = self._("【IdeaSearcher】 %s 的 IdeaSearch 正在运行，此次运行每个岛屿会演化 %d 个 epoch ！") % (program_name, additional_interaction_num)
            )
                
            self._total_interaction_num += len(self._islands) * additional_interaction_num
            
            for island_id in self._islands:
                island = self._islands[island_id]
                island.fuel(additional_interaction_num)
                
            if self._first_time_run:
                self._models_backup = deepcopy(models)
                self._load_model_score_config()
                self._load_database_assessment_config()
                self._first_time_run = False
            else:
                if models != self._models_backup:
                    self._load_model_score_config()
                    if not self._model_temperatures or \\
                        len(self._model_temperatures) != len(models):
                        self._model_temperatures = \\
                            [self._default_model_temperature] * len(models)
                self._models_backup = deepcopy(models)
                if self._assess_on:
                    self._expand_database_assessment_range()
                if self._model_assess_save_result:
                    self._expand_model_score_range()
                
            max_workers_num = 0
            for island_id in self._islands:
                island = self._islands[island_id]
                max_workers_num += len(island.samplers)
            
            with ThreadPoolExecutor(
                max_workers = max_workers_num
            ) as executor:
            
                futures = {{executor.submit(sampler.run): (island_id, sampler.id)
                    for island_id in self._islands
                    for sampler in self._islands[island_id].samplers
                }}
                for future in as_completed(futures):
                    island_id, sampler_id = futures[future]
                    try:
                        _ = future.result()
                    except Exception as e:
                        append_to_file(
                            file_path = diary_path,
                            content = self._("【IdeaSearcher】 %d号岛屿的%d号采样器在运行过程中出现错误：\\n%s\\n调用栈：\\n%s\\nIdeaSearch意外终止！") % (island_id, sampler_id, e, traceback.format_exc()),
                        )
                        exit()
"""

    get_model = """    def get_model(
        self
    )-> Tuple[str, float]:
        
        with self._lock:
            
            self._show_model_scores()
            
            models = self._models
            model_temperatures = self._model_temperatures
            model_sample_temperature = self._model_sample_temperature
            assert models is not None
            assert model_temperatures is not None
            assert model_sample_temperature is not None
            
            selected_index = make_boltzmann_choice(
                energies = self._model_scores,
                temperature = model_sample_temperature,
            )
            assert isinstance(selected_index, int)
            
            selected_model_name = models[selected_index]
            selected_model_temperature = model_temperatures[selected_index]
            
            return selected_model_name, selected_model_temperature
"""

    repopulate_islands = """    def repopulate_islands(
        self,
    )-> None:
    
        \"""
        Redistribute ideas among islands by colonization from top-performing islands to lower-ranked ones.

        This method sorts all islands by their _best_score in descending order, then copies the best ideas from the top half 
        of islands to the bottom half to promote idea sharing and improve overall search performance.

        Logs the start and completion of the redistribution process to the diary file.

        It helps to prevent local optima stagnation by enabling migration of high-quality ideas across islands.
        \"""
    
        diary_path = self._diary_path
        assert diary_path is not None
        
        with self._user_lock:
        
            with self._console_lock:
                append_to_file(
                    file_path = diary_path,
                    content = self._("【IdeaSearcher】 现在 ideas 开始在岛屿间重分布")
                )
            
            island_ids = self._islands.keys()
            
            island_ids = sorted(
                island_ids,
                key = lambda id: self._islands[id]._best_score,
                reverse = True,
            )
            
            N = len(island_ids)
            M = N // 2
            
            for index in range(M):
            
                island_to_colonize = self._islands[island_ids[index]]
                assert island_to_colonize._best_idea is not None
                
                self._islands[island_ids[-index]].accept_colonization(
                    [island_to_colonize._best_idea]
                )
                
            with self._console_lock:
                append_to_file(
                    file_path = diary_path,
                    content = self._("【IdeaSearcher】 此次 ideas 在岛屿间的重分布已完成")
                )
"""

    get_answer = f"""    def _get_answer(
        self,
        model: str, 
        temperature: Optional[float],
        system_prompt: str,
        prompt: str,
        images: List[Any],
        image_placeholder: str,
    ):
        
        return self._model_manager.get_answer(
            model = model,
            prompt = prompt,
            temperature = temperature,
            system_prompt = system_prompt,
            images = images,
            image_placeholder = image_placeholder,
            top_p = self._top_p,
            max_completion_tokens = self._max_completion_tokens,
        )
"""

    check_runnability = f"""    def _check_runnability(
        self,
        exemptions: List[str] = [],
    )-> Optional[str]:
        
        missing_param = None
        def _update_missing_param(candidate):
            nonlocal missing_param
            if missing_param is None and candidate not in exemptions:
                missing_param = candidate
        
        if self._database_path is None:
            _update_missing_param("database_path")
        
        if self._program_name is None:
            _update_missing_param("program_name")
            
        if self._prologue_section is None and self._generate_prompt_func is None:
            _update_missing_param("prologue_section")

        if self._epilogue_section is None and self._generate_prompt_func is None:
            _update_missing_param("epilogue_section")
            
        if self._evaluate_func is None:
            _update_missing_param("evaluate_func")
           
        if self._models is None:
            _update_missing_param("models")

        if self._assess_func is not None:
            if self._assess_interval is None:
                _update_missing_param("assess_interval")

        if self._mutation_func is not None:
            if self._mutation_interval is None:
                _update_missing_param("mutation_interval")
            if self._mutation_num is None:
                _update_missing_param("mutation_num")
            if self._mutation_temperature is None:
                _update_missing_param("mutation_temperature")
         
        if self._crossover_func is not None:
            if self._crossover_interval is None:
                _update_missing_param("crossover_interval")
            if self._crossover_num is None:
                _update_missing_param("crossover_num")
            if self._crossover_temperature is None:
                _update_missing_param("crossover_temperature")
                
        if missing_param is not None: return missing_param
        
        database_path = self._database_path
        models = self._models
        assert database_path is not None
        
        if "models" not in exemptions:
            assert models is not None
            if self._model_temperatures is None:
                self._model_temperatures = [self._default_model_temperature] * len(models)
        
        if self._diary_path is None:
            self._diary_path = f"{{database_path}}{{seperator}}log{{seperator}}diary.txt"
       
        if self._system_prompt is None:
            self._system_prompt = "You're a helpful assistant."
    
        if self._assess_func is not None:
            if self._assess_result_data_path is None:
                self._assess_result_data_path = f"{{database_path}}{{seperator}}data{{seperator}}database_assessment.npz"
            if self._assess_result_pic_path is None:
                self._assess_result_pic_path = f"{{database_path}}{{seperator}}pic{{seperator}}database_assessment.png"
                
        if self._model_assess_save_result:
            if self._model_assess_result_data_path is None:
                self._model_assess_result_data_path = f"{{database_path}}{{seperator}}data{{seperator}}model_scores.npz"
            if self._model_assess_result_pic_path is None:
                self._model_assess_result_pic_path = f"{{database_path}}{{seperator}}pic{{seperator}}model_scores.png"
                
        if self._backup_path is None:
            self._backup_path = f"{{database_path}}{{seperator}}ideas{{seperator}}backup"
                
        return None
"""

    load_model_score_config = f"""    def _load_model_score_config(
        self,
    )-> None:
        
        models = self._models
        model_assess_save_result = self._model_assess_save_result
        model_assess_window_size = self._model_assess_window_size
        model_assess_initial_score = self._model_assess_initial_score
        assert models is not None
    
        self._model_recent_scores = []
        self._model_scores = []
        
        for _ in range(len(models)):
            self._model_recent_scores.append(
                np.full((model_assess_window_size,), model_assess_initial_score)
            )
            self._model_scores.append(model_assess_initial_score)
            
        if model_assess_save_result:
            self._scores_of_models = np.zeros((1+self._total_interaction_num, len(models)))
            self._scores_of_models_length = 0
            self._scores_of_models_x_axis = np.linspace(
                start = 0, 
                stop = self._total_interaction_num, 
                num = 1 + self._total_interaction_num, 
                endpoint = True
            )
            self._sync_model_score_result()
"""

    expand_model_score_range = f"""    def _expand_model_score_range(
        self,
    )-> None:
    
        models = self._models
        assert models is not None

        new_scores_of_models = np.zeros((1+self._total_interaction_num, len(models)))
        new_scores_of_models[:len(self._scores_of_models)] = self._scores_of_models
        self._scores_of_models = new_scores_of_models
        
        self._scores_of_models_x_axis = np.linspace(
            start = 0, 
            stop = self._total_interaction_num, 
            num = 1 + self._total_interaction_num, 
            endpoint = True
        )
"""

    show_model_scores = """    def _show_model_scores(
        self
    )-> None:
        
        diary_path = self._diary_path
        models = self._models
        model_temperatures = self._model_temperatures
        
        assert diary_path is not None
        assert models is not None
        assert model_temperatures is not None
            
        with self._console_lock:
            
            append_to_file(
                file_path = diary_path,
                content = self._("【IdeaSearcher】 各模型目前评分情况如下："),
            )
            for index, model in enumerate(models):
                
                model_temperature = model_temperatures[index]
                
                append_to_file(
                    file_path = diary_path,
                    content = (
                        f"  {index+1}. {model}(T={model_temperature:.2f}): {self._model_scores[index]:.2f}"
                    ),
                )
"""

    sync_database_assessment_result = f"""    def _sync_database_assessment_result(
        self,
        is_initialization: bool,
        get_database_score_success: bool,
    )-> None:
    
        if self._total_interaction_num == 0: return
        
        diary_path = self._diary_path
        score_range = self._score_range
        assess_result_data_path = self._assess_result_data_path
        assess_result_pic_path = self._assess_result_pic_path
        assess_baseline = self._assess_baseline
        
        assert diary_path is not None
        assert assess_result_data_path is not None
        assert assess_result_pic_path is not None
        
        np.savez_compressed(
            file = assess_result_data_path, 
            interaction_num = self._assess_result_ndarray_x_axis,
            database_scores = self._assess_result_ndarray,
        )
        
        point_num = len(self._assess_result_ndarray_x_axis)
        auto_markersize = get_auto_markersize(point_num)
        
        range_expand_ratio = 0.08
        x_axis_range = (0, self._total_interaction_num)
        x_axis_range_delta = (x_axis_range[1] - x_axis_range[0]) * range_expand_ratio
        x_axis_range = (
            int(math.floor(x_axis_range[0] - x_axis_range_delta)), 
            int(math.ceil(x_axis_range[1] + x_axis_range_delta))
        )
        score_range_delta = (score_range[1] - score_range[0]) * range_expand_ratio
        score_range = (score_range[0] - score_range_delta, score_range[1] + score_range_delta)
        
        plt.figure(figsize=(10, 6))
        plt.plot(
            self._assess_result_ndarray_x_axis[:self._assess_result_ndarray_length], 
            self._assess_result_ndarray[:self._assess_result_ndarray_length], 
            label='Database Score', 
            color='dodgerblue', 
            marker='o',
            markersize = auto_markersize,
        )
        if assess_baseline is not None:
            plt.axhline(
                y = assess_baseline,
                color = "red",
                linestyle = "--",
                label = "Baseline",
            )
        plt.title("Database Assessment")
        plt.xlabel("Total Interaction No.")
        plt.ylabel("Database Score")
        plt.xlim(x_axis_range)
        plt.ylim(score_range)
        plt.grid(True)
        plt.legend()
        plt.savefig(assess_result_pic_path)
        plt.close()
        
        if get_database_score_success:
            if is_initialization:
                append_to_file(
                        file_path = diary_path,
                        content = self._("【IdeaSearcher】 初始质量评估结束， %s 与 %s 已更新！") % (basename(assess_result_data_path), basename(assess_result_pic_path)),
                    )
            else:
                with self._console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content = self._("【IdeaSearcher】 此轮质量评估结束， %s 与 %s 已更新！") % (basename(assess_result_data_path), basename(assess_result_pic_path)),
                    )
"""

    sync_model_score_result = f"""    def _sync_model_score_result(self):
    
        if self._total_interaction_num == 0: return
        
        diary_path = self._diary_path
        model_assess_result_data_path = self._model_assess_result_data_path
        model_assess_result_pic_path = self._model_assess_result_pic_path
        models = self._models
        model_temperatures = self._model_temperatures
        score_range = self._score_range
        
        assert diary_path is not None
        assert model_assess_result_data_path is not None
        assert model_assess_result_pic_path is not None
        assert models is not None
        assert model_temperatures is not None
        
        guarantee_file_exist(model_assess_result_data_path)
        guarantee_file_exist(model_assess_result_pic_path)
        
        self._scores_of_models[self._scores_of_models_length] = self._model_scores
        self._scores_of_models_length += 1
        
        scores_of_models = self._scores_of_models.T
        
        scores_of_models_dict = {{}}
        for model_name, model_temperature, model_scores in zip(models, model_temperatures, scores_of_models):
            scores_of_models_dict[f"{{model_name}}(T={{model_temperature:.2f}})"] = model_scores
        
        np.savez_compressed(
            file = model_assess_result_data_path,
            interaction_num = self._scores_of_models_x_axis,
            **scores_of_models_dict
        )
        
        point_num = len(self._scores_of_models_x_axis)
        auto_markersize = get_auto_markersize(point_num)
        
        range_expand_ratio = 0.08
        x_axis_range = (0, self._total_interaction_num)
        x_axis_range_delta = (x_axis_range[1] - x_axis_range[0]) * range_expand_ratio
        x_axis_range = (
            int(math.floor(x_axis_range[0] - x_axis_range_delta)), 
            int(math.ceil(x_axis_range[1] + x_axis_range_delta))
        )
        score_range_delta = (score_range[1] - score_range[0]) * range_expand_ratio
        score_range = (score_range[0] - score_range_delta, score_range[1] + score_range_delta)

        plt.figure(figsize=(10, 6))
        for model_label, model_scores in scores_of_models_dict.items():
            plt.plot(
                self._scores_of_models_x_axis[:self._scores_of_models_length],
                model_scores[:self._scores_of_models_length],
                label=model_label,
                marker='o',
                markersize = auto_markersize,
            )
        plt.title("Model Scores")
        plt.xlabel("Interaction No.")
        plt.ylabel("Model Score")
        plt.xlim(x_axis_range)
        plt.ylim(score_range)
        plt.grid(True)
        plt.legend()
        plt.savefig(model_assess_result_pic_path)
        plt.close()
        
        with self._console_lock:
            append_to_file(
                file_path=diary_path,
                content=(
                    f"【IdeaSearcher】 "
                    f" {{basename(model_assess_result_data_path)}} 与 {{basename(model_assess_result_pic_path)}} 已更新！"
                ),
            )
"""

    load_database_assessment_config = f"""    def _load_database_assessment_config(
        self,
    )-> None:
    
        diary_path = self._diary_path
        assess_func = self._assess_func
        assess_interval = self._assess_interval
        assess_result_data_path = self._assess_result_data_path
        assess_result_pic_path = self._assess_result_pic_path
        
        assert diary_path is not None
        assert assess_result_data_path is not None
        assert assess_result_pic_path is not None
        
        if assess_func is not None:
        
            assert assess_interval is not None

            self._assess_on = True
            self._assess_interaction_count = 0
            
            self._assess_result_ndarray = np.zeros((1 + (self._total_interaction_num // assess_interval),))
            self._assess_result_ndarray_length = 1
            self._assess_result_ndarray_x_axis = np.linspace(
                start = 0, 
                stop = self._total_interaction_num, 
                num = 1 + (self._total_interaction_num // assess_interval), 
                endpoint = True
            )
            
            guarantee_file_exist(assess_result_data_path)
            guarantee_file_exist(assess_result_pic_path)
            
            ideas: list[str] = []
            scores: list[float] = []
            infos: list[Optional[str]] = []
                            
            for island_id in self._islands:
                island = self._islands[island_id]
                for current_idea in island.ideas:
                    
                    assert current_idea.content is not None
                    assert current_idea.score is not None
                    
                    ideas.append(current_idea.content)
                    scores.append(current_idea.score)
                    infos.append(current_idea.info)
                    
            get_database_initial_score_success = False
            
            try:
                database_initial_score = assess_func(
                    ideas,
                    scores,
                    infos,
                )
                get_database_initial_score_success = True
                with self._console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content = self._("【IdeaSearcher】 初始 ideas 的整体质量评分为：%.2f！") % database_initial_score,
                    )
                    
            except Exception as error:
                database_initial_score = 0.0
                with self._console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content = self._("【IdeaSearcher】 评估库中初始 ideas 的整体质量时遇到错误：\\n%s") % error,
                    )
                    
            self._assess_result_ndarray[0] = database_initial_score
            self._sync_database_assessment_result(
                is_initialization = True,
                get_database_score_success = get_database_initial_score_success,
            )
            
        else:
            self._assess_on = False
"""

    expand_database_assessment_range = f"""    def _expand_database_assessment_range(
        self,
    )-> None:
    
        assess_interval = self._assess_interval
        assert assess_interval is not None
    
        new_assess_result_ndarray = np.zeros((1 + (self._total_interaction_num // assess_interval),))
        new_assess_result_ndarray[:len(self._assess_result_ndarray)] = self._assess_result_ndarray
        self._assess_result_ndarray = new_assess_result_ndarray
        
        self._assess_result_ndarray_x_axis = np.linspace(
            start = 0, 
            stop = self._total_interaction_num, 
            num = 1 + (self._total_interaction_num // assess_interval), 
            endpoint = True
        )  
"""

    assess_database = f"""    def assess_database(
        self,
    )-> None:
        
        with self._lock:
        
            if not self._assess_on: return
        
            diary_path = self._diary_path
            assess_func = self._assess_func
            assess_interval = self._assess_interval
            
            assert diary_path is not None
            assert assess_func is not None
            assert assess_interval is not None
        
            self._assess_interaction_count += 1
            if self._assess_interaction_count % assess_interval != 0: return

            start_time = perf_counter()
            
            with self._console_lock:
                append_to_file(
                    file_path = diary_path,
                    content = self._("【IdeaSearcher】 现在开始评估数据库中 ideas 的整体质量！"),
                )
                
            ideas: list[str] = []
            scores: list[float] = []
            infos: list[Optional[str]] = []
            
            for island_id in self._islands:
                island = self._islands[island_id]
                for idea in island.ideas:
                    
                    assert idea.content is not None
                    assert idea.score is not None
                    
                    ideas.append(idea.content)
                    scores.append(idea.score)
                    infos.append(idea.info)
                
            get_database_score_success = False
            try:
                database_score = assess_func(
                    ideas,
                    scores,
                    infos,
                )
                get_database_score_success = True
                
                end_time = perf_counter()
                total_time = end_time - start_time
                
                with self._console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content = self._("【IdeaSearcher】 数据库中 ideas 的整体质量评分为：%.2f！评估用时：%.2f秒。") % (database_score, total_time),
                    )
                    
            except Exception as error:
                database_score = 0
                with self._console_lock:
                    append_to_file(
                        file_path = diary_path,
                        content = self._("【IdeaSearcher】 评估库中 ideas 的整体质量时遇到错误：\\n%s") % error,
                    )
                    
            self._assess_result_ndarray[self._assess_result_ndarray_length] = database_score
            self._assess_result_ndarray_length += 1
            
            self._sync_database_assessment_result(
                is_initialization = False,
                get_database_score_success = get_database_score_success,
            )
"""

    get_idea_uid = f"""    def get_idea_uid(
        self,
    )-> str:
    
        with self._lock:
        
            idea_uid_length = self._idea_uid_length
            
            idea_uid = ''.join(random.choices(
                population = string.ascii_lowercase, 
                k = idea_uid_length,
            ))
            
            while idea_uid in self._assigned_idea_uids:
                idea_uid = ''.join(random.choices(
                    population = string.ascii_lowercase, 
                    k = idea_uid_length,
                ))
                
            self._assigned_idea_uids.add(idea_uid)
            
            return idea_uid
"""

    get_best_score = f"""    def _get_best_score(
        self,
    )-> float:
    
        missing_param = self._check_runnability()
        if missing_param is not None:
            raise RuntimeError(self._("【IdeaSearcher】 参数`%s`未传入，在当前设置下无法进行 get_best_score 动作！") % missing_param)
        
        scores: list[float] = []
        
        for island_id in self._islands:
            island = self._islands[island_id]
            for idea in island.ideas:
                assert idea.score is not None
                scores.append(idea.score)
                
        if not scores: raise RuntimeError(self._("【IdeaSearcher】 目前各岛屿均无 ideas ，无法进行 get_best_score 动作！"))
            
        return max(scores)
    
    def get_best_score(
        self,
    )-> float:
    
        \"""
        Return the highest score among all ideas across islands.
        Raises a RuntimeError if any required parameter is missing or if there are no ideas available.
        \"""
    
        with self._user_lock:
            return self._get_best_score()
"""

    get_best_idea = f"""    def get_best_idea(
        self,
    )-> str:
    
        \"""
        Return the content of the idea with the highest score across all islands.
        Raises a RuntimeError if any required parameter is missing or if there are no ideas available.
        \"""
    
        with self._user_lock:
        
            missing_param = self._check_runnability()
            if missing_param is not None:
                raise RuntimeError(self._("【IdeaSearcher】 参数`%s`未传入，在当前设置下无法进行 get_best_idea 动作！") % missing_param)
        
            scores: list[float] = []
            ideas: list[str] = []
            
            for island_id in self._islands:
                island = self._islands[island_id]
                for idea in island.ideas:
                    assert idea.score is not None
                    assert idea.content is not None
                    scores.append(idea.score)
                    ideas.append(idea.content)
                    
            if not scores: raise RuntimeError(self._("【IdeaSearcher】 目前各岛屿均无 ideas ，无法进行 get_best_idea 动作！"))
                
            return ideas[scores.index(max(scores))]
"""

    record_ideas_in_backup = f"""    def record_ideas_in_backup(
        self,
        ideas_to_record,
    ):
    
        with self._lock:
        
            database_path = self._database_path
            backup_path = self._backup_path
            backup_on = self._backup_on
            assert database_path is not None
            assert backup_path is not None
            
            if not backup_on: return
            
            guarantee_file_exist(f"{{backup_path}}{{seperator}}score_sheet_backup.json")
        
            for idea in ideas_to_record:
                
                if basename(idea.path) not in self._recorded_idea_names:
                    
                    self._recorded_ideas.append(idea)
                    self._recorded_idea_names.add(basename(idea.path))
                
                    with open(
                        file = f"{{backup_path}}{{seperator}}{{basename(idea.path)}}",
                        mode = "w",
                        encoding = "UTF-8",
                    ) as file:

                        file.write(idea.content)
                        
            score_sheet = {{
                basename(idea.path): {{
                    "score": idea.score,
                    "info": idea.info if idea.info is not None else "",
                    "source": idea.source,
                    "level": idea.level,
                    "created_at": idea.created_at,
                }}
                for idea in self._recorded_ideas
            }}

            with open(
                file = f"{{backup_path}}{{seperator}}score_sheet_backup.json", 
                mode = "w", 
                encoding = "UTF-8",
            ) as file:
                
                json.dump(
                    obj = score_sheet, 
                    fp = file, 
                    ensure_ascii = False,
                    indent = 4
                )
"""

    set_language = '''    def set_language(
        self,
        value: str,
    ) -> None:
    
        """
        Set the parameter `language` to the given value, if it is of the type str.
        This parameter sets the language for the user interface and translations; currently, only 'zh_CN' and 'en' are supported.
        The shorthand 'zh' will be converted to 'zh_CN'.
        """

        if not isinstance(value, str):
            raise TypeError(self._("【IdeaSearcher】 参数`language`类型应为str，实为%s") % str(type(value)))

        with self._user_lock:
            if value == "zh": value = "zh_CN"
            self._language = value
            self._translation = gettext.translation(_DOMAIN, _LOCALE_DIR, languages=[self._language], fallback=True)
            self._ = self._translation.gettext
'''

    get_language = """    def get_language(
        self,
    )-> str:
    
        \"""
        Get the current value of the `language` parameter.
        This parameter determines the active language for translations and interface text.
        \"""
        
        return self._language
"""

    dir_code = """    def __dir__(self):
        # 返回类的所有属性和方法
        return [
            attr for attr in super().__dir__() 
            if not attr.startswith('_') and not callable(getattr(self, attr))
        ] + [
            'run', 'load_models', 'shutdown_models', 'get_best_score', 
            'get_best_idea', 'add_island', 'delete_island', 
            'repopulate_islands', 'get_idea_uid', 'record_ideas_in_backup',
            'assess_database', 'get_model'
        ]
"""

    add_initial_ideas = f"""    def _add_initial_ideas(
        self,
        initial_ideas: List[str],
    )-> None:
    
        database_path = self._database_path
        if database_path is None:
            raise RuntimeError(self._(
                "【IdeaSearcher】 添加初始 ideas 失败：应先设置数据库路径！"
            ))
            
        initial_ideas_path = f"{{database_path}}{{seperator}}ideas{{seperator}}initial_ideas"
        guarantee_file_exist(
            file_path = initial_ideas_path,
            is_directory = True,
        )
        
        for initial_idea in initial_ideas:
        
            with open(
                file = f"{{initial_ideas_path}}{{seperator}}added_initial_idea{{self._added_initial_idea_no}}.idea",
                mode = "w",
                encoding = "UTF-8",
            ) as file:
            
                file.write(initial_idea)
                
            self._added_initial_idea_no += 1
    
    
    def add_initial_ideas(
        self,
        initial_ideas: List[str],
    ):
        with self._user_lock: self._add_initial_ideas(initial_ideas)
"""

    bind_helper = f"""    def bind_helper(
        self,
        helper: object,
    )-> None:
    
        with self._lock:
        
            if self._database_path is None:
            
                raise RuntimeError(self._(
                    "【IdeaSearcher】 绑定 helper 失败：应先设置数据库路径！"
                ))
        
            if not hasattr(helper, "prologue_section"):
            
                raise ValueError(self._(
                    "【IdeaSearcher】 绑定 helper 失败：helper 缺失属性 `prologue_section` ！"
                ))
                
            if not hasattr(helper, "epilogue_section"):
            
                raise ValueError(self._(
                    "【IdeaSearcher】 绑定 helper 失败：helper 缺失属性 `epilogue_section` ！"
                ))
                
            if not hasattr(helper, "evaluate_func"):
            
                raise ValueError(self._(
                    "【IdeaSearcher】 绑定 helper 失败：helper 缺失属性 `evaluate_func` ！"
                ))
                
            self._prologue_section = helper.prologue_section # type: ignore
            self._epilogue_section = helper.epilogue_section # type: ignore
            self._evaluate_func = helper.evaluate_func # type: ignore
            if hasattr(helper, "initial_ideas"): self._add_initial_ideas(helper.initial_ideas) # type: ignore
            if hasattr(helper, "system_prompt"): self._system_prompt = helper.system_prompt # type: ignore
            if hasattr(helper, "assess_func"): self._mutation_func = helper.assess_func # type: ignore
            if hasattr(helper, "mutation_func"): self._mutation_func = helper.mutation_func # type: ignore
            if hasattr(helper, "crossover_func"): self._crossover_func = helper.crossover_func # type: ignore
            if hasattr(helper, "filter_func"): self._filter_func = helper.filter_func # type: ignore
            if hasattr(helper, "postprocess_func"): self._postprocess_func = helper.postprocess_func # type: ignore
"""

    about_grouping_and_potential_calculation = """    def update_potential(
        self,
        example_ideas: List[str],
        generated_ideas: List[str],
    )-> None:
    
        diary_path = self._diary_path
        assert diary_path is not None

        with self._lock:
        
            src_indices = [self._get_or_create_group_index(idea) for idea in example_ideas]
            dst_indices = [self._get_or_create_group_index(idea) for idea in generated_ideas]
            
            for s_idx in src_indices:
                self._group_source_counts[s_idx] = self._group_source_counts.get(s_idx, 0) + len(dst_indices)
                
                for d_idx in dst_indices:
                    if s_idx == d_idx: continue
                    pair = (s_idx, d_idx)
                    self._transition_counts[pair] = self._transition_counts.get(pair, 0) + 1
            
            start_time = perf_counter()
            self._calculate_potential()
            end_time = perf_counter()
            total_time = end_time - start_time
            with self._console_lock:
                append_to_file(
                    file_path = diary_path,
                    content = self._("【IdeaSearcher】 potential calculation time cost: %s; group count: %d" % (f"{total_time:.2f} seconds", len(self._group_representatives))),
                )
    

    def get_potentials(
        self,
        ideas: List[str],
    )-> List[float]:
    
        with self._lock:
            
            results = []
            for idea in ideas:
                idx = self._get_or_create_group_index(idea)
                results.append(self._group_potentials.get(idx, 0.0))

            return results


    def get_potential_group_sizes(
        self, 
        ideas: List[str],
    )-> List[int]:

        with self._lock:
            
            group_indices = [self._get_or_create_group_index(idea) for idea in ideas]
            group_counts = Counter(group_indices)
            results = [group_counts[idx] for idx in group_indices]
            
            return results

    
    def _get_or_create_group_index(
        self,
        idea: str,
    )-> int:
    
        identify_func = self._identify_func
        diary_path = self._diary_path
        assert diary_path is not None
    
        if idea in self._idea_to_group_index:
            return self._idea_to_group_index[idea]
        
        # 线性扫描代表元，性能敏感点
        start_time = perf_counter()
        try:
            for idx, representative in enumerate(self._group_representatives):
                idea_in_group = (identify_func is not None and identify_func(representative, idea)) \\
                    or representative == idea
                if idea_in_group:
                    self._idea_to_group_index[idea] = idx
                    return idx
            
            new_idx = len(self._group_representatives)
            self._group_representatives.append(idea)
            self._idea_to_group_index[idea] = new_idx
            self._group_potentials[new_idx] = 0.0 
            return new_idx
        finally:
            end_time = perf_counter()
            total_time = end_time - start_time
            with self._console_lock:
                append_to_file(
                    file_path = diary_path,
                    content = self._("【IdeaSearcher】 _get_or_create_group_index time cost (cache unhit): %s" % (f"{total_time:.2f} seconds")),
                )
            

    def _calculate_potential(
        self
    )-> None:
    
        n_threshold = self._potential_sample_threshold
        reject_mode = self._potential_normalization_mode
        alpha = self._potential_regularization_alpha
        database_path = self._database_path
        assert database_path is not None
    
        active_nodes = list(self._group_source_counts.keys())
        if not active_nodes: return
        
        num_groups = len(self._group_representatives)
        nodes = range(num_groups)
        
        n_0_map: Dict[int, float] = {}
        filtered_nodes: Set[int] = set()
        
        if reject_mode == 'high_reject':
            val = 20000.0 / num_groups if num_groups > 0 else 1.0
            for node in nodes:
                n_0_map[node] = val
        else: # low_reject
            for node in nodes:
                cnt = self._group_source_counts.get(node, 0)
                n_0_map[node] = float(cnt)
                if cnt <= n_threshold:
                    filtered_nodes.add(node)

        edges_data = []
        valid_source_count = 0
        
        for node in nodes:
             if node not in filtered_nodes and n_0_map.get(node, 0) > n_threshold:
                 valid_source_count += 1
        
        if valid_source_count == 0:
            return

        # 构建边列表
        for (u, v), count in self._transition_counts.items():
            if reject_mode == 'low_reject':
                if u in filtered_nodes or v in filtered_nodes:
                    continue
            
            n_0_u = n_0_map.get(u, 1.0)
            if n_0_u > n_threshold:
                # w(f->g) = min(1, N(f->g) / N_0(f))
                weight = min(1.0, count / n_0_u)
                edges_data.append({'u': u, 'v': v, 'w': weight})
        
        def K(x: Any) -> Any:
            return np.exp(-x / 2)

        def objective_and_grad(v_array: np.ndarray) -> Tuple[float, np.ndarray]:
            action_numerator = 0.0
            grads = np.zeros_like(v_array)
            
            for edge in edges_data:
                u, v_idx, w = edge['u'], edge['v'], edge['w']
                delta_v = v_array[u] - v_array[v_idx]
                k_val = K(delta_v)
                action_numerator += w * k_val
                common_grad_term = w * (-0.5 * k_val)
                grads[u] += common_grad_term
                grads[v_idx] -= common_grad_term
            
            action_loss = action_numerator / valid_source_count
            action_grads = grads / valid_source_count
            current_mean = np.mean(v_array)
            # 零均值正则项
            reg_loss = alpha * (current_mean ** 2)
            reg_grad_val = alpha * 2 * current_mean / len(v_array)
            reg_grads = np.full_like(v_array, reg_grad_val)
            total_loss = action_loss + reg_loss
            total_grads = action_grads + reg_grads
            return float(total_loss), total_grads

        # 初始值使用上一轮的 potentials 以加速收敛
        x0 = np.zeros(num_groups)
        for i in range(num_groups):
            x0[i] = self._group_potentials.get(i, 0.0)
        # 零上界 -> 零均值
        x0 -= np.mean(x0)

        res = minimize(
            fun = objective_and_grad,
            x0 = x0,
            jac = True,
            method = 'L-BFGS-B',
            options = {'maxiter': 100, 'disp': False} 
        )

        optimized_v = res.x
        # 零上界归一化
        optimized_v -= np.max(optimized_v)
        
        for i, val in enumerate(optimized_v):
            self._group_potentials[i] = val
        
        self._sync_group_potential_result(f"{database_path}{seperator}ideas{seperator}group_potential.json")
        
    
    def _sync_group_potential_result(
        self,
        save_path: str,
    )-> None:
        
        group_sizes = Counter(self._idea_to_group_index.values())
        
        transitions_nested: Dict[int, Dict[int, int]] = {}
        for (u, v), count in self._transition_counts.items():
            if u not in transitions_nested:
                transitions_nested[u] = {}
            transitions_nested[u][v] = count

        output_data = []
        num_groups = len(self._group_representatives)
        
        for idx in range(num_groups):
            group_info = {
                "group_index": idx,
                "representative": self._group_representatives[idx],
                "member_count": group_sizes.get(idx, 0),
                "potential": float(self._group_potentials.get(idx, 0.0)),
                "transitions_out": transitions_nested.get(idx, {})
            }
            output_data.append(group_info)
            
        time_stamp = get_time_stamp(show_minute=True, show_second=True)
        temp_path = f"{save_path}.halfway_{time_stamp}"
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, save_path)
        except:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
             
               
    @property
    def ideas_num(
        self
    )-> int:
        return len(self._idea_to_group_index)
"""
    
    ideasearcher_code = f"""{import_section}

class IdeaSearcher:
    
    # ----------------------------- IdeaSearhcer 初始化 ----------------------------- 

{init_code}

{dir_code}
    # ----------------------------- 核心功能 ----------------------------- 
    
    # ⭐️ Important
{run}

{check_runnability}
    # ----------------------------- API4LLMs 相关 ----------------------------- 

{load_models}  

{get_answer}
    # ----------------------------- Ideas 管理相关 ----------------------------- 
    
    # ⭐️ Important
{get_best_score}

    # ⭐️ Important
{get_best_idea}

    # ⭐️ Important
{add_initial_ideas}
    
{get_idea_uid}

{record_ideas_in_backup}
    # ----------------------------- 岛屿相关 ----------------------------- 

    # ⭐️ Important
{add_island}
           
    # ⭐️ Important 
{delete_island}
    
    # ⭐️ Important
{repopulate_islands}
    # ----------------------------- Model Score 相关 ----------------------------- 
    
{load_model_score_config}

{expand_model_score_range}

{update_model_score}

{sync_model_score_result}
                        
{get_model}
       
{show_model_scores}
    # ----------------------------- Database Assessment 相关 ----------------------------- 
            
{load_database_assessment_config}
            
{expand_database_assessment_range}

{assess_database}

{sync_database_assessment_result}
    # ----------------------------- Helper 拓展相关 ----------------------------- 
            
{bind_helper}
    # ----------------------------- 分组与势函数计算相关 ----------------------------- 
            
{about_grouping_and_potential_calculation}
    # ----------------------------- Getters and Setters ----------------------------- 
    
{set_language}
        
{set_code}

{get_language}

{get_code}
"""
    
    with open(
        file = ideasearcher_path,
        mode = "w",
        encoding = "UTF-8",
    ) as file:
        
        file.write(ideasearcher_code)


if __name__ == "__main__":

    main()