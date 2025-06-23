# About IdeaSearch

## Project Overview

`IdeaSearch` framework is an **AI-powered Research Idea Generation and Optimization System**. In `IdeaSearch`, an **Idea** specifically refers to a text file with the `.idea` extension, which contains the creative content readable and processable by the system. It is **inspired by the FunSearch framework, launched by Google in 2023.** The FunSearch framework pioneered the discovery of new mathematical structures and algorithms by combining large language models with evaluation programs. Building upon this foundation, `IdeaSearch` aims to construct a more user-friendly and streamlined integrated framework, with the goal of supporting various fields in scientific research and education.

Compared to FunSearch, `IdeaSearch` introduces several innovative features that significantly enhance the system's flexibility and exploratory capabilities, including:

- **Generation Bonus (generation_bonus)**: Provides additional score rewards for newly generated Ideas. This mechanism encourages the system to continuously explore and produce novel, more vibrant Ideas, effectively preventing it from getting stuck in local optima too early, and promoting broad exploration of the Idea space.
- **Mutation (mutation)**: Introduces randomness, allowing minor modifications and perturbations to existing Ideas. This injects serendipity and diversity into the Idea search process, helping to discover unexpected new directions or optimize existing Ideas, even in seemingly saturated Idea spaces.
- **Crossover (crossover)**: Generates new hybrid Ideas by combining elements from two or more existing Ideas. This classic operation from genetic algorithms is enhanced in `IdeaSearch`; it facilitates more complex evolutionary paths, capable of merging the strengths of different excellent Ideas to produce novel combinations beyond the limitations of single Ideas.
- **Prompt Prologue (prologue_section) and Epilogue (epilogue_section)**: Allow users to more flexibly and modularly define the opening and closing parts of the prompts sent to the large language model. This makes it easy for users to provide context, set task objectives, or guide output format without rewriting the entire prompt each time. Furthermore, if the user chooses, they can completely control the prompt generation logic through a custom `generate_prompt_func` function, offering great flexibility for various complex scenarios.
- **Evaluator Information (evaluator info)**: Beyond providing a quantified score for an Idea, the evaluation function can now return additional string information. This allows users not only to know the quantitative 'goodness' of an Idea but also to understand 'why it's good,' 'where it can be improved,' or 'what makes it unique' through this supplementary information, providing richer context and deeper insights for subsequent Idea optimization and system analysis.

## Key Features

- **Multi-Island Parallel Search**: Supports the creation of multiple independent "islands," each equipped with its own Samplers and Evaluators, to explore the Idea space in parallel, enhancing search efficiency and diversity.
- **Large Language Model (LLM) Integration**: Manages various LLM models via `ModelManager` and dynamically selects models for Idea generation.
- **Evolutionary Strategies**:
  - **Sampling**: Selects high-quality Ideas as references for generating new ones, based on their scores and temperature parameters.
  - **Evaluation**: Scores each generated Idea using a user-defined `evaluate_func`, with optional additional information return.
  - **Mutation**: Modifies Ideas slightly through a user-defined `mutation_func` to introduce diversity.
  - **Crossover**: Combines existing Ideas using a user-defined `crossover_func` to produce new hybrid Ideas.
- **Dynamic Model Selection and Assessment**: Adjusts the probability of models being selected in future rounds based on their performance in generating high-quality Ideas, encouraging better-performing models. Provides visualization of model scores.
- **System Overall Assessment and Visualization**: Periodically evaluates the overall quality of the entire Idea database using a user-defined `assess_func` and generates charts to display quality trends throughout the evolutionary process.
- **Data Persistence and Backup**: Automatically manages Idea files and score data, supporting backup functionalities to ensure data security during the search process.
- **Highly Configurable**: Offers a rich set of parameters (via `set_` methods) for users to customize search behavior, including model temperatures, sampling strategies, evaluation intervals, similarity thresholds, and more.
- **Internationalization Support**: Includes built-in `gettext` internationalization mechanism, supporting multiple languages (currently `zh_CN` for Simplified Chinese and `en` for English).

## Core API

The following methods in the `IdeaSearcher` class are marked as "⭐️ Important" and constitute the primary interface for user interaction:

- `__init__()`:
  - **Function**: Initializes an `IdeaSearcher` instance. Sets all default parameters and initializes internal states such as locks, model manager, and island dictionary.
  - **Importance**: The class constructor, serving as the starting point for all search parameters.
- `run(additional_interaction_num: int)`:
  - **Function**: Initiates the Idea search process. This method initializes and runs the samplers for all islands, allowing each island to evolve for `additional_interaction_num` epochs (rounds).
  - **Importance**: The core method to start the entire Idea search cycle.
- `load_models()`:
  - **Function**: Loads API keys for all LLM models from the configuration file specified by `api_keys_path`.
  - **Importance**: This method must be called before any LLM interaction can occur.
- `add_island()`:
  - **Function**: Adds a new island to the system and returns its `island_id`. The first time an island is added, it performs necessary initialization cleanup (e.g., clearing logs, old Idea directories, and backups).
  - **Importance**: Defines the parallelism of the search and creates independent search units.
- `delete_island(island_id: int)`:
  - **Function**: Deletes the island with the specified `island_id` from the system.
  - **Importance**: Allows for dynamic management of search resources and strategies.
- `repopulate_islands()`:
  - **Function**: Redistributes Ideas among islands. This method sorts all islands by their "best score" and then copies the best Ideas from the top half of islands to the bottom half, promoting Idea sharing and preventing local optima stagnation.
  - **Importance**: A crucial operation in evolutionary algorithms for global optimization.
- `get_best_score()`:
  - **Function**: Returns the highest Idea score across all islands.
  - **Importance**: Retrieves the quality metric of the best Idea found so far.
- `get_best_idea()`:
  - **Function**: Returns the content of the Idea with the highest score across all islands.
  - **Importance**: Retrieves the content of the best Idea found so far.

## Important Configuration Parameters

`IdeaSearcher` provides a rich set of `set_` methods to configure its behavior. The following are the most critical parameters that must be set before calling the `run()` method:

- `set_program_name(value: str)`: The name of the project, used for logging and identification.
- `set_prologue_section(value: str)`: The fixed introductory part of the prompt for LLMs.
- `set_epilogue_section(value: str)`: The fixed concluding part of the prompt for LLMs.
  - **Note**: `prologue_section` and `epilogue_section` can be replaced by a custom `generate_prompt_func`. If `generate_prompt_func` is set, these two parameters can be `None`.
- `set_database_path(value: str)`: The root path for the Idea database. The system will automatically create subdirectories under this path: `ideas/` (for Idea files), `data/` (for data files like evaluation results), `pic/` (for images like trend charts), and `log/` (for logs).
- `set_models(value: List[str])`: A list of LLM model names participating in Idea generation.
- `set_model_temperatures(value: List[float])`: The sampling temperature for each model, corresponding to the `models` list. The length of this list must match the length of the `models` list.
- `set_evaluate_func(value: Callable[[str], Tuple[float, Optional[str]]])`: A function used to evaluate a single Idea. It accepts an Idea string as input and returns a tuple containing a float score and optional additional information (Optional[str]).
- `set_api_keys_path(value: str)`: The path to the JSON/YAML file containing API key configurations.

  **API Keys File Format (`api_keys.json`):**
  This file should be a JSON object where each top-level key corresponds to a unique model alias that you will use in `set_models()`. The value for each model alias is a list of dictionaries, where each dictionary represents an instance of that model. This allows you to configure multiple instances (e.g., with different API keys or base URLs) for the same logical model, and the system will automatically manage them.

  ```json
  {
    "Deepseek_V3": [
      {
        "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat"
      }
    ],
    "Deepseek_R1": [
      {
        "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-reasoner"
      }
    ],
    "Qwen_Plus": [
      {
        "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus"
      }
    ],
    "Qwen_Max": [
      {
        "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-max"
      }
    ],
    "Gemini_2.0_Flash_Thinking_Experimental": [
      {
        "api_key": "AIzaSyXXXX-XXXXXXXXXXXXXXXXXXXXXXXX",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/",
        "model": "gemini-2.0-flash-thinking-exp"
      }
    ],
    "Gemini_1.5_Flash": [
      {
        "api_key": "AIzaSyXXXX-XXXXXXXXXXXXXXXXXXXXXXXX",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/",
        "model": "gemini-1.5-flash"
      }
    ],
    "Moonshot_V1_32k": [
      {
        "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-32k"
      }
    ],
    "Moonshot_V1_8k": [
      {
        "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-32k"
      }
    ]
  }
  ```

All parameters have corresponding `get_` methods to retrieve their current values. For example, `ideasearcher.get_program_name()`.

## Workflow Overview

A typical `IdeaSearch` usage flow is as follows:

1.  **Instantiation**:
    ```python
    from IdeaSearch.ideasearcher import IdeaSearcher
    ideasearcher = IdeaSearcher()
    ```
2.  **Configuration of Core Parameters**:

    ```python
    # Example evaluation function
    def my_custom_evaluate_function(idea: str) -> Tuple[float, Optional[str]]:
        # Your evaluation logic goes here, returning a score and optional info
        score = len(idea) # Simple example: Longer Idea gets higher score
        info = "Idea length is " + str(len(idea))
        return float(score), info

    ideasearcher.set_program_name("My AI Idea Project")
    ideasearcher.set_database_path("./my_idea_database") # Ensure this path exists and contains an ideas/initial_ideas/ subdirectory
    ideasearcher.set_models(["gpt-3.5-turbo", "gpt-4"])
    ideasearcher.set_model_temperatures([0.7, 0.5])
    ideasearcher.set_evaluate_func(my_custom_evaluate_function)
    ideasearcher.set_api_keys_path("./api_keys.json") # Assuming your API key file is here
    ideasearcher.set_prologue_section("Please generate innovative business Ideas for the following topic, be concise:")
    ideasearcher.set_epilogue_section("Each Idea must start with 'Idea:'.")

    # More optional configurations...
    ideasearcher.set_samplers_num(2) # 2 samplers per island
    ideasearcher.set_evaluators_num(1) # 1 evaluator per island
    ideasearcher.set_assess_interval(5) # Assess database every 5 rounds
    ideasearcher.set_generation_bonus(5.0) # New Ideas get a 5-point bonus
    # ... other set_ methods
    ```

3.  **Loading Model API Keys**:
    ```python
    ideasearcher.load_models()
    ```
4.  **Adding Islands**:
    ```python
    # Add an island. You can call add_island() multiple times to increase parallel search islands.
    # Idea files from the initial_ideas directory will be loaded into the first island.
    island_id_1 = ideasearcher.add_island()
    print(f"Island {island_id_1} added.")
    ```
5.  **Running the Search**:
    ```python
    # Start the multi-threaded search process; each island will evolve for the specified number of rounds.
    print("Starting IdeaSearch...")
    ideasearcher.run(additional_interaction_num=50) # Each island runs for 50 epochs
    print("IdeaSearch finished.")
    ```
6.  **(Optional) Idea Repopulation Across Islands**:
    ```python
    # Call during or after the search to facilitate the spread and merging of excellent Ideas.
    ideasearcher.repopulate_islands()
    ```
7.  **Retrieving Best Results**:
    ```python
    best_score = ideasearcher.get_best_score()
    best_idea = ideasearcher.get_best_idea()
    print(f"\nCurrent Best Idea Score: {best_score}")
    print(f"Current Best Idea Content: {best_idea}")
    ```
8.  **Shutting Down Models**:
    ```python
    ideasearcher.shutdown_models()
    print("Models shut down.")
    ```

## Internationalization

`IdeaSearch` supports internationalization of its interface text.
You can set the system language using the `set_language(value: str)` method.
For example, `ideasearcher.set_language('en')` will switch the interface and log texts to English, while `ideasearcher.set_language('zh_CN')` will switch to Simplified Chinese.
The default language for `IdeaSearch` is Simplified Chinese (`zh_CN`).
