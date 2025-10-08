# About IdeaSearch

## Project Overview

`IdeaSearch` is an **AI-powered system for research idea generation and optimization**. It is an open-source project concurrent with Google's AlphaEvolve framework, launched in 2025. Within `IdeaSearch`, an **Idea** specifically refers to a text file with the `.idea` extension, containing creative content that can be read and processed by the system. It is **inspired by the FunSearch framework**, introduced in 2023. FunSearch pioneered the discovery of new mathematical structures and algorithms by combining large language models with evaluation programs. Building on this foundation, `IdeaSearch` aims to construct a more user-friendly, streamlined, and highly extensible integrated framework to support various fields in scientific research and education.

Compared to FunSearch, `IdeaSearch` introduces several innovative features that significantly enhance the system's flexibility and exploratory capabilities, including:

- **Prompt Prologue (`prologue_section`) and Epilogue (`epilogue_section`)**: Allow users to define the opening and closing parts of prompts sent to the LLM in a more flexible and modular way. This enables users to easily provide context, set task objectives, or guide output formatting without rewriting the entire prompt each time. Furthermore, users can opt for complete control over the prompt generation logic via a custom `generate_prompt_func`, offering immense flexibility for complex scenarios.

- **Evaluator Information (`evaluator_info`)**: In addition to a quantitative score, the evaluation function can now return supplementary string information. This allows users to understand not just _how good_ an Idea is, but also _why_ it is good, _where it can be improved_, or _what makes it unique_. This provides richer context and deeper insights for subsequent Idea optimization and system analysis.

- **Mutation (`mutation`)**: Introduces stochasticity by allowing minor modifications and perturbations to existing Ideas. This injects serendipity and diversity into the search process, helping to discover unexpected new directions or optimize existing Ideas, even in seemingly saturated search spaces.

- **Crossover (`crossover`)**: Generates new hybrid Ideas by combining elements from two or more existing Ideas. This classic operation from genetic algorithms is enhanced in `IdeaSearch` to facilitate more complex evolutionary paths, capable of merging the strengths of different excellent Ideas to produce novel combinations that transcend the limitations of a single Idea.

- **Generation Bonus (`generation_bonus`)**: Provides an additional score reward for newly generated Ideas. This mechanism encourages the system to continuously explore and produce novel, more vibrant Ideas, effectively preventing premature convergence to local optima and promoting a broad exploration of the Idea space.

## Key Features

- **Multi-Island Parallel Search**: Supports the creation of multiple independent "islands," each equipped with its own Samplers and Evaluators, to explore the Idea space in parallel, enhancing search efficiency and diversity.

- **Large Language Model (LLM) Integration**: Automatically manages API key loading and concurrent requests for multiple LLM models via a `ModelManager`.

- **Vision Language Model (VLM) Support**: Supports embedding images within prompts for multi-modal interaction with VLMs, significantly expanding the dimensions of Idea generation.

- **Evolutionary Strategies**:

  - **Sampling**: Selects high-quality Ideas based on their scores and temperature parameters to serve as references for generating new Ideas.
  - **Evaluation**: Scores each generated Idea using a user-defined `evaluate_func`, with an option to return supplementary information.
  - **Mutation**: Slightly modifies Ideas using a user-defined `mutation_func` to introduce diversity.
  - **Crossover**: Combines existing Ideas using a user-defined `crossover_func` to produce new hybrid Ideas.

- **Dynamic Model Selection and Assessment**: Dynamically adjusts the selection probability of models in subsequent rounds based on their performance in generating high-quality Ideas, favoring better-performing models. Includes visualization of model scores.

- **System-wide Assessment and Visualization**: Periodically evaluates the overall quality of the entire Idea database using a user-defined `assess_func` and generates charts to display the quality trend throughout the evolutionary process.

- **Modularity and Extensibility**: Through custom callback functions like `filter_func` and `postprocess_func`, and the `bind_helper` interface, users can easily integrate their own logic into the core workflow or build higher-level applications on top of `IdeaSearch`.

- **Data Persistence and Backup**: Automatically manages Idea files and score data, with support for backup functionality to ensure data integrity during the search process.

- **Highly Configurable**: Offers a rich set of parameters (via `set_` methods) for users to customize search behavior, including model temperatures, sampling strategies, evaluation intervals, and similarity thresholds.

- **Internationalization Support**: Includes a built-in `gettext` internationalization mechanism, supporting multiple languages (currently Simplified Chinese `zh_CN` and English `en`).

## Core API

The following methods in the `IdeaSearcher` class constitute the primary user interface:

### Core Workflow Methods

- `__init__()`: ⭐️ **Important**

  - **Function**: Initializes an `IdeaSearcher` instance. Sets all default parameters and initializes internal states such as locks, the model manager, and the island dictionary.
  - **Importance**: The class constructor, serving as the entry point for all search configurations.

- `run(additional_interaction_num: int)`: ⭐️ **Important**

  - **Function**: Starts the Idea search process. This method initializes and runs the samplers for all islands, allowing each island to evolve for `additional_interaction_num` generations (rounds).
  - **Importance**: The core method that initiates the entire Idea Search cycle.

### Island and Population Management

- `add_island()`: ⭐️ **Important**

  - **Function**: Adds a new island to the system and returns its `island_id`. The first time an island is added, it performs necessary initialization and cleanup (e.g., clearing logs, old Idea directories, and backups).
  - **Importance**: Defines the parallelism of the search and creates independent search units.

- `delete_island(island_id: int)`:

  - **Function**: Deletes the island with the specified `island_id` from the system.
  - **Importance**: Allows for dynamic management of search resources and strategies.

- `repopulate_islands()`: ⭐️ **Important**

  - **Function**: Redistributes Ideas among islands. This method sorts all islands by their "best score" and then copies the best Ideas from the top-half islands to the bottom-half islands, promoting idea sharing and preventing stagnation in local optima.
  - **Importance**: A key operation in evolutionary algorithms for global optimization.

### Result Retrieval

- `get_best_score()`: ⭐️ **Important**

  - **Function**: Returns the highest Idea score across all islands.
  - **Importance**: Retrieves the quality metric of the best Idea found so far.

- `get_best_idea()`: ⭐️ **Important**

  - **Function**: Returns the content of the highest-scoring Idea across all islands.
  - **Importance**: Retrieves the content of the best Idea found so far.

### Convenience Configuration Methods

- `add_initial_ideas(ideas: List[str])`:

  - **Function**: Programmatically adds a list of initial ideas, serving as an alternative to placing `.idea` files in the `database/ideas/initial_ideas/` directory. This allows users to provide seed ideas without direct file system manipulation.
  - **Importance**: Simplifies the setup of the initial population.

- `bind_helper(helper: object)`: ⭐️ **Important**

  - **Function**: Binds a "helper" object and uses its attributes to quickly configure multiple core parameters of the `IdeaSearcher`. This is a convenient end-to-end setup method, especially useful for building interface tools on top of the `IdeaSearch` framework (e.g., [IdeaSearch-Fitter](https://github.com/IdeaSearch/IdeasSearch-fit)).
  - **Helper Object Attributes**:
    - **Required**: `prologue_section` (str), `epilogue_section` (str), `evaluate_func` (Callable)
    - **Optional**: `initial_ideas` (List[str]), `system_prompt` (str), `assess_func` (Callable), `mutation_func` (Callable), `crossover_func` (Callable), `filter_func` (Callable), `postprocess_func` (Callable), etc.
  - **Importance**: Greatly simplifies parameter configuration and provides a standard interface for extending the framework.

## Configuration Parameters

`IdeaSearcher` provides a rich set of `set_` methods to configure its behavior. These parameters are logically grouped by functionality to facilitate understanding and management.

### Project and File System

- **`program_name`**: `str`
  - The name of the project, used for identification in logs and outputs.
- **`database_path`**: `str`
  - The root path for the database. **Prerequisite**: Unless using `add_initial_ideas()`, this path must contain an `ideas/initial_ideas/` subdirectory with initial `.idea` files. The system will automatically create subdirectories for island-specific ideas (`ideas/island*/`), data (`data/`), visualizations (`pic/`), and logs (`log/`). **This is the only location on the file system that IdeaSearch will modify**.
- **`diary_path`**: `Optional[str]` (Default: `None`)
  - The path to the log file. If `None`, defaults to `{database_path}/log/diary.txt`.
- **`backup_path`**: `Optional[str]` (Default: `None`)
  - The path for storing backups. If `None`, defaults to `{database_path}/ideas/backup/`.
- Other path-related parameters (e.g., `model_assess_result_data_path`, `assess_result_pic_path`) follow a similar pattern, using default locations under `database_path` if set to `None`.

### Initialization

- **`load_idea_skip_evaluation`**: `bool` (Default: `True`)
  - If `True`, the system will attempt to load scores from a `score_sheet.json` file in the initial ideas directory, skipping re-evaluation.
- **`initialization_cleanse_threshold`**: `float` (Default: `-1.0`)
  - The minimum score an idea must achieve to survive the initial screening phase. Ideas below this threshold will be removed.
- **`delete_when_initial_cleanse`**: `bool` (Default: `False`)
  - If `True`, ideas scoring below `initialization_cleanse_threshold` during the initial screening are permanently deleted.

### Sampling

- **`samplers_num`**: `int` (Default: `3`)
  - The number of parallel Sampler threads to run for each island.
- **`sample_temperature`**: `float` (Default: `50.0`)
  - The softmax temperature used to control randomness when sampling historical ideas as examples for the prompt. Higher values increase randomness.
- **`generation_bonus`**: `float` (Default: `0.0`)
  - A score bonus added to ideas from more recent generations during sampling, encouraging the exploration of newer evolutionary paths.

### Prompt Engineering

- **`system_prompt`**: `Optional[str]` (Default: `None`)
  - A system-level instruction for the LLM that sets the overall context and persona.
- **`explicit_prompt_structure`**: `bool` (Default: `True`)
  - If `True`, automatically includes structural headers (e.g., "Examples:") in the prompt for better organization.
- **`prologue_section`**: `str`
  - A user-defined string that appears at the beginning of every prompt, typically for instructions or context.
- **`epilogue_section`**: `str`
  - A user-defined string that appears at the end of every prompt, often for formatting instructions or final commands.
- **`filter_func`**: `Optional[Callable[[str], str]]` (Default: `None`)
  - A custom function to preprocess idea content before it is sampled and included in a prompt.
- **`examples_num`**: `int` (Default: `3`)
  - The number of historical ideas to include as examples in the prompt for each generation round.
- **`include_info_in_prompt`**: `bool` (Default: `True`)
  - If `True`, the supplementary `info` string returned by `evaluate_func` will be included alongside the idea's content and score in the prompt.
- **`images`**: `List[Any]` (Default: `[]`)
  - A list of images to be passed to a Vision Language Model (VLM). Use placeholders in `prologue_section` or `epilogue_section` to position them; URLs, local file paths, and raw bytes are automatically resolved.
- **`image_placeholder`**: `str` (Default: `"<image>"`)
  - The placeholder string used in prompt sections to indicate where an image from the `images` list should be inserted.
- **`generate_prompt_func`**: `Optional[Callable[[List[str], List[float], List[Optional[str]]], str]]` (Default: `None`)
  - A custom function that provides complete control over prompt generation, overriding the default structure (prologue, examples, epilogue). **Note**: This is an experimental feature and may be unstable.

### Model Configuration

- **`api_keys_path`**: `str`
  - The file path to the JSON configuration file containing API keys and model endpoint information.
- **`models`**: `List[str]`
  - A list of model aliases (e.g., `'GPT4_o'`, `'Deepseek_V3'`) to be used for idea generation. These aliases must be a subset of the keys in the `api_keys_path` file.
- **`model_temperatures`**: `List[float]`
  - A list of sampling temperatures for the LLMs. The length and order of this list must match the `models` list.
- **`model_sample_temperature`**: `float` (Default: `50.0`)
  - The softmax temperature for selecting which model to use for the next generation. Higher values increase randomness in model selection.
- **`top_p`**: `Optional[float]` (Default: `None`)
  - The nucleus sampling parameter `top_p`, corresponding to the standard API parameter.
- **`max_completion_tokens`**: `Optional[int]` (Default: `None`)
  - The maximum number of tokens to generate in a completion, corresponding to the standard API parameter.

**API Keys File Format (`api_keys.json`):**

This file should be a JSON object where each top-level key is a unique model alias used in `set_models()`. The value for each alias is a list of dictionaries, where each dictionary represents an instance of that model. This allows you to configure multiple instances (e.g., with different API keys or base URLs) for the same logical model, and the system will manage them automatically.

```json
{
  "Deepseek-V3": [
    {
      "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      "base_url": "https://api.deepseek.com/v1",
      "model": "deepseek-chat"
    }
  ],
  "Gemini-2.5-Pro": [
    {
      "api_key": "AIzaSyXXXX-XXXXXXXXXXXXXXXXXXXXXXXX",
      "base_url": "https://generativelanguage.googleapis.com/v1beta/",
      "model": "gemini-1.5-flash"
    }
  ]
}
```

### Generation and Post-processing

- **`generate_num`**: `int` (Default: `1`)
  - The number of new ideas each Sampler thread will attempt to generate in a single round.
- **`postprocess_func`**: `Optional[Callable[[str], str]]` (Default: `None`)
  - A custom function to clean or format the raw text generated by the LLM before it is saved as an idea file.
- **`hand_over_threshold`**: `float` (Default: `0.0`)
  - The minimum score a newly generated idea must achieve from the Evaluator to be accepted into an island's population.

### Evaluator

- **`evaluators_num`**: `int` (Default: `3`)
  - The number of parallel Evaluator threads to run for each island.
- **`evaluate_func`**: `Callable[[str], Tuple[float, Optional[str]]]`
  - The core evaluation function. It takes an idea's content (string) and must return a tuple `(score: float, info: Optional[str])`.
- **`score_range`**: `Tuple[float, float]` (Default: `(0.0, 100.0)`)
  - A tuple `(min_score, max_score)` defining the expected output range of `evaluate_func`, used for normalization and visualization.

### Database Assessment

- **`assess_func`**: `Optional[Callable[[List[str], List[float], List[Optional[str]]], float]]` (Default: `default_assess_func`)
  - A custom function to assess the overall state of the entire idea database, providing a holistic quality metric.
- **`assess_interval`**: `Optional[int]` (Default: `1`)
  - The frequency (in rounds) at which the `assess_func` is called to evaluate the entire database.
- **`assess_baseline`**: `Optional[float]` (Default: `60.0`)
  - A baseline score value to be drawn as a horizontal line on the database assessment graph for easy performance comparison.

### Model Assessment

- **`model_assess_window_size`**: `int` (Default: `20`)
  - The number of recent ideas generated by a model to consider when calculating its moving average performance score.
- **`model_assess_initial_score`**: `float` (Default: `100.0`)
  - The initial score assigned to each model. A high value encourages initial exploration of all available models.
- **`model_assess_average_order`**: `float` (Default: `1.0`)
  - The order $p$ for the p-norm (generalized mean) used to calculate the moving average of model scores. $p=1$ is the arithmetic mean; higher values give more weight to high scores.
- **`model_assess_save_result`**: `bool` (Default: `True`)
  - If `True`, saves the model assessment data and visualization to their specified paths.

### Mutation

- **`mutation_func`**: `Optional[Callable[[str], str]]` (Default: `None`)
  - A custom function that takes an idea's content and returns a slightly modified version.
- **`mutation_interval`**: `Optional[int]` (Default: `None`)
  - The frequency (in rounds) at which the mutation operation is performed on an island's population.
- **`mutation_num`**: `Optional[int]` (Default: `None`)
  - The number of new ideas to be generated via mutation each time the operation is triggered.
- **`mutation_temperature`**: `Optional[float]` (Default: `None`)
  - The softmax temperature for selecting parent ideas for mutation. Higher values increase the chance of lower-scoring ideas being mutated.

### Crossover

- **`crossover_func`**: `Optional[Callable[[str, str], str]]` (Default: `None`)
  - A custom function that takes two ideas' content and returns a new idea combining elements of both.
- **`crossover_interval`**: `Optional[int]` (Default: `None`)
  - The frequency (in rounds) at which the crossover operation is performed.
- **`crossover_num`**: `Optional[int]` (Default: `None`)
  - The number of new ideas to be generated via crossover each time the operation is triggered.
- **`crossover_temperature`**: `Optional[float]` (Default: `None`)
  - The softmax temperature for selecting parent ideas for crossover. Higher values increase randomness in parent selection.

### Similarity (Infrequently Used)

- **`similarity_threshold`**: `float` (Default: `-1.0`)
  - The distance threshold below which two ideas are considered similar. A value of `-1.0` disables similarity checks except for exact duplicates.
- Other related parameters like `similarity_distance_func` are available for more complex similarity control.

### Miscellaneous

- **`idea_uid_length`**: `int` (Default: `6`)
  - The character length of the Unique Identifier (UID) used in `.idea` filenames.
- **`record_prompt_in_diary`**: `bool` (Default: `False`)
  - If `True`, the full prompt sent to the LLM in each generation round will be recorded in the log file.
- **`backup_on`**: `bool` (Default: `True`)
  - If `True`, enables automatic backup of the `ideas` directory at the start of the `run` method.
- **`shutdown_score`**: `float` (Default: `float('inf')`)
  - If the best score across all islands reaches this value, the IdeaSearch process will terminate gracefully.

## Workflow Overview

A typical `IdeaSearch` workflow demonstrates how to structure your code for a complete evolutionary loop, including the creation of multiple islands, periodic population migration, and continuous idea generation. The following example, inspired by the [IdeaSearch-test repository](https://github.com/IdeaSearch/IdeaSearch-test), represents a common practical pattern.

```python
# pip install IdeaSearch
from IdeaSearch import IdeaSearcher
from user_code.prompt import prologue_section, epilogue_section
from user_code.evaluation import evaluate
from user_code.initial_ideas import initial_ideas


def main():
    # 1. Initialization
    ideasearcher = IdeaSearcher()

    # 2. Basic Configuration
    ideasearcher.set_language("en")  # Set language (default: 'zh_CN'; available: 'zh_CN', 'en')
    ideasearcher.set_api_keys_path("api_keys.json")
    ideasearcher.set_program_name("TemplateProgram")
    ideasearcher.set_database_path("database")

    # 3. Core Logic Configuration (Evaluation Function)
    ideasearcher.set_evaluate_func(evaluate)

    # 4. Prompt Engineering Configuration
    ideasearcher.set_prologue_section(prologue_section)
    ideasearcher.set_epilogue_section(epilogue_section)

    # 5. Model Configuration
    ideasearcher.set_models([
        "Deepseek_V3",
    ])
    ideasearcher.set_model_temperatures([
        0.6,
    ])

    # 6. (Optional) Other Configurations
    ideasearcher.set_record_prompt_in_diary(True)

    # 7. Add Initial Ideas (avoids manual file system operations)
    ideasearcher.add_initial_ideas(initial_ideas)

    # 8. Define and Execute the Evolutionary Loop
    island_num = 2             # Number of islands
    cycle_num = 3              # Number of migration cycles
    unit_interaction_num = 10  # Number of evolution rounds per cycle

    # Create initial islands
    for _ in range(island_num):
        ideasearcher.add_island()

    # Run the evolution
    for cycle in range(cycle_num):
        print(f"---[ Cycle {cycle + 1}/{cycle_num} ]---")

        # Before each cycle (except the first), perform inter-island migration
        if cycle != 0:
            print("Repopulating islands...")
            ideasearcher.repopulate_islands()

        # Run evolution for the specified number of rounds within the current cycle
        ideasearcher.run(unit_interaction_num)

    # 9. Retrieve and Utilize the Final Result
    print("\n---[ Search Complete ]---")
    best_idea_content = ideasearcher.get_best_idea()
    print("Best Idea Content:")
    print(best_idea_content)


if __name__ == "__main__":
    main()
```

## Internationalization

`IdeaSearch` supports the internationalization of its interface text.

You can set the system language using the `set_language(value: str)` method.

For example, `ideasearcher.set_language('en')` will switch the interface and log text to English, while `ideasearcher.set_language('zh_CN')` will switch to Simplified Chinese.

The default language of `IdeaSearch` is Simplified Chinese (`zh_CN`).
