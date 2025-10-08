# 关于 IdeaSearch

## 项目概述

`IdeaSearch` 框架是一个**由人工智能驱动的研究思路生成与优化系统**，是 **Google 在 2025 年推出的 AlphaEvolve 框架的同期开源工作**。在 `IdeaSearch` 中，一个**思路 (Idea)** 特指一个以 `.idea` 为扩展名的文本文件，其中包含了可被系统读取和处理的创造性内容。它的**灵感来源于 2023 年的 FunSearch 框架**。FunSearch 框架开创性地通过结合大型语言模型和评估程序，来发现新的数学结构和算法。`IdeaSearch` 在此基础上，旨在构建一个用户友好、流程简便、高开放性的集成框架，以期为科学研究和教育的各个领域提供支持。

与 FunSearch 相比，`IdeaSearch` 引入了多项创新特性，显著增强了系统的灵活性和探索能力，包括：

- **提示词序言 (prologue_section) 和跋语 (epilogue_section)**: 允许用户更灵活、模块化地定义发送给大语言模型提示的开头和结尾部分。这使用户可以轻松地提供上下文、设定任务目标或指导输出格式，而无需每次都重写整个提示。此外，如果用户选择，他们可以通过自定义的 `generate_prompt_func` 函数完全控制提示生成逻辑，为各种复杂场景提供了极大的灵活性。

- **评估器信息 (evaluator info)**: 除了为思路提供量化分数外，评估函数现在还可以返回额外的字符串信息。这使用户不仅能知道一个思路的定量“好坏”，还能通过这些补充信息了解“它为什么好”、“可以在哪里改进”或“它有何独特之处”，为后续的思路优化和系统分析提供了更丰富的背景和更深入的洞察。

- **变异 (mutation)**: 引入随机性，允许对现有思路进行微小的修改和扰动。这为思路搜索过程注入了偶然性和多样性，有助于发现意想不到的新方向或优化现有思路，即使在看似饱和的思路空间中也是如此。

- **交叉 (crossover)**: 通过组合两个或多个现有思路的元素来生成新的混合思路。这一遗传算法中的经典操作在 `IdeaSearch` 中得到了增强；它促进了更复杂的进化路径，能够融合不同优秀思路的优点，产生超越单个思路局限的新颖组合。

- **代际奖励 (generation_bonus)**: 为新生成的思路提供额外的分数奖励。此机制鼓励系统不断探索和产生新颖、更具活力的思路，有效防止其过早陷入局部最优，促进对思路空间的广泛探索。

## 关键特性

- **多岛屿并行搜索**: 支持创建多个独立的“岛屿”，每个岛屿都配备自己的采样器 (Sampler) 和评估器 (Evaluator)，以并行方式探索思路空间，提高搜索效率和多样性。

- **大型语言模型 (LLM) 集成**: 通过 `ModelManager` 自动管理多种 LLM 模型的 API 密钥加载与并发请求。

- **视觉语言模型 (VLM) 支持**: 支持在提示词中嵌入图像，与 VLM 进行多模态交互，极大地扩展了思路生成的维度。

- **进化策略**:

  - **采样**: 根据思路的分数和温度参数，选择高质量的思路作为生成新思路的参考。
  - **评估**: 使用用户定义的 `evaluate_func` 对每个生成的思路进行打分，并可选地返回附加信息。
  - **变异**: 通过用户定义的 `mutation_func` 对思路进行轻微修改以引入多样性。
  - **交叉**: 使用用户定义的 `crossover_func` 组合现有思路以产生新的混合思路。

- **动态模型选择与评估**: 根据模型在生成高质量思路方面的表现，调整其在未来轮次中被选中的概率，鼓励表现更优的模型。提供模型分数的可视化。

- **系统整体评估与可视化**: 使用用户定义的 `assess_func` 定期评估整个思路数据库的总体质量，并生成图表以展示整个进化过程中的质量趋势。

- **模块化与可扩展性**: 通过 `filter_func`、`postprocess_func` 等自定义回调函数，以及 `bind_helper` 接口，用户可以轻松地将自己的逻辑集成到系统核心流程中，或基于 `IdeaSearch` 构建更上层的应用。

- **数据持久化与备份**: 自动管理思路文件和分数数据，支持备份功能，确保搜索过程中数据的安全。

- **高度可配置**: 提供丰富的参数设置（通过 `set_` 方法），供用户定制搜索行为，包括模型温度、采样策略、评估间隔、相似度阈值等。

- **国际化支持**: 内置 `gettext` 国际化机制，支持多种语言（目前支持简体中文 `zh_CN` 和英文 `en`）。

## 核心 API

`IdeaSearcher` 类中的以下方法构成了用户交互的主要接口：

### 核心流程方法

- `__init__()`: ⭐️ **重要**

  - **功能**: 初始化一个 `IdeaSearcher` 实例。设置所有默认参数并初始化内部状态，如锁、模型管理器和岛屿字典。
  - **重要性**: 类的构造函数，是所有搜索参数设置的起点。

- `run(additional_interaction_num: int)`: ⭐️ **重要**

  - **功能**: 启动思路搜索过程。该方法会初始化并运行所有岛屿的采样器，让每个岛屿进化 `additional_interaction_num` 个世代（轮次）。
  - **重要性**: 启动整个思路搜索周期的核心方法。

### 岛屿与种群管理

- `add_island()`: ⭐️ **重要**

  - **功能**: 向系统中添加一个新岛屿，并返回其 `island_id`。首次添加岛屿时，会执行必要的初始化清理工作（例如，清理日志、旧的思路目录和备份）。
  - **重要性**: 定义搜索的并行度，并创建独立的搜索单元。

- `delete_island(island_id: int)`:

  - **功能**: 从系统中删除指定 `island_id` 的岛屿。
  - **重要性**: 允许动态管理搜索资源和策略。

- `repopulate_islands()`: ⭐️ **重要**

  - **功能**: 在岛屿之间重新分配思路。此方法按“最佳分数”对所有岛屿进行排序，然后将排名前半部分岛屿的最佳思路复制到后半部分的岛屿中，促进思路共享并防止陷入局部最优。
  - **重要性**: 进化算法中用于全局优化的关键操作。

### 结果获取

- `get_best_score()`: ⭐️ **重要**

  - **功能**: 返回所有岛屿中最高的思路分数。
  - **重要性**: 获取迄今为止找到的最佳思路的质量度量。

- `get_best_idea()`: ⭐️ **重要**

  - **功能**: 返回所有岛屿中分数最高的思路内容。
  - **重要性**: 获取迄今为止找到的最佳思路的内容。

### 便捷配置方法

- `add_initial_ideas(ideas: List[str])`:

  - **功能**: 以编程方式添加初始思路列表，作为 `database/ideas/initial_ideas/` 目录下存放 `.idea` 文件的替代方案。这使得用户无需直接操作文件系统即可提供种子思路。
  - **重要性**: 简化了初始种群的设置流程。

- `bind_helper(helper: object)`: ⭐️ **重要**

  - **功能**: 绑定一个 "helper" 对象，并使用其属性来快速配置 `IdeaSearcher` 的多个核心参数。这是一个便捷的端到端设置方法，尤其适用于构建基于 `IdeaSearch` 框架的接口工具（例如 [IdeaSearch-Fitter](https://github.com/IdeaSearch/IdeaSearch-fit)）。
  - **Helper 对象属性**:
    - **必需**: `prologue_section` (str), `epilogue_section` (str), `evaluate_func` (Callable)
    - **可选**: `initial_ideas` (List[str]), `system_prompt` (str), `assess_func` (Callable), `mutation_func` (Callable), `crossover_func` (Callable), `filter_func` (Callable), `postprocess_func` (Callable) 等。
  - **重要性**: 极大地简化了参数配置，并为框架的二次开发提供了标准接口。

## 配置参数详解

`IdeaSearcher` 提供了丰富的 `set_` 方法来配置其行为。这些参数逻辑上可划为多个类别，以便于理解与管理。

### 项目与文件系统 (Project and File System)

- **`program_name`**: `str`
  - 项目的名称，用于日志和输出中的识别。
- **`database_path`**: `str`
  - 数据库的根路径。**前置条件**: 除非使用 `add_initial_ideas()`，否则必须包含 `ideas/initial_ideas/` 子目录并在其中存放初始的 `.idea` 文件。系统将在此路径下自动创建岛屿专属的思路目录 (`ideas/island*/`)、数据目录 (`data/`)、可视化图表目录 (`pic/`) 和日志目录 (`log/`)。**这是 IdeaSearch 将会修改的唯一文件系统位置**。
- **`diary_path`**: `Optional[str]` (默认: `None`)
  - 日志文件的路径。如果为 `None`，则默认为 `{database_path}/log/diary.txt`。
- **`backup_path`**: `Optional[str]` (默认: `None`)
  - 备份存储路径。如果为 `None`，则默认为 `{database_path}/ideas/backup/`。
- 其他路径参数（如 `model_assess_result_data_path`, `assess_result_pic_path` 等）也遵循类似的模式，若为 `None` 则使用 `database_path` 下的默认位置。

### 初始化 (Initialization)

- **`load_idea_skip_evaluation`**: `bool` (默认: `True`)
  - 如果为 `True`，系统会尝试从初始思路所在目录的 `score_sheet.json` 文件中加载分数，从而跳过对这些思路的重新评估。
- **`initialization_cleanse_threshold`**: `float` (默认: `-1.0`)
  - 一个思路在初始净化阶段必须达到的最低分数。低于此阈值的思路将被删除。
- **`delete_when_initial_cleanse`**: `bool` (默认: `False`)
  - 如果为 `True`，在初始净化阶段分数低于 `initialization_cleanse_threshold` 的思路将被永久删除。

### 采样 (Sampling)

- **`samplers_num`**: `int` (默认: `3`)
  - 每个岛屿并行运行的采样器 (Sampler) 线程数量。
- **`sample_temperature`**: `float` (默认: `50.0`)
  - 用于采样历史思路作为提示词上下文的 softmax 温度。值越高，随机性越大。
- **`generation_bonus`**: `float` (默认: `0.0`)
  - 在采样过程中，为较新代际的思路添加的分数奖励。这鼓励系统探索更新的进化路径。

### 提示词工程 (Prompt Engineering)

- **`system_prompt`**: `Optional[str]` (默认: `None`)
  - 发送给大语言模型的系统级指令，用于设定整体上下文和角色。
- **`explicit_prompt_structure`**: `bool` (默认: `True`)
  - 如果为 `True`，将在提示词中自动包含结构化标题（如 "Examples:"），以获得更好的组织性。
- **`prologue_section`**: `str`
  - 用户定义的字符串，出现在每个提示词的开头，通常用于提供指令或背景。
- **`epilogue_section`**: `str`
  - 用户定义的字符串，出现在每个提示词的末尾，常用于格式化指令或最终命令。
- **`filter_func`**: `Optional[Callable[[str], str]]` (默认: `None`)
  - 一个自定义函数，用于在思路内容被采样并包含到提示词中之前对其进行预处理。
- **`examples_num`**: `int` (默认: `3`)
  - 在每一轮生成中，作为示例包含在提示词中的历史思路数量。
- **`include_info_in_prompt`**: `bool` (默认: `True`)
  - 如果为 `True`，将在提示词中包含由 `evaluate_func` 返回的补充 `info` 字符串，与思路内容和分数一起展示。
- **`images`**: `List[Any]` (默认: `[]`)
  - 要传递给视觉语言模型 (VLM) 的图像列表。在 `prologue_section` 或 `epilogue_section` 中使用占位符来定位它们；可自动解析 url、图片文件路径和裸字节。
- **`image_placeholder`**: `str` (默认: `"<image>"`)
  - 在提示词部分用于指示应插入 `images` 列表中图像的占位符字符串。
- **`generate_prompt_func`**: `Optional[Callable[[List[str], List[float], List[Optional[str]]], str]]` (默认: `None`)
  - 一个自定义函数，可以完全控制提示词的生成，覆盖默认的（序言、示例、跋语）结构。**注意**: 这是一个实验性功能，可能不稳定。

### 模型配置 (Model Configuration)

- **`api_keys_path`**: `str`
  - 指向包含 API 密钥和模型端点信息的 JSON 配置文件路径。
- **`models`**: `List[str]`
  - 用于思路生成的模型别名列表（例如, `'GPT4_o'`, `'Deepseek_V3'`）。这些别名必须是 `api_keys_path` 文件中键的子集。
- **`model_temperatures`**: `List[float]`
  - LLM 的采样温度列表。此列表的长度和顺序必须与 `models` 列表匹配。
- **`model_sample_temperature`**: `float` (默认: `50.0`)
  - 用于选择下一轮生成使用哪个模型的 softmax 温度。值越高，模型选择的随机性越大。
- **`top_p`**: `Optional[float]` (默认: `None`)
  - `top_p` 核采样参数，控制令牌选择的累积概率。对应于标准的 API 参数。
- **`max_completion_tokens`**: `Optional[int]` (默认: `None`)
  - 在一次补全中生成的最大令牌数。对应于标准的 API 参数。

**API 密钥文件格式 (`api_keys.json`):**

此文件应为一个 JSON 对象，其中每个顶级键对应于您将在 `set_models()` 中使用的唯一模型别名。每个模型别名的值是一个字典列表，每个字典代表该模型的一个实例。这允许您为同一个逻辑模型配置多个实例（例如，使用不同的 API 密钥或基础 URL），系统将自动管理它们。

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

### 生成与后处理 (Generation and Post-processing)

- **`generate_num`**: `int` (默认: `1`)
  - 每个采样器线程在单轮中尝试生成的新思路数量。
- **`postprocess_func`**: `Optional[Callable[[str], str]]` (默认: `None`)
  - 一个自定义函数，用于在 LLM 生成的原始文本保存为思路文件之前对其进行清理或格式化。
- **`hand_over_threshold`**: `float` (默认: `0.0`)
  - 新生成的思路必须从评估器获得最低分数才能被接纳到岛屿的种群中。

### 评估器 (Evaluator)

- **`evaluators_num`**: `int` (默认: `3`)
  - 每个岛屿并行运行的评估器 (Evaluator) 线程数量。
- **`evaluate_func`**: `Callable[[str], Tuple[float, Optional[str]]]`
  - 核心评估函数。它接受一个思路的内容（字符串）作为输入，并且必须返回一个元组 `(score: float, info: Optional[str])`。
- **`score_range`**: `Tuple[float, float]` (默认: `(0.0, 100.0)`)
  - 一个元组 `(min_score, max_score)`，定义了 `evaluate_func` 的预期输出范围。用于归一化和可视化。

### 数据库评估 (Database Assessment)

- **`assess_func`**: `Optional[Callable[[List[str], List[float], List[Optional[str]]], float]]` (默认: `default_assess_func`)
  - 一个自定义函数，用于评估整个思路数据库的总体状态，提供一个整体的质量度量。
- **`assess_interval`**: `Optional[int]` (默认: `1`)
  - 调用 `assess_func` 来评估整个数据库的频率（以轮次为单位）。
- **`assess_baseline`**: `Optional[float]` (默认: `60.0`)
  - 一个基线分数值，将在数据库评估图上绘制为水平线，以便于性能比较。

### 模型评估 (Model Assessment)

- **`model_assess_window_size`**: `int` (默认: `20`)
  - 在计算模型移动平均性能得分时，所考虑的由该模型最近生成的思路数量。
- **`model_assess_initial_score`**: `float` (默认: `100.0`)
  - 分配给每个模型的初始分数。较高的值鼓励对所有可用模型的初始探索。
- **`model_assess_average_order`**: `float` (默认: `1.0`)
  - 用于计算模型得分移动平均的 p-范数（广义平均值）的阶数 $p$。$p=1$ 是算术平均值，较高的值会给予高分更大的权重。
- **`model_assess_save_result`**: `bool` (默认: `True`)
  - 如果为 `True`，将模型评估数据和可视化结果保存到指定的路径。

### 变异 (Mutation)

- **`mutation_func`**: `Optional[Callable[[str], str]]` (默认: `None`)
  - 一个自定义函数，接受一个思路的内容并返回一个轻微修改后的版本。
- **`mutation_interval`**: `Optional[int]` (默认: `None`)
  - 在岛屿种群上执行变异操作的频率（以轮次为单位）。
- **`mutation_num`**: `Optional[int]` (默认: `None`)
  - 每次触发操作时通过变异生成的新思路数量。
- **`mutation_temperature`**: `Optional[float]` (默认: `None`)
  - 用于选择变异父代思路的 softmax 温度。值越高，得分较低的思路被变异的机会就越大。

### 交叉 (Crossover)

- **`crossover_func`**: `Optional[Callable[[str, str], str]]` (默认: `None`)
  - 一个自定义函数，接受两个思路的内容并返回一个结合了两方元素的新思路。
- **`crossover_interval`**: `Optional[int]` (默认: `None`)
  - 执行交叉操作的频率（以轮次为单位）。
- **`crossover_num`**: `Optional[int]` (默认: `None`)
  - 每次触发操作时通过交叉生成的新思路数量。
- **`crossover_temperature`**: `Optional[float]` (默认: `None`)
  - 用于选择交叉父代思路的 softmax 温度。值越高，父代选择的随机性越大。

### 相似度 (Similarity) - 不常用

- **`similarity_threshold`**: `float` (默认: `-1.0`)
  - 距离阈值，低于此值的两个思路被认为是相似的。值为 `-1.0` 表示除完全重复外，禁用相似性检查。
- 其他相关参数如 `similarity_distance_func` 等用于更复杂的相似性控制。

### 杂项 (Miscellaneous)

- **`idea_uid_length`**: `int` (默认: `6`)
  - `.idea` 文件名中使用的唯一标识符 (UID) 的字符长度。
- **`record_prompt_in_diary`**: `bool` (默认: `False`)
  - 如果为 `True`，每轮生成中发送给 LLM 的完整提示词将被记录在日志文件中。
- **`backup_on`**: `bool` (默认: `True`)
  - 如果为 `True`，则在 `run` 方法开始时启用对 `ideas` 目录的自动备份。
- **`shutdown_score`**: `float` (默认: `float('inf')`)
  - 如果所有岛屿中的最佳分数达到此值，IdeaSearch 进程将优雅地终止。

## 工作流程概览

一个典型的 `IdeaSearch` 使用流程展示了如何组织代码以实现完整的进化循环，包括多岛屿的创建、周期性的种群交换和持续的思路生成。以下示例借鉴自 [IdeaSearch-test 仓库](https://github.com/IdeaSearch/IdeaSearch-test)，代表了一种常见的实践模式。

```python
# pip install IdeaSearch
from IdeaSearch import IdeaSearcher
from user_code.prompt import prologue_section, epilogue_section
from user_code.evaluation import evaluate
from user_code.initial_ideas import initial_ideas


def main():
    # 1. 初始化
    ideasearcher = IdeaSearcher()

    # 2. 基础配置
    ideasearcher.set_language("zh_CN")  # 设置语言 (默认: 'zh_CN'; 可选: 'zh_CN', 'en')
    ideasearcher.set_api_keys_path("api_keys.json")
    ideasearcher.set_program_name("TemplateProgram")
    ideasearcher.set_database_path("database")

    # 3. 核心逻辑配置 (评估函数)
    ideasearcher.set_evaluate_func(evaluate)

    # 4. 提示词工程配置
    ideasearcher.set_prologue_section(prologue_section)
    ideasearcher.set_epilogue_section(epilogue_section)

    # 5. 模型配置
    ideasearcher.set_models([
        "Deepseek_V3",
    ])
    ideasearcher.set_model_temperatures([
        0.6,
    ])

    # 6. (可选) 其他配置
    ideasearcher.set_record_prompt_in_diary(True)

    # 7. 添加初始思路 (无需在文件系统中操作)
    ideasearcher.add_initial_ideas(initial_ideas)

    # 8. 定义并执行进化循环
    island_num = 2             # 岛屿数量
    cycle_num = 3              # 迁徙周期数
    unit_interaction_num = 10  # 每个周期的进化轮次

    # 创建初始岛屿
    for _ in range(island_num):
        ideasearcher.add_island()

    # 运行进化
    for cycle in range(cycle_num):
        print(f"---[ 周期 {cycle + 1}/{cycle_num} ]---")

        # 在每个周期开始前 (除第一个周期外) 进行岛屿间思路迁徙
        if cycle != 0:
            print("正在重新繁衍岛屿...")
            ideasearcher.repopulate_islands()

        # 在当前周期内运行指定轮次的进化
        ideasearcher.run(unit_interaction_num)

    # 9. 获取并利用最终结果
    print("\n---[ 搜索完成 ]---")
    best_idea_content = ideasearcher.get_best_idea()
    print("最佳思路内容:")
    print(best_idea_content)


if __name__ == "__main__":
    main()
```

## 国际化

`IdeaSearch` 支持其界面文本的国际化。

你可以使用 `set_language(value: str)` 方法来设置系统语言。

例如, `ideasearcher.set_language('en')` 会将界面和日志文本切换为英文，而 `ideasearcher.set_language('zh_CN')` 会切换为简体中文。

`IdeaSearch` 的默认语言是简体中文 (`zh_CN`)。
