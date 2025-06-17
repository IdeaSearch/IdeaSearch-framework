# About IdeaSearch

## 项目概述

`IdeaSearch` 框架是一个 **AI 驱动的研究 Idea 生成与优化系统**。在 `IdeaSearch` 中，一个 **Idea** 特指以 `.idea` 为后缀名的文本文件，这些文件包含了系统可读取和处理的创意内容。**它在设计上受到 Google 于 2023 年推出的 FunSearch 框架的启发。FunSearch 框架通过结合大语言模型与评估程序，开创性地实现了发现新的数学结构和算法。IdeaSearch 则在此基础上，致力于构建一个更加用户友好、界面简洁的整合框架，目标在于支持科研与教育的多个领域。**

相较于 FunSearch，`IdeaSearch` 引入了多个创新特性，增强了系统的灵活性和探索能力，包括：

- **生成奖励 (generation_bonus)**：为新生成的 Idea 提供额外的分数奖励。此机制鼓励系统持续探索和产出新颖、更具活力的 Idea，有效避免系统过早陷入局部最优，促进 Idea 空间的广度探索。
- **变异 (mutation)**：引入随机性，允许对现有 Idea 进行小幅度的修改和扰动。这为 Idea 搜索过程注入了偶然性和多样性，有助于发现意想不到的新方向或优化现有 Idea，即便在看似饱和的 Idea 空间中也能带来突破。
- **交叉 (crossover)**：通过组合两个或多个现有 Idea 的元素，生成新的混合 Idea。这一遗传算法中的经典操作在 `IdeaSearch` 中得到了增强，它促进了更复杂的进化路径，能够融合不同优秀 Idea 的优点，产生超越单一 Idea 局限性的新颖组合。
- **提示词的前言（prologue_section）和结语（epilogue_section）**：允许用户更灵活、模块化地定义发送给大语言模型的提示词的开头和结尾部分。这使得用户可以轻松地为模型提供上下文、设定任务目标或指导输出格式，而无需每次都重写整个提示词。同时，如果用户选择，也可以通过自定义的 `generate_prompt_func` 函数完全掌控提示词的生成逻辑，提供了极大的自由度以适应各种复杂场景。
- **评估器信息 (evaluator info)**：除了提供量化的 Idea 分数，评估函数现在可以返回额外的字符串信息。这使得用户不仅能知道 Idea “好不好”的量化结果，还能通过这些附加信息了解“为什么好”、“哪里可以改进”或“其独特之处”，为后续的 Idea 优化和系统分析提供了更丰富的上下文和深层洞察。

## 主要功能特性

- **多岛屿并行搜索**: 支持创建多个独立的“岛屿”，每个岛屿都包含自己的采样器（Sampler）和评估器（Evaluator），并行探索 Idea 空间，提高搜索效率和多样性。
- **大语言模型 (LLM) 集成**: 通过 `ModelManager` 管理多种 LLM 模型，并动态选择模型进行 Idea 生成。
- **进化策略**:
  - **采样 (Sampling)**: 根据现有 Idea 的分数和温度参数，选择高质量的 Idea 作为生成新 Idea 的参考。
  - **评估 (Evaluation)**: 通过用户自定义的 `evaluate_func` 对每个生成的 Idea 进行评分，并可选择性地返回额外信息。
  - **变异 (Mutation)**: 通过用户自定义的 `mutation_func` 对 Idea 进行小幅修改，引入多样性。
  - **交叉 (Crossover)**: 通过用户自定义的 `crossover_func` 组合现有 Idea，生成新的混合 Idea。
- **动态模型选择与评估**: 根据模型在生成高质量 Idea 方面的表现，动态调整模型在未来轮次中被选择的概率，以鼓励表现更好的模型。提供模型分数的可视化。
- **系统整体评估与可视化**: 定期通过用户自定义的 `assess_func` 评估整个 Idea 数据库的整体质量，并生成图表展示进化过程中的质量趋势。
- **数据持久化与备份**: 自动管理 Idea 文件和评分数据，支持备份功能，确保搜索过程中的数据安全。
- **高度可配置**: 提供了丰富的参数（通过 `set_` 方法）供用户定制搜索行为，包括模型温度、采样策略、评估间隔、相似度阈值等。
- **国际化支持**: 内置 `gettext` 国际化机制，支持多种语言（目前支持中文 `zh_CN` 和英文 `en`）。

## 核心 API

以下是 `IdeaSearcher` 类中标记为“⭐️ Important”的关键方法，它们构成了与用户交互的主要接口：

- `__init__()`:
  - **功能**: 初始化 `IdeaSearcher` 实例。设置所有默认参数，并初始化内部状态，如锁、模型管理器、岛屿字典等。
  - **重要性**: 类的构造函数，所有搜索参数的起点。
- `run(additional_interaction_num: int)`:
  - **功能**: 启动 Idea 搜索过程。该方法会初始化并运行所有岛屿的采样器，使每个岛屿演化 `additional_interaction_num` 个 epoch（轮次）。
  - **重要性**: 核心的启动方法，开始整个 Idea 搜索循环。
- `load_models()`:
  - **功能**: 从 `api_keys_path` 指定的配置文件中加载所有 LLM 模型的 API 密钥。
  - **重要性**: 在进行任何 LLM 交互之前必须调用此方法。
- `add_island()`:
  - **功能**: 添加一个新的岛屿到系统中，并返回其 `island_id`。首次添加岛屿会进行初始化清理工作（如清空日志、旧的 Idea 目录和备份）。
  - **重要性**: 定义搜索的并行度，创建独立的搜索单元。
- `delete_island(island_id: int)`:
  - **功能**: 从系统中删除指定 ID 的岛屿。
  - **重要性**: 允许动态管理搜索资源和策略。
- `repopulate_islands()`:
  - **功能**: 在岛屿之间重新分配 Idea。该方法会根据岛屿的“最佳分数”对所有岛屿进行排序，然后将排名靠前一半岛屿的最佳 Idea 复制到排名靠后一半的岛屿中，以促进 Idea 共享和避免局部最优。
  - **重要性**: 进化算法中的关键操作，用于全局优化。
- `get_best_score()`:
  - **功能**: 返回所有岛屿中最高 Idea 的分数。
  - **重要性**: 获取当前搜索到的最佳 Idea 质量指标。
- `get_best_idea()`:
  - **功能**: 返回所有岛屿中最高分 Idea 的具体内容。
  - **重要性**: 获取当前搜索到的最佳 Idea 内容。

## 重要配置参数

`IdeaSearcher` 提供了丰富的 `set_` 方法来配置其行为。以下是一些最关键的参数，它们必须在 `run()` 方法调用之前设置：

- `set_program_name(value: str)`: 项目的名称，用于日志和识别。
- `set_prologue_section(value: str)`: 生成提示词的固定序言部分。
- `set_epilogue_section(value: str)`: 生成提示词的固定结语部分。
  - **注意**: `prologue_section` 和 `epilogue_section` 可以被 `generate_prompt_func` 替代，如果 `generate_prompt_func` 被设置，则这两个参数可以为 `None`。
- `set_database_path(value: str)`: Idea 数据库的根路径。系统将在此路径下自动创建 `ideas/` (存放 Idea 文件)、`data/` (存放数据文件，如评估结果)、`pic/` (存放图片，如趋势图) 和 `log/` (存放日志) 等子目录。
- `set_models(value: List[str])`: 参与 Idea 生成的大语言模型名称列表。
- `set_model_temperatures(value: List[float])`: 与 `models` 列表对应的每个模型的采样温度。此列表的长度必须与 `models` 列表相同。
- `set_evaluate_func(value: Callable[[str], Tuple[float, Optional[str]]])`: 用于评估单个 Idea 的函数。它接收一个 Idea 字符串作为输入，并返回一个包含分数（float）和可选的额外信息（Optional[str]）的元组。
- `set_api_keys_path(value: str)`: 包含 API 密钥配置的 JSON/YAML 文件的路径。

  **API 密钥文件格式 (`api_keys.json`)：**
  此文件应为一个 JSON 对象，其中每个顶级键对应一个您将在 `set_models()` 中使用的唯一模型别名。每个模型别名的值是一个字典列表，其中每个字典代表该模型的一个实例。**这允许您为同一个逻辑模型配置多个实例（例如，使用不同的 API 密钥或基础 URL），系统将自动管理它们。**

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

所有参数都有对应的 `get_` 方法用于获取当前值。例如，`ideasearcher.get_program_name()`。

## 工作流程概览

一个典型的 `IdeaSearch` 使用流程如下：

1.  **实例化**:
    ```python
    from IdeaSearch.ideasearcher import IdeaSearcher
    ideasearcher = IdeaSearcher()
    ```
2.  **配置核心参数**:

    ```python
    # 示例评估函数
    def my_custom_evaluate_function(idea: str) -> Tuple[float, Optional[str]]:
        # 这里是你的评估逻辑，返回分数和可选信息
        score = len(idea) # 简单示例：Idea 越长分数越高
        info = "Idea length is " + str(len(idea))
        return float(score), info

    ideasearcher.set_program_name("我的AI Idea 项目")
    ideasearcher.set_database_path("./my_idea_database") # 请确保此路径存在且包含 ideas/initial_ideas/ 子目录
    ideasearcher.set_models(["Deepseek_V3", "Qwen_Max"]) # 使用 api_keys.json 中定义的模型别名
    ideasearcher.set_model_temperatures([0.7, 0.5])
    ideasearcher.set_evaluate_func(my_custom_evaluate_function)
    ideasearcher.set_api_keys_path("./api_keys.json") # 假设你的API密钥文件在此
    ideasearcher.set_prologue_section("请为以下主题生成创新的商业 Idea，要求简洁明了：")
    ideasearcher.set_epilogue_section("每个 Idea 必须以“Idea：”开头。")

    # 更多可选配置...
    ideasearcher.set_samplers_num(2) # 每个岛屿2个采样器
    ideasearcher.set_evaluators_num(1) # 每个岛屿1个评估器
    ideasearcher.set_assess_interval(5) # 每5轮评估一次数据库
    ideasearcher.set_generation_bonus(5.0) # 新生成 Idea 有5分奖励
    # ... 其他 set_ 方法
    ```

3.  **加载模型 API 密钥**:
    ```python
    ideasearcher.load_models()
    ```
4.  **添加岛屿**:
    ```python
    # 添加一个岛屿。可以多次调用 add_island() 来增加并行搜索的岛屿数量。
    # initial_ideas 目录下的 Idea 文件会被加载到第一个岛屿中。
    island_id_1 = ideasearcher.add_island()
    print(f"Island {island_id_1} added.")
    ```
5.  **运行搜索**:
    ```python
    # 启动多线程的搜索过程，每个岛屿将演化指定的轮次。
    print("Starting IdeaSearch...")
    ideasearcher.run(additional_interaction_num=50) # 每个岛屿运行50个epoch
    print("IdeaSearch finished.")
    ```
6.  **（可选）岛屿间 Idea 重分布**:
    ```python
    # 在搜索过程中或结束后调用，以促进优秀 Idea 的传播和融合。
    ideasearcher.repopulate_islands()
    ```
7.  **获取最佳结果**:
    ```python
    best_score = ideasearcher.get_best_score()
    best_idea = ideasearcher.get_best_idea()
    print(f"\n当前最佳 Idea 分数: {best_score}")
    print(f"当前最佳 Idea 内容: {best_idea}")
    ```
8.  **关闭模型**:
    ```python
    ideasearcher.shutdown_models()
    print("Models shut down.")
    ```

## 国际化

`IdeaSearch` 支持界面文本的国际化。
您可以通过 `set_language(value: str)` 方法设置系统语言。
例如，`ideasearcher.set_language('en')` 将把界面和日志中的文本切换为英文，`ideasearcher.set_language('zh_CN')` 则切换为简体中文。
`IdeaSearch` 的默认语言是简体中文 (`zh_CN`)。
