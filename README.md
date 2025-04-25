## 🚀 运行项目指南

### 📥 克隆框架仓库

在你希望创建框架文件夹的位置，运行以下命令以克隆 IdeaSearch 框架仓库：

```bash
git clone https://github.com/IdeaSearch/IdeaSearch-framework IdeaSearch
```

克隆完成后，你将看到如下的项目结构：

```text
📦 项目根目录/
├── programs/
│   └── TemplateProgram/
│       ├── database/
│       │   └── initial_idea.idea
│       ├── evaluator/
│       │   └── evaluator.py
│       └── prompt.py
│   └── .gitignore
│
├── src/
│   ├── API4LLMs/
│   │   ├── api_keys_example.json
│   │   └── api_keys.json 📝 (需手动创建)
│   ├── FunSearch/
│   ├── my_script_template.py
│   ├── my_script.py 🛠 (需手动创建)
│   └── diary.txt 📔 (程序运行时自动生成)
│
└── README.md
```

---

### 🧱 创建你的项目

你可以在 `programs/` 下创建与 `TemplateProgram` 平级的自定义项目 `MyProgram`，目录结构建议参考 `TemplateProgram`，包括以下部分：

📁 `database/`  
用于存放大语言模型生成的 idea 文件。每个 idea 应为以 `.idea` 结尾的 UTF-8 编码纯文本文件，项目初始化时，**至少要包含一个 idea 文件**。

📁 `evaluator/`  
包含 `evaluator.py` 文件，提供一个名为 `evaluate` 的函数，用于评估大语言模型生成的 idea。函数签名请参考：
`programs/TemplateProgram/evaluator/evaluator.py`。

📄 `prompt.py`  
提供两个字符串变量：`prologue_section` 与 `epilogue_section`。IdeaSearch 在生成 prompt 时会将这两个部分与中间的 examples 一起拼接，构成完整的提示词。

> ✅ 注意：`programs/.gitignore` 会忽略 `TemplateProgram` 以外的所有子项目。建议使用 VS Code，运行：
>
> ```bash
> code programs/MyProgram
> ```
>
> 打开你的子项目，并在此页面中创建 Git 仓库进行开发。

📡 在开发期间，可以将远程仓库设为**私有仓库**保护隐私；项目完成后若希望分享，也可将其迁移至 `IdeaSearch` 组织的私有仓库，方便组织成员共同查阅。

🗂 推荐以 **`IdeaSearch-xxxxxx`** 命名你的远端仓库，便于识别。

---

### 📦 克隆已有项目

若想将某个已有项目（例如 `https://github.com/user_id/IdeaSearch-xxxxxx`）克隆进你的框架中，在项目根目录下运行：

```bash
git clone https://github.com/user_id/IdeaSearch-xxxxxx programs/xxxxxx
```

---

### 🎯 运行 IdeaSearch

在开始 IdeaSearch 之前，请完成以下准备工作：

1. ✍ **创建密钥文件**  
   在 `src/API4LLMs/` 下，**仿照 `api_keys_example.json`** 创建 `api_keys.json` 文件，并填写你的 API 密钥。

2. 🛠 **准备运行脚本**  
   在 `src/` 下，**仿照 `my_script_template.py`** 创建 `my_script.py`，并在其中导入要运行的项目的 `evaluate` 函数与 `prompt`，设置 IdeaSearch 的相关参数。

准备就绪后，在项目根目录下运行：

```bash
python -u -m src.my_script > src/diary.txt
```

📒 日志输出会被自动记录到 `src/diary.txt` 中，方便你随时回顾每一次 IdeaSearch 的全过程！
