__all__ = [
    "system_prompt",
    "prologue_section",
    "epilogue_section",
]


system_prompt = "你是一个哲学家，总是思考事物的深层意义。每次回答都带有深刻的哲理，或者提问让人思考人生的奥义。"


prologue_section = (
    "我现在正在测试 IdeaSearch IdeaSearcher能否顺利运行。"
    " IdeaSearch IdeaSearcher会依据一个智能算法不断从岛屿中选择 idea ，"
    "然后说给你（大语言模型）听，让你知道我们搜寻 idea 的目的与已有的 idea ，看看你能否提出更好的点子。\n"
    "每次说给你听的 prompt 包含三个部分，现在这个部分是 prologue section 。"
    "接下来是 examples section ：\n"
)


epilogue_section = (
    "最后，这里是 epilogue section 。你可以看到，由于这只是一个用于测试IdeaSearcher运行的模板项目，"
    "所有的 examples 的得分都是随机的，请你也随便说点啥吧。"
)