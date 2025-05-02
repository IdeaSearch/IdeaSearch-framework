__all__ = [
    "prologue_section",
    "epilogue_section",
]


prologue_section = (
    "我现在正在测试 IdeaSearch 系统能否顺利运行。"
    " IdeaSearch 系统会依据一个智能算法不断从数据库中选择 idea ，"
    "然后说给你（大语言模型）听，让你知道我们搜寻 idea 的目的与已有的 idea ，看看你能否提出更好的点子。\n"
    "每次说给你听的 prompt 包含三个部分，现在这个部分是 prologue section 。"
    "接下来是 examples section ：\n"
)

epilogue_section = (
    "最后，这里是 epilogue section 。你可以看到，由于这只是一个用于测试系统运行的玩具项目，"
    "所有的 examples 的得分都是随机的，评语都是“非常好！”，请你也随便说点啥吧。"
)