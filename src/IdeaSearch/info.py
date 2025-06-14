import gettext
import os
from pathlib import Path

# 设置本地化目录
_LOCALE_DIR = Path(__file__).parent / "locales"
_DOMAIN = "ideasearch"

# 初始化gettext
gettext.bindtextdomain(_DOMAIN, _LOCALE_DIR)
gettext.textdomain(_DOMAIN)
_ = gettext.gettext

def get_info():
    """获取包信息，支持国际化"""
    return {
        "name": _("IdeaSearch Framework"),
        "description": _("A powerful framework for idea search and management"),
        "version": "1.0.0",
        "author": _("Development Team"),
        "license": "MIT",
        "homepage": "https://github.com/your-org/ideasearch-framework"
    }

def get_info_text():
    """获取格式化的包信息文本"""
    info = get_info()
    return f"""{info['name']} v{info['version']}
{info['description']}
{_('Author')}: {info['author']}
{_('License')}: {info['license']}
{_('Homepage')}: {info['homepage']}"""

def set_language(lang_code):
    """设置语言"""
    global _
    translation = gettext.translation(_DOMAIN, _LOCALE_DIR, languages=[lang_code], fallback=True)
    _ = translation.gettext
    # 重新生成INFO变量以使用新的语言
    global INFO
    INFO = get_info_text()

# 为了保持向后兼容性，保留原有的INFO变量
INFO = get_info_text()

# 如果直接运行此文件，打印信息
if __name__ == "__main__":
    print(get_info_text())
    
    # 演示语言切换
    print("\n" + _("Switching to English:"))
    set_language('en')
    print(get_info_text())
    
    print("\n" + _("Switching to Chinese:"))
    set_language('zh_CN')
    print(get_info_text())