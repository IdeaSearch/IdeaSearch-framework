#!/usr/bin/env python3
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional


class I18nManager:
    """国际化管理器"""
    
    def __init__(self, project_root: Optional[str] = None):
        """
        初始化国际化管理器
        
        Args:
            project_root: 项目根目录路径，默认为脚本所在目录的上级目录
        """
        if project_root is None:
            # 获取脚本所在目录的上级目录作为项目根目录
            script_dir = Path(__file__).parent
            self.project_root = script_dir.parent
        else:
            self.project_root = Path(project_root)
            
        # 设置相关路径
        self.src_dir = self.project_root
        self.locales_dir = self.project_root / "IdeaSearch" / "locales"
        self.babel_cfg = self.src_dir / "IdeaSearch" / "babel.cfg"
        self.pot_file = self.locales_dir / "ideasearch.pot"
        
        # 支持的语言列表
        self.supported_languages = ["en", "zh_CN"]
        
    def check_babel_installed(self) -> bool:
        """检查Babel是否已安装"""
        try:
            result = subprocess.run(
                ["pybabel", "--version"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            print(f"✓ Babel已安装: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ 错误: Babel未安装或不在PATH中")
            print("请运行: pip install Babel")
            return False
    
    def ensure_directories(self):
        """确保必要的目录存在"""
        self.locales_dir.mkdir(exist_ok=True)
        print(f"✓ 确保目录存在: {self.locales_dir}")
        
        for lang in self.supported_languages:
            lang_dir = self.locales_dir / lang / "LC_MESSAGES"
            lang_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ 确保语言目录存在: {lang_dir}")
    
    def check_babel_config(self) -> bool:
        """检查Babel配置文件是否存在"""
        if self.babel_cfg.exists():
            print(f"✓ Babel配置文件存在: {self.babel_cfg}")
            return True
        else:
            print(f"✗ Babel配置文件不存在: {self.babel_cfg}")
            return False
    
    def extract_messages(self) -> bool:
        """提取可翻译的消息到POT文件"""
        print("\n--- 提取可翻译消息 ---")
        
        if not self.check_babel_config():
            return False
            
        try:
            cmd = [
                "pybabel", "extract",
                "--no-location",
                # "--no-wrap",
                "--sort-by-file",
                "-F", str(self.babel_cfg),
                "-k", "_",  # 标记函数名
                "-k", "gettext",
                "-k", "ngettext",
                "-o", str(self.pot_file),
                str(self.src_dir)
            ]
            
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            if self.pot_file.exists():
                print(f"✓ 成功提取消息到: {self.pot_file}")
                
                # 显示提取的消息数量
                with open(self.pot_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    msgid_count = content.count('msgid ')
                    print(f"✓ 提取了 {msgid_count} 个可翻译字符串")
                return True
            else:
                print("✗ POT文件未生成")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"✗ 提取消息失败: {e}")
            if e.stderr:
                print(f"错误详情: {e.stderr}")
            return False
    
    def init_language(self, language: str) -> bool:
        """
        为指定语言初始化PO文件
        
        Args:
            language: 语言代码 (如 'zh_CN', 'en', 'ja')
        """
        print(f"\n--- 初始化语言: {language} ---")
        
        if not self.pot_file.exists():
            print(f"✗ POT文件不存在: {self.pot_file}")
            print("请先运行 extract 命令")
            return False
        
        # 修改为ideasearch域的PO文件
        po_file = self.locales_dir / language / "LC_MESSAGES" / "ideasearch.po"
        
        if po_file.exists():
            print(f"! 警告: PO文件已存在: {po_file}")
            response = input("是否覆盖? (y/N): ").lower()
            if response != 'y':
                print("操作已取消")
                return False
        
        try:
            cmd = [
                "pybabel", "init",
                "-i", str(self.pot_file),
                "-d", str(self.locales_dir),
                "-D", "ideasearch",  # 指定域名
                "-l", language
            ]
            
            print(f"执行命令: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            print(f"✓ 成功为语言 {language} 初始化PO文件: {po_file}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ 初始化语言失败: {e}")
            return False

    def update_translations(self, language: Optional[str] = None) -> bool:
        """
        更新翻译文件
        
        Args:
            language: 指定语言代码，如果为None则更新所有语言
        """
        if language:
            languages = [language]
            print(f"\n--- 更新翻译: {language} ---")
        else:
            # 查找所有存在的语言
            languages = []
            for lang in self.supported_languages:
                # 修改为ideasearch域的PO文件
                po_file = self.locales_dir / lang / "LC_MESSAGES" / "ideasearch.po"
                if po_file.exists():
                    languages.append(lang)
            print(f"\n--- 更新所有翻译: {', '.join(languages)} ---")
        
        if not languages:
            print("✗ 没有找到要更新的翻译文件")
            return False
            
        if not self.pot_file.exists():
            print(f"✗ POT文件不存在: {self.pot_file}")
            print("请先运行 extract 命令")
            return False
        
        success = True
        for lang in languages:
            # 修改为ideasearch域的PO文件
            po_file = self.locales_dir / lang / "LC_MESSAGES" / "ideasearch.po"
            
            if not po_file.exists():
                print(f"✗ PO文件不存在: {po_file}")
                print(f"请先为语言 {lang} 运行 init 命令")
                success = False
                continue
            
            try:
                cmd = [
                    "pybabel", "update",
                    "--omit-header",
                    "-i", str(self.pot_file),
                    "-d", str(self.locales_dir),
                    "-D", "ideasearch",  # 指定域名
                    "-l", lang
                ]
                
                print(f"更新 {lang}: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                print(f"✓ 成功更新语言 {lang}")
                
            except subprocess.CalledProcessError as e:
                print(f"✗ 更新语言 {lang} 失败: {e}")
                success = False
        
        return success
    
    def compile_translations(self, language: Optional[str] = None) -> bool:
        """
        编译翻译文件为MO文件
        
        Args:
            language: 指定语言代码，如果为None则编译所有语言
        """
        if language:
            languages = [language]
            print(f"\n--- 编译翻译: {language} ---")
        else:
            # 查找所有存在的语言
            languages = []
            for lang in self.supported_languages:
                po_file = self.locales_dir / lang / "LC_MESSAGES" / "ideasearch.po"
                if po_file.exists():
                    languages.append(lang)
            print(f"\n--- 编译所有翻译: {', '.join(languages)} ---")
        
        if not languages:
            print("✗ 没有找到要编译的翻译文件")
            return False
        
        success = True
        for lang in languages:
            # 修改为ideasearch域的PO和MO文件
            po_file = self.locales_dir / lang / "LC_MESSAGES" / "ideasearch.po"
            mo_file = self.locales_dir / lang / "LC_MESSAGES" / "ideasearch.mo"
            
            if not po_file.exists():
                print(f"✗ PO文件不存在: {po_file}")
                success = False
                continue
            
            try:
                cmd = [
                    "pybabel", "compile",
                    "-d", str(self.locales_dir),
                    "-D", "ideasearch",  # 指定域名
                    "-l", lang
                ]
                
                print(f"编译 {lang}: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                
                if mo_file.exists():
                    print(f"✓ 成功编译语言 {lang}: {mo_file}")
                else:
                    print(f"✗ 编译语言 {lang} 失败: MO文件未生成")
                    success = False
                
            except subprocess.CalledProcessError as e:
                print(f"✗ 编译语言 {lang} 失败: {e}")
                success = False
        
        return success
    
    def list_languages(self):
        """列出所有支持的语言和其状态"""
        print("\n--- 支持的语言状态 ---")
        print(f"{'语言代码':<10} {'PO文件':<8} {'MO文件':<8} {'路径'}")
        print("-" * 60)
        
        for lang in self.supported_languages:
            po_file = self.locales_dir / lang / "LC_MESSAGES" / "ideasearch.po"
            mo_file = self.locales_dir / lang / "LC_MESSAGES" / "ideasearch.mo"
            
            po_status = "✓" if po_file.exists() else "✗"
            mo_status = "✓" if mo_file.exists() else "✗"
            
            print(f"{lang:<10} {po_status:<8} {mo_status:<8} {po_file.parent}")
    
    def clean(self):
        """清理生成的文件"""
        print("\n--- 清理生成的文件 ---")
        
        files_removed = 0
        
        # 删除POT文件
        if self.pot_file.exists():
            self.pot_file.unlink()
            print(f"✓ 删除POT文件: {self.pot_file}")
            files_removed += 1
        
        # 删除所有MO文件
        for lang in self.supported_languages:
            # 修改为ideasearch域的MO文件
            mo_file = self.locales_dir / lang / "LC_MESSAGES" / "ideasearch.mo"
            if mo_file.exists():
                mo_file.unlink()
                print(f"✓ 删除MO文件: {mo_file}")
                files_removed += 1
        
        if files_removed == 0:
            print("没有需要清理的文件")
        else:
            print(f"✓ 总共清理了 {files_removed} 个文件")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="IdeaSearch项目国际化管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s extract                    # 提取可翻译字符串
  %(prog)s init zh_CN                # 为中文初始化翻译文件
  %(prog)s update                     # 更新所有翻译文件
  %(prog)s update zh_CN               # 更新中文翻译文件
  %(prog)s compile                    # 编译所有翻译文件
  %(prog)s compile zh_CN              # 编译中文翻译文件
  %(prog)s list                       # 列出语言状态
  %(prog)s clean                      # 清理生成的文件
        """
    )
    
    parser.add_argument(
        "command", 
        choices=["extract", "init", "update", "compile", "list", "clean"],
        help="要执行的命令"
    )
    
    parser.add_argument(
        "language", 
        nargs="?", 
        help="语言代码 (如: zh_CN, en, ja)"
    )
    
    parser.add_argument(
        "--project-root",
        help="项目根目录路径"
    )
    
    args = parser.parse_args()
    
    # 创建国际化管理器
    i18n_manager = I18nManager(args.project_root)
    
    print(f"项目根目录: {i18n_manager.project_root}")
    print(f"源代码目录: {i18n_manager.src_dir}")
    print(f"翻译目录: {i18n_manager.locales_dir}")
    
    # 检查Babel是否安装
    if not i18n_manager.check_babel_installed():
        sys.exit(1)
    
    # 确保目录存在
    i18n_manager.ensure_directories()
    
    # 执行相应命令
    success = True
    
    if args.command == "extract":
        success = i18n_manager.extract_messages()
        
    elif args.command == "init":
        if not args.language:
            print("✗ init命令需要指定语言代码")
            parser.print_help()
            sys.exit(1)
        success = i18n_manager.init_language(args.language)
        
    elif args.command == "update":
        success = i18n_manager.update_translations(args.language)
        
    elif args.command == "compile":
        success = i18n_manager.compile_translations(args.language)
        
    elif args.command == "list":
        i18n_manager.list_languages()
        
    elif args.command == "clean":
        i18n_manager.clean()
    
    if success:
        print("\n✓ 操作成功完成")
        sys.exit(0)
    else:
        print("\n✗ 操作失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
