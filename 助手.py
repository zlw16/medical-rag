"""
desktop_app.py - 桌面应用 (使用tkinter)
支持 settings.json 配置文件和设置对话框
"""

import sys
import os
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, simpledialog
import threading
from datetime import datetime

# PyInstaller 打包后资源文件的路径
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)

# 设置文件路径（exe 同目录 或 项目根目录）
if getattr(sys, 'frozen', False):
    CONFIG_DIR = os.path.dirname(sys.executable)
else:
    CONFIG_DIR = BASE_DIR

SETTINGS_FILE = os.path.join(CONFIG_DIR, "settings.json")

# 默认设置
DEFAULT_SETTINGS = {
    "DEEPSEEK_API_KEY": "",
    "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1",
}


def load_settings() -> dict:
    """从 settings.json 加载配置"""
    settings = dict(DEFAULT_SETTINGS)
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                settings.update(loaded)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    return settings


def save_settings(settings: dict):
    """保存配置到 settings.json"""
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存配置文件失败: {e}")
        return False


def apply_settings(settings: dict):
    """将设置应用到环境变量（在 RAG 初始化前调用）"""
    for key, value in settings.items():
        os.environ[key] = value  # 空值也写入，覆盖 .env 中的旧值


# 打包模式下设置路径和默认值
if getattr(sys, 'frozen', False):
    exe_dir = os.path.dirname(sys.executable)
    os.environ.setdefault("DOC_FOLDER", os.path.join(exe_dir, "medical_knowledge"))
    os.environ.setdefault("CACHE_FOLDER", os.path.join(exe_dir, "cache"))
    os.environ.setdefault("LOG_FOLDER", os.path.join(exe_dir, "logs"))
    os.environ.setdefault("SEMANTIC_WEIGHT", "0")

# 加载并应用设置
_settings = load_settings()
apply_settings(_settings)

from src.rag_engine import MedicalRAG


class SettingsDialog:
    """设置对话框"""

    def __init__(self, parent, settings: dict):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("设置")
        self.dialog.geometry("480x250")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.result = None

        frame = ttk.Frame(self.dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        # API Key
        ttk.Label(frame, text="DeepSeek API Key:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.api_key_var = tk.StringVar(value=settings.get("DEEPSEEK_API_KEY", ""))
        api_entry = ttk.Entry(frame, textvariable=self.api_key_var, width=50, show="*")
        api_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=(0, 5))

        # 显示/隐藏按钮
        def toggle_key_visibility():
            if api_entry.cget("show") == "*":
                api_entry.configure(show="")
                toggle_btn.configure(text="隐藏")
            else:
                api_entry.configure(show="*")
                toggle_btn.configure(text="显示")

        toggle_btn = ttk.Button(frame, text="显示", command=toggle_key_visibility, width=6)
        toggle_btn.grid(row=0, column=2, padx=(5, 0), pady=(0, 5))

        # Base URL
        ttk.Label(frame, text="API Base URL:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        self.base_url_var = tk.StringVar(value=settings.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
        ttk.Entry(frame, textvariable=self.base_url_var, width=50).grid(
            row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))

        # 说明
        ttk.Label(frame, text="配置 API Key 后，回答将使用 DeepSeek 大模型生成，效果更好。",
                  foreground="#666", font=("", 9)).grid(
            row=2, column=0, columnspan=3, pady=(10, 5))
        ttk.Label(frame, text="留空则使用规则生成答案（不需要网络）。",
                  foreground="#666", font=("", 9)).grid(
            row=3, column=0, columnspan=3, pady=(0, 15))

        # 按钮
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=4, column=0, columnspan=3)
        ttk.Button(btn_frame, text="保存", command=self.on_save, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=self.dialog.destroy, width=10).pack(side=tk.LEFT, padx=5)

        frame.columnconfigure(1, weight=1)

    def on_save(self):
        self.result = {
            "DEEPSEEK_API_KEY": self.api_key_var.get().strip(),
            "DEEPSEEK_BASE_URL": self.base_url_var.get().strip(),
        }
        self.dialog.destroy()


class MedicalRAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("医疗RAG助手")
        self.root.geometry("800x600")

        # 设置
        self.settings = load_settings()

        # 打包模式：首次运行将知识库从临时目录复制到 exe 旁边
        if getattr(sys, 'frozen', False):
            kb_dir = os.environ["DOC_FOLDER"]
            if not os.path.exists(kb_dir):
                import shutil
                src_kb = os.path.join(BASE_DIR, "medical_knowledge")
                if os.path.exists(src_kb):
                    os.makedirs(os.path.dirname(kb_dir), exist_ok=True)
                    shutil.copytree(src_kb, kb_dir)
                    print(f"知识库已复制到: {kb_dir}")

        # 初始化RAG系统
        self.rag = MedicalRAG()

        # 设置UI
        self.setup_ui()

        # 绑定关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        """设置界面"""
        # 菜单栏
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="重新加载知识库", command=self.reload_kb)
        file_menu.add_command(label="设置", command=self.open_settings)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.on_closing)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self.show_about)

        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # 聊天历史区域
        self.chat_area = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            font=("微软雅黑", 10),
            state='disabled'
        )
        self.chat_area.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # 配置聊天区域的标签样式
        self.chat_area.tag_config("user", foreground="#667eea", font=("微软雅黑", 10, "bold"))
        self.chat_area.tag_config("assistant", foreground="#2c3e50", font=("微软雅黑", 10))
        self.chat_area.tag_config("time", foreground="#999", font=("微软雅黑", 8))

        # 输入区域框架
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        input_frame.columnconfigure(0, weight=1)

        # 输入框
        self.input_text = scrolledtext.ScrolledText(
            input_frame,
            height=4,
            wrap=tk.WORD,
            font=("微软雅黑", 10)
        )
        self.input_text.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))

        # 按钮框架
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=0, column=1, sticky=(tk.N))

        # 发送按钮
        self.send_btn = ttk.Button(
            button_frame,
            text="发送 (Enter)",
            command=self.send_message
        )
        self.send_btn.pack(pady=(0, 5))

        # 清空按钮
        clear_btn = ttk.Button(
            button_frame,
            text="清空对话",
            command=self.clear_chat
        )
        clear_btn.pack()

        # 绑定快捷键
        self.input_text.bind("<Return>", self.on_enter)
        self.input_text.bind("<Control-Return>", self.on_ctrl_enter)

        # 显示欢迎消息
        self.add_message("系统", "欢迎使用医疗RAG助手！请输入您的医疗问题。", "system")
        self.add_message(
            "助手",
            "您好！我是医疗智能助手，可以为您提供医疗知识查询服务。\n\n"
            "⚠️ 温馨提示：本系统仅供参考，不能替代专业医疗诊断。",
            "assistant"
        )

        # 显示知识库状态
        api_status = "已配置" if self.settings.get("DEEPSEEK_API_KEY") else "未配置"
        doc_count = len(self.rag.documents)
        self.add_message("系统", f"知识库已加载，共 {doc_count} 个知识片段（API: {api_status}）", "system")

    def open_settings(self):
        """打开设置对话框"""
        dialog = SettingsDialog(self.root, self.settings)
        self.root.wait_window(dialog.dialog)

        if dialog.result is not None:
            self.settings = dialog.result
            if save_settings(self.settings):
                # 更新配置模块（使新的 API Key 立即生效，无需重启）
                import config as cfg
                cfg.config.DEEPSEEK_API_KEY = self.settings.get("DEEPSEEK_API_KEY", "")
                cfg.config.DEEPSEEK_BASE_URL = self.settings.get("DEEPSEEK_BASE_URL", "")
                cfg.config.USE_LLM = bool(cfg.config.DEEPSEEK_API_KEY)

                # 重新初始化RAG系统
                self.rag = MedicalRAG()
                api_status = "已配置" if cfg.config.DEEPSEEK_API_KEY else "未配置"
                doc_count = len(self.rag.documents)
                self.add_message("系统", f"设置已保存，知识库已重新加载（API: {api_status}）", "system")
                messagebox.showinfo("成功", "设置已保存！")
            else:
                messagebox.showerror("错误", "设置保存失败")

    def send_message(self):
        """发送消息"""
        question = self.input_text.get("1.0", tk.END).strip()

        if not question:
            return

        # 清空输入框
        self.input_text.delete("1.0", tk.END)

        # 显示用户问题
        self.add_message("用户", question, "user")

        # 在新线程中处理（避免界面卡顿）
        thread = threading.Thread(target=self.get_answer, args=(question,))
        thread.daemon = True
        thread.start()

    def get_answer(self, question):
        """获取回答（在新线程中运行）"""
        try:
            # 显示加载提示
            self.root.after(0, self.add_message, "助手", "正在思考中...", "assistant", True)

            # 获取答案
            answer = self.rag.query(question)

            # 更新回答
            self.root.after(0, self.update_last_message, answer)
        except Exception as e:
            self.root.after(0, self.update_last_message, f"抱歉，出错了：{str(e)}")

    def add_message(self, sender, content, msg_type, temporary=False):
        """添加消息到聊天区域"""
        self.chat_area.configure(state='normal')

        # 添加时间戳
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_area.insert(tk.END, f"[{timestamp}] ", "time")

        # 添加发送者
        if msg_type == "user":
            self.chat_area.insert(tk.END, f"{sender}: ", "user")
        elif msg_type == "assistant":
            self.chat_area.insert(tk.END, f"{sender}: ", "assistant")
        else:
            self.chat_area.insert(tk.END, f"{sender}: ", "time")

        # 添加内容
        tag = "assistant" if msg_type == "assistant" else "user"
        self.chat_area.insert(tk.END, f"{content}\n\n", tag)

        self.chat_area.configure(state='disabled')
        self.chat_area.see(tk.END)

        # 如果是临时消息，返回位置标记
        if temporary:
            return self.chat_area.index(tk.END)

        return None

    def update_last_message(self, new_content):
        """更新最后一条消息"""
        self.chat_area.configure(state='normal')

        # 删除最后一条消息
        last_line = self.chat_area.index("end-2l")
        self.chat_area.delete(last_line, tk.END)

        # 重新添加消息
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_area.insert(tk.END, f"[{timestamp}] ", "time")
        self.chat_area.insert(tk.END, "助手: ", "assistant")
        self.chat_area.insert(tk.END, f"{new_content}\n\n", "assistant")

        self.chat_area.configure(state='disabled')
        self.chat_area.see(tk.END)

    def clear_chat(self):
        """清空聊天记录"""
        self.chat_area.configure(state='normal')
        self.chat_area.delete("1.0", tk.END)
        self.chat_area.configure(state='disabled')
        self.add_message("系统", "对话已清空", "system")

    def reload_kb(self):
        """重新加载知识库"""
        self.rag.load_knowledge_base()
        doc_count = len(self.rag.documents)
        self.add_message("系统", f"知识库已重新加载，共 {doc_count} 个知识片段", "system")
        messagebox.showinfo("成功", f"知识库已重新加载！\n共 {doc_count} 个知识片段")

    def show_about(self):
        """显示关于对话框"""
        about_text = """医疗RAG系统 v1.0

基于TF-IDF和BM25混合检索的医疗问答系统

技术栈：
- 检索：TF-IDF + BM25 + 语义检索
- 分词：jieba
- 界面：Tkinter

⚠️ 免责声明：
本系统仅供参考，不能替代专业医疗诊断。
如身体不适，请及时就医。
"""
        messagebox.showinfo("关于", about_text)

    def on_enter(self, event):
        """回车键发送消息"""
        if not event.state & 0x4:  # 没有按Ctrl
            self.send_message()
            return "break"  # 阻止默认的换行

    def on_ctrl_enter(self, event):
        """Ctrl+Enter换行"""
        self.input_text.insert(tk.INSERT, "\n")
        return "break"

    def on_closing(self):
        """关闭窗口"""
        if messagebox.askokcancel("退出", "确定要退出吗？"):
            self.root.destroy()


# 运行应用
if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalRAGApp(root)
    root.mainloop()
