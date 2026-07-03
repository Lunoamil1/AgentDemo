"""
提示词加载器

统一的提示词加载接口，根据 prompts.yml 中的 key 加载对应文件内容。
"""

from utils.config_handler import load_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

_prompts_conf = load_config("prompts")


def load_prompt(key: str) -> str:
    """
    根据 prompts.yml 中的 key 加载提示词文件内容。

    参数:
        key: prompts.yml 中的配置键名，如 "main_prompt", "rag_summarize_prompt_path"
    返回:
        提示词文本内容
    """
    try:
        relative_path = _prompts_conf[key]
    except KeyError:
        logger.error(f"提示词配置项不存在: {key}")
        raise ValueError(f"提示词配置项不存在: {key}")

    abs_path = get_abs_path(relative_path)

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.debug(f"成功加载提示词: {key} ({abs_path})")
        return content
    except FileNotFoundError:
        logger.error(f"提示词文件不存在: {abs_path}")
        raise
    except Exception as e:
        logger.error(f"读取提示词文件失败: {abs_path}, 错误: {e}")
        raise


# 兼容旧接口
def load_system_prompt():
    return load_prompt("main_prompt")


def load_rag_prompt():
    return load_prompt("rag_summarize_prompt_path")


def load_report_prompt():
    return load_prompt("report_prompt_path")


if __name__ == '__main__':
    print(load_prompt("main_prompt")[:200])
