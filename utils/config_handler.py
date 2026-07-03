"""
配置管理器

统一的 YAML 配置加载接口，支持按名称加载对应配置文件。
使用 yaml.safe_load 替代 yaml.load，防止任意代码执行。
"""

import yaml

from utils.path_tool import get_abs_path
from utils.logger_handler import logger

# 配置文件名称到路径的映射
_CONFIG_MAP = {
    "rag": "config/rag.yml",
    "faiss": "config/faiss.yml",
    "prompts": "config/prompts.yml",
    "agent": "config/agent.yml",
}

# 缓存已加载的配置，避免重复读取文件
_config_cache: dict[str, dict] = {}


def load_config(name: str) -> dict:
    """
    根据配置名称加载 YAML 配置。

    参数:
        name: 配置名称，支持 "rag" / "faiss" / "prompts" / "agent"
              也可传入完整路径作为 fallback
    返回:
        解析后的字典
    """
    # 优先从缓存读取
    if name in _config_cache:
        return _config_cache[name]

    # 查找配置路径
    config_path = _CONFIG_MAP.get(name, name)
    abs_path = get_abs_path(config_path)

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"配置文件不存在: {abs_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"配置文件解析失败: {abs_path}, 错误: {e}")
        raise

    # 缓存
    _config_cache[name] = conf or {}
    return _config_cache[name]


# 快捷访问（兼容旧代码）
rag_conf = load_config("rag")
faiss_conf = load_config("faiss")
prompts_conf = load_config("prompts")
agent_conf = load_config("agent")


if __name__ == "__main__":
    print(rag_conf.get("chunk_size"))
