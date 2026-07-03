"""
日志工具模块
提供日志记录功能，支持控制台和文件输出
"""
import logging  # 导入Python标准日志模块
import os  # 导入操作系统接口模块

from utils.path_tool import get_abs_path  # 从自定义路径工具模块获取绝对路径函数
from datetime import datetime  # 导入日期时间处理模块



# 设置日志根目录
LOG_ROOT = get_abs_path('logs')  # 获取logs目录的绝对路径



# 确保日志目录存在
os.makedirs(LOG_ROOT, exist_ok=True)  # 创建日志目录，如果已存在则不报错

# 默认日志格式
DEFAULT_LOG_FORMAT =logging.Formatter (
    '%(asctime)s-%(name)s-%(levelname)s-%(filename)s-%(lineno)d-%(message)s'
)  # 定义日志输出格式，包含时间、级别、文件名、行号和消息


def get_logger(
        name: str = "agent",  # 日志器名称，默认为"agent"
        console_level: int = logging.INFO,  # 控制台日志级别，默认为INFO
        file_level: int = logging.DEBUG,  # 文件日志级别，默认为DEBUG
        log_file= None,  # 日志文件路径，默认为None

)->logging.Logger:
    """
    获取日志记录器
    参数:
        name: 日志器名称
        console_level: 控制台日志级别
        file_level: 文件日志级别
        log_file: 日志文件路径
    返回:
        配置好的日志记录器实例
    """
    # 创建或获取指定名称的日志器
    logger=logging.getLogger(name)
    # 设置日志器最低级别
    logger.setLevel(logging.DEBUG)


    # 如果日志器已有处理器，直接返回，避免重复添加
    if logger.handlers:
        return logger



    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    # 设置控制台处理器日志级别
    console_handler.setLevel(console_level)
    # 设置控制台处理器日志格式
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    # 将控制台处理器添加到日志器
    logger.addHandler(console_handler)


    # 如果未指定日志文件，则使用默认命名规则
    if not log_file:
        # 生成日志文件名，包含日志器名称和当前日期
        log_file= os.path.join(LOG_ROOT, f'{name}_{datetime.now().strftime('%Y%m%d')}.log')

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file,encoding='utf-8')
    # 设置文件处理器日志级别
    file_handler.setLevel(file_level)
    # 设置文件处理器日志格式
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    # 将文件处理器添加到日志器
    logger.addHandler(file_handler)


    # 返回配置好的日志器
    return logger


# 快捷获取日志管理器实例
logger = get_logger()

# 如果作为主程序运行，则测试日志功能
if __name__ == '__main__':

    # 测试不同级别的日志输出
    logger.info('test')  # 输出INFO级别日志
    logger.error('cuowu')  # 输出ERROR级别日志
    logger.debug('warn')  # 输出DEBUG级别日志
    logger.warning('debug')  # 输出WARNING级别日志