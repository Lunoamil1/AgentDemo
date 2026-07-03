"""文件处理"""
import os.path  # 导入操作系统路径模块
import hashlib  # 导入哈希库

from langchain_core.documents import Document  # 从langchain_core导入Document类

from logger_handler import logger  # 导入日志处理器
from langchain_community.document_loaders import PyPDFLoader,TextLoader  # 导入文档加载器


def get_file_md5_hex(filepath:str):

    """
    计算文件的MD5值

    参数:
        filepath: 文件路径

    返回:
        文件的MD5哈希值(十六进制字符串)，如果文件不存在或发生错误则返回None
    """
    if not os.path.exists(filepath):  # 检查文件是否存在
        logger.error(f"文件不存在:{filepath}")
        return
    if not os.path.isfile(filepath):  # 检查路径是否为文件
        logger.error(f"路径不是文件:{filepath}")
        return
    md5_obj=hashlib.md5()  # 创建MD5对象
    chunk_size=4096  # 设置读取块大小
    try:
        with open(filepath,"rb") as f:  # 以二进制模式打开文件
            while chunk:=f.read(chunk_size):  # 读取文件块
                md5_obj.update(chunk)  # 更新MD5对象

            md5_hex= md5_obj.hexdigest()  # 获取MD5哈希值
            return md5_hex

    except Exception as e:  # 捕获异常
        logger.error(f"读取文件失败:{filepath},{str(e)}")





def listdir_with_allowed_type(path:str,allowed_types:tuple[str]):
    """
    获取指定路径下允许类型的文件列表
    参数:
        path: 要搜索的目录路径
        allowed_types: 允许的文件类型元组，如('.txt', '.pdf')
    返回:
        包含所有允许类型文件的完整路径元组
    """
    files = []
    if not os.path.isdir(path):  # 检查路径是否为目录
        logger.error(f"[listdir_with_allowed_type]路径不是文件夹:{path}")
        return allowed_types

    for f in os.listdir(path):  # 遍历目录
        if f.endswith(allowed_types):  # 检查文件类型
            files.append(os.path.join(path,f))  # 添加完整路径到列表

    return tuple(files)


def pdf_loader(filepath:str,password=None)->list[Document]:
    """
    加载PDF文件内容
    参数:
        filepath: PDF文件路径
        password: PDF文件密码(如果有)
    返回:
        包含PDF文档内容的Document对象列表
    """
    return PyPDFLoader(filepath,password).load()



def text_loader(filepath:str)->list[Document]:
    """
    加载文本文件内容
    参数:
        filepath: 文本文件路径
    返回:
        包含文本文件内容的Document对象列表
    """
    return TextLoader(filepath,encoding='utf-8').load()
    pass