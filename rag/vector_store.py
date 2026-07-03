"""
向量存储服务

集成三层文档切割策略：
  第一层：结构感知预切割（StructureParser） — 按正则识别的文档结构边界切割
  第二层：语义分块（SemanticSplitter） — 对超长块用 embedding 相似度检测语义转折点
  第三层：字符级兜底（RecursiveCharacterTextSplitter） — 兜底切割仍超长的块
"""

import os
from pathlib import Path
import hashlib

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.doc_parser import StructureParser
from rag.semantic_splitter import SemanticSplitter
from utils.config_handler import rag_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


class VectorStoreService():
    def __init__(self):
        db_dir = get_abs_path(rag_conf["persist_directory"])
        os.makedirs(db_dir, exist_ok=True)

        self.embedding = DashScopeEmbeddings(model=rag_conf["embedding_model_name"])
        self.vector_db = None

        # 初始化三层切割器
        self.structure_parser = StructureParser()
        self.semantic_splitter = SemanticSplitter(self.embedding)
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=rag_conf["chunk_size"],
            chunk_overlap=rag_conf["chunk_overlap"],
            separators=rag_conf["separators"]
        )

        # 加载或创建向量库
        index_path = Path(db_dir) / "index.faiss"
        if index_path.exists():
            self.vector_db = FAISS.load_local(
                folder_path=db_dir,
                embeddings=self.embedding,
                allow_dangerous_deserialization=True
            )
            logger.info(f"已加载向量库: {db_dir}")
        else:
            temp_text = ["init empty vector store"]
            self.vector_db = FAISS.from_texts(texts=temp_text, embedding=self.embedding)
            self.vector_db.index.reset()
            self.vector_db.save_local(db_dir)
            logger.info("空向量库已持久化到faiss_db")

    def load_document(self):
        """加载文档并执行三层切割后存入向量库"""

        # --- MD5 去重（一次性加载到 set，O(1) 查找） ---
        md5_path = get_abs_path(rag_conf["md5_hex_store"])
        existing_md5s = self._load_md5_set(md5_path)

        # --- 遍历数据目录，加载新文档 ---
        data_dir = get_abs_path(rag_conf["data_path"])
        if not os.path.exists(data_dir):
            logger.error(f"数据目录不存在: {data_dir}")
            return

        all_docs = []
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if not os.path.isfile(file_path):
                continue

            # 计算文件 MD5（分块读取，避免大文件 OOM）
            md5_str = self._compute_file_md5(file_path)
            if md5_str is None:
                continue

            # MD5 去重
            if md5_str in existing_md5s:
                logger.info(f"文件已存在，跳过: {filename}")
                continue

            # 加载文档
            docs = self._load_file(file_path)
            if not docs:
                continue

            # 增强元数据：添加文件名和文档类别
            for doc in docs:
                doc.metadata["filename"] = filename
                doc.metadata["doc_category"] = self._infer_category(filename)

            all_docs.extend(docs)
            existing_md5s.add(md5_str)
            self._append_md5(md5_path, md5_str)
            logger.info(f"成功加载文件: {filename}")

        if not all_docs:
            logger.info("没有新文档需要处理")
            return

        # --- 三层切割 ---
        logger.info(f"开始三层切割，原始文档数: {len(all_docs)}")

        # 第一层：结构感知预切割
        layer1_docs = self.structure_parser.parse(all_docs)
        logger.info(f"[第一层-结构切割] {len(all_docs)} → {len(layer1_docs)} 个块")

        # 第二层：语义分块（仅处理超长块）
        layer2_docs = self.semantic_splitter.split(layer1_docs)
        logger.info(f"[第二层-语义分块] {len(layer1_docs)} → {len(layer2_docs)} 个块")

        # 第三层：字符级兜底（仅处理仍超长的块）
        layer3_docs = self._char_split_fallback(layer2_docs)
        logger.info(f"[第三层-字符兜底] {len(layer2_docs)} → {len(layer3_docs)} 个块")

        # 清理临时 metadata 字段
        for doc in layer3_docs:
            doc.metadata.pop("needs_semantic_split", None)

        # --- 存入向量库 ---
        try:
            self.vector_db.add_documents(layer3_docs)
            self.vector_db.save_local(get_abs_path(rag_conf["persist_directory"]))
            logger.info(f"成功存入 {len(layer3_docs)} 个文档块到向量库")
        except Exception as e:
            logger.error(f"文档向量化存储失败: {e}")

    def get_retriever(self):
        """获取检索器，支持配置 search_type 和 search_kwargs"""
        retriever_conf = rag_conf.get("retriever", {})
        search_type = retriever_conf.get("search_type", "similarity")
        search_kwargs = retriever_conf.get("search_kwargs", {"k": 4})
        return self.vector_db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

    # ===== 私有方法 =====

    def _load_md5_set(self, md5_path: str) -> set:
        """一次性加载 MD5 集合到 set，O(1) 查找"""
        if not os.path.exists(md5_path):
            return set()
        with open(md5_path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())

    def _append_md5(self, md5_path: str, md5_str: str):
        """追加 MD5 到文件"""
        with open(md5_path, "a", encoding="utf-8") as f:
            f.write(md5_str + "\n")

    @staticmethod
    def _compute_file_md5(file_path: str, chunk_size: int = 8192) -> str | None:
        """分块计算文件 MD5，避免大文件 OOM"""
        md5_obj = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(chunk_size):
                    md5_obj.update(chunk)
            return md5_obj.hexdigest()
        except Exception as e:
            logger.error(f"计算MD5失败: {file_path}, 错误: {e}")
            return None

    @staticmethod
    def _load_file(file_path: str) -> list:
        """根据文件扩展名选择对应的 Loader 加载文档"""
        ext = os.path.splitext(file_path)[1].lower()
        allow_extensions = rag_conf.get("allow_knowledge_file", [])
        if ext.lstrip(".") not in allow_extensions:
            logger.warning(f"不支持的文件类型: {file_path}")
            return []
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                logger.warning(f"未实现的文件加载器: {ext}")
                return []
            return loader.load()
        except Exception as e:
            logger.error(f"加载文件失败: {file_path}, 错误: {e}")
            return []

    @staticmethod
    def _infer_category(filename: str) -> str:
        """根据文件名推断文档类别"""
        name = filename.lower()
        if "保养" in name or "维护" in name:
            return "维护保养"
        elif "故障" in name or "排除" in name or "排查" in name:
            return "故障排除"
        elif "选购" in name or "购机" in name:
            return "选购指南"
        elif "100问" in name or "问答" in name:
            return "知识问答"
        else:
            return "其他"

    def _char_split_fallback(self, docs: list) -> list:
        """
        第三层兜底：对仍然超长的块用 RecursiveCharacterTextSplitter 切割。
        仅切割超长的块，短块直接保留。
        """
        max_len = rag_conf["chunk_size"] * 2
        needs_split = []
        keep_as_is = []

        for doc in docs:
            if len(doc.page_content) > max_len:
                needs_split.append(doc)
            else:
                keep_as_is.append(doc)

        if not needs_split:
            return docs

        # 批量切割超长块
        split_result = self.char_splitter.split_documents(needs_split)

        # 为兜底切割的块添加元数据
        for doc in split_result:
            doc.metadata["split_method"] = "char_fallback"

        logger.info(f"[字符兜底] {len(needs_split)} 个超长块 → {len(split_result)} 个块")
        return keep_as_is + split_result


if __name__ == "__main__":
    vector_store = VectorStoreService()
    vector_store.load_document()
