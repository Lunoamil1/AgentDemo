"""
语义分块器（第二层切割）

基于 embedding 余弦相似度检测语义转折点：
  1. 将文本按句子拆分
  2. 计算相邻句子的 embedding 余弦相似度
  3. 在相似度显著下降处（语义转折点）切割

仅对结构预切割后超长的块启用，避免不必要的 API 调用开销。
"""

import re
from typing import List

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from utils.config_handler import rag_conf
from utils.logger_handler import logger


class SemanticSplitter:
    """基于 embedding 相似度的语义分块器"""

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        semantic_conf = rag_conf.get("semantic", {})
        self.enabled = semantic_conf.get("enabled", True)
        self.breakpoint_threshold = semantic_conf.get("breakpoint_threshold", 0.5)
        self.breakpoint_type = semantic_conf.get("breakpoint_type", "percentile")
        self.max_chunk_length = rag_conf.get("chunk_size", 800) * 2

    def split(self, docs: List[Document]) -> List[Document]:
        """
        对需要语义分块的文档进行切割。
        跳过不需要语义分块的文档（metadata 中 needs_semantic_split=False）。
        """
        if not self.enabled:
            logger.info("[语义分块] 已禁用，跳过")
            return docs

        result = []
        semantic_split_count = 0

        for doc in docs:
            # 跳过不需要语义分块的文档
            if not doc.metadata.get("needs_semantic_split", False):
                result.append(doc)
                continue

            # 文本不够长，无需语义分块
            if len(doc.page_content) <= self.max_chunk_length:
                doc.metadata["needs_semantic_split"] = False
                result.append(doc)
                continue

            # 执行语义分块
            chunks = self._semantic_chunk(doc)
            semantic_split_count += 1
            result.extend(chunks)

        logger.info(f"[语义分块] 处理了 {semantic_split_count} 个超长块，"
                     f"输出 {len(result)} 个块（含未处理的 {len(docs) - semantic_split_count} 个）")
        return result

    def _semantic_chunk(self, doc: Document) -> List[Document]:
        """对单个超长文档执行语义分块"""
        content = doc.page_content
        metadata = doc.metadata.copy()

        # 1. 按句子拆分
        sentences = self._split_sentences(content)
        if len(sentences) <= 1:
            metadata["needs_semantic_split"] = False
            return [Document(page_content=content, metadata=metadata)]

        # 2. 批量计算 embedding
        try:
            embeddings = self.embeddings.embed_documents(sentences)
        except Exception as e:
            logger.warning(f"[语义分块] embedding 计算失败，保留原文: {e}")
            metadata["needs_semantic_split"] = False
            return [Document(page_content=content, metadata=metadata)]

        # 3. 计算相邻句子间的余弦相似度
        similarities = self._compute_similarities(embeddings)

        # 4. 根据断点策略确定切割位置
        breakpoints = self._find_breakpoints(similarities)

        # 5. 在断点处合并句子为块
        chunks = self._merge_sentences(sentences, breakpoints, metadata)

        logger.info(f"[语义分块] 1 个超长文档 → {len(chunks)} 个语义块 "
                     f"(句子数={len(sentences)}, 断点数={len(breakpoints)})")
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        中文友好的句子拆分。
        按中文句号、问号、感叹号等断句，保留分隔符。
        """
        # 按中文标点断句，保留标点
        parts = re.split(r'(?<=[。！？；\n])', text)
        # 过滤空句，去除首尾空白
        sentences = [s.strip() for s in parts if s.strip()]
        return sentences

    def _compute_similarities(self, embeddings: List[List[float]]) -> List[float]:
        """计算相邻 embedding 间的余弦相似度"""
        similarities = []
        for i in range(len(embeddings) - 1):
            a = np.array(embeddings[i])
            b = np.array(embeddings[i + 1])
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                sim = 0.0
            else:
                sim = float(np.dot(a, b) / (norm_a * norm_b))
            similarities.append(sim)
        return similarities

    def _find_breakpoints(self, similarities: List[float]) -> List[int]:
        """
        根据相似度序列找到语义转折点索引。

        支持三种断点策略：
          - percentile: 相似度低于百分位阈值处切割
          - standard_deviation: 相似度低于均值-N倍标准差处切割
          - interquartile: 相似度低于Q1-1.5*IQR处切割
        """
        if not similarities:
            return []

        sim_array = np.array(similarities)

        if self.breakpoint_type == "percentile":
            # 低于指定百分位数的相似度视为断点
            threshold = np.percentile(sim_array, self.breakpoint_threshold * 100)
            breakpoints = [i for i, sim in enumerate(similarities) if sim < threshold]

        elif self.breakpoint_type == "standard_deviation":
            mean = np.mean(sim_array)
            std = np.std(sim_array)
            threshold = mean - self.breakpoint_threshold * std
            breakpoints = [i for i, sim in enumerate(similarities) if sim < threshold]

        elif self.breakpoint_type == "interquartile":
            q1 = np.percentile(sim_array, 25)
            q3 = np.percentile(sim_array, 75)
            iqr = q3 - q1
            threshold = q1 - 1.5 * iqr
            breakpoints = [i for i, sim in enumerate(similarities) if sim < threshold]

        else:
            logger.warning(f"[语义分块] 未知断点类型: {self.breakpoint_type}，使用 percentile")
            threshold = np.percentile(sim_array, self.breakpoint_threshold * 100)
            breakpoints = [i for i, sim in enumerate(similarities) if sim < threshold]

        return breakpoints

    def _merge_sentences(self, sentences: List[str], breakpoints: List[int],
                         base_metadata: dict) -> List[Document]:
        """
        在断点处合并句子为文档块。
        断点索引 i 表示在第 i 句和第 i+1 句之间切割。
        """
        if not breakpoints:
            base_metadata["needs_semantic_split"] = False
            return [Document(page_content="".join(sentences), metadata=base_metadata)]

        chunks = []
        start = 0
        # 在每个断点后切割
        cut_points = sorted(set(breakpoints))
        for bp in cut_points:
            chunk_text = "".join(sentences[start:bp + 1])
            chunk_metadata = base_metadata.copy()
            chunk_metadata["needs_semantic_split"] = False
            chunk_metadata["split_method"] = "semantic"
            chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))
            start = bp + 1

        # 最后一个块
        if start < len(sentences):
            chunk_text = "".join(sentences[start:])
            chunk_metadata = base_metadata.copy()
            chunk_metadata["needs_semantic_split"] = False
            chunk_metadata["split_method"] = "semantic"
            chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        return chunks
