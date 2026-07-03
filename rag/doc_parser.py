"""
结构感知文档预切割器

根据文档内容自动识别结构模式（问答对/章节/故障条目），
在语义边界处切割，保证每个文档块的语义完整性。

三层切割策略：
  第一层：结构感知预切割（本模块） — 按正则识别的结构边界切割
  第二层：语义分块 — 对超长块用 embedding 相似度检测语义转折点
  第三层：字符级兜底 — RecursiveCharacterTextSplitter 兜底切割
"""

import re
from typing import List, Optional

from langchain_core.documents import Document
from utils.config_handler import rag_conf
from utils.logger_handler import logger


class StructureParser:
    """基于正则的结构感知文档预切割器"""

    def __init__(self):
        parser_conf = rag_conf.get("parser", {})
        self.qa_pattern = parser_conf.get("qa_pattern", r"(?=\n\d+\.\s)")
        self.chapter_pattern = parser_conf.get("chapter_pattern", r"(?=\n[一二三四五六七八九十]+、)")
        self.fault_pattern = parser_conf.get("fault_pattern", r"(?=\n\d+\.\s.+?\n故障现象)")
        # 结构预切割后单个块的最大长度，超过此值将交由语义分块处理
        self.max_chunk_length = rag_conf.get("chunk_size", 800) * 2

    def parse(self, docs: List[Document]) -> List[Document]:
        """
        对文档列表进行结构感知预切割。
        自动检测每个文档的结构类型并选择对应切割策略。
        """
        result = []
        for doc in docs:
            content = doc.page_content
            metadata = doc.metadata.copy()
            source = metadata.get("source", "")

            # 自动检测文档结构类型
            structure_type = self._detect_structure(content, source)
            logger.info(f"[结构检测] 文件: {source}, 类型: {structure_type}")

            # 按结构类型切割
            if structure_type == "qa":
                chunks = self._split_qa(content, metadata)
            elif structure_type == "fault":
                chunks = self._split_fault(content, metadata)
            elif structure_type == "chapter":
                chunks = self._split_chapter(content, metadata)
            else:
                # 无明确结构，原样保留，交由后续语义分块处理
                chunks = [Document(page_content=content, metadata=metadata)]

            result.extend(chunks)

        logger.info(f"[结构预切割] 输入 {len(docs)} 个文档，输出 {len(result)} 个块")
        return result

    def _detect_structure(self, content: str, source: str = "") -> str:
        """
        自动检测文档结构类型。

        优先级：故障条目 > 问答对 > 章节 > 无结构
        """
        # 检测故障条目结构：包含 "故障现象" "排查原因" "解决方法" 关键词
        fault_keywords = ["故障现象", "排查原因", "解决方法"]
        fault_count = sum(1 for kw in fault_keywords if kw in content)
        if fault_count >= 2:
            return "fault"

        # 检测问答对结构：包含 "数字. 问题" + "答：" 模式
        qa_matches = re.findall(r"\n\d+\.\s", content)
        answer_matches = re.findall(r"答[：:]", content)
        if len(qa_matches) >= 3 and len(answer_matches) >= 3:
            return "qa"

        # 检测章节结构：包含 "一、" "二、" 等中文数字章节
        chapter_matches = re.findall(r"[一二三四五六七八九十]+、", content)
        if len(chapter_matches) >= 2:
            return "chapter"

        return "plain"

    def _split_qa(self, content: str, metadata: dict) -> List[Document]:
        """
        按问答对切割。
        每个 Q&A 对为一个独立块，保留完整的问题和答案。
        """
        # 按 "数字. 问题" 模式切割
        parts = re.split(self.qa_pattern, content)
        # 过滤空片段
        parts = [p.strip() for p in parts if p.strip()]

        chunks = []
        for i, part in enumerate(parts):
            # 提取问题编号
            q_num_match = re.match(r"(\d+)\.\s", part)
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_type"] = "qa"
            chunk_metadata["chunk_index"] = i + 1
            if q_num_match:
                chunk_metadata["question_num"] = int(q_num_match.group(1))

            # 如果单个问答对超长，不再拆分（交由语义分块处理）
            chunk_metadata["needs_semantic_split"] = len(part) > self.max_chunk_length

            chunks.append(Document(page_content=part, metadata=chunk_metadata))

        return chunks

    def _split_fault(self, content: str, metadata: dict) -> List[Document]:
        """
        按故障条目切割。
        每个故障条目（故障现象+排查原因+解决方法）为一个独立块。
        """
        parts = re.split(self.fault_pattern, content)
        parts = [p.strip() for p in parts if p.strip()]

        # 如果第一条是前言/说明（不包含"故障现象"），作为单独块保留
        chunks = []
        for i, part in enumerate(parts):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_type"] = "fault"

            # 提取故障编号
            fault_num_match = re.match(r"(\d+)\.\s", part)
            if fault_num_match:
                chunk_metadata["fault_num"] = int(fault_num_match.group(1))
                # 提取故障标题
                title_match = re.match(r"\d+\.\s+(.+?)(?:\n|$)", part)
                if title_match:
                    chunk_metadata["fault_title"] = title_match.group(1).strip()

            chunk_metadata["chunk_index"] = i + 1
            chunk_metadata["needs_semantic_split"] = len(part) > self.max_chunk_length

            chunks.append(Document(page_content=part, metadata=chunk_metadata))

        return chunks

    def _split_chapter(self, content: str, metadata: dict) -> List[Document]:
        """
        按章节切割。
        每个"一、二、三"章节为一个独立块，章节内部子项保持完整。
        """
        parts = re.split(self.chapter_pattern, content)
        parts = [p.strip() for p in parts if p.strip()]

        chunks = []
        chapter_num = 0
        cn_nums = "一二三四五六七八九十"

        for i, part in enumerate(parts):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_type"] = "chapter"

            # 提取章节标题
            title_match = re.match(r"([一二三四五六七八九十]+)、\s*(.+?)(?:\n|$)", part)
            if title_match:
                chapter_num = cn_nums.find(title_match.group(1)) + 1
                chunk_metadata["chapter_title"] = title_match.group(2).strip()
                chunk_metadata["chapter_num"] = chapter_num

            chunk_metadata["chunk_index"] = i + 1
            chunk_metadata["needs_semantic_split"] = len(part) > self.max_chunk_length

            chunks.append(Document(page_content=part, metadata=chunk_metadata))

        return chunks
