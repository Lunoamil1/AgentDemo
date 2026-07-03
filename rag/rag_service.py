"""
RAG 检索总结服务

接收用户查询 → 向量检索相关文档 → 组装 context → LLM 总结输出
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from rag.vector_store import VectorStoreService
from model.factory import chat_model
from utils.prompt_loader import load_prompt
from utils.logger_handler import logger


class RagSummarizeService:
    def __init__(self):
        self.vector_db = VectorStoreService()
        self.prompt_text = load_prompt("rag_summarize_prompt_path")
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.retriever = self.vector_db.get_retriever()
        self.model = chat_model
        self.chain = self._init_chain()

    def rag_summarize(self, query: str) -> str:
        """检索并总结"""
        context_docs = self.retriever_docs(query)
        context = self._build_context(context_docs)
        return self.chain.invoke({"input": query, "context": context})

    def retriever_docs(self, query: str) -> list[Document]:
        """检索相关文档"""
        return self.retriever.invoke(query)

    def _build_context(self, docs: list[Document]) -> str:
        """
        组装 context，精简元数据，仅保留对 LLM 有用的信息。
        过滤掉冗余的内部元数据字段（如 needs_semantic_split 等）。
        """
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", doc.metadata.get("source", "未知来源"))
            category = doc.metadata.get("doc_category", "")
            category_tag = f"[{category}]" if category else ""

            # 提取结构化元数据标签
            extra_tags = []
            if doc.metadata.get("question_num"):
                extra_tags.append(f"第{doc.metadata['question_num']}问")
            if doc.metadata.get("fault_title"):
                extra_tags.append(doc.metadata["fault_title"])
            if doc.metadata.get("chapter_title"):
                extra_tags.append(doc.metadata["chapter_title"])

            tag_str = " ".join(extra_tags)
            header = f"参考资料{i}{category_tag}"
            if tag_str:
                header += f" {tag_str}"
            header += f"[来源:{source}]:"

            parts.append(f"{header}\n{doc.page_content}")

        return "\n\n".join(parts)

    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain


if __name__ == '__main__':
    print(RagSummarizeService().rag_summarize("扫拖一体机器人适用什么人群"))
