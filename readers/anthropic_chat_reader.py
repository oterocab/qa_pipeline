from typing import List
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from .base_reader import BaseReader

class AnthropicReader(BaseReader):
    def __init__(self, system_prompt: str, model_config_kargs: dict = None):
        self.llm = ChatAnthropic(**(model_config_kargs or {}))
        self.system_prompt = system_prompt

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question", "system_prompt"],
            template=(
                "System_prompt:\n{system_prompt}\n\n"
                "Context:\n{context}\n\n"
                "Query:\n{question}\n\n"
                "Answer:"
            )
        )

    def build_context(self, documents: List[dict]) -> str:
        formatted_docs = []
        for i, doc in enumerate(documents, start=1):
            doc_id = doc.get("id", f"doc_{i}")
            chunk = doc.get("chunk", "-")
            score = doc.get("rerank_score") or doc.get("score")
            text = doc.get("content", "").strip()
            formatted_docs.append(
                f"Document {i} (ID: {doc_id}, Chunk: {chunk}, Score: {score:.4f}):\n{text}"
            )
        return "\n\n".join(formatted_docs)

    def answer(self, query: str, documents: List[dict]) -> str:
        context = self.build_context(documents)
        prompt = self.prompt_template.format(
            context=context,
            question=query,
            system_prompt=self.system_prompt
        )
        response = self.llm.invoke(prompt)
        return response.content.strip()
