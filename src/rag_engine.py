from typing import Dict, List

from openai import OpenAI

from .config import Config
from .embedder import Embedder
from .milvus_store import MilvusStore


class RagEngine:
    def __init__(self, config: Config, milvus_store: MilvusStore, embedder: Embedder) -> None:
        self.config = config
        self.milvus_store = milvus_store
        self.embedder = embedder
        self.client = OpenAI(
            api_key=self.config.openai_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model_name = self.config.openrouter_model

    def _build_prompt(self, context: str, question: str) -> Dict[str, str]:
        system_prompt = """
Human: You are an AI assistant. You can find the answer to the question from the provided context paragraphs.
"""
        user_prompt = f"""
Please use the information fragments enclosed in <context> tags below to answer the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""
        return {"system": system_prompt, "user": user_prompt}

    def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    def query(self, question: str) -> str:
        query_embedding = self.embedder.encode_queries([question])[0]
        matches = self.milvus_store.search(
            query_embedding=query_embedding,
            limit=self.config.max_retrievals,
            metric_type=self.config.metric_type,
        )

        if not matches:
            raise RuntimeError("No similar content found in Milvus.")

        context = "\n".join(match["text"] for match in matches)
        prompts = self._build_prompt(context=context, question=question)
        return self._generate_response(
            system_prompt=prompts["system"], user_prompt=prompts["user"]
        )
