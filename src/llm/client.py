from collections.abc import AsyncIterator
from dataclasses import dataclass

from openai import AsyncOpenAI

from src.config import settings
from src.llm.models import EMBEDDING_DIMENSIONS


@dataclass
class ChatMessage:
    role: str
    content: str


class LLMClient:
    def __init__(self):
        self._client = AsyncOpenAI(
            base_url=settings.nvidia_base_url,
            api_key=settings.nvidia_api_key,
        )

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIMENSIONS.get(settings.llm_embedding_model, 1024)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        response = await self._client.embeddings.create(
            model=settings.llm_embedding_model,
            input=texts,
            extra_body={"input_type": "query", "truncate": "END"},
        )
        return [item.embedding for item in response.data]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for documents (uses passage input_type)."""
        response = await self._client.embeddings.create(
            model=settings.llm_embedding_model,
            input=texts,
            extra_body={"input_type": "passage", "truncate": "END"},
        )
        return [item.embedding for item in response.data]

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 8192,
    ) -> AsyncIterator[str]:
        """Stream a chat completion, yielding content deltas."""
        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        stream = await self._client.chat.completions.create(
            model=settings.llm_chat_model,
            messages=api_messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
            stream=True,
        )

        async for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
            if reasoning:
                yield f"[思考] {reasoning}"
            content = getattr(delta, "content", None)
            if content:
                yield content

    async def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 8192,
    ) -> str:
        """Non-streaming chat completion."""
        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        response = await self._client.chat.completions.create(
            model=settings.llm_chat_model,
            messages=api_messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


llm_client = LLMClient()
