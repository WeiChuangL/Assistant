from collections.abc import AsyncIterator

from src.agent.prompts import SYSTEM_PROMPT
from src.config import settings
from src.knowledge.retrieval import RetrievalResult, search_chunks
from src.llm.client import ChatMessage, llm_client
from src.memory.long_term import MemoryEntry, extract_and_store_memories, search_memories
from src.memory.profile import extract_and_update_profile, get_profile
from src.memory.short_term import ShortTermMemory


class Agent:
    def __init__(self):
        self.short_term = ShortTermMemory()

    async def chat_stream(self, user_input: str) -> AsyncIterator[str]:
        """Process user input and stream the response."""
        # Step 1: Generate embedding and run parallel retrievals
        query_embedding, chunks, memories, profile = await self._parallel_retrieve(user_input)

        # Step 2: Build context
        system_msg = self._build_system_prompt(chunks, memories, profile)

        # Step 3: Build messages for LLM
        messages = [ChatMessage(role="system", content=system_msg)]
        for m in self.short_term.get_all():
            messages.append(ChatMessage(role=m["role"], content=m["content"]))
        messages.append(ChatMessage(role="user", content=user_input))

        # Step 4: Stream response
        full_response = ""
        async for token in llm_client.chat_stream(messages=messages):
            full_response += token
            yield token

        # Step 5: Update memory
        self.short_term.add("user", user_input)
        self.short_term.add("assistant", full_response)

        # Step 6: Async post-processing (don't block user)
        await self._post_process(user_input, full_response)

    async def _parallel_retrieve(self, query: str):
        """Run retrieval operations in parallel."""
        chunks_task = search_chunks(query, top_k=settings.agent_top_k_chunks)
        memories_task = search_memories(query, top_k=settings.agent_top_k_memories)
        profile_task = get_profile()

        chunks = await chunks_task
        memories = await memories_task
        profile = await profile_task

        return None, chunks, memories, profile  # query_embedding not needed separately

    def _build_system_prompt(
        self,
        chunks: list[RetrievalResult],
        memories: list[MemoryEntry],
        profile: dict[str, str],
    ) -> str:
        profile_str = "\n".join(f"- {k}: {v}" for k, v in profile.items()) if profile else "暂无用户画像"

        if memories:
            memory_str = "\n".join(
                f"- [{m.memory_type}] {m.summary or m.content} (重要度: {m.importance})"
                for m in memories
            )
        else:
            memory_str = "暂无相关长期记忆"

        if chunks:
            chunk_str = "\n\n---\n\n".join(
                f"[来源: {c.filename}] (相关度: {c.score:.2f})\n{c.content}" for c in chunks
            )
        else:
            chunk_str = "暂无相关知识库内容"

        convo = ""
        if self.short_term:
            convo = "## 最近对话\n" + "\n".join(
                f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:200]}"
                for m in self.short_term.get_last_n(6)
            )

        return SYSTEM_PROMPT.format(
            user_profile=profile_str,
            long_term_memory=memory_str,
            retrieved_chunks=chunk_str,
            conversation_history=convo,
        )

    async def _post_process(self, user_input: str, assistant_response: str):
        """Extract and store memories and profile updates."""
        convo = f"用户: {user_input}\n助手: {assistant_response}"
        await extract_and_store_memories(convo)
        await extract_and_update_profile(convo)

    def clear_memory(self):
        self.short_term.clear()


agent = Agent()
