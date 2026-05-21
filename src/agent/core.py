import json
from collections.abc import AsyncIterator

from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tool import tool_registry
from src.config import settings
from src.knowledge.retrieval import RetrievalResult, search_chunks
from src.llm.client import ChatMessage, llm_client
from src.memory.long_term import MemoryEntry, extract_and_store_memories, search_memories
from src.memory.profile import extract_and_update_profile, get_profile
from src.memory.short_term import ShortTermMemory

MAX_REACT_LOOPS = 5

_skills_loaded = False


def _init_skills():
    global _skills_loaded
    if _skills_loaded:
        return
    from src.skill import load_skills
    load_skills()
    _skills_loaded = True


class Agent:
    def __init__(self, session_id: int | None = None):
        self.short_term = ShortTermMemory()
        self.session_id = session_id
        # Ensure tools are imported and registered
        import src.agent.tools  # noqa: F401
        _init_skills()

    async def chat_stream(self, user_input: str) -> AsyncIterator[str]:
        """Process user input and stream the response."""
        # Step 1-2: Parallel retrieval + build context
        chunks, memories, profile = await self._parallel_retrieve(user_input)
        system_msg = self._build_system_prompt(chunks, memories, profile)

        # Step 3: ReAct loop with tools
        messages = [ChatMessage(role="system", content=system_msg)]
        for m in self.short_term.get_all():
            messages.append(ChatMessage(role=m["role"], content=m["content"]))
        messages.append(ChatMessage(role="user", content=user_input))

        full_response = ""
        has_tools = bool(tool_registry)

        if has_tools:
            # ReAct loop: LLM decides to call tools or answer directly
            for _ in range(MAX_REACT_LOOPS):
                content, tool_calls = await llm_client.chat_with_tools(
                    messages=messages,
                    tools=tool_registry.get_all_for_llm(),
                )

                if tool_calls:
                    # Add assistant's tool_calls message
                    messages.append(ChatMessage(
                        role="assistant",
                        content="",
                        tool_calls=[
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"],
                                },
                            }
                            for tc in tool_calls
                        ],
                    ))
                    # Execute tools and add results
                    for tc in tool_calls:
                        try:
                            args = json.loads(tc["arguments"])
                        except json.JSONDecodeError:
                            args = {}
                        result = await tool_registry.execute(tc["name"], args)
                        messages.append(ChatMessage(
                            role="tool",
                            content=result,
                            tool_call_id=tc["id"],
                        ))
                    continue
                else:
                    # LLM provided a final text response — yield it
                    final_content = content or ""
                    if final_content:
                        full_response = final_content
                        yield final_content
                    break
            else:
                # Max loops reached — force a summary
                messages.append(ChatMessage(
                    role="user",
                    content="请基于已获取的工具结果，用中文给出简洁的回答。"
                ))
                content, _ = await llm_client.chat_with_tools(
                    messages=messages, tools=tool_registry.get_all_for_llm()
                )
                final = content or "抱歉，暂时无法回答这个问题。"
                full_response = final
                yield final
        else:
            # No tools — classic RAG streaming
            async for token in llm_client.chat_stream(messages=messages):
                full_response += token
                yield token

        # Step 5: Update memory
        self.short_term.add("user", user_input)
        self.short_term.add("assistant", full_response)
        await self._post_process(user_input, full_response)

    async def _parallel_retrieve(self, query: str):
        chunks_task = search_chunks(query, top_k=settings.agent_top_k_chunks)
        memories_task = search_memories(query, top_k=settings.agent_top_k_memories, session_id=self.session_id)
        profile_task = get_profile()

        chunks = await chunks_task
        memories = await memories_task
        profile = await profile_task

        return chunks, memories, profile

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

        tools_section = ""
        if tool_registry:
            tools_list = tool_registry.list_all()
            builtin = [t for t in tools_list if t["source"] == "builtin"]
            mcp_tools = [t for t in tools_list if t["source"].startswith("mcp:")]
            parts = []
            if builtin:
                names = ", ".join(t["name"] for t in builtin)
                parts.append(f"内置工具: {names}")
            if mcp_tools:
                names = ", ".join(t["name"] for t in mcp_tools)
                parts.append(f"MCP工具: {names}")
            if parts:
                tools_section = "\n## 可用工具\n" + "\n".join(parts) + "\n在需要实时数据时主动调用工具。"

        # Append enabled skill prompts
        from src.skill.registry import skill_registry
        skill_prompt = skill_registry.get_prompt_augmentation()

        return SYSTEM_PROMPT.format(
            user_profile=profile_str,
            long_term_memory=memory_str,
            retrieved_chunks=chunk_str,
            conversation_history=convo,
            tools_section=tools_section,
        ) + ("\n\n" + skill_prompt if skill_prompt else "")

    async def _post_process(self, user_input: str, assistant_response: str):
        convo = f"用户: {user_input}\n助手: {assistant_response}"
        await extract_and_store_memories(convo, session_id=self.session_id)
        await extract_and_update_profile(convo)
        if self.session_id is not None:
            from src.session.manager import touch_session
            await touch_session(self.session_id)

    def clear_memory(self):
        self.short_term.clear()



