from collections import deque

from src.config import settings


class ShortTermMemory:
    """Sliding window conversation memory."""

    def __init__(self, max_size: int | None = None):
        self.max_size = max_size or settings.agent_short_term_size
        self._messages: deque[dict[str, str]] = deque(maxlen=self.max_size)

    def add(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})

    def get_all(self) -> list[dict[str, str]]:
        return list(self._messages)

    def get_last_n(self, n: int) -> list[dict[str, str]]:
        return list(self._messages)[-n:]

    def clear(self):
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def __bool__(self) -> bool:
        return len(self._messages) > 0
