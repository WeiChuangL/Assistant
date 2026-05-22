from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    execute: Callable[..., Any]
    source: str = "builtin"  # "builtin" | "mcp:<server_name>"

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def unregister(self, name: str):
        self._tools.pop(name, None)

    def unregister_by_source(self, source: str):
        """Remove all tools from a given source (e.g. 'mcp:filesystem')."""
        to_remove = [n for n, t in self._tools.items() if t.source == source]
        for name in to_remove:
            del self._tools[name]

    def register_batch(self, tools: list[Tool]):
        for t in tools:
            self._tools[t.name] = t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_all_for_llm(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def list_all(self) -> list[dict]:
        """Return tool metadata for UI display."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "source": t.source,
            }
            for t in self._tools.values()
        ]

    async def execute(self, name: str, arguments: dict) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"Error: tool '{name}' not found"
        try:
            result = await tool.execute(**arguments)
            return str(result)
        except Exception as e:
            return f"Error executing {name}: {e}"

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __bool__(self) -> bool:
        return len(self._tools) > 0


tool_registry = ToolRegistry()


def register_tool(name: str, description: str, parameters: dict, source: str = "builtin"):
    """Decorator to register a function as a tool."""
    def decorator(func: Callable):
        tool = Tool(name=name, description=description, parameters=parameters, execute=func, source=source)
        tool_registry.register(tool)
        return func
    return decorator
