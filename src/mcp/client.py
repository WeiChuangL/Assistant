import json
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from src.agent.tool import Tool, tool_registry


@dataclass
class MCPConnection:
    server_id: int
    server_name: str
    session: object = None  # ClientSession
    _exit_stack: AsyncExitStack | None = None
    tools: list[dict] = field(default_factory=list)
    connected: bool = False


class MCPClientManager:
    """Manages multiple MCP server connections and their tools."""

    def __init__(self):
        self._connections: dict[int, MCPConnection] = {}

    async def connect(self, server_id: int, config: dict) -> list[dict]:
        """Connect to an MCP server and discover its tools."""
        transport = config.get("transport", "stdio")

        if server_id in self._connections:
            await self.disconnect(server_id)

        conn = MCPConnection(server_id=server_id, server_name=config["name"])

        try:
            if transport == "stdio":
                session, exit_stack = await self._connect_stdio(config)
            elif transport in ("http", "streamable_http"):
                session, exit_stack = await self._connect_http(config)
            elif transport == "sse":
                session, exit_stack = await self._connect_sse(config)
            else:
                return []

            # Discover tools
            result = await session.list_tools()
            tools = []
            for t in result.tools:
                tool_name = f"mcp__{config['name']}__{t.name}"
                tools.append({
                    "name": tool_name,
                    "description": t.description or "",
                    "parameters": t.inputSchema if t.inputSchema else {"type": "object", "properties": {}},
                })

            conn.session = session
            conn._exit_stack = exit_stack
            conn.tools = tools
            conn.connected = True
            self._connections[server_id] = conn

            self._register_tools(config["name"], tools)
            return tools

        except Exception:
            # Clean up on failure — must close both session and transport
            await self._cleanup_connection(conn)
            return []

    async def _connect_stdio(self, config: dict) -> tuple[object, AsyncExitStack]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        args = _parse_json_field(config, "args", [])
        env = _parse_json_field(config, "env", {})

        params = StdioServerParameters(
            command=config["command"],
            args=args,
            env=env if env else None,
        )

        exit_stack = AsyncExitStack()
        try:
            transport = await exit_stack.enter_async_context(stdio_client(params))
            read, write = transport  # stdio_client yields (read_stream, write_stream)
            session = await exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            return session, exit_stack
        except Exception:
            await _close_exit_stack(exit_stack)
            raise

    async def _connect_http(self, config: dict) -> tuple[object, AsyncExitStack]:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        headers = _parse_json_field(config, "headers", {})

        exit_stack = AsyncExitStack()
        try:
            transport = await exit_stack.enter_async_context(
                streamablehttp_client(config["url"], headers=headers if headers else None)
            )
            read, write = transport
            session = await exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            return session, exit_stack
        except Exception:
            await _close_exit_stack(exit_stack)
            raise

    async def _connect_sse(self, config: dict) -> tuple[object, AsyncExitStack]:
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        exit_stack = AsyncExitStack()
        try:
            transport = await exit_stack.enter_async_context(sse_client(config["url"]))
            read, write = transport
            session = await exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            return session, exit_stack
        except Exception:
            await _close_exit_stack(exit_stack)
            raise

    async def _cleanup_connection(self, conn: MCPConnection):
        """Properly close both session and transport via the exit stack."""
        if conn._exit_stack:
            await _close_exit_stack(conn._exit_stack)
            conn._exit_stack = None
        conn.session = None
        conn.connected = False

    async def disconnect(self, server_id: int):
        conn = self._connections.pop(server_id, None)
        if conn:
            source = f"mcp:{conn.server_name}"
            tool_registry.unregister_by_source(source)
            await self._cleanup_connection(conn)

    def _register_tools(self, server_name: str, tools: list[dict]):
        source = f"mcp:{server_name}"

        async def _make_executor(server_id: int, tool_name: str):
            async def _execute(**kwargs):
                conn = self._connections.get(server_id)
                if not conn or not conn.session:
                    return f"Error: MCP server '{server_name}' is not connected"
                try:
                    from mcp import types
                    orig_name = tool_name.replace(f"mcp__{server_name}__", "", 1)
                    result = await conn.session.call_tool(orig_name, arguments=kwargs)
                    texts = []
                    for block in result.content:
                        if isinstance(block, types.TextContent):
                            texts.append(block.text)
                        elif hasattr(block, "text"):
                            texts.append(block.text)
                    return "\n".join(texts) if texts else str(result.content)
                except Exception as e:
                    return f"Error: {e}"
            return _execute

        for t in tools:
            server_id = None
            for sid, c in self._connections.items():
                if c.server_name == server_name:
                    server_id = sid
                    break

            exec_fn = _make_executor(server_id or 0, t["name"])
            tool = Tool(
                name=t["name"],
                description=f"[MCP:{server_name}] {t['description']}",
                parameters=t["parameters"],
                execute=exec_fn,
                source=source,
            )
            tool_registry.register(tool)

    async def connect_auto(self, servers: list[dict]):
        """Connect to servers with enabled=True AND auto_connect=True."""
        for s in servers:
            if s.get("enabled", True) and s.get("auto_connect", True):
                await self.connect(s["id"], s)

    async def connect_all_enabled(self, servers: list[dict]):
        """Backward-compatible: connect all enabled servers regardless of auto_connect."""
        for s in servers:
            if s.get("enabled", True):
                await self.connect(s["id"], s)

    def get_connection_status(self, server_id: int) -> str:
        conn = self._connections.get(server_id)
        if not conn:
            return "disconnected"
        return "connected" if conn.connected else "disconnected"

    def get_tool_count(self, server_id: int) -> int:
        conn = self._connections.get(server_id)
        return len(conn.tools) if conn else 0

    def get_server_tools(self, server_id: int) -> list[dict]:
        conn = self._connections.get(server_id)
        return conn.tools if conn else []

    async def call_tool(self, server_id: int, tool_name: str, args: dict) -> str:
        conn = self._connections.get(server_id)
        if not conn or not conn.session:
            return f"Error: server not connected"
        try:
            from mcp import types
            result = await conn.session.call_tool(tool_name, arguments=args)
            texts = []
            for block in result.content:
                if isinstance(block, types.TextContent):
                    texts.append(block.text)
                elif hasattr(block, "text"):
                    texts.append(block.text)
            return "\n".join(texts) if texts else str(result.content)
        except Exception as e:
            return f"Error: {e}"


def _parse_json_field(config: dict, key: str, default):
    val = config.get(key, default)
    if isinstance(val, str):
        try:
            return json.loads(val) if val.strip() else default
        except json.JSONDecodeError:
            return default
    return val if val is not None else default


async def _close_exit_stack(stack: AsyncExitStack):
    """Safely close an AsyncExitStack, swallowing all exceptions."""
    try:
        await stack.aclose()
    except Exception:
        pass


mcp_client = MCPClientManager()
