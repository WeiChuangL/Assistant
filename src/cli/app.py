from rich.console import Console

from src.agent.core import Agent
from src.cli.commands import handle_slash_command
from src.session.manager import ensure_default_session, get_session

console = Console()


class CLISession:
    def __init__(self):
        self.current_session_id: int | None = None
        self.agent: Agent | None = None

    async def init(self):
        session = await ensure_default_session()
        self.current_session_id = session.id
        self.agent = Agent(session_id=session.id)

    async def switch(self, session_id: int):
        session = await get_session(session_id)
        if not session:
            console.print(f"[red]会话 {session_id} 不存在[/]")
            return False
        self.current_session_id = session_id
        self.agent = Agent(session_id=session_id)
        console.print(f"[green]已切换到会话: {session.title}[/]")
        return True

    async def get_title(self) -> str:
        if self.current_session_id is None:
            return "无会话"
        session = await get_session(self.current_session_id)
        return session.title if session else "未知"


cli_session = CLISession()


async def run_cli():
    await cli_session.init()

    # Connect auto-connect MCP servers
    try:
        from src.mcp.client import mcp_client
        from src.mcp.manager import list_servers as mcp_list_servers
        servers = await mcp_list_servers()
        await mcp_client.connect_auto(servers)
    except Exception:
        pass

    console.print("[bold blue]智能助手已启动[/]")
    console.print("[dim]输入 /help 查看命令，/exit 退出[/]\n")

    while True:
        try:
            title = await cli_session.get_title()
            user_input = console.input(f"[bold green][{title}] > [/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]再见！[/]")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            try:
                await handle_slash_command(user_input, cli_session)
            except SystemExit:
                break
            console.print()
            continue

        console.print()
        try:
            async for token in cli_session.agent.chat_stream(user_input):
                console.print(token, end="")
        except Exception as e:
            console.print(f"\n[red]错误: {e}[/]")
        console.print("\n")
