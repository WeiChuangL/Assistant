from rich.console import Console
from rich.markdown import Markdown

from src.agent.core import agent
from src.cli.commands import handle_slash_command

console = Console()


async def run_cli():
    console.print("[bold blue]智能助手已启动[/]")
    console.print("[dim]输入 /help 查看命令，/exit 退出[/]\n")

    while True:
        try:
            user_input = console.input("[bold green]> [/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]再见！[/]")
            break

        if not user_input:
            continue

        # Check for slash commands
        if user_input.startswith("/"):
            try:
                await handle_slash_command(user_input)
            except SystemExit:
                break
            console.print()
            continue

        # Normal chat
        console.print()
        try:
            async for token in agent.chat_stream(user_input):
                console.print(token, end="")
        except Exception as e:
            console.print(f"\n[red]错误: {e}[/]")
        console.print("\n")
