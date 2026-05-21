import os
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.knowledge.ingestion import ingest_directory, ingest_file
from src.knowledge.manager import delete_document, list_documents
from src.knowledge.retrieval import search_chunks
from src.memory.long_term import search_memories
from src.memory.profile import delete_profile_value, get_profile, set_profile_value

console = Console()


async def handle_slash_command(cmd: str):
    """Parse and execute a slash command. Returns True if it was a command, False otherwise."""
    parts = cmd.strip().split(maxsplit=1)
    action = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    match action:
        case "/help":
            _show_help()

        case "/kb":
            await _kb_subcommand(args)

        case "/memory":
            await _memory_subcommand(args)

        case "/profile":
            await _profile_subcommand(args)

        case "/clear":
            from src.agent.core import agent
            agent.clear_memory()
            console.print("[green]对话记忆已清除[/]")

        case "/exit" | "/quit":
            console.print("[yellow]再见！[/]")
            raise SystemExit(0)

        case _:
            return False

    return True


def _show_help():
    console.print("""
[bold]可用命令:[/]

[bold]/kb add <文件/目录>[/]  - 导入文档到知识库
[bold]/kb list[/]              - 列出所有已导入文档
[bold]/kb search <查询>[/]     - 搜索知识库
[bold]/kb delete <ID>[/]       - 删除指定文档

[bold]/memory search <查询>[/] - 搜索长期记忆

[bold]/profile show[/]         - 查看用户画像
[bold]/profile set <键> <值>[/] - 设置用户画像
[bold]/profile del <键>[/]     - 删除画像条目

[bold]/clear[/]                - 清除当前对话记忆
[bold]/help[/]                 - 显示帮助
[bold]/exit[/]                 - 退出
""", markup=True)


async def _kb_subcommand(args: str):
    parts = args.split(maxsplit=1)
    sub = parts[0].lower() if parts else "list"
    sub_args = parts[1] if len(parts) > 1 else ""

    match sub:
        case "add":
            if not sub_args:
                console.print("[red]用法: /kb add <文件路径或目录>[/]")
                return
            path = Path(sub_args.strip())
            if not path.exists():
                console.print(f"[red]路径不存在: {path}[/]")
                return
            console.print(f"[yellow]正在导入: {path}[/]")
            if path.is_dir():
                doc_ids = await ingest_directory(path)
                console.print(f"[green]导入完成，共 {len(doc_ids)} 个文件[/]")
            else:
                doc_id = await ingest_file(path)
                console.print(f"[green]导入完成，文档 ID: {doc_id}[/]")

        case "list":
            docs = await list_documents()
            if not docs:
                console.print("[yellow]知识库为空[/]")
                return
            table = Table(title="知识库文档列表")
            table.add_column("ID", style="cyan")
            table.add_column("文件名", style="green")
            table.add_column("类型")
            table.add_column("分块数")
            table.add_column("导入时间")
            for d in docs:
                table.add_row(str(d.id), d.filename, d.file_type or "", str(d.chunk_count), d.created_at[:19])
            console.print(table)

        case "search":
            if not sub_args:
                console.print("[red]用法: /kb search <查询内容>[/]")
                return
            results = await search_chunks(sub_args)
            if not results:
                console.print("[yellow]未找到相关内容[/]")
                return
            for i, r in enumerate(results, 1):
                console.print(f"\n[cyan]#{i}[/] [green]{r.filename}[/] (相关度: {r.score:.2f})")
                console.print(r.content[:300] + ("..." if len(r.content) > 300 else ""))

        case "delete":
            if not sub_args:
                console.print("[red]用法: /kb delete <文档ID>[/]")
                return
            try:
                doc_id = int(sub_args.strip())
            except ValueError:
                console.print("[red]请输入有效的文档 ID[/]")
                return
            if await delete_document(doc_id):
                console.print(f"[green]文档 {doc_id} 已删除[/]")
            else:
                console.print(f"[red]文档 {doc_id} 不存在[/]")

        case _:
            console.print("[red]未知子命令。可用: add | list | search | delete[/]")


async def _memory_subcommand(args: str):
    parts = args.split(maxsplit=1)
    sub = parts[0].lower() if parts else "search"
    sub_args = parts[1] if len(parts) > 1 else ""

    match sub:
        case "search":
            if not sub_args:
                console.print("[red]用法: /memory search <查询内容>[/]")
                return
            results = await search_memories(sub_args)
            if not results:
                console.print("[yellow]未找到相关记忆[/]")
                return
            for i, r in enumerate(results, 1):
                console.print(f"\n[cyan]#{i}[/] [{r.memory_type}] (相关度: {r.score:.2f}, 重要度: {r.importance})")
                console.print(r.content[:300] + ("..." if len(r.content) > 300 else ""))

        case _:
            console.print("[red]未知子命令。可用: search[/]")


async def _profile_subcommand(args: str):
    parts = args.split(maxsplit=2)
    sub = parts[0].lower() if parts else "show"
    sub_args1 = parts[1] if len(parts) > 1 else ""
    sub_args2 = parts[2] if len(parts) > 2 else ""

    match sub:
        case "show":
            profile = await get_profile()
            if not profile:
                console.print("[yellow]暂无用户画像[/]")
                return
            console.print("[bold]用户画像:[/]")
            for k, v in profile.items():
                console.print(f"  [cyan]{k}[/]: {v}")

        case "set":
            if not sub_args1 or not sub_args2:
                console.print("[red]用法: /profile set <键> <值>[/]")
                return
            await set_profile_value(sub_args1, sub_args2)
            console.print(f"[green]已设置: {sub_args1} = {sub_args2}[/]")

        case "del":
            if not sub_args1:
                console.print("[red]用法: /profile del <键>[/]")
                return
            if await delete_profile_value(sub_args1):
                console.print(f"[green]已删除: {sub_args1}[/]")
            else:
                console.print(f"[yellow]键不存在: {sub_args1}[/]")

        case _:
            console.print("[red]未知子命令。可用: show | set | del[/]")
