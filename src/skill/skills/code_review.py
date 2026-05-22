SKILL = {
    "name": "code-review",
    "display_name": "代码审查",
    "description": "对代码进行审查，检查安全漏洞、性能问题和代码规范",
    "icon": "🔍",
    "prompt_append": (
        "## 代码审查模式\n"
        "你正在以代码审查专家模式工作。审查代码时请关注以下方面:\n"
        "1. **安全漏洞**: SQL注入、XSS、命令注入、敏感信息泄露\n"
        "2. **性能问题**: 不必要的循环、重复查询、内存泄漏风险\n"
        "3. **可读性**: 命名规范、函数长度、注释清晰度\n"
        "4. **最佳实践**: 错误处理、类型安全、设计模式\n"
        "请用中文输出审查意见，给出具体的改进建议。"
    ),
    "tools": [],
    "enabled": True,
    "auto_trigger": True,
    "trigger_keywords": ["/code-review", "/review", "审查代码", "代码审查", "review code"],
}
