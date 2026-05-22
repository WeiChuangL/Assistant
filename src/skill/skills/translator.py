SKILL = {
    "name": "translator",
    "display_name": "翻译助手",
    "description": "多语言翻译专家，支持中英日韩法德等语言互译",
    "icon": "🌐",
    "prompt_append": (
        "## 翻译模式\n"
        "你正在以专业翻译模式工作。翻译时请注意:\n"
        "1. **准确性**: 忠实原文意思，不增不减\n"
        "2. **流畅性**: 符合目标语言的表达习惯\n"
        "3. **术语**: 技术术语使用业界标准译法\n"
        "4. **语气**: 保持原文的语气和风格\n"
        "翻译结果请标注源语言和目标语言。"
    ),
    "tools": [],
    "enabled": True,
    "auto_trigger": True,
    "trigger_keywords": ["/translator", "/translate", "翻译", "翻译成", "translate"],
}
