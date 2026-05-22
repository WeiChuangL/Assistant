import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Skill:
    name: str
    display_name: str
    description: str
    icon: str
    prompt_append: str
    tools: list[str]
    enabled: bool = True
    auto_trigger: bool = True
    trigger_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "icon": self.icon,
            "prompt_append": self.prompt_append,
            "tools": self.tools,
            "enabled": self.enabled,
            "auto_trigger": self.auto_trigger,
            "trigger_keywords": self.trigger_keywords,
            "tools_count": len(self.tools),
        }


STATE_FILE = Path(os.path.dirname(__file__)).parent.parent.parent / "data" / "skill_state.json"


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def _load_state(self) -> dict[str, dict]:
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_state(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state = {
            name: {"enabled": s.enabled, "auto_trigger": s.auto_trigger}
            for name, s in self._skills.items()
        }
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def register(self, skill_def: dict):
        name = skill_def["name"]
        state = self._load_state()
        saved = state.get(name, {})
        # Backward compat: old format stored bool directly
        if isinstance(saved, bool):
            saved = {"enabled": saved, "auto_trigger": True}
        enabled = saved.get("enabled", skill_def.get("enabled", True))
        auto_trigger = saved.get("auto_trigger", skill_def.get("auto_trigger", True))
        skill = Skill(
            name=name,
            display_name=skill_def.get("display_name", name),
            description=skill_def.get("description", ""),
            icon=skill_def.get("icon", "⚡"),
            prompt_append=skill_def.get("prompt_append", ""),
            tools=skill_def.get("tools", []),
            enabled=enabled,
            auto_trigger=auto_trigger,
            trigger_keywords=skill_def.get("trigger_keywords", []),
        )
        self._skills[name] = skill

    def enable(self, name: str):
        if name in self._skills:
            self._skills[name].enabled = True
            self._save_state()

    def disable(self, name: str):
        if name in self._skills:
            self._skills[name].enabled = False
            self._save_state()

    def toggle_auto_trigger(self, name: str, enabled: bool):
        if name in self._skills:
            self._skills[name].auto_trigger = enabled
            self._save_state()

    def is_enabled(self, name: str) -> bool:
        return self._skills[name].enabled if name in self._skills else False

    def remove(self, name: str):
        self._skills.pop(name, None)
        self._save_state()

    def find_triggered_skills(self, message: str) -> list[str]:
        """Return list of skill names whose trigger_keywords match the message."""
        triggered = []
        msg_lower = message.lower()
        for s in self._skills.values():
            if not s.enabled:
                continue
            for kw in s.trigger_keywords:
                if kw.lower() in msg_lower:
                    triggered.append(s.name)
                    break
        return triggered

    def get_prompt_augmentation(self, triggered_skills: list[str] | None = None) -> str:
        """Build system prompt additions from enabled skills.

        If triggered_skills is provided, only auto_trigger skills and the
        specified triggered skills are included. Otherwise all enabled skills
        are included (backward-compatible default).
        """
        parts = []
        for s in self._skills.values():
            if not s.enabled or not s.prompt_append:
                continue
            if triggered_skills is not None:
                # Conditional mode: include if auto_trigger or explicitly triggered
                if s.auto_trigger or s.name in triggered_skills:
                    parts.append(s.prompt_append)
            else:
                parts.append(s.prompt_append)
        return "\n\n".join(parts) if parts else ""

    def list_all(self) -> list[dict]:
        return [s.to_dict() for s in self._skills.values()]

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __bool__(self) -> bool:
        return len(self._skills) > 0


skill_registry = SkillRegistry()
