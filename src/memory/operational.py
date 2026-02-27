"""Self-modifying operational memory (markdown files).

- safety-rules.md: Append-only, agent can add but not remove
- preferences.md: Learned preferences, overridable
- operational-notes.md: Strategy and methodology notes
"""

from pathlib import Path
from datetime import datetime, timezone


class OperationalMemory:
    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self.safety_rules_path = memory_dir / "safety-rules.md"
        self.preferences_path = memory_dir / "preferences.md"
        self.operational_notes_path = memory_dir / "operational-notes.md"

    def initialize(self):
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        for path, header in [
            (self.safety_rules_path, "# Safety Rules\n\n"),
            (self.preferences_path, "# Preferences\n\n"),
            (self.operational_notes_path, "# Operational Notes\n\n"),
        ]:
            if not path.exists():
                path.write_text(header)

    def append_safety_rule(self, rule: str):
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        with open(self.safety_rules_path, "a") as f:
            f.write(f"- [{timestamp}] {rule}\n")

    def update_preference(self, key: str, value: str):
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        with open(self.preferences_path, "a") as f:
            f.write(f"- **{key}** [{timestamp}]: {value}\n")

    def add_operational_note(self, note: str):
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        with open(self.operational_notes_path, "a") as f:
            f.write(f"- [{timestamp}] {note}\n")

    def read_all(self) -> dict[str, str]:
        result = {}
        for key, path in [
            ("safety_rules", self.safety_rules_path),
            ("preferences", self.preferences_path),
            ("operational_notes", self.operational_notes_path),
        ]:
            result[key] = path.read_text() if path.exists() else ""
        return result
