"""Versioned prompt loading; the version is recorded in the audit trail.

Prompts live as markdown files rather than string literals for one operational
reason: when a customer disputes an answer months later, reconstructing what the
system actually said requires knowing which prompt produced it. Every generated
message stores ``prompt_version``, and that only means something if the version
is declared in the file and read from it.

Header format (leading comment lines, stripped before use):

    # Generate - grounded answer under the citation contract
    # version: v1
    # temperature: 0.1
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent / "templates"

_META = re.compile(r"^#\s*(version|temperature)\s*:\s*(.+?)\s*$", re.I)


@dataclass(slots=True, frozen=True)
class Prompt:
    name: str
    version: str
    body: str
    temperature: float | None = None

    def render(self, **kwargs: object) -> str:
        """Substitute ``{placeholders}``.

        Uses ``str.replace`` rather than ``format`` on purpose: prompt bodies
        contain literal braces (JSON schemas, output contracts) that ``format``
        would try to interpret.
        """
        text = self.body
        for key, value in kwargs.items():
            text = text.replace("{" + key + "}", str(value))
        return text

    @property
    def qualified(self) -> str:
        """What gets written to ``messages.prompt_version``."""
        return f"{self.name}@{self.version}"


@lru_cache(maxsize=32)
def load(name: str) -> Prompt:
    path = TEMPLATE_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"No prompt template '{name}' in {TEMPLATE_DIR}")

    version = "v0"
    temperature: float | None = None
    body_lines: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") and (m := _META.match(line)):
            key, value = m.group(1).lower(), m.group(2)
            if key == "version":
                version = value
            else:
                temperature = float(value)
            continue
        # Drop the leading comment header; keep markdown headings in the body.
        if line.startswith("#") and not body_lines and not line.startswith("##"):
            continue
        body_lines.append(line)

    return Prompt(
        name=name,
        version=version,
        body="\n".join(body_lines).strip(),
        temperature=temperature,
    )


def available() -> list[str]:
    return sorted(p.stem for p in TEMPLATE_DIR.glob("*.md"))
