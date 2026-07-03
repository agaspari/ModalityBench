"""afus — Abstract Functional UI Syntax (user's Schema A).

Affordance-oriented line grammar; see ``docs/serializers/afus.md`` for the authoritative
spec. Scope markers (``@page``, ``@sec``, ``~region``) plus element lines
``type:ref "name" flags``. No indentation.
"""

from __future__ import annotations

from urllib.parse import urlparse

from modalitybench.observations.base import PageGraph, PageNode
from modalitybench.observations.registry import register_strategy
from modalitybench.observations.serializers._common import make_observation, short_type


def _path(url: str) -> str:
    if not url:
        return ""
    p = urlparse(url)
    return (p.path or "") + (f"?{p.query}" if p.query else "")


def _flags(node: PageNode) -> list[str]:
    role = node.role or ""
    tag = node.tag
    out: list[str] = []
    if role in {"checkbox", "radio", "switch"}:
        out.append("on" if "checked" in node.flags else "off")
    elif tag == "select" or role == "combobox":
        if node.value:
            out.append(f'="{node.value}"')
        opts = node.attrs.get("opts")
        if opts:
            out.append(f"opts={opts}")
    elif role in {"textbox", "searchbox", "spinbutton"} or tag in {"input", "textarea"}:
        if "required" in node.flags:
            out.append("req")
        if node.value:
            out.append(f'="{node.value}"')
        else:
            out.append("empty")
    elif role == "link":
        href = node.attrs.get("href")
        if href:
            out.append(href)
    if "required" in node.flags and not (
        role in {"textbox", "searchbox", "spinbutton"} or tag in {"input", "textarea"}
    ):
        out.append("req")
    if "disabled" in node.flags:
        out.append("disabled")
    if "readonly" in node.flags:
        out.append("ro")
    mx = next((f for f in node.flags if f.startswith("max=")), None)
    if mx:
        out.append(mx.replace("max=", "max="))
    return out


class AfusSerializer:
    name = "afus"

    def observe(self, graph: PageGraph, *, task_text=None):
        lines: list[str] = []
        header = f'@page "{graph.title}" {_path(graph.url)}'.rstrip()
        lines.append(header)
        last_scope: object = object()
        for node in graph.nodes():
            if node.ref is None or not node.visible:
                continue
            scope = node.scope
            if scope != last_scope:
                if scope:
                    lines.append(f"@sec {scope}")
                last_scope = scope
            if node.is_interactive:
                sig = short_type(node)
                flags = _flags(node)
                name = node.name or ""
                line = f'{sig}:{node.ref} "{name}"'
                if flags:
                    line += " " + " ".join(flags)
                lines.append(line)
            elif node.text:
                lines.append(f'txt "{node.text}"')
        return make_observation(graph, self.name, "\n".join(lines))


@register_strategy("afus")
def _make() -> AfusSerializer:
    return AfusSerializer()
