"""fct — Flattened Contextual Tree (user's Schema B).

Pipe-delimited records with a scope column and ditto (``"``) compression; see
``docs/serializers/fct.md``. Header row is always ``ref|t|name|val|flags|scope``.
"""

from __future__ import annotations

from modalitybench.observations.base import PageGraph, PageNode
from modalitybench.observations.registry import register_strategy
from modalitybench.observations.serializers._common import make_observation

_HEADER = "ref|t|name|val|flags|scope"

# Two-letter type codes per the FCT spec.
_TYPE2 = {
    "link": "ln", "button": "bt", "textbox": "in", "searchbox": "in", "spinbutton": "in",
    "combobox": "se", "checkbox": "ck", "radio": "rd", "image": "im", "heading": "hd",
    "listitem": "li", "option": "op", "tab": "tb", "switch": "sw",
}


def _type2(node: PageNode) -> str:
    role = (node.role or "").lower()
    if role in _TYPE2:
        return _TYPE2[role]
    tag = node.tag.lower()
    if tag == "select":
        return "se"
    if tag in {"input", "textarea"}:
        return "in"
    if tag == "a":
        return "ln"
    if tag == "button":
        return "bt"
    return "tx"


def _esc(s: str) -> str:
    return s.replace("|", "∣")


def _flags(node: PageNode) -> str:
    out: list[str] = []
    if "required" in node.flags:
        out.append("req")
    if node.role in {"checkbox", "radio", "switch"}:
        out.append("on" if "checked" in node.flags else "off")
    elif (node.role in {"textbox", "searchbox", "spinbutton"}
          or node.tag in {"input", "textarea"}) and not node.value:
        out.append("empty")
    if "disabled" in node.flags:
        out.append("dis")
    if "readonly" in node.flags:
        out.append("ro")
    mx = next((f for f in node.flags if f.startswith("max=")), None)
    if mx:
        out.append(mx.replace("max=", "max"))
    opts = node.attrs.get("opts")
    if opts and (node.tag == "select" or node.role == "combobox"):
        out.append(f"opts={opts}")
    return ",".join(out)


class FctSerializer:
    name = "fct"

    def observe(self, graph: PageGraph, *, task_text=None):
        rows = [_HEADER]
        prev_scope: object = object()
        for node in graph.nodes():
            if node.ref is None or not node.visible:
                continue
            name = _esc(node.name or node.text or "")
            val = _esc(node.value or "")
            flags = _flags(node)
            scope = node.scope or ""
            scope_cell = '"' if scope == prev_scope else _esc(scope)
            prev_scope = scope
            rows.append(f"{node.ref}|{_type2(node)}|{name}|{val}|{flags}|{scope_cell}")
        return make_observation(graph, self.name, "\n".join(rows))


@register_strategy("fct")
def _make() -> FctSerializer:
    return FctSerializer()
