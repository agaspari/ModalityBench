"""Shared helpers for serializers.

Every serializer builds its ``RefRegistry`` from the same ``PageGraph`` via
:func:`build_registry`, so a given ref resolves to the same element no matter which
serializer produced the observation — that interchangeability is the core invariant.
"""

from __future__ import annotations

from modalitybench.observations.base import (
    Observation,
    PageGraph,
    PageNode,
    RefRegistry,
    TextBlock,
)


def build_registry(graph: PageGraph) -> RefRegistry:
    reg = RefRegistry()
    for node in graph.nodes():
        if node.ref is not None:
            reg.add(node.ref, node.locator or {"type": "backend_id", "value": node.ref})
    return reg


def salient_nodes(graph: PageGraph) -> list[PageNode]:
    return [n for n in graph.nodes() if n.ref is not None]


def make_observation(graph: PageGraph, serializer: str, text: str) -> Observation:
    reg = build_registry(graph)
    obs = Observation(
        content_blocks=[TextBlock(text=text)],
        ref_registry=reg,
        meta={
            "serializer": serializer,
            "chars": len(text),
            "bytes": len(text.encode("utf-8")),
            "element_count": len(reg),
        },
    )
    return obs


_VOID_TAGS = {"input", "img", "br", "hr", "meta", "link", "source", "area", "col"}


def render_html(
    node: PageNode,
    *,
    prune_hidden: bool = False,
    allow_attrs: set[str] | None = None,
    max_text: int | None = None,
    inject_ref: bool = True,
    _depth: int = 0,
) -> str:
    """Re-serialize a PageNode subtree to HTML.

    Used by ``body_html`` (light clean) and ``pruned_html`` (aggressive: drop hidden nodes,
    tight attribute allowlist, truncate text). When ``inject_ref`` is set, salient elements
    carry a ``ref="eN"`` attribute so the model can reference them.
    """
    if prune_hidden and not node.visible:
        return ""
    tag = node.tag or "div"
    attrs: dict[str, str] = {}
    if allow_attrs is not None:
        attrs = {k: v for k, v in node.attrs.items() if k in allow_attrs}
    else:
        attrs = dict(node.attrs)
    if inject_ref and node.ref is not None:
        attrs["ref"] = node.ref
    attr_str = "".join(f' {k}="{_esc(v)}"' for k, v in attrs.items())

    text = node.text or ""
    if max_text is not None and len(text) > max_text:
        text = text[:max_text] + "…"

    inner_parts = [_esc(text)] if text else []
    for child in node.children:
        rendered = render_html(
            child,
            prune_hidden=prune_hidden,
            allow_attrs=allow_attrs,
            max_text=max_text,
            inject_ref=inject_ref,
            _depth=_depth + 1,
        )
        if rendered:
            inner_parts.append(rendered)
    inner = "".join(inner_parts)

    if tag in _VOID_TAGS and not inner:
        return f"<{tag}{attr_str}>"
    if not inner and node.ref is None and not attrs:
        return ""  # drop empty structural wrappers
    return f"<{tag}{attr_str}>{inner}</{tag}>"


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def short_type(node: PageNode) -> str:
    """Two/three-letter affordance code shared by compact serializers."""
    role = (node.role or "").lower()
    tag = node.tag.lower()
    mapping = {
        "link": "lnk", "button": "btn", "textbox": "in", "searchbox": "in",
        "combobox": "sel", "checkbox": "chk", "radio": "rd", "slider": "sld",
        "spinbutton": "in", "option": "opt", "heading": "hd", "image": "img",
        "listitem": "li", "tab": "tab", "switch": "sw", "menuitem": "mi",
    }
    if role in mapping:
        return mapping[role]
    if tag == "select":
        return "sel"
    if tag in {"input", "textarea"}:
        return "in"
    if tag == "a":
        return "lnk"
    if tag == "button":
        return "btn"
    return "txt"
