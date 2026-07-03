"""DOM capture: build the shared intermediate ``PageGraph`` from a page.

Two entry points, one output type:

* :func:`graph_from_html` — offline, ``lxml``-based. Used by the serializer bench, the
  golden tests, and the Mind2Web loader. Fully deterministic, no browser.
* :func:`graph_from_page` — live, Playwright. Injects ``data-mb-ref`` attributes and reads
  computed role / accessible name / visibility / bounding boxes via one in-page script, so
  refs resolve to real locators and serializers that need geometry (2D map) get bboxes.

Both assign refs identically (same traversal order → same ``e1..eN``) so a ref means the same
element regardless of which path produced the graph, and every serializer is interchangeable.
"""

from __future__ import annotations

import re
from typing import Any

from modalitybench.observations.base import PageGraph, PageNode

# ---------------------------------------------------------------------------
# Shared role / salience tables
# ---------------------------------------------------------------------------

_SKIP_TAGS = {"script", "style", "noscript", "template", "head", "meta", "link", "title"}

_TAG_ROLE = {
    "a": "link",
    "button": "button",
    "select": "combobox",
    "textarea": "textbox",
    "img": "image",
    "nav": "navigation",
    "main": "main",
    "header": "banner",
    "footer": "contentinfo",
    "aside": "complementary",
    "form": "form",
    "h1": "heading", "h2": "heading", "h3": "heading",
    "h4": "heading", "h5": "heading", "h6": "heading",
    "ul": "list", "ol": "list", "li": "listitem",
    "table": "table", "option": "option", "label": "label",
    "summary": "button",
}

_INPUT_TYPE_ROLE = {
    "text": "textbox", "email": "textbox", "search": "searchbox", "url": "textbox",
    "tel": "textbox", "password": "textbox", "number": "spinbutton",
    "checkbox": "checkbox", "radio": "radio", "range": "slider",
    "submit": "button", "button": "button", "reset": "button",
    "hidden": "hidden",
}

_LANDMARK_TAGS = {"nav", "main", "header", "footer", "aside", "section", "form"}

_ATTR_ALLOW = {
    "id", "name", "type", "href", "src", "alt", "title", "placeholder", "value",
    "role", "aria-label", "aria-labelledby", "aria-describedby", "for", "checked",
    "selected", "disabled", "required", "readonly", "aria-hidden", "backend_node_id",
    "data-mb-ref",
}

_WS = re.compile(r"\s+")


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return _WS.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Offline (lxml) capture
# ---------------------------------------------------------------------------


def graph_from_html(
    html: str,
    *,
    url: str = "",
    title: str = "",
    id_attr: str = "backend_node_id",
    source: str = "html",
) -> PageGraph:
    """Build a ``PageGraph`` from an HTML string.

    ``id_attr`` names an attribute (if present on elements) whose value becomes the node's
    stable backend id in the locator — Mind2Web's cleaned HTML carries ``backend_node_id``,
    which lets offline element-selection scoring match against ground-truth candidates.
    """
    import lxml.html as LH

    doc = LH.fromstring(html) if html.strip() else LH.Element("html")
    if not title:
        t = doc.find(".//title")
        title = _clean(t.text_content()) if t is not None else ""

    counter = _RefCounter()
    body = doc.find(".//body")
    root_el = body if body is not None else doc
    root = _build_node(root_el, scope=None, counter=counter, id_attr=id_attr)
    if root is None:
        root = PageNode(tag="body")
    graph = PageGraph(root=root, url=url, title=title, source=source)
    _resolve_labels(graph)
    _renumber(graph)
    graph.meta["html"] = html
    return graph


class _RefCounter:
    def __init__(self) -> None:
        self.n = 0

    def next(self) -> str:
        self.n += 1
        return f"e{self.n}"


_CONTROL_ROLES = {
    "textbox", "searchbox", "spinbutton", "checkbox", "radio", "combobox",
    "slider", "switch", "listbox",
}
_CONTROL_TAGS = {"input", "select", "textarea"}


def _is_control(node: PageNode) -> bool:
    return (node.role in _CONTROL_ROLES) or (node.tag in _CONTROL_TAGS)


def _resolve_labels(graph: PageGraph) -> None:
    """Fold ``<label>`` text into its control's accessible name, then suppress the label.

    Offline HTML has no layout engine, so we resolve label→control association here:
    ``<label for=id>`` targets ``#id``; a wrapping ``<label>`` targets its first control
    descendant. The label node is then blanked so it doesn't appear as a separate row.
    """
    by_id = {n.attrs["id"]: n for n in graph.nodes() if n.attrs.get("id")}
    for node in graph.nodes():
        if node.tag != "label":
            continue
        label_text = node.name or node.text
        target: PageNode | None = None
        for_id = node.attrs.get("for")
        if for_id:
            target = by_id.get(for_id)
        else:
            target = next((c for c in node.walk() if c is not node and _is_control(c)), None)
        if target is not None and label_text:
            target.name = label_text
        # Blank the label so serializers skip it (no ref/role/name/text).
        node.ref = None
        node.role = None
        node.name = None
        node.text = None
        node.locator = None


def _renumber(graph: PageGraph) -> None:
    """Reassign contiguous ``e1..eN`` refs in document order (fills gaps from dropped nodes)."""
    counter = _RefCounter()
    for node in graph.nodes():
        if node.ref is None:
            continue
        old = node.ref
        new = counter.next()
        node.ref = new
        if node.locator and node.locator.get("type") == "backend_id" \
                and node.locator.get("value") == old:
            node.locator["value"] = new


def _build_node(
    el: Any, *, scope: str | None, counter: _RefCounter, id_attr: str
) -> PageNode | None:
    tag = str(getattr(el, "tag", "")).lower()
    if not isinstance(tag, str) or tag in _SKIP_TAGS or not tag.isidentifier() and tag != "":
        # Comments / processing instructions have non-str tags → skip.
        if not isinstance(el.tag, str):
            return None
    if tag in _SKIP_TAGS:
        return None

    attrs_raw = {k.lower(): (v or "") for k, v in el.attrib.items()}
    hidden = _is_hidden(tag, attrs_raw)

    role = _role_for(tag, attrs_raw)
    # Update scope when entering a landmark region.
    node_scope = scope
    if tag in _LANDMARK_TAGS or role in {"navigation", "main", "banner", "contentinfo",
                                         "complementary", "form", "region"}:
        node_scope = _landmark_label(el, tag, attrs_raw) or scope

    node = PageNode(
        tag=tag,
        role=role,
        visible=not hidden,
        scope=scope,  # the scope this element *belongs to* (its parent's landmark)
        attrs={k: v for k, v in attrs_raw.items() if k in _ATTR_ALLOW and v != ""},
    )

    # Own (direct) text, excluding text contributed by child elements.
    own_text = _clean(el.text)

    # A <select> is one affordance; don't descend into its <option> children.
    if tag == "select":
        node.attrs["opts"] = str(len(el.findall(".//option")))
    else:
        for child in el:
            child_node = _build_node(
                child, scope=node_scope, counter=counter, id_attr=id_attr
            )
            if child_node is not None:
                node.children.append(child_node)
            tail = _clean(child.tail)
            if tail:
                own_text = f"{own_text} {tail}".strip()

    node.name = _accessible_name(el, tag, attrs_raw, own_text)
    node.value = _value_for(el, tag, attrs_raw)
    node.flags = _flags_for(tag, attrs_raw)
    node.text = own_text or None

    interactive = node.is_interactive and role != "hidden"
    is_text_leaf = bool(node.text) and not node.children and role not in {"hidden"}
    if (interactive or is_text_leaf) and role not in {"hidden"}:
        node.ref = counter.next()
        backend = attrs_raw.get(id_attr) or attrs_raw.get("id")
        if backend:
            node.attrs.setdefault(id_attr, attrs_raw.get(id_attr, backend))
            node.locator = {"type": "backend_id", "value": backend}
        else:
            node.locator = {"type": "backend_id", "value": node.ref}
    return node


def _role_for(tag: str, attrs: dict[str, str]) -> str | None:
    if attrs.get("role"):
        return attrs["role"].strip().lower()
    if tag == "input":
        return _INPUT_TYPE_ROLE.get(attrs.get("type", "text").lower(), "textbox")
    return _TAG_ROLE.get(tag)


def _is_hidden(tag: str, attrs: dict[str, str]) -> bool:
    if tag == "input" and attrs.get("type", "").lower() == "hidden":
        return True
    if "hidden" in attrs:
        return True
    if attrs.get("aria-hidden", "").lower() == "true":
        return True
    style = attrs.get("style", "").replace(" ", "").lower()
    if "display:none" in style or "visibility:hidden" in style:
        return True
    return False


def _landmark_label(el: Any, tag: str, attrs: dict[str, str]) -> str | None:
    if attrs.get("aria-label"):
        return _clean(attrs["aria-label"])
    # Nearest heading inside the region.
    for h in el.iter("h1", "h2", "h3", "h4", "h5", "h6"):
        txt = _clean(h.text_content())
        if txt:
            return txt[:40]
    return tag


def _accessible_name(el: Any, tag: str, attrs: dict[str, str], own_text: str) -> str | None:
    if attrs.get("aria-label"):
        return _clean(attrs["aria-label"])
    if tag == "img":
        return _clean(attrs.get("alt")) or None
    if tag == "input":
        itype = attrs.get("type", "text").lower()
        if itype in {"submit", "button", "reset"}:
            return _clean(attrs.get("value")) or itype
        return _clean(attrs.get("placeholder")) or _clean(attrs.get("name")) or None
    # For links/buttons/generic: full visible text.
    text = _clean(el.text_content()) if tag in {"a", "button", "label", "option",
                                                "summary"} else own_text
    if text:
        return text[:200]
    return _clean(attrs.get("title")) or None


def _value_for(el: Any, tag: str, attrs: dict[str, str]) -> str | None:
    if tag == "input" and attrs.get("type", "text").lower() not in {
        "submit", "button", "reset", "checkbox", "radio"
    }:
        return _clean(attrs.get("value")) or None
    if tag == "select":
        for opt in el.iter("option"):
            if "selected" in {k.lower() for k in opt.attrib}:
                return _clean(opt.text_content()) or None
        first = el.find(".//option")
        return _clean(first.text_content()) if first is not None else None
    if tag == "textarea":
        return _clean(el.text_content()) or None
    return None


def _flags_for(tag: str, attrs: dict[str, str]) -> set[str]:
    flags: set[str] = set()
    keys = {k.lower() for k in attrs}
    if "required" in keys or attrs.get("aria-required", "").lower() == "true":
        flags.add("required")
    if "disabled" in keys or attrs.get("aria-disabled", "").lower() == "true":
        flags.add("disabled")
    if "readonly" in keys:
        flags.add("readonly")
    if "checked" in keys:
        flags.add("checked")
    if tag == "input" and attrs.get("type", "").lower() in {"checkbox", "radio"} \
            and "checked" not in keys:
        flags.add("unchecked")
    if attrs.get("maxlength"):
        flags.add(f"max={attrs['maxlength']}")
    return flags


# ---------------------------------------------------------------------------
# Live (Playwright) capture
# ---------------------------------------------------------------------------

# In-page script: tag every element with a stable data-mb-ref, then return a flat node list
# with parent indices and computed properties. Assembled into PageNodes in Python.
_CAPTURE_JS = r"""
() => {
  const INTERACTIVE = new Set(['a','button','input','select','textarea','option','label','summary']);
  const nodes = [];
  let counter = 0;
  function isVisible(el) {
    const s = window.getComputedStyle(el);
    if (s.display === 'none' || s.visibility === 'hidden' || parseFloat(s.opacity) === 0) return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  }
  function accName(el) {
    const al = el.getAttribute('aria-label'); if (al) return al.trim();
    const tag = el.tagName.toLowerCase();
    if (tag === 'img') return (el.getAttribute('alt') || '').trim();
    if (tag === 'input') {
      const t = (el.getAttribute('type')||'text').toLowerCase();
      if (['submit','button','reset'].includes(t)) return (el.value||t).trim();
      return (el.placeholder || el.name || '').trim();
    }
    if (INTERACTIVE.has(tag)) return (el.innerText||'').trim().slice(0,200);
    return '';
  }
  function walk(el, parentIdx, scope) {
    const tag = el.tagName ? el.tagName.toLowerCase() : '';
    if (['script','style','noscript','template','head'].includes(tag)) return;
    const idx = nodes.length;
    const vis = isVisible(el);
    const r = el.getBoundingClientRect();
    const role = el.getAttribute('role') || '';
    const interactive = INTERACTIVE.has(tag) || role;
    let ref = null;
    if (interactive && vis) { ref = 'e' + (++counter); el.setAttribute('data-mb-ref', ref); }
    const ownText = Array.from(el.childNodes)
        .filter(n => n.nodeType === 3).map(n => n.textContent).join(' ').replace(/\s+/g,' ').trim();
    const opts = tag === 'select' ? String(el.querySelectorAll('option').length) : '';
    nodes.push({
      tag, role, parent: parentIdx, ref, visible: vis,
      name: accName(el), value: (el.value !== undefined ? String(el.value||'') : ''),
      text: ownText,
      bbox: [r.x, r.y, r.width, r.height],
      flags: {
        required: el.required || false, disabled: el.disabled || false,
        checked: el.checked || false, readonly: el.readOnly || false,
      },
      scope,
      attrs: { id: el.id||'', name: el.getAttribute('name')||'', type: el.getAttribute('type')||'',
               href: el.getAttribute('href')||'', placeholder: el.getAttribute('placeholder')||'',
               opts },
    });
    let childScope = scope;
    if (['nav','main','header','footer','aside','section','form'].includes(tag)) {
      childScope = (el.getAttribute('aria-label') || tag);
    }
    if (tag !== 'select') for (const c of el.children) walk(c, idx, childScope);
  }
  walk(document.body, -1, null);
  return { url: location.href, title: document.title, nodes,
           viewport: [window.innerWidth, window.innerHeight] };
}
"""


def graph_from_page(page: Any, *, screenshot: bool = True) -> PageGraph:
    """Build a ``PageGraph`` from a live Playwright page.

    Injects ``data-mb-ref`` attributes and reads computed roles / names / visibility /
    bboxes. When ``screenshot`` is set, also attaches the page PNG bytes to
    ``graph.meta['screenshot']`` so image strategies can run from the same capture.
    """
    data = page.evaluate(_CAPTURE_JS)
    graph = _assemble_live(data)
    if screenshot:
        try:
            graph.meta["screenshot"] = page.screenshot()
        except Exception:
            pass
    return graph


def _assemble_live(data: dict[str, Any]) -> PageGraph:
    raw = data["nodes"]
    page_nodes: list[PageNode] = []
    for r in raw:
        flags = {k for k, v in r["flags"].items() if v}
        if r["tag"] == "input" and not r["flags"].get("checked") and \
                r["attrs"].get("type", "") in {"checkbox", "radio"}:
            flags.add("unchecked")
        attrs = {k: v for k, v in r["attrs"].items() if v}
        node = PageNode(
            tag=r["tag"],
            ref=r["ref"],
            role=r["role"] or None,
            name=(r["name"] or None),
            value=(r["value"] or None),
            text=(r["text"] or None),
            flags=flags,
            visible=r["visible"],
            bbox=tuple(r["bbox"]) if r.get("bbox") else None,
            scope=r.get("scope"),
            attrs=attrs,
        )
        if r["ref"]:
            node.locator = {"type": "selector", "value": f'[data-mb-ref="{r["ref"]}"]'}
        page_nodes.append(node)
    # Link children by parent index.
    root = PageNode(tag="body")
    for i, r in enumerate(raw):
        parent = r["parent"]
        target = page_nodes[parent] if parent >= 0 else root
        target.children.append(page_nodes[i])
    vp = data.get("viewport")
    return PageGraph(
        root=root,
        url=data.get("url", ""),
        title=data.get("title", ""),
        viewport=tuple(vp) if vp else None,
        source="live",
    )
