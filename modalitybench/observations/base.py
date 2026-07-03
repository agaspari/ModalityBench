"""Core observation types and the ObservationStrategy protocol.

Design invariant (see plan): **one capture, many serializers**. A single intermediate
``PageGraph`` is produced once per page; every DOM serializer renders from it, so token
comparisons are apples-to-apples and offline snapshots reuse the exact same code path.

The **ref registry is the contract** between a serializer and the action executor. A
serializer emits stable refs (``e1``, ``e2``, …); the registry maps each ref to a locator
descriptor (a CSS selector for live pages, or a backend node id for offline snapshots) so
the executor can act on it regardless of which serializer produced the observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Page graph: the shared intermediate representation
# ---------------------------------------------------------------------------


@dataclass
class PageNode:
    """A single node in the intermediate page graph.

    Nodes form a tree via ``children``. Only *interesting* nodes (interactive elements,
    landmarks, non-trivial text) are assigned a ``ref``; structural wrappers may have
    ``ref=None`` and exist only to preserve hierarchy for HTML/axtree serializers.
    """

    tag: str
    ref: str | None = None
    role: str | None = None
    name: str | None = None  # accessible name
    value: str | None = None
    text: str | None = None  # own text content (for text-bearing nodes)
    flags: set[str] = field(default_factory=set)  # required, disabled, checked, hidden, ...
    visible: bool = True
    bbox: tuple[float, float, float, float] | None = None  # x, y, w, h
    scope: str | None = None  # nearest landmark/section label (for FCT scope column)
    attrs: dict[str, str] = field(default_factory=dict)  # allowlisted attributes
    # Locator descriptor used to resolve this ref back to a real element.
    #   live pages:   {"type": "selector", "value": '[data-mb-ref="e1"]'}
    #   offline:      {"type": "backend_id", "value": 42}
    locator: dict[str, Any] | None = None
    children: list["PageNode"] = field(default_factory=list)

    @property
    def is_interactive(self) -> bool:
        return self.role in _INTERACTIVE_ROLES or self.tag in _INTERACTIVE_TAGS

    def walk(self):
        """Depth-first pre-order traversal yielding this node then descendants."""
        yield self
        for child in self.children:
            yield from child.walk()


_INTERACTIVE_TAGS = {
    "a", "button", "input", "select", "textarea", "option", "label", "summary",
}
_INTERACTIVE_ROLES = {
    "button", "link", "textbox", "checkbox", "radio", "combobox", "listbox",
    "menuitem", "tab", "switch", "searchbox", "slider", "spinbutton", "option",
}


@dataclass
class PageGraph:
    """Intermediate representation of a page, produced once and shared by all serializers."""

    root: PageNode
    url: str = ""
    title: str = ""
    viewport: tuple[int, int] | None = None  # width, height
    source: str = ""  # "live" | "html" | "mind2web" — provenance for debugging
    meta: dict[str, Any] = field(default_factory=dict)

    def nodes(self):
        """Iterate every node depth-first."""
        yield from self.root.walk()

    def by_ref(self) -> dict[str, PageNode]:
        return {n.ref: n for n in self.nodes() if n.ref is not None}


# ---------------------------------------------------------------------------
# Observation: what actually gets sent to the model
# ---------------------------------------------------------------------------


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ImageBlock:
    """An image observation. ``data`` is raw bytes; ``media_type`` e.g. 'image/png'."""

    data: bytes
    media_type: str = "image/png"
    width: int | None = None
    height: int | None = None
    type: str = "image"


ContentBlock = TextBlock | ImageBlock


@dataclass
class ToolSpec:
    """A tool exposed to the model (used by tools-mode, which sends query tools instead of
    an upfront DOM dump)."""

    name: str
    description: str
    input_schema: dict[str, Any]


class RefRegistry:
    """Maps element refs emitted by a serializer to locator descriptors.

    The executor calls :meth:`resolve` to turn a model-supplied ref into something it can
    act on. Refs are opaque strings; the registry is the single source of truth.
    """

    def __init__(self) -> None:
        self._by_ref: dict[str, dict[str, Any]] = {}

    def add(self, ref: str, locator: dict[str, Any]) -> None:
        self._by_ref[ref] = locator

    def resolve(self, ref: str) -> dict[str, Any] | None:
        return self._by_ref.get(ref)

    def __contains__(self, ref: str) -> bool:
        return ref in self._by_ref

    def __len__(self) -> int:
        return len(self._by_ref)

    def refs(self) -> list[str]:
        return list(self._by_ref)


@dataclass
class Observation:
    """The output of an observation strategy: content the model sees + how to act on it.

    ``content_blocks`` is what gets rendered into the model prompt (text and/or images).
    ``tools`` is non-empty only for tools-mode strategies. ``ref_registry`` resolves the
    refs the model may reference in its actions. ``meta`` carries measured size stats.
    """

    content_blocks: list[ContentBlock] = field(default_factory=list)
    tools: list[ToolSpec] = field(default_factory=list)
    ref_registry: RefRegistry = field(default_factory=RefRegistry)
    meta: dict[str, Any] = field(default_factory=dict)  # tokens, bytes, element_count, ...

    def text(self) -> str:
        """Concatenated text of all text blocks (for token counting / inspection)."""
        return "\n".join(b.text for b in self.content_blocks if isinstance(b, TextBlock))

    @property
    def has_image(self) -> bool:
        return any(isinstance(b, ImageBlock) for b in self.content_blocks)


# ---------------------------------------------------------------------------
# The strategy protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ObservationStrategy(Protocol):
    """Turn a page into an Observation.

    Implementations may be pure serializers over a ``PageGraph`` (the common case), or
    wrappers (salience) / tools emitters (tools-mode). The runner treats them uniformly.
    """

    name: str

    def observe(self, graph: PageGraph, *, task_text: str | None = None) -> Observation:
        """Produce an observation from an already-captured page graph.

        ``task_text`` is provided so task-aware strategies (salience, retrieval) can rank
        against the goal; task-agnostic serializers ignore it.
        """
        ...
