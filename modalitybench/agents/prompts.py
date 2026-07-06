"""Prompt construction shared by the offline element-selection eval and the live agent loop.

Kept deliberately small and format-agnostic: the observation text (whatever serializer
produced it) is dropped in verbatim, so prompt tokens differ across strategies only by the
observation itself — which is exactly what the benchmark measures.
"""

from __future__ import annotations

from modalitybench.agents.actions import action_space_prompt
from modalitybench.observations.base import ContentBlock, Observation, TextBlock

SYSTEM_PROMPT = (
    "You are a web-browsing agent. You are shown a compact representation of a web page and "
    "a goal. Choose the single best next action and respond with ONE JSON object only — no "
    "prose, no code fences.\n\n" + action_space_prompt()
)


def build_user_blocks(
    goal: str,
    observation: Observation,
    *,
    history: list[str] | None = None,
    instruction: str | None = None,
) -> list[ContentBlock]:
    """Assemble the user-turn content: goal, history, the observation, and a final ask.

    The observation's content blocks (text and/or image) are inserted as-is so image
    strategies work without special-casing.
    """
    header_lines = [f"GOAL: {goal}"]
    if history:
        header_lines.append("\nACTIONS SO FAR:")
        header_lines += [f"  {i + 1}. {h}" for i, h in enumerate(history)]
    header_lines.append("\nPAGE OBSERVATION:")
    blocks: list[ContentBlock] = [TextBlock(text="\n".join(header_lines))]
    # A serializer can legitimately produce an empty observation (e.g. no visible interactive
    # elements at this page state). Anthropic rejects empty text blocks ("text content blocks
    # must be non-empty" → 400); substitute a placeholder so the request is valid and the model
    # gets an explicit "nothing here" signal instead of a void. (Token accounting counts the
    # raw observation blocks separately, so this does not skew measured observation size.)
    for b in observation.content_blocks:
        if isinstance(b, TextBlock) and not b.text.strip():
            blocks.append(TextBlock(text="(no interactive elements visible on this page)"))
        else:
            blocks.append(b)
    ask = instruction or (
        "Respond with the JSON action for the single best next step. Reference elements by "
        'their ref id (e.g. {"action": "click", "ref": "e5"}).'
    )
    blocks.append(TextBlock(text="\n" + ask))
    return blocks
