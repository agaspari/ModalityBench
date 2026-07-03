"""screenshot — image observations, parameterized by scale / grayscale / format.

Reads the raw screenshot captured alongside the live page (``graph.meta['screenshot']``) and
applies a transform (downscale, grayscale, JPEG re-encode). Image-token cost is estimated as
``(w*h)/750`` for the meta; live runs bill the real image tokens via the API.

Refs still resolve (the registry comes from the DOM capture), so a vision agent can act on
elements by ref even though the observation itself is an image. Set-of-Marks annotation (draw
numbered marks) is on the roadmap.
"""

from __future__ import annotations

from modalitybench.observations.base import ImageBlock, Observation, PageGraph
from modalitybench.observations.registry import register_strategy
from modalitybench.observations.serializers._common import build_registry
from modalitybench.metrics.tokens import estimate_image_tokens


class ScreenshotStrategy:
    def __init__(
        self,
        *,
        scale: float = 1.0,
        grayscale: bool = False,
        fmt: str = "png",
        jpeg_quality: int = 80,
        name: str | None = None,
    ) -> None:
        self.scale = scale
        self.grayscale = grayscale
        self.fmt = fmt.lower()
        self.jpeg_quality = jpeg_quality
        self.name = name or self._default_name()

    def _default_name(self) -> str:
        parts = ["screenshot"]
        if self.scale != 1.0:
            parts.append(f"{int(self.scale * 100)}pct")
        if self.grayscale:
            parts.append("gray")
        if self.fmt != "png":
            parts.append(self.fmt)
        return "_".join(parts)

    def observe(self, graph: PageGraph, *, task_text=None) -> Observation:
        raw = graph.meta.get("screenshot")
        if not raw:
            raise ValueError(
                "screenshot strategy requires a live capture with a screenshot "
                "(graph.meta['screenshot'] is empty)"
            )
        data, media_type, (w, h) = self._transform(raw)
        block = ImageBlock(data=data, media_type=media_type, width=w, height=h)
        return Observation(
            content_blocks=[block],
            ref_registry=build_registry(graph),
            meta={
                "serializer": self.name,
                "bytes": len(data),
                "chars": 0,
                "element_count": len(graph.by_ref()),
                "image_size": [w, h],
                "est_image_tokens": estimate_image_tokens(w, h),
            },
        )

    def _transform(self, raw: bytes) -> tuple[bytes, str, tuple[int, int]]:
        import io

        from PIL import Image  # lazy: only needed when a screenshot strategy actually runs

        img = Image.open(io.BytesIO(raw))
        if self.scale != 1.0:
            img = img.resize(
                (max(1, int(img.width * self.scale)), max(1, int(img.height * self.scale)))
            )
        if self.grayscale:
            img = img.convert("L")
        buf = io.BytesIO()
        if self.fmt in {"jpg", "jpeg"}:
            img.convert("RGB").save(buf, format="JPEG", quality=self.jpeg_quality)
            media = "image/jpeg"
        else:
            img.save(buf, format="PNG")
            media = "image/png"
        return buf.getvalue(), media, (img.width, img.height)


@register_strategy("screenshot")
def _full() -> ScreenshotStrategy:
    return ScreenshotStrategy()


@register_strategy("screenshot_gray")
def _gray() -> ScreenshotStrategy:
    return ScreenshotStrategy(grayscale=True)


@register_strategy("screenshot_50pct")
def _half() -> ScreenshotStrategy:
    return ScreenshotStrategy(scale=0.5)


@register_strategy("screenshot_50pct_gray")
def _half_gray() -> ScreenshotStrategy:
    return ScreenshotStrategy(scale=0.5, grayscale=True)
