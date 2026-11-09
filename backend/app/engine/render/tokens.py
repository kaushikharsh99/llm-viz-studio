# Vendored & modified from llmviz (Apache-2.0, h9-tec/llmviz); see THIRD_PARTY_NOTICES.md. Renamed to LLMViz Studio engine.
"""Raschka-style design tokens: fixed grayscale grammar + one accent color per model.

Values pixel-sampled from Sebastian Raschka's LLM Architecture Gallery figures
(Llama 3 8B, GPT-OSS 20B, DeepSeek V3/R1 671B).
"""

from __future__ import annotations

CANVAS = "#ffffff"
OUTER_GRAY = "#d5d5d5"  # outer model container
MID_GRAY = "#929292"  # dense-model block container (his Llama 3)
ATTN_DARK = "#5e5e5e"  # attention box, white text
INK = "#000000"
PILL = "#ffffff"
INSET_GRAY = "#cccccc"  # FeedForward module inset fill

# curated accents, sampled + extended in the same register
ACCENTS = {
    "blue": "#4d76ea",
    "coral": "#ff664f",
    "green": "#1e9e63",
    "violet": "#8b5cf6",
    "amber": "#e8961e",
    "teal": "#0f9b9b",
}
_ACCENT_ORDER = list(ACCENTS.values())

# family assignments chosen to match his figures where they exist
_FAMILY_ACCENT = {
    "gpt_oss": ACCENTS["blue"],
    "deepseek_v3": ACCENTS["coral"],
    "llama": ACCENTS["blue"],
    "gpt2": ACCENTS["blue"],
    "qwen3": ACCENTS["violet"],
    "qwen3_moe": ACCENTS["violet"],
    "gemma3": ACCENTS["teal"],
    "gemma3_text": ACCENTS["teal"],
    "mistral": ACCENTS["amber"],
}


def accent_for(model_type: str) -> str:
    if model_type in _FAMILY_ACCENT:
        return _FAMILY_ACCENT[model_type]
    return _ACCENT_ORDER[sum(model_type.encode()) % len(_ACCENT_ORDER)]


def block_tint(accent: str) -> str:
    """The repeated-block container fill: accent nudged ~33% toward white (sampled ratio)."""
    a = accent.lstrip("#")
    ch = [round(int(a[i : i + 2], 16) * 0.67 + 255 * 0.33) for i in (0, 2, 4)]
    return "#" + "".join(f"{c:02x}" for c in ch)


# Arial resolves to Liberation Sans on Linux (metric-compatible), real Arial/Helvetica
# elsewhere; cairosvg only honors the first family, which fontconfig aliases correctly.
FONT = 'Arial, "Helvetica Neue", "Liberation Sans", "DejaVu Sans", sans-serif'
MONO = '"Courier New", "Liberation Mono", "DejaVu Sans Mono", monospace'


def accent_on_dark(accent: str) -> str:
    """Accent lightened for legibility on the dark attention box."""
    a = accent.lstrip("#")
    ch = [round(int(a[i : i + 2], 16) * 0.55 + 255 * 0.45) for i in (0, 2, 4)]
    return "#" + "".join(f"{c:02x}" for c in ch)


ANIM_CSS = "@keyframes llmviz-studio-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }"


def stylesheet(accent: str) -> str:
    """Selectors are scoped under a per-accent id: SVG <style> is document-global when
    several figures are inlined into one HTML page — without scoping, the last figure's
    accent repaints all of them."""
    sc = f"#llmviz-studio-{accent.lstrip('#')}"
    css = _rules(accent)
    return "\n".join(
        (
            f"  {sc} {line.strip()}"
            if line.strip() and "{" in line and not line.strip().startswith(("}", "stroke", "fill"))
            else line
        )
        for line in css.splitlines()
    )


def scope_id(accent: str) -> str:
    return f"llmviz-studio-{accent.lstrip('#')}"


def _rules(accent: str) -> str:
    return f"""
  text {{ font-family: {FONT}; fill: {INK}; }}
  .mono {{ font-family: {MONO}; }}
  .accent {{ fill: {accent}; }}
  .white {{ fill: #ffffff; }}
  .surface {{ fill: {CANVAS}; }}
  .pill {{ fill: {PILL}; stroke: {INK}; stroke-width: 2.5; }}
  .outer {{ fill: {OUTER_GRAY}; stroke: {INK}; stroke-width: 2.5; }}
  .blockc {{ fill: {block_tint(accent)}; stroke: {INK}; stroke-width: 2.5; }}
  .attnbox {{ fill: {ATTN_DARK}; stroke: {INK}; stroke-width: 2.5; }}
  .inset {{ fill: {INSET_GRAY}; stroke: {INK}; stroke-width: 3; stroke-dasharray: 2 7;
            stroke-linecap: round; }}
  .inset-accent {{ fill: {CANVAS}; stroke: {accent}; stroke-width: 3.5;
                   stroke-dasharray: 2 8; stroke-linecap: round; }}
  .moepill {{ fill: {INSET_GRAY}; stroke: {INK}; stroke-width: 2.5; }}
  .router {{ fill: {PILL}; stroke: {accent}; stroke-width: 3.5; }}
  .tagdark {{ fill: {INK}; }}
  .tagaccent {{ fill: {accent}; }}
  .flow {{ stroke: {INK}; stroke-width: 2.5; fill: none; }}
  .flowhead {{ fill: {INK}; }}
  .leader {{ stroke: {INK}; stroke-width: 3; stroke-dasharray: 2 7; fill: none;
             stroke-linecap: round; }}
  .leader-accent {{ stroke: {accent}; stroke-width: 4; stroke-dasharray: 2 8; fill: none;
                    stroke-linecap: round; }}
  .plusc {{ fill: {PILL}; stroke: {INK}; stroke-width: 2.5; }}
  .accent-dark {{ fill: {accent_on_dark(accent)}; }}
"""
