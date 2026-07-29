"""Schematic of the method under test, and of where it breaks.

A reader arriving at this application needs to know in about ten seconds what
the method claims to do and what this analysis found. A table of nine claims
does not achieve that; a diagram of the measurement chain does.

The chain is drawn as it is *claimed* to work, with the nuisance channels shown
entering at the point where they actually enter - the loading, upstream of the
strain ratio - because that is the whole argument. The crack signal and the
environmental noise arrive at the same node by the same path, so no amount of
downstream processing separates them.

Plain inline SVG: no external assets, scales cleanly, and readable in both a
browser and a printed page.
"""

from __future__ import annotations

__all__ = ["method_chain_svg"]

INK = "#14181c"
DIM = "#5b646d"
LINE = "#c9ced4"
ACCENT = "#1a5fb4"
WARN = "#b3261e"
GOOD = "#1a7f43"


def _box(x, y, w, h, title, sub, colour=INK, fill="#ffffff", bold=False):
    weight = "660" if bold else "560"
    return f"""
  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4"
        fill="{fill}" stroke="{colour}" stroke-width="{2 if bold else 1.2}"/>
  <text x="{x + w / 2}" y="{y + h / 2 - 3}" text-anchor="middle"
        font-size="12.5" font-weight="{weight}" fill="{INK}">{title}</text>
  <text x="{x + w / 2}" y="{y + h / 2 + 13}" text-anchor="middle"
        font-size="10.5" fill="{DIM}">{sub}</text>"""


def _arrow(x1, y1, x2, y2, colour=LINE, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'\n  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" '
            f'stroke-width="1.6" marker-end="url(#ah)"{d}/>')


def method_chain_svg(
    nuisance_pct: float | None = None,
    signature_pct: float = 11.1,
    verdict: str = "",
) -> str:
    """The measurement chain, annotated with this run's numbers where known."""
    n_txt = f"{nuisance_pct:.1f}%" if nuisance_pct is not None else "—"
    ratio_txt = (
        f"{nuisance_pct / signature_pct:.2f}x the signal"
        if nuisance_pct is not None and signature_pct
        else "not yet computed"
    )
    bad = nuisance_pct is not None and nuisance_pct > signature_pct / 3.0
    verdict_colour = WARN if bad else GOOD

    return f"""
<svg viewBox="0 0 1000 330" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;height:auto;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
  <defs>
    <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6"
            markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{LINE}"/>
    </marker>
  </defs>

  <text x="0" y="14" font-size="11" font-weight="700" fill="{DIM}"
        letter-spacing="1.2">THE METHOD UNDER TEST</text>

  {_box(0, 30, 150, 54, "Tidal current", "M2, rotary ellipse", ACCENT)}
  {_arrow(150, 57, 186, 57)}
  {_box(188, 30, 150, 54, "Morison drag", "force ∝ speed²", ACCENT)}
  {_arrow(338, 57, 374, 57)}
  {_box(376, 30, 150, 54, "Jacket bends", "3D frame solve", ACCENT)}
  {_arrow(526, 57, 562, 57)}
  {_box(564, 30, 168, 54, "Two FBG gauges", "bracketing the K-joint", ACCENT)}
  {_arrow(732, 57, 768, 57)}
  {_box(770, 30, 150, 54, "Strain ratio", "lower ÷ upper", ACCENT, bold=True)}

  <text x="0" y="126" font-size="11" font-weight="700" fill="{DIM}"
        letter-spacing="1.2">WHAT IS SUPPOSED TO MOVE IT</text>
  {_box(770, 140, 150, 50, "Crack at the joint", f"claimed {signature_pct:.1f}%", GOOD)}
  <line x1="845" y1="140" x2="845" y2="88" stroke="{GOOD}" stroke-width="1.6"
        marker-end="url(#ah)"/>

  <text x="0" y="228" font-size="11" font-weight="700" fill="{WARN}"
        letter-spacing="1.2">WHAT ALSO MOVES IT — THE PROBLEM</text>
  {_box(0, 242, 150, 50, "Current direction", "rotary ellipse", WARN, "#fdf6f5")}
  {_box(158, 242, 150, 50, "Spring / neap", "range modulation", WARN, "#fdf6f5")}
  {_box(316, 242, 150, 50, "Wind + waves", "storm-driven", WARN, "#fdf6f5")}
  {_box(474, 242, 150, 50, "Growth + scour", "over 20 years", WARN, "#fdf6f5")}
  {_box(632, 242, 150, 50, "Gauge drift", "differential", WARN, "#fdf6f5")}

  <line x1="75" y1="242" x2="75" y2="90" stroke="{WARN}" stroke-width="1.4"
        stroke-dasharray="4 3" marker-end="url(#ah)"/>
  <line x1="233" y1="242" x2="233" y2="90" stroke="{WARN}" stroke-width="1.4"
        stroke-dasharray="4 3" marker-end="url(#ah)"/>
  <line x1="391" y1="242" x2="391" y2="90" stroke="{WARN}" stroke-width="1.4"
        stroke-dasharray="4 3" marker-end="url(#ah)"/>
  <line x1="549" y1="242" x2="549" y2="90" stroke="{WARN}" stroke-width="1.4"
        stroke-dasharray="4 3" marker-end="url(#ah)"/>
  <line x1="707" y1="242" x2="707" y2="90" stroke="{WARN}" stroke-width="1.4"
        stroke-dasharray="4 3" marker-end="url(#ah)"/>

  <rect x="800" y="212" width="200" height="96" rx="4" fill="#ffffff"
        stroke="{verdict_colour}" stroke-width="2"/>
  <text x="900" y="236" text-anchor="middle" font-size="10.5" fill="{DIM}">
    measured nuisance
  </text>
  <text x="900" y="266" text-anchor="middle" font-size="26" font-weight="660"
        fill="{verdict_colour}" font-family="ui-monospace,Menlo,Consolas,monospace">
    {n_txt}
  </text>
  <text x="900" y="286" text-anchor="middle" font-size="11" fill="{INK}">
    {ratio_txt}
  </text>
  <text x="900" y="302" text-anchor="middle" font-size="10.5" font-weight="700"
        fill="{verdict_colour}">{verdict}</text>

  <text x="0" y="322" font-size="10.5" fill="{DIM}">
    The nuisance channels enter the same node as the crack, by the same path — so no
    downstream processing separates them.
  </text>
</svg>"""
