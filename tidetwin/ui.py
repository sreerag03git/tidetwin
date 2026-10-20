"""Shared UI components. The provenance chip is non-negotiable and lives here.

Every displayed quantity in the application goes through :func:`quantity`, so
there is exactly one place where a number can reach the screen, and it cannot do
so without a provenance class attached.

Visual design
-------------
A white base with a near-black text colour and a single blue accent. Body text
is set in the reader's system UI font, which is the most legible face available
on any machine and needs no download; only *numerals* are set in monospace, with
tabular figures, so that digits align down a column and a value can be compared
against the one above it at a glance. That split is deliberate: prose wants a
proportional face, engineering quantities want a fixed one.

Colour never carries meaning alone. Every status and provenance class is spelled
out in words beside its chip, and the palette clears WCAG AA contrast on white.
"""

from __future__ import annotations

import traceback
from contextlib import contextmanager

import plotly.graph_objects as go
import streamlit as st

from .claims.registry import Status
from .provenance import Provenance, Quantity

__all__ = [
    "inject_theme",
    "loading_screen",
    "running_reporter",
    "masthead",
    "cover",
    "svg_figure",
    "static_export_status",
    "quantity",
    "panel",
    "status_chip",
    "figure_block",
    "unavailable_panel",
    "verdict_block",
    "section",
    "provenance_legend",
    "dataframe",
    "MONO",
    "SANS",
    "PLOTLY_TEMPLATE",
]

SANS = (
    '-apple-system,BlinkMacSystemFont,"Segoe UI","Helvetica Neue",'
    'Arial,"Noto Sans",sans-serif'
)
MONO = '"SF Mono",Menlo,Consolas,"Liberation Mono","Courier New",monospace'

INK = "#14181c"
DIM = "#5b646d"
LINE = "#e1e5e9"
PANEL = "#f7f8fa"
ACCENT = "#1a5fb4"
GOOD = "#1a7f43"  # the PASS green, reused by the running panel's completed stages

_CSS = f"""
<style>
:root {{
  --tt-bg:#ffffff; --tt-panel:{PANEL}; --tt-line:{LINE};
  --tt-ink:{INK}; --tt-dim:{DIM}; --tt-accent:{ACCENT};
}}

.stApp, [data-testid="stAppViewContainer"] {{ background:var(--tt-bg); }}
html, body, [class*="css"], .stApp, p, li, span, label, div {{ color:var(--tt-ink); }}
html, body, .stApp {{ font-family:{SANS}; font-size:16px; line-height:1.55; }}

/* Numerals only: fixed width so digits line up down a column. */
code, pre, .tt-num {{
  font-family:{MONO}; font-variant-numeric:tabular-nums lining-nums;
}}

h1,h2,h3,h4,h5 {{ color:var(--tt-ink); font-family:{SANS}; letter-spacing:-.01em; }}
h1 {{ font-size:1.75rem; font-weight:650; }}
h2 {{ font-size:1.3rem; font-weight:640; }}
h3 {{ font-size:1.08rem; font-weight:640; }}
h4 {{ font-size:.98rem; font-weight:640; }}
hr {{ border:0; border-top:1px solid var(--tt-line); margin:1.4rem 0; }}
a {{ color:var(--tt-accent); }}

/* ---------- cover ---------- */
.tt-cover {{ padding:.2rem 0 1.4rem; }}
.tt-cover .mark {{
  font-family:{SANS}; font-size:2.6rem; font-weight:700; letter-spacing:-.03em;
  color:var(--tt-ink); line-height:1;
}}
.tt-cover .mark .dot {{ color:var(--tt-accent); }}
.tt-cover .tag {{
  font-family:{SANS}; font-size:.76rem; font-weight:600; letter-spacing:.18em;
  text-transform:uppercase; color:var(--tt-dim); margin-top:.55rem;
}}
.tt-cover .lead {{
  font-size:1.02rem; line-height:1.65; color:var(--tt-ink);
  max-width:62rem; margin-top:.9rem;
}}
.tt-creds {{ display:flex; flex-wrap:wrap; gap:.4rem; margin:1.1rem 0 .2rem; }}
.tt-creds span {{
  font-family:{MONO}; font-size:.68rem; color:var(--tt-dim); background:#fff;
  border:1px solid var(--tt-line); border-radius:2px; padding:.24rem .5rem;
  white-space:nowrap;
}}
.tt-chain {{
  display:grid; gap:0; margin:1.4rem 0 .4rem;
  grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  border-top:1px solid var(--tt-line); border-bottom:1px solid var(--tt-line);
}}
.tt-chain .step {{ padding:.85rem 1rem .9rem 1rem; border-left:1px solid var(--tt-line); }}
.tt-chain .step:first-child {{ border-left:0; padding-left:0; }}
.tt-chain .n {{
  font-family:{MONO}; font-size:.64rem; letter-spacing:.14em; color:var(--tt-accent);
  display:block; margin-bottom:.3rem;
}}
.tt-chain .h {{ font-size:.84rem; font-weight:600; color:var(--tt-ink); display:block; }}
.tt-chain .d {{ font-size:.76rem; line-height:1.5; color:var(--tt-dim); display:block; margin-top:.22rem; }}
.tt-note {{
  font-size:.76rem; color:var(--tt-dim); margin-top:.9rem;
  padding-left:.7rem; border-left:2px solid var(--tt-line);
}}
@media (max-width:640px) {{
  .tt-cover .mark {{ font-size:2rem; }}
  .tt-chain .step {{ border-left:0; padding-left:0; }}
}}
/* ---------- masthead ---------- */
.tt-head {{
  display:flex; align-items:baseline; gap:.85rem; flex-wrap:wrap;
  padding-bottom:.7rem; margin-bottom:1rem; border-bottom:2px solid var(--tt-ink);
}}
.tt-head .name {{ font-size:1.5rem; font-weight:680; letter-spacing:-.02em; }}
.tt-head .tag {{ color:var(--tt-dim); font-size:.92rem; }}

/* ---------- quantity row ---------- */
.tt-q {{
  display:flex; align-items:baseline; gap:.7rem; flex-wrap:wrap;
  padding:.5rem .1rem; border-bottom:1px solid var(--tt-line);
}}
.tt-q-label {{ color:var(--tt-dim); font-size:.9rem; min-width:14rem; flex:1 1 14rem; }}
.tt-q-value {{
  font-family:{MONO}; font-variant-numeric:tabular-nums lining-nums;
  font-size:1.05rem; font-weight:600; color:var(--tt-ink);
}}
.tt-chip {{
  font-family:{SANS}; font-size:.68rem; font-weight:700; letter-spacing:.055em;
  padding:.14rem .45rem; border-radius:3px; border:1px solid currentColor;
  text-transform:uppercase; white-space:nowrap; background:#fff;
}}

/* ---------- status ---------- */
.tt-status {{
  font-family:{SANS}; font-size:.8rem; font-weight:700; letter-spacing:.05em;
  padding:.26rem .62rem; border-radius:3px; border:1.5px solid currentColor;
  display:inline-block; white-space:nowrap; background:#fff;
}}

/* ---------- verdict banner ---------- */
.tt-verdict {{
  border:1px solid var(--tt-line); border-left:5px solid;
  background:var(--tt-panel); padding:1rem 1.2rem; margin:.3rem 0 1.3rem;
  border-radius:4px;
}}
.tt-verdict .vh {{
  display:flex; align-items:center; gap:.7rem; flex-wrap:wrap; margin-bottom:.45rem;
}}
.tt-verdict .vt {{ font-size:1.05rem; font-weight:670; }}
.tt-verdict p {{ margin:0; font-size:.94rem; line-height:1.6; color:var(--tt-ink); }}

/* ---------- headline finding ---------- */
.tt-hero {{
  border:1px solid var(--tt-line); border-top:4px solid; background:#fff;
  padding:1.35rem 1.5rem 1.2rem; margin:.2rem 0 1.4rem; border-radius:5px;
  box-shadow:0 1px 3px rgba(20,24,28,.05);
}}
.tt-hero .eyebrow {{
  font-family:{MONO}; font-size:.7rem; letter-spacing:.12em; text-transform:uppercase;
  color:var(--tt-dim); margin-bottom:.5rem;
}}
.tt-hero .headline {{
  font-size:1.32rem; font-weight:660; line-height:1.35; letter-spacing:-.015em;
  margin-bottom:.7rem; max-width:62rem;
}}
.tt-hero .facts {{
  display:flex; gap:2.2rem; flex-wrap:wrap; margin:.9rem 0 .8rem;
  padding:.85rem 0; border-top:1px solid var(--tt-line);
  border-bottom:1px solid var(--tt-line);
}}
.tt-hero .fact .k {{
  font-family:{MONO}; font-size:1.42rem; font-weight:650;
  font-variant-numeric:tabular-nums; line-height:1.1;
}}
.tt-hero .fact .l {{ color:var(--tt-dim); font-size:.76rem; margin-top:.2rem; }}
.tt-hero .body {{ font-size:.92rem; line-height:1.62; color:var(--tt-ink); max-width:62rem; }}

/* ---------- claims strip ---------- */
.tt-strip {{ display:flex; gap:.4rem; flex-wrap:wrap; margin:.2rem 0 1.1rem; }}
.tt-strip .c {{
  flex:1 1 5.2rem; min-width:5.2rem; border:1px solid var(--tt-line);
  border-top:3px solid; border-radius:4px; padding:.5rem .55rem; background:#fff;
}}
.tt-strip .c .id {{ font-family:{MONO}; font-weight:700; font-size:.9rem; }}
.tt-strip .c .v {{
  font-size:.66rem; font-weight:700; letter-spacing:.04em; margin-top:.15rem;
  text-transform:uppercase; line-height:1.25;
}}
.tt-strip .c .n {{
  font-family:{MONO}; font-size:.68rem; color:var(--tt-dim); margin-top:.25rem;
  font-variant-numeric:tabular-nums; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap;
}}

/* ---------- data-unavailable panel ---------- */
.tt-unavail {{
  border:1px solid #c9ced4; border-left:5px solid var(--tt-dim);
  background:var(--tt-panel); padding:.85rem 1.05rem; border-radius:4px;
  font-size:.9rem; color:var(--tt-ink); margin:.5rem 0;
}}
.tt-unavail b {{ color:var(--tt-ink); }}

/* ---------- section heading ---------- */
.tt-section {{
  display:flex; align-items:baseline; gap:.7rem; flex-wrap:wrap;
  margin:1.6rem 0 .3rem; padding-bottom:.35rem; border-bottom:1px solid var(--tt-line);
}}
.tt-section .t {{ font-size:1.12rem; font-weight:650; }}
.tt-section .s {{ color:var(--tt-dim); font-size:.86rem; }}

/* ---------- legend ---------- */
.tt-legend {{ display:flex; gap:.5rem; flex-wrap:wrap; margin:.3rem 0 .9rem; }}
.tt-legend .item {{ display:flex; align-items:center; gap:.3rem; font-size:.8rem;
  color:var(--tt-dim); }}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap:.15rem; border-bottom:1px solid var(--tt-line); }}
.stTabs [data-baseweb="tab"] {{
  font-family:{SANS}; font-size:.88rem; font-weight:600; letter-spacing:.01em;
  text-transform:none; background:transparent; color:var(--tt-dim);
  padding:.55rem .9rem;
}}
.stTabs [aria-selected="true"] {{ color:var(--tt-ink); }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{ background:var(--tt-panel); border-right:1px solid var(--tt-line); }}
[data-testid="stSidebar"] .stSlider label, [data-testid="stSidebar"] label {{ font-size:.86rem; }}
[data-testid="stMetricValue"] {{
  font-family:{MONO}; font-variant-numeric:tabular-nums; font-weight:650;
}}
[data-testid="stMetricLabel"] {{ color:var(--tt-dim); }}

/* Tables read better with fixed-width numerals. */
[data-testid="stDataFrame"] {{ font-variant-numeric:tabular-nums; }}

.stButton button, .stDownloadButton button {{ font-family:{SANS}; font-weight:600; }}

/* ---------- polish ---------- */
/* Cards lift slightly on hover, so the strip reads as a set of objects rather
   than a flat wall. Kept subtle; motion is disabled for reduced-motion users. */
.tt-strip .c, .tt-hero {{ transition:box-shadow .15s ease, transform .15s ease; }}
.tt-strip .c:hover {{
  box-shadow:0 3px 10px rgba(20,24,28,.09); transform:translateY(-1px);
}}
.tt-cover .tt-creds span {{ transition:border-color .15s ease, color .15s ease; }}
.tt-cover .tt-creds span:hover {{ border-color:var(--tt-accent); color:var(--tt-accent); }}

/* Active tab carries the accent as an underline, not just a colour shift. */
.stTabs [data-baseweb="tab"] {{ border-bottom:2px solid transparent; margin-bottom:-1px; }}
.stTabs [aria-selected="true"] {{ border-bottom-color:var(--tt-accent); font-weight:670; }}
.stTabs [data-baseweb="tab"]:hover {{ color:var(--tt-ink); }}

/* Primary button in the brand accent; a clear affordance for Run. */
.stButton button[kind="primary"], .stButton button[data-testid="baseButton-primary"] {{
  background:var(--tt-accent); border-color:var(--tt-accent); color:#fff;
}}
.stButton button[kind="primary"]:hover {{ background:#17539f; border-color:#17539f; }}
.stButton button {{ transition:background .12s ease, border-color .12s ease; }}

/* Section headings get a short accent tick before the title. */
.tt-section .t::before {{
  content:""; display:inline-block; width:.5rem; height:.5rem; margin-right:.5rem;
  background:var(--tt-accent); border-radius:1px; vertical-align:.08em;
}}

/* Metric row reads as cards, lightly bounded. */
[data-testid="stMetric"] {{
  background:#fff; border:1px solid var(--tt-line); border-radius:5px;
  padding:.55rem .8rem;
}}

@media (prefers-reduced-motion: reduce) {{
  .tt-strip .c, .tt-hero, .stButton button {{ transition:none; }}
  .tt-strip .c:hover {{ transform:none; }}
}}
</style>
"""

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family=SANS, size=13, color=INK),
        title=dict(font=dict(family=SANS, size=15, color=INK)),
        # Colour-blind-safe ordering: blue, vermillion, amber, green, grey, purple.
        colorway=["#1a5fb4", "#b3261e", "#8a5300", "#1a7f43", "#5b646d", "#6a4bab"],
        xaxis=dict(
            gridcolor="#eceff2", zerolinecolor="#c9ced4", linecolor="#9aa3ab",
            ticks="outside", tickcolor="#9aa3ab", tickfont=dict(family=MONO, size=12),
            title=dict(font=dict(family=SANS, size=13)),
        ),
        yaxis=dict(
            gridcolor="#eceff2", zerolinecolor="#c9ced4", linecolor="#9aa3ab",
            ticks="outside", tickcolor="#9aa3ab", tickfont=dict(family=MONO, size=12),
            title=dict(font=dict(family=SANS, size=13)),
        ),
        legend=dict(bgcolor="rgba(255,255,255,.85)", bordercolor=LINE, borderwidth=1),
        hovermode="x unified",
        hoverlabel=dict(font=dict(family=MONO, size=12), bgcolor="#ffffff",
                        bordercolor="#9aa3ab"),
        margin=dict(l=70, r=26, t=48, b=56),
    )
)


def inject_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


_BOOT_CSS = f"""
<style>
@keyframes ttslide {{
  0% {{ transform:translateX(-100%); }}
  100% {{ transform:translateX(320%); }}
}}
@keyframes ttrise {{
  from {{ opacity:0; transform:translateY(8px); }}
  to   {{ opacity:1; transform:translateY(0); }}
}}
@keyframes ttdraw {{ from {{ width:0; }} to {{ width:3.6rem; }} }}
@keyframes ttpulse {{
  0%,100% {{ opacity:1; }} 50% {{ opacity:.55; }}
}}
@keyframes ttripple {{
  0% {{ transform:scale(.6); opacity:.55; }}
  100% {{ transform:scale(2.4); opacity:0; }}
}}
.tt-boot {{
  min-height: 66vh; display:flex; flex-direction:column;
  align-items:center; justify-content:center; text-align:center;
  font-family:{SANS}; color:{INK};
}}
.tt-boot .glyph {{
  position:relative; width:44px; height:44px; margin-bottom:1rem;
}}
.tt-boot .glyph .dot {{
  position:absolute; inset:0; margin:auto; width:12px; height:12px; border-radius:50%;
  background:{ACCENT}; animation:ttpulse 1.6s ease-in-out infinite;
}}
.tt-boot .glyph .ring {{
  position:absolute; inset:0; margin:auto; width:16px; height:16px; border-radius:50%;
  border:2px solid {ACCENT};
}}
.tt-boot .glyph .ring.r1 {{ animation:ttripple 1.9s ease-out infinite; }}
.tt-boot .glyph .ring.r2 {{ animation:ttripple 1.9s ease-out .95s infinite; }}
.tt-boot .mark {{
  font-size:2.15rem; font-weight:680; letter-spacing:-.03em; margin-bottom:.2rem;
  animation:ttrise .55s ease both;
}}
.tt-boot .mark .dot {{ color:{ACCENT}; }}
.tt-boot .rule {{
  width:3.6rem; height:2px; background:{ACCENT}; margin:.6rem 0 .95rem;
  animation:ttdraw .6s .15s ease both;
}}
.tt-boot .say {{
  color:{DIM}; font-size:1rem; max-width:34rem; line-height:1.65;
  animation:ttrise .55s .12s ease both;
}}
.tt-boot .beats {{
  display:flex; gap:2.4rem; flex-wrap:wrap; justify-content:center;
  margin:2rem 0 .4rem; max-width:46rem;
}}
.tt-boot .beat {{ flex:1 1 11rem; max-width:13rem; text-align:left; animation:ttrise .6s ease both; }}
.tt-boot .beat:nth-child(1) {{ animation-delay:.20s; }}
.tt-boot .beat:nth-child(2) {{ animation-delay:.34s; }}
.tt-boot .beat:nth-child(3) {{ animation-delay:.48s; }}
.tt-boot .beat .n {{
  font-family:{MONO}; font-size:.7rem; color:{ACCENT}; letter-spacing:.1em;
  margin-bottom:.3rem;
}}
.tt-boot .beat .h {{ font-size:.88rem; font-weight:640; margin-bottom:.2rem; }}
.tt-boot .beat .d {{ font-size:.8rem; color:{DIM}; line-height:1.55; }}
.tt-boot .step {{
  font-family:{MONO}; font-size:.8rem; color:{ACCENT};
  margin-top:1.8rem; letter-spacing:.02em; animation:ttrise .6s .55s ease both;
}}
.tt-boot .bar {{
  width:min(22rem, 70vw); height:3px; border-radius:2px; background:{LINE};
  margin-top:.7rem; overflow:hidden; position:relative;
}}
.tt-boot .bar::after {{
  content:""; position:absolute; inset:0; width:35%; border-radius:2px;
  background:linear-gradient(90deg,{ACCENT},#3b82d4);
  animation:ttslide 1.15s ease-in-out infinite;
}}
@media (prefers-reduced-motion: reduce) {{
  .tt-boot .bar::after {{ animation:none; width:100%; opacity:.35; }}
  .tt-boot .glyph .ring {{ animation:none; opacity:.3; }}
  .tt-boot .glyph .dot {{ animation:none; }}
  .tt-boot .mark, .tt-boot .rule, .tt-boot .say, .tt-boot .beat, .tt-boot .step {{
    animation:none;
  }}
}}
</style>
"""


def loading_screen(step: str = "Assembling the jacket and solving the frame") -> str:
    """Markup for the first-load screen.

    Shown while the frame is assembled and solved, which is a real few seconds of
    sparse linear algebra rather than an artificial delay. It names what it is
    doing, so the wait is legible rather than a blank page.
    """
    beats = (
        ("01", "The claim", "A crack in an offshore jacket shifts the tidal strain "
                            "ratio between two gauges by 11.1 percent."),
        ("02", "The test", "Solve the OC4 jacket under real measured tides, then ask how "
                           "much the sea moves that ratio with no crack at all."),
        ("03", "The rule", "Every number is computed here and carries its provenance. "
                           "Nothing is asserted."),
    )
    cards = "".join(
        f'<div class="beat"><div class="n">{n}</div><div class="h">{h}</div>'
        f'<div class="d">{d}</div></div>'
        for n, h, d in beats
    )
    return _BOOT_CSS + (
        '<div class="tt-boot">'
        '<div class="glyph"><span class="ring r1"></span>'
        '<span class="ring r2"></span><span class="dot"></span></div>'
        '<div class="mark">Tide<span class="dot">·</span>Twin</div>'
        '<div class="rule"></div>'
        '<div class="say">An adversarial test bench for a fatigue digital twin. '
        "Built to falsify the method, not to demonstrate it.</div>"
        f'<div class="beats">{cards}</div>'
        f'<div class="step">{step}</div>'
        '<div class="bar"></div>'
        "</div>"
    )


_RUN_CSS = f"""
<style>
.tt-run {{
  border:1px solid var(--tt-line); border-top:3px solid var(--tt-accent);
  background:var(--tt-panel); border-radius:6px; padding:1.5rem 1.6rem 1.7rem;
  margin:.2rem 0 1.2rem; font-family:{SANS};
}}
.tt-run .mark {{ font-size:1.15rem; font-weight:660; letter-spacing:-.01em; }}
.tt-run .sub {{ color:{DIM}; font-size:.82rem; margin-top:.2rem; }}
.tt-run .track {{
  height:8px; border-radius:5px; background:{LINE}; overflow:hidden; margin:1.1rem 0 .6rem;
}}
.tt-run .fill {{
  height:100%; border-radius:5px; background:linear-gradient(90deg,{ACCENT},#3b82d4);
  transition:width .35s ease;
}}
.tt-run .msg {{
  font-family:{MONO}; font-size:.82rem; color:var(--tt-ink);
  font-variant-numeric:tabular-nums; display:flex; justify-content:space-between; gap:1rem;
}}
.tt-run .msg .pct {{ color:{ACCENT}; font-weight:600; }}
.tt-run .stages {{
  display:flex; flex-wrap:wrap; gap:.4rem; margin-top:1rem;
}}
.tt-run .stage {{
  font-family:{MONO}; font-size:.66rem; letter-spacing:.02em; padding:.22rem .5rem;
  border:1px solid var(--tt-line); border-radius:3px; color:{DIM}; background:#fff;
}}
.tt-run .stage.done {{ color:{GOOD}; border-color:{GOOD}; }}
.tt-run .stage.active {{ color:{ACCENT}; border-color:{ACCENT}; font-weight:600; }}
</style>
"""

#: The pipeline stages, in the order run_full reports them, with the fraction at
#: which each becomes active. Used to light up the running panel.
_RUN_STAGES: tuple[tuple[float, str], ...] = (
    (0.00, "Frame"),
    (0.05, "C1·C2"),
    (0.30, "C3"),
    (0.60, "C4"),
    (0.75, "C6"),
    (0.88, "C8·C9"),
    (1.00, "Ledger"),
)


def running_reporter(placeholder):
    """A progress callback that renders a full-width 'running' panel.

    run_full reports ``(fraction, message)`` at each claim. This turns that into
    a live panel with a filling bar, the current step, and the pipeline stages
    lighting up as they complete - so pressing Run shows the machine working
    rather than a bare grey progress line.
    """

    def report(frac: float, msg: str) -> None:
        f = float(min(max(frac, 0.0), 1.0))
        pct = int(round(f * 100))
        chips = []
        for i, (start, name) in enumerate(_RUN_STAGES):
            nxt = _RUN_STAGES[i + 1][0] if i + 1 < len(_RUN_STAGES) else 1.01
            cls = "done" if f >= nxt else ("active" if f >= start else "")
            chips.append(f'<span class="stage {cls}">{name}</span>')
        placeholder.markdown(
            _RUN_CSS
            + '<div class="tt-run">'
            '<div class="mark">Running the analysis</div>'
            '<div class="sub">Nine claims &middot; a Monte Carlo over eight nuisance '
            "channels &middot; a few hundred frame solves</div>"
            f'<div class="track"><div class="fill" style="width:{pct}%"></div></div>'
            f'<div class="msg"><span>{msg}</span><span class="pct">{pct}%</span></div>'
            f'<div class="stages">{"".join(chips)}</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    return report


def hero_result(
    eyebrow: str, headline: str, body: str, colour: str, facts: list[tuple[str, str]]
) -> None:
    """The application's headline finding, presented as a result rather than an alarm.

    The verdict is not softened - a FAIL says so, in those words, with the
    numbers that produced it. But it is styled as a paper's key result, because a
    red error banner reads as "this software is broken" rather than "this is what
    the analysis found", and that misrepresents the work.
    """
    cells = "".join(
        f'<div class="fact"><div class="k" style="color:{colour}">{k}</div>'
        f'<div class="l">{lab}</div></div>'
        for k, lab in facts
    )
    st.markdown(
        f'<div class="tt-hero" style="border-top-color:{colour}">'
        f'<div class="eyebrow">{eyebrow}</div>'
        f'<div class="headline">{headline}</div>'
        + (f'<div class="facts">{cells}</div>' if facts else "")
        + f'<div class="body">{body}</div></div>',
        unsafe_allow_html=True,
    )


def claims_strip(rows: list[tuple[str, str, str, str]]) -> None:
    """Nine claims at a glance: id, status, colour, one-line computed value."""
    cards = "".join(
        f'<div class="c" style="border-top-color:{colour}">'
        f'<div class="id">{cid}</div>'
        f'<div class="v" style="color:{colour}">{status}</div>'
        f'<div class="n" title="{value}">{value}</div></div>'
        for cid, status, colour, value in rows
    )
    st.markdown(f'<div class="tt-strip">{cards}</div>', unsafe_allow_html=True)


def masthead(subtitle: str) -> None:
    st.markdown(
        f'<div class="tt-head"><span class="name">TideTwin</span>'
        f'<span class="tag">{subtitle}</span></div>',
        unsafe_allow_html=True,
    )



def svg_figure(svg: str, height: int = 320) -> None:
    """Render an inline SVG reliably, inside a sandboxed iframe.

    st.markdown(..., unsafe_allow_html=True) runs its HTML through a sanitizer
    that strips <svg> and its shape children while keeping the <text> nodes.
    That is why the method diagram showed as a tall blank gap with its labels
    spilling out as loose text: the drawing was removed and only the words
    survived. st.iframe embeds a raw HTML document in a sandboxed iframe with no
    sanitiser, so the diagram actually draws. It supersedes the deprecated
    st.components.v1.html (Streamlit removes that after 2026-06-01); st.iframe
    treats a string that is not a URL or file path as raw HTML and defaults to
    width="stretch", so it fills the column exactly as the old call did.

    The SVG keeps its own ``width:100%;height:auto`` styling; this only ensures
    it preserves its aspect ratio and sits on a transparent, margin-free page so
    it lines up with the Streamlit column above and below it. ``overflow:hidden``
    stands in for the old ``scrolling=False``: the frame is a fixed height and
    must not sprout a scrollbar if the content rounds a pixel over it.
    """
    svg_tag = svg
    if "preserveAspectRatio" not in svg_tag:
        svg_tag = svg_tag.replace("<svg ", '<svg preserveAspectRatio="xMidYMid meet" ', 1)
    doc = (
        "<!doctype html><meta charset='utf-8'>"
        "<style>html,body{margin:0;padding:0;overflow:hidden;background:transparent;"
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}</style>"
        f"<body>{svg_tag}</body>"
    )
    st.iframe(doc, height=height)


def cover(
    tagline: str,
    lead: str,
    credentials,
    chain,
    note: str = "",
) -> None:
    """The cover block at the top of the landing page.

    A reader arriving here has to learn three things before any number means
    anything: what this is, what it is built on, and how it gets from a tide to a
    verdict. The page used to open on a claims table and nothing else, which told
    a first-time reader none of them.

    ``credentials`` are the sources and methods actually in the code, so the row
    is a claim that can be checked against the Provenance tab rather than
    decoration. ``chain`` is ``(heading, description)`` per step, numbered here.
    """
    creds = "".join(f"<span>{c}</span>" for c in credentials)
    steps = "".join(
        f'<div class="step"><span class="n">{i + 1:02d}</span>'
        f'<span class="h">{h}</span><span class="d">{d}</span></div>'
        for i, (h, d) in enumerate(chain)
    )
    st.markdown(
        f'<div class="tt-cover">'
        f'<div class="mark">Tide<span class="dot">·</span>Twin</div>'
        f'<div class="tag">{tagline}</div>'
        f'<div class="lead">{lead}</div>'
        f'<div class="tt-creds">{creds}</div>'
        f'<div class="tt-chain">{steps}</div>'
        + (f'<div class="tt-note">{note}</div>' if note else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="tt-section"><span class="t">{title}</span>'
        f'<span class="s">{subtitle}</span></div>',
        unsafe_allow_html=True,
    )


def provenance_legend() -> None:
    """Explain the chips once, where they are first used."""
    items = "".join(
        f'<span class="item"><span class="tt-chip" style="color:{p.colour}">'
        f"{p.value}</span>{t}</span>"
        for p, t in (
            (Provenance.MEASURED, "real dataset"),
            (Provenance.PUBLISHED, "standard or paper"),
            (Provenance.DERIVED, "our solver"),
            (Provenance.ASSUMED, "your input"),
        )
    )
    st.markdown(f'<div class="tt-legend">{items}</div>', unsafe_allow_html=True)


def quantity(q: Quantity, label: str | None = None, sig: int = 4) -> None:
    """Render a quantity with its provenance chip and expandable citation chain.

    This is the only sanctioned way to put a number on screen.
    """
    p = q.provenance
    name = label or q.name or "quantity"
    st.markdown(
        f'<div class="tt-q">'
        f'<span class="tt-q-label">{name}</span>'
        f'<span class="tt-q-value">{q.format(sig)}</span>'
        f'<span class="tt-chip" style="color:{p.colour}">{p.value}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Where this number comes from", expanded=False):
        if q.citation:
            st.markdown(f"**Source** &mdash; {q.citation}")
            if q.citation.url:
                st.markdown(f"<{q.citation.url}>")
        if q.operation:
            st.markdown(f"**Computed by** &mdash; {q.operation}")
        if q.note:
            st.caption(q.note)
        if q.contaminated:
            st.markdown(
                f'<span class="tt-chip" style="color:{Provenance.ASSUMED.colour}">'
                "RESTS ON ASSUMPTIONS</span>",
                unsafe_allow_html=True,
            )
            st.caption("Specifically: " + ", ".join(q.blocking_assumptions))
        chain = q.chain()[:-1]
        if chain:
            st.markdown("**Input chain**")
            for c in chain:
                st.markdown(
                    f'<div class="tt-q"><span class="tt-q-label">{c.name or "?"}</span>'
                    f'<span class="tt-q-value">{c.format()}</span>'
                    f'<span class="tt-chip" style="color:{c.provenance.colour}">'
                    f"{c.provenance.value}</span></div>",
                    unsafe_allow_html=True,
                )
                if c.citation:
                    st.caption(str(c.citation))


def status_chip(status: Status) -> str:
    """Inline HTML for a claim status. FAIL is styled exactly like PASS."""
    return f'<span class="tt-status" style="color:{status.colour}">{status.value}</span>'


def verdict_block(title: str, body: str, colour: str, status: str = "") -> None:
    chip = (
        f'<span class="tt-status" style="color:{colour}">{status}</span>' if status else ""
    )
    st.markdown(
        f'<div class="tt-verdict" style="border-left-color:{colour}">'
        f'<div class="vh"><span class="vt">{title}</span>{chip}</div>'
        f"<p>{body}</p></div>",
        unsafe_allow_html=True,
    )


def unavailable_panel(title: str, message: str, remedy: str = "") -> None:
    body = message.replace("\n", "<br>")
    if remedy:
        body += f"<br><br><b>How to fix it</b> &mdash; {remedy}"
    st.markdown(
        f'<div class="tt-unavail"><b>{title}</b><br>{body}</div>', unsafe_allow_html=True
    )


@st.cache_resource(show_spinner=False)
def static_export_status() -> tuple[bool, str]:
    """Can this deployment actually render a static image?

    Importing ``kaleido`` is not the question. Kaleido 1.x drives a headless
    Chrome, and Streamlit Cloud has no Chrome binary, so the import succeeds and
    ``to_image`` then raises at render time. This probes the real capability once
    by rendering a trivial figure, and caches the answer for the session.
    """
    import importlib.util
    import os

    if os.environ.get("TIDETWIN_NO_STATIC_EXPORT"):
        return False, "Static export disabled by TIDETWIN_NO_STATIC_EXPORT."
    # find_spec, not import: importing kaleido can trigger plotly's Chrome
    # discovery, which stalls on a host that has none. A feature nobody has asked
    # for yet must never delay a page load.
    if importlib.util.find_spec("kaleido") is None:
        return False, (
            "kaleido is not installed, which is deliberate for cloud deployment. "
            "Interactive HTML export is vector and loses no fidelity. For 300 dpi PNG "
            "and vector PDF, run locally with `pip install kaleido==1.3.0` then "
            "`plotly_get_chrome`."
        )
    try:
        go.Figure().to_image(format="png", width=8, height=8)
        return True, "Static image export is available."
    except Exception as exc:  # noqa: BLE001 - any failure means unavailable
        msg = str(exc).strip().splitlines()
        head = msg[0] if msg else type(exc).__name__
        return False, (
            "kaleido is installed but cannot render: it needs a Chrome binary, which "
            f"this host does not provide ({head[:120]}). Interactive HTML export still "
            "works and keeps full vector fidelity."
        )


def figure_block(fig: go.Figure, name: str, caption: str = "", height: int = 420) -> None:
    """Render a figure, with paper-ready export offered only when it can work.

    Static images are generated **on request**, never on page render. Generating
    a PNG and a PDF for every figure on every rerun is slow, and if the renderer
    is unavailable it takes down the whole tab - which is exactly what happened
    on Streamlit Cloud, where kaleido imports but has no Chrome to drive.
    """
    fig.update_layout(template=PLOTLY_TEMPLATE, height=height)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    if caption:
        st.caption(caption)

    with st.expander("Download this figure"):
        st.download_button(
            "Interactive HTML (vector, keeps full resolution)",
            fig.to_html(include_plotlyjs="cdn").encode(),
            file_name=f"{name}.html",
            mime="text/html",
            key=f"dl_html_{name}",
            width="stretch",
        )
        ok, why = static_export_status()
        if not ok:
            st.caption(f"PNG and PDF unavailable here. {why}")
            return
        if not st.checkbox(
            "Prepare 300 dpi PNG and vector PDF",
            key=f"dl_prep_{name}",
            help="Rendered on demand, because generating them for every figure on every "
            "rerun is slow.",
        ):
            return
        try:
            # 1005 px across a 3.35 in two-column figure is exactly 300 dpi.
            png = fig.to_image(format="png", width=1005, height=int(height * 2.4), scale=1)
            pdf = fig.to_image(format="pdf", width=241, height=int(height * 0.58))
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the tab
            st.caption(f"Static export failed for this figure: {type(exc).__name__}: {exc}")
            return
        cols = st.columns(2)
        with cols[0]:
            st.download_button(
                "PNG 300 dpi", png, file_name=f"{name}.png", mime="image/png",
                key=f"dl_png_{name}", width="stretch",
            )
        with cols[1]:
            st.download_button(
                "PDF (vector)", pdf, file_name=f"{name}.pdf", mime="application/pdf",
                key=f"dl_pdf_{name}", width="stretch",
            )


def dataframe(df, **kw) -> None:
    st.dataframe(df, width="stretch", hide_index=True, **kw)


@contextmanager
def panel(name: str):
    """Isolate a block so its failure cannot blank the rest of the page.

    Streamlit renders top to bottom and an uncaught exception stops the script,
    so one broken figure takes down every tab after it. That is how a missing
    Chrome binary turned into an app with no graphs at all. Each section now
    reports its own failure in place, with the traceback, and the rest still
    renders.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - the point is to contain it
        st.error(f"**{name}** could not be rendered: {type(exc).__name__}: {exc}")
        with st.expander(f"Traceback for {name}"):
            st.code("".join(traceback.format_exception(exc)), language="text")
