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
    "masthead",
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
.tt-boot {{
  min-height: 62vh; display:flex; flex-direction:column;
  align-items:center; justify-content:center; text-align:center;
  font-family:{SANS}; color:{INK};
}}
.tt-boot .mark {{
  font-size:2.1rem; font-weight:680; letter-spacing:-.03em; margin-bottom:.2rem;
}}
.tt-boot .rule {{
  width:3.2rem; height:2px; background:{INK}; margin:.55rem 0 .9rem;
}}
.tt-boot .say {{ color:{DIM}; font-size:.95rem; max-width:30rem; line-height:1.6; }}
.tt-boot .step {{
  font-family:{MONO}; font-size:.8rem; color:{ACCENT};
  margin-top:1.5rem; letter-spacing:.02em;
}}
.tt-boot .bar {{
  width:min(22rem, 70vw); height:2px; background:{LINE};
  margin-top:.7rem; overflow:hidden; position:relative;
}}
.tt-boot .bar::after {{
  content:""; position:absolute; inset:0; width:35%;
  background:{ACCENT}; animation:ttslide 1.15s ease-in-out infinite;
}}
@keyframes ttslide {{
  0% {{ transform:translateX(-100%); }}
  100% {{ transform:translateX(320%); }}
}}
@media (prefers-reduced-motion: reduce) {{
  .tt-boot .bar::after {{ animation:none; width:100%; opacity:.35; }}
}}
</style>
"""


def loading_screen(step: str = "Assembling the jacket and solving the frame") -> str:
    """Markup for the first-load screen.

    Shown while the frame is assembled and solved, which is a real few seconds of
    sparse linear algebra rather than an artificial delay. It names what it is
    doing, so the wait is legible rather than a blank page.
    """
    return _BOOT_CSS + (
        '<div class="tt-boot">'
        '<div class="mark">TideTwin</div>'
        '<div class="rule"></div>'
        '<div class="say">An adversarial test bench for the claims of a tidal-calibration '
        "fatigue digital twin. Every number is computed at runtime and carries its "
        "provenance.</div>"
        f'<div class="step">{step}</div>'
        '<div class="bar"></div>'
        "</div>"
    )


def masthead(subtitle: str) -> None:
    st.markdown(
        f'<div class="tt-head"><span class="name">TideTwin</span>'
        f'<span class="tag">{subtitle}</span></div>',
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
