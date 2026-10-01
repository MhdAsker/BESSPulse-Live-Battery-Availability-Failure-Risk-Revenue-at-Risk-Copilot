"""Central industrial-energy visual system."""

import streamlit as st

COLORS = {
    "background": "#0A0F14",
    "surface": "#111923",
    "surface_high": "#17222E",
    "border": "#263544",
    "text": "#E8EEF4",
    "muted": "#8FA1B3",
    "primary": "#63D48A",
    "secondary": "#4FC3D7",
    "warning": "#E8B04A",
    "critical": "#ED6A5A",
    "simulated": "#A990E8",
    "grid": "#263544",
}

PLOTLY_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": COLORS["text"], "family": "Inter, system-ui, sans-serif"},
    "margin": {"l": 40, "r": 20, "t": 48, "b": 40},
    "hoverlabel": {"bgcolor": COLORS["surface_high"], "font_color": COLORS["text"]},
    "legend": {"orientation": "h", "y": 1.12, "x": 0},
    "xaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["muted"]},
    "yaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["muted"]},
}


def apply_theme() -> None:
    st.markdown(
        """
<style>
:root { --bp-bg:#0A0F14; --bp-surface:#111923; --bp-high:#17222E; --bp-border:#263544;
--bp-text:#E8EEF4; --bp-muted:#8FA1B3; --bp-green:#63D48A; --bp-cyan:#4FC3D7;
--bp-amber:#E8B04A; --bp-red:#ED6A5A; --bp-purple:#A990E8; }
.stApp { background: radial-gradient(circle at 75% -20%, #142432 0, var(--bp-bg) 38%); }
.block-container { padding-top: 1.35rem; padding-bottom: 3rem; max-width: 1680px; }
[data-testid="stSidebar"] { background: #0D141C; border-right: 1px solid var(--bp-border); }
h1,h2,h3 { letter-spacing:-0.025em; }
.bp-header { display:flex; align-items:center; justify-content:space-between; gap:1rem;
padding: .4rem 0 1.1rem; border-bottom:1px solid var(--bp-border); margin-bottom:1rem; }
.bp-brand { font-size:1.45rem; font-weight:760; letter-spacing:-.04em; }
.bp-brand-mark { color:var(--bp-green); margin-right:.42rem; }
.bp-kicker { font-size:.73rem; color:var(--bp-muted); text-transform:uppercase; letter-spacing:.12em; }
.bp-card { background:linear-gradient(145deg,var(--bp-surface),#0E161F); border:1px solid var(--bp-border);
border-radius:12px; padding:1rem 1.05rem; min-height:112px; transition:transform .18s,border-color .18s; }
.bp-card:hover { transform:translateY(-2px); border-color:#3A5064; }
.bp-card-label { color:var(--bp-muted); font-size:.72rem; letter-spacing:.07em; text-transform:uppercase; }
.bp-card-value { color:var(--bp-text); font-weight:700; font-size:1.62rem; line-height:1.2; margin:.35rem 0; }
.bp-card-note { color:var(--bp-muted); font-size:.73rem; }
.bp-badge { display:inline-flex; align-items:center; border-radius:999px; padding:.18rem .52rem;
font-size:.65rem; font-weight:750; letter-spacing:.06em; border:1px solid currentColor; margin:.1rem .2rem .1rem 0; }
.bp-badge-real { color:var(--bp-cyan); background:#10252B; }
.bp-badge-simulated { color:var(--bp-purple); background:#211C32; }
.bp-badge-derived { color:var(--bp-green); background:#12281C; }
.bp-badge-model { color:#78B7FF; background:#14243A; }
.bp-badge-counterfactual { color:var(--bp-amber); background:#2B2212; }
.bp-badge-decision { color:#F3A8BE; background:#301B25; }
.bp-status { padding:.65rem .85rem; border-radius:9px; border:1px solid var(--bp-border);
background:var(--bp-surface); font-weight:700; }
.bp-status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:7px;
background:var(--bp-green); box-shadow:0 0 0 0 rgba(99,212,138,.5); animation:bp-pulse 2.2s infinite; }
.bp-critical { color:var(--bp-red); animation:bp-alert 2s ease-in-out infinite; }
.bp-empty { padding:1.4rem; border:1px dashed #334657; border-radius:12px; color:var(--bp-muted);
text-align:center; background:rgba(17,25,35,.58); }
.bp-disclaimer { border-left:3px solid var(--bp-amber); background:#201B12; padding:.8rem 1rem;
border-radius:0 8px 8px 0; color:#F2D49A; }
.bp-flow { display:flex; align-items:center; justify-content:center; gap:.8rem; color:var(--bp-cyan); }
.bp-arrow { letter-spacing:.25rem; animation:bp-flow 1.5s ease-in-out infinite; }
.bp-rack { border:1px solid var(--bp-border); border-radius:8px; padding:.55rem; min-height:74px; }
@keyframes bp-pulse { 70% { box-shadow:0 0 0 7px rgba(99,212,138,0); } 100% { box-shadow:0 0 0 0 rgba(99,212,138,0); } }
@keyframes bp-alert { 50% { opacity:.68; } }
@keyframes bp-flow { 50% { transform:translateX(5px); opacity:.55; } }
@media(max-width:900px) { .bp-card-value{font-size:1.28rem}.block-container{padding-left:1rem;padding-right:1rem} }
</style>
""",
        unsafe_allow_html=True,
    )
