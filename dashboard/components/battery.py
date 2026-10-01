"""Inline battery and power-flow visualization."""

from html import escape

import streamlit as st

from dashboard.utils import finite, format_number, format_percent


def battery_visual(
    soc: float | None,
    available_energy_mwh: float | None,
    charge_headroom_mwh: float | None,
    mode: str = "IDLE",
) -> None:
    safe_soc = max(0.0, min(1.0, finite(soc) or 0.0))
    fill = safe_soc * 100
    mode = escape(mode.upper())
    color = "#63D48A" if fill >= 30 else "#E8B04A" if fill >= 15 else "#ED6A5A"
    st.markdown(
        f"""
<div class="bp-card" style="min-height:260px;text-align:center">
  <div class="bp-card-label">ENERGY STATE · {mode}</div>
  <div style="position:relative;width:72%;height:112px;border:4px solid #8FA1B3;border-radius:14px;
       margin:1.4rem auto .7rem;padding:7px">
    <div style="position:absolute;right:-13px;top:35px;width:10px;height:38px;background:#8FA1B3;border-radius:0 4px 4px 0"></div>
    <div style="height:100%;width:{fill:.1f}%;border-radius:7px;background:{color};transition:width .8s ease;
         box-shadow:0 0 24px {color}44"></div>
    <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
         color:#E8EEF4;font-size:1.8rem;font-weight:800">{format_percent(safe_soc)}</div>
  </div>
  <div style="display:flex;justify-content:space-around;color:#8FA1B3;font-size:.78rem">
    <span>Discharge energy<br><b style="color:#E8EEF4">{format_number(available_energy_mwh, "MWh")}</b></span>
    <span>Charge headroom<br><b style="color:#E8EEF4">{format_number(charge_headroom_mwh, "MWh")}</b></span>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def power_flow(mode: str) -> None:
    normalized = mode.upper()
    if normalized == "DISCHARGING":
        left, right, arrow = "BATTERY", "GRID", "→ → →"
    elif normalized == "CHARGING":
        left, right, arrow = "GRID", "BATTERY", "→ → →"
    else:
        left, right, arrow = "BATTERY", "GRID", "— IDLE —"
    animation = "bp-arrow" if normalized != "IDLE" else ""
    st.markdown(
        f'<div class="bp-flow"><b>{left}</b><span class="{animation}">{arrow}</span><b>{right}</b></div>',
        unsafe_allow_html=True,
    )
