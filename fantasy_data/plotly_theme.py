"""Midnight Broadcast Plotly theme.

Registered once as the default template so the 20+ scattered chart builds in
views.py inherit it without touching their individual update_layout calls.
Import this module for its side effect (views.py does) before building figures.
"""
import plotly.graph_objects as go
import plotly.io as pio

# Fixed per-owner palette — assign once via TeamOwnerMapping order and reuse
# anywhere owners are colored (charts, avatars, tables).
OWNER_COLORS = ["#2f6fed", "#b0592f", "#3f9e57", "#8a52c9", "#c9527e",
                "#6b4a2f", "#2f5b6b", "#56662f", "#3f8f9e", "#a08a2f"]

pio.templates["dunn"] = go.layout.Template(layout=dict(
    paper_bgcolor="#14171d", plot_bgcolor="#14171d",
    font=dict(family="Barlow, sans-serif", size=13, color="#b6bcc8"),
    title=dict(font=dict(family="Barlow Condensed, sans-serif",
                         size=20, color="#eef1f5")),
    colorway=OWNER_COLORS,
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)",
               zerolinecolor="rgba(255,255,255,0.12)", linecolor="rgba(255,255,255,0.12)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)",
               zerolinecolor="rgba(255,255,255,0.12)", linecolor="rgba(255,255,255,0.12)"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=50, r=30, t=56, b=48),
))

pio.templates.default = "plotly_dark+dunn"
