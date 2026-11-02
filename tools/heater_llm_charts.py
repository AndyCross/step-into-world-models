#!/usr/bin/env python3
"""Draw the charts for the world models 106 post from the CSVs that
`tools/heater_llm.ts score` writes. Same look and helpers as heater_charts.py.

    node --no-warnings tools/heater_llm.ts score <dir>
    python3 tools/heater_llm_charts.py

Colours follow the entity across all three charts: the real room S1, the 104
model S2, the physics-shaped model S3, Claude S4.
"""
from __future__ import annotations

from heater_charts import (AXIS, GRID, INK, INK2, MUTED, S1, S2, S3, S4, SURFACE, W,
                           esc, read, svg_open, text, write)


def chart_forecast_errors():
    """One row per forecaster, one dot per run, error on a shared axis."""
    rows = read("heater-llm-forecast-errors")
    groups = [
        ("104 model", S2, "same answer every time"),
        ("Claude, readings only", S4, "three runs"),
        ("Claude, with the log", S4, "three runs"),
        ("physics-shaped model", S3, "same answer every time"),
    ]
    H = 280
    L, R, T, B = 190, 40, 30, 52
    pw = W - L - R
    row_h = (H - T - B) / len(groups)
    vmax = 1.4

    def X(v):
        return L + v / vmax * pw

    o = [svg_open(H, "How far off each five-step forecast of the open-door room was",
                  "Root mean square error in degrees, lower is better. The 104 model scored 0.53. Claude with the readings only scored 0.66, 0.18 and 0.16 over three runs. Claude with the log scored 0.15, 1.28 and 0.07. The physics-shaped model scored 0.10.")]
    for tv in (0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4):
        x = X(tv)
        o.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H - B}" stroke="{GRID}" stroke-width="1"/>')
        o.append(text(x, H - B + 18, f"{tv:.1f}°", 11, MUTED, "middle"))
    o.append(text(L + pw, H - B + 36, "forecast error (RMSE), lower is better", 11, MUTED, "end"))
    o.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H - B}" stroke="{AXIS}" stroke-width="1"/>')
    for i, (who, color, note) in enumerate(groups):
        cy = T + row_h * (i + 0.5)
        o.append(text(L - 12, cy + 1, who, 12, INK, "end", 600))
        o.append(text(L - 12, cy + 15, note, 11, MUTED, "end"))
        vals = [float(r["rmse"]) for r in rows if r["who"] == who]
        for run, v in enumerate(vals, 1):
            label = f"{who}, run {run}: {v:.2f}°" if len(vals) > 1 else f"{who}: {v:.2f}°"
            o.append(f'<circle cx="{X(v):.1f}" cy="{cy:.1f}" r="6" fill="{color}" stroke="{SURFACE}" stroke-width="2"><title>{esc(label)}</title></circle>')
        if len(vals) == 1:
            o.append(text(X(vals[0]) + 12, cy + 4, f"{vals[0]:.2f}", 12, INK2, "start", 600))
        else:
            worst = max(vals)
            o.append(text(X(worst) + 12, cy + 4, f"{worst:.2f}", 12, INK2, "start", 600))
    write("heater-llm-forecast-errors", "".join(o))


def chart_plans():
    """Two panels: the plan Claude chose with each model as its tool, as the
    model predicted it and as it went in the real room, with the heater steps under it."""
    rows = read("heater-llm-plans")
    steps = [int(r["step"]) for r in rows]
    panels = [
        ("With the 104 model as its tool", "plan_104", S2, "13 heater steps"),
        ("With the physics-shaped model", "plan_physics", S3, "16 heater steps"),
    ]
    H = 380
    T, B = 56, 78
    gutter, L0, R0 = 36, 52, 16
    pw = (W - L0 - R0 - gutter) / 2
    ph = H - T - B
    vmin, vmax = 10, 20
    strip_y = T + ph + 28

    o = [svg_open(H, "Claude's plan to keep the room at or above 12 degrees, predicted against real",
                  "Left: with the 104 model as its tool, Claude chose 13 heater steps. The model predicted the room would stay just above 12; in the real room the mean falls to 12.1 at step 10 and 10.6 at step 20. Right: with the physics-shaped model, Claude chose 16 heater steps. The model predicted a low of 12.4; the real room levels off at 13.2.")]
    for p, (title, key, color, note) in enumerate(panels):
        L = L0 + p * (pw + gutter)

        def X(s, L=L):
            return L + (s - steps[0]) / (steps[-1] - steps[0]) * pw

        def Y(v):
            return T + ph - (v - vmin) / (vmax - vmin) * ph

        o.append(text(L, 20, title, 13, INK, "start", 600))
        o.append(text(L, 38, note, 11, MUTED))
        for tv in range(vmin, vmax + 1, 2):
            y = Y(tv)
            o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
            if p == 0:
                o.append(text(L - 8, y + 4, f"{tv}°", 11, MUTED, "end"))
        o.append(f'<line x1="{L}" y1="{Y(12):.1f}" x2="{L + pw:.1f}" y2="{Y(12):.1f}" stroke="{INK2}" stroke-width="1" stroke-dasharray="4 4"/>')
        o.append(text(L + 4, Y(12) + 16, "12° target", 11, INK2))
        for s in (5, 10, 15, 20, 25):
            o.append(text(X(s), T + ph + 16, str(s), 11, MUTED, "middle"))
        pred = [float(r[f"{key}_predicted_c"]) for r in rows]
        real = [float(r[f"{key}_real_mean_c"]) for r in rows]
        for label, vals, dash in (("predicted", pred, ' stroke-dasharray="5 4"'), ("real room", real, "")):
            pts = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in zip(steps, vals))
            o.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"{dash}/>')
            for s, v in zip(steps, vals):
                o.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="7" fill="transparent"><title>{esc(f"{label}, step {s}: {v:.2f}°C")}</title></circle>')
        lo = min(real)
        slo = steps[real.index(lo)]
        o.append(f'<circle cx="{X(slo):.1f}" cy="{Y(lo):.1f}" r="4" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>')
        if p == 0:
            o.append(text(X(slo) - 4, Y(lo) + 18, f"real {lo:.1f}°", 12, INK, "end", 600))
            o.append(text(X(14), Y(13.6), "predicted", 11, INK2, "middle"))
        else:
            o.append(text(X(25), Y(real[-1]) - 10, f"real {real[-1]:.1f}°", 12, INK, "end", 600))
            o.append(text(X(25), Y(pred[-1]) + 30, f"predicted {pred[-1]:.1f}°", 12, INK2, "end", 600))
        # heater strip: one cell per step, filled when the heater was on into that reading
        if p == 0:
            o.append(text(L - 6, strip_y + 10, "heater", 11, MUTED, "end"))
        cw = pw / (steps[-1] - steps[0])
        for r in rows[1:]:
            s = int(r["step"])
            on = r[f"{key}_heater"] == "1"
            x = X(s) - cw + 1
            fill = color if on else GRID
            o.append(f'<rect x="{x:.1f}" y="{strip_y:.1f}" width="{cw - 2:.1f}" height="12" rx="2" fill="{fill}"><title>{esc(f"step {s - 1} to {s}: heater {"on" if on else "off"}")}</title></rect>')
    # legend for the line styles
    lx = L0
    ly = H - 10
    o.append(f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 24}" y2="{ly - 4}" stroke="{INK2}" stroke-width="2"/>')
    o.append(text(lx + 30, ly, "real room, mean of 200 runs", 11, INK2))
    lx += 220
    o.append(f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 24}" y2="{ly - 4}" stroke="{INK2}" stroke-width="2" stroke-dasharray="5 4"/>')
    o.append(text(lx + 30, ly, "what the model predicted", 11, INK2))
    write("heater-llm-plans", "".join(o))


def chart_heater_off():
    rows = read("heater-llm-heater-off")
    steps = [int(r["step"]) for r in rows]
    series = [
        ("Real room", "real_mean_c", S1),
        ("104 model", "model_104_c", S2),
        ("Physics-shaped model", "physics_shaped_c", S3),
    ]
    H = 340
    L, R, T, B = 48, 150, 36, 48
    pw, ph = W - L - R, H - T - B
    vmin, vmax = 0, 20

    def X(s):
        return L + (s - steps[0]) / (steps[-1] - steps[0]) * pw

    def Y(v):
        return T + ph - (v - vmin) / (vmax - vmin) * ph

    o = [svg_open(H, "Heater off for twenty steps at minus four outside, door shut",
                  "Three lines from 19.15 degrees at step 5. The real room falls to 1.4 by step 25. The physics-shaped model follows it to 1.15. The 104 model agrees for about five steps, then falls more slowly and ends at 5.25.")]
    for tv in range(vmin, vmax + 1, 4):
        y = Y(tv)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(text(L - 8, y + 4, f"{tv}°", 11, MUTED, "end"))
    for s in (5, 10, 15, 20, 25):
        o.append(text(X(s), T + ph + 20, str(s), 11, MUTED, "middle"))
    o.append(text(L + pw, T + ph + 36, "step", 11, MUTED, "end"))
    o.append(f'<line x1="{L}" y1="{Y(vmin):.1f}" x2="{L + pw}" y2="{Y(vmin):.1f}" stroke="{AXIS}" stroke-width="1"/>')
    for label, key, c in series:
        vals = [float(r[key]) for r in rows]
        pts = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in zip(steps, vals))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for s, v in zip(steps, vals):
            o.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="3.5" fill="{c}" stroke="{SURFACE}" stroke-width="2"><title>{esc(f"{label}, step {s}: {v:.2f}°C")}</title></circle>')
    xe = X(steps[-1])
    end = {key: float(rows[-1][key]) for _, key, _ in series}
    o.append(text(xe + 12, Y(end["model_104_c"]) + 4, f"104 model {end['model_104_c']:.2f}°", 12, INK, "start", 600))
    o.append(text(xe + 12, Y(end["real_mean_c"]) - 6, f"real {end['real_mean_c']:.2f}°", 12, INK, "start", 600))
    o.append(text(xe + 12, Y(end["physics_shaped_c"]) + 14, f"physics-shaped {end['physics_shaped_c']:.2f}°", 12, INK, "start", 600))
    lx = L
    for label, _, c in series:
        o.append(f'<rect x="{lx}" y="7" width="12" height="12" rx="2" fill="{c}"/>')
        o.append(text(lx + 18, 17, label, 12, INK2))
        lx += 18 + 7 * len(label) + 24
    write("heater-llm-heater-off", "".join(o))


if __name__ == "__main__":
    chart_forecast_errors()
    chart_plans()
    chart_heater_off()
    print("wrote heater-llm-forecast-errors.svg, heater-llm-plans.svg, heater-llm-heater-off.svg")
