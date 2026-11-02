#!/usr/bin/env python3
"""Draw the charts for the world models 107 post from the CSVs that
`tools/heater_plan.ts csv` writes. Same look and helpers as heater_charts.py.

    node --no-warnings tools/heater_plan.ts csv
    python3 tools/heater_plan_charts.py

Colours follow the entity from 106: the real room S1, the 104 model S2, the
physics-shaped model S3. A dashed line is a variant of the solid one in its colour.
In the evening chart the planner is S3, because it plans on the physics-shaped
model, and the thermostat takes S4.
"""
from __future__ import annotations

from heater_charts import (AXIS, GRID, INK, INK2, MUTED, S1, S2, S3, S4, SURFACE, W,
                           esc, read, svg_open, text, write)


def chart_heater_off():
    rows = read("heater-plan-heater-off")
    steps = [int(r["step"]) for r in rows]
    series = [
        ("Real room", "real_mean_c", S1, ""),
        ("Physics-shaped", "physics_shaped_c", S3, ""),
        ("with a bias", "physics_with_bias_c", S3, "6 4"),
        ("104 model", "model_104_c", S2, ""),
        ("colder logs", "model_104_cold_logs_c", S2, "6 4"),
    ]
    H = 380
    L, R, T, B = 48, 190, 40, 48
    pw, ph = W - L - R, H - T - B
    vmin, vmax = -8, 20

    def X(s):
        return L + (s - steps[0]) / (steps[-1] - steps[0]) * pw

    def Y(v):
        return T + ph - (v - vmin) / (vmax - vmin) * ph

    o = [svg_open(H, "Heater off for twenty steps at minus four outside, door shut, five models",
                  "From 19.15 degrees at step 5. The real room falls to 1.42 by step 25 and the physics-shaped model to 1.15. "
                  "The same model with a bias term falls to minus 3.96. The 104 model ends warm at 5.25, and the 104 model "
                  "refitted on logs down to minus six ends at minus 4.85, colder than the air outside.")]
    for tv in range(vmin, vmax + 1, 4):
        y = Y(tv)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(text(L - 8, y + 4, f"{tv}°", 11, MUTED, "end"))
    for s in (5, 10, 15, 20, 25):
        o.append(text(X(s), T + ph + 20, str(s), 11, MUTED, "middle"))
    o.append(text(L + pw, T + ph + 36, "step", 11, MUTED, "end"))
    o.append(f'<line x1="{L}" y1="{Y(-4):.1f}" x2="{L + pw}" y2="{Y(-4):.1f}" stroke="{INK2}" stroke-width="1" stroke-dasharray="2 4"/>')
    o.append(text(L + 6, Y(-4) - 6, "outside, −4°", 11, INK2))
    ends = []
    for label, key, c, dash in series:
        vals = [float(r[key]) for r in rows]
        pts = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in zip(steps, vals))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"{d}/>')
        for s, v in zip(steps, vals):
            o.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="6" fill="transparent"><title>{esc(f"{label}, step {s}: {v:.2f}°C")}</title></circle>')
        o.append(f'<circle cx="{X(steps[-1]):.1f}" cy="{Y(vals[-1]):.1f}" r="3.5" fill="{c}" stroke="{SURFACE}" stroke-width="2"/>')
        ends.append([Y(vals[-1]) + 4, f"{label} {vals[-1]:.2f}°"])
    ends.sort()
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 15:
            ends[i][0] = ends[i - 1][0] + 15
    for y, s in ends:
        o.append(text(X(steps[-1]) + 10, y, s, 12, INK, "start", 600))
    lx = L
    # legend: colour is the model, dashes are the variant that failed
    items = [("Real room", S1, ""), ("Physics-shaped", S3, ""), ("104 model", S2, ""), ("dashed: with a bias, or refitted on colder logs", INK2, "6 4")]
    for label, c, dash in items:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o.append(f'<line x1="{lx}" y1="14" x2="{lx + 22}" y2="14" stroke="{c}" stroke-width="2"{d}/>')
        o.append(text(lx + 28, 18, label, 12, INK2))
        lx += 28 + 7 * len(label) + 22
    write("heater-plan-heater-off", "".join(o))


def chart_evening():
    rows = read("heater-plan-evening")
    steps = [int(r["step"]) for r in rows]
    H = 410
    L, R, T, B = 84, 130, 44, 96
    pw, ph = W - L - R, H - T - B
    vmin, vmax = 10, 22

    def X(s):
        return L + s / steps[-1] * pw

    def Y(v):
        return T + ph - (v - vmin) / (vmax - vmin) * ph

    o = [svg_open(H, "One evening at three degrees outside: the planner against a thermostat tuned for five",
                  "The target is 12 degrees until step 25 and 18 after. The planner heats almost from the start and reads 17.8 "
                  "at step 25, then stays above 18. The thermostat, switched eleven steps early, reads 17.0 when the person "
                  "arrives and is short for five readings.")]
    for tv in range(vmin, vmax + 1, 2):
        y = Y(tv)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(text(L - 8, y + 4, f"{tv}°", 11, MUTED, "end"))
    for s in (0, 10, 20, 25, 30, 40):
        o.append(text(X(s), T + ph + 16, str(s), 11, MUTED, "middle"))
    # the target, as a step
    tgt = [float(r["target_c"]) for r in rows]
    pts = []
    for i, (s, v) in enumerate(zip(steps, tgt)):
        if i and v != tgt[i - 1]:
            pts.append(f"{X(s):.1f},{Y(tgt[i - 1]):.1f}")
        pts.append(f"{X(s):.1f},{Y(v):.1f}")
    o.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{INK2}" stroke-width="1.5" stroke-dasharray="4 4"/>')
    o.append(text(X(1), Y(12) + 16, "wanted: 12° until step 25", 11, INK2))
    o.append(text(X(25) + 6, Y(22) - 4, "18° from step 25", 11, INK2))
    lx = L
    for label, c in (("Room under the planner", S3), ("Room under the thermostat, lead tuned for 5°", S4)):
        o.append(f'<line x1="{lx}" y1="16" x2="{lx + 22}" y2="16" stroke="{c}" stroke-width="2"/>')
        o.append(text(lx + 28, 20, label, 12, INK2))
        lx += 28 + 7 * len(label) + 22
    series = [("planner", "planner_c", "planner_heater", S3), ("thermostat", "thermostat_c", "thermostat_heater", S4)]
    ends = []
    for label, key, _, c in series:
        vals = [float(r[key]) for r in rows]
        pts = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in zip(steps, vals))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for s, v, t in zip(steps, vals, tgt):
            short = ", short" if v < t else ""
            o.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="6" fill="transparent"><title>{esc(f"{label}, step {s}: {v:.2f}°C{short}")}</title></circle>')
        ends.append([Y(vals[-1]) + 4, f"{label} {vals[-1]:.1f}°"])
    ends.sort()
    if ends[1][0] - ends[0][0] < 15:
        ends[1][0] = ends[0][0] + 15
    for y, s in ends:
        o.append(text(X(steps[-1]) + 10, y, s, 12, INK, "start", 600))
    # heater strips
    cw = pw / steps[-1]
    for k, (label, _, hkey, c) in enumerate(series):
        y = T + ph + 30 + k * 22
        n = sum(1 for r in rows[1:] if r[hkey] == "1")
        o.append(text(L - 8, y + 10, label, 11, MUTED, "end"))
        for r in rows[1:]:
            s = int(r["step"])
            on = r[hkey] == "1"
            o.append(f'<rect x="{X(s) - cw + 1:.1f}" y="{y:.1f}" width="{cw - 2:.1f}" height="12" rx="2" fill="{c if on else GRID}"><title>{esc(f"{label}, step {s - 1} to {s}: heater {"on" if on else "off"}")}</title></rect>')
        o.append(text(L + pw + 10, y + 10, f"{n} heater steps", 11, INK2))
    write("heater-plan-evening", "".join(o))


if __name__ == "__main__":
    chart_heater_off()
    chart_evening()
    print("wrote heater-plan-heater-off.svg, heater-plan-evening.svg")
