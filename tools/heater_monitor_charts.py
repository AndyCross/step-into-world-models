#!/usr/bin/env python3
"""Draw the charts for the world models 109 post from the CSVs that
`tools/heater_monitor.ts csv` writes. Same look and helpers as heater_charts.py.

    node --no-warnings tools/heater_monitor.ts csv
    python3 tools/heater_monitor_charts.py

Each chart is one watched day of 240 readings in stacked panels on a shared step
axis. The residual and the running sum are S2, the leak the model believes in is
the physics-shaped model's green, S3, and every threshold is a dashed line in
secondary ink. The door is shaded, as in the 108 charts.
"""
from __future__ import annotations

from heater_charts import (GRID, INK, INK2, MUTED, S2, S3, SURFACE, W, esc, read,
                           svg_open, text, write)

L, R = 56, 24


def thresholds():
    return {r["detector"]: float(r["threshold"]) for r in read("heater-monitor-thresholds")}


def day_chart(name, title, desc, panels, fault_at=None, rings=None):
    rows = read(name)
    n = len(rows)
    pw = W - L - R
    T, gap, B = 30, 34, 40
    H = T + sum(p["h"] for p in panels) + gap * (len(panels) - 1) + B

    def X(s):
        return L + (s - 1) / (n - 1) * pw

    o = [svg_open(H, title, desc)]
    y0 = T
    for k, p in enumerate(panels):
        h, (vmin, vmax) = p["h"], p["range"]

        def Y(v, y0=y0, h=h, vmin=vmin, vmax=vmax):
            v = max(vmin, min(vmax, v))
            return y0 + h - (v - vmin) / (vmax - vmin) * h

        # the door, shaded in runs
        i = 0
        while i < n:
            if rows[i]["door"] == "open":
                j = i
                while j + 1 < n and rows[j + 1]["door"] == "open":
                    j += 1
                o.append(f'<rect x="{X(i + 0.5):.1f}" y="{y0}" width="{max(1.0, X(j + 1.5) - X(i + 0.5)):.1f}" height="{h}" fill="{GRID}" opacity="0.8"/>')
                i = j + 1
            else:
                i += 1
        for tv in p["ticks"]:
            y = Y(tv)
            o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
            o.append(text(L - 8, y + 4, p["fmt"](tv), 11, MUTED, "end"))
        o.append(text(L, y0 - 8, p["label"], 12, INK, "start", 600))
        for tv, lab in p.get("lines", []):
            y = Y(tv)
            o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}" stroke="{INK2}" stroke-width="1.5" stroke-dasharray="5 4"/>')
            if lab:
                o.append(text(L + pw - 4, y - 5, lab, 11, INK2, "end"))
        vals = [float(r[p["col"]]) for r in rows]
        pts = " ".join(f"{X(s + 1):.1f},{Y(v):.1f}" for s, v in enumerate(vals))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{p["color"]}" stroke-width="{p.get("stroke", 1.5)}" stroke-linejoin="round" stroke-linecap="round"/>')
        for s, v in enumerate(vals):
            if (v > vmax or v < vmin) and not p.get("clip_note"):
                o.append(f'<circle cx="{X(s + 1):.1f}" cy="{Y(v):.1f}" r="2.5" fill="{p["color"]}"/>')
        if p.get("clip_note"):
            o.append(text(L + pw - 4, y0 + h - 8, p["clip_note"], 11, INK2, "end"))
        for s, (r, v) in enumerate(zip(rows, vals)):
            tip = f"step {s + 1}: {p['tip']} {p['fmt'](v)}, heater {r['heater']}, door {r['door']}"
            o.append(f'<rect x="{X(s + 1) - pw / n / 2:.1f}" y="{y0}" width="{pw / n:.1f}" height="{h}" fill="transparent"><title>{esc(tip)}</title></rect>')
        y0 += h + gap
    bottom = y0 - gap
    for at, lab, anchor in ((fault_at, "breaks", "end"), (rings, "monitor rings", "start")):
        if at:
            x = X(at)
            o.append(f'<line x1="{x:.1f}" y1="{T - 4}" x2="{x:.1f}" y2="{bottom}" stroke="{INK}" stroke-width="1" opacity="0.7"/>')
            o.append(text(x + (-5 if anchor == "end" else 5), T + 6 + (0 if anchor == "end" else 14), lab, 11, INK, anchor, 600))
    for s in (1, 40, 80, 120, 160, 200, 240):
        o.append(text(X(s), bottom + 16, str(s), 11, MUTED, "middle"))
    o.append(text(L + pw, bottom + 32, "step", 11, MUTED, "end"))
    o.append(text(L, bottom + 32, "shaded: the door open, which the model is never told", 11, INK2))
    write(name, "".join(o))


def main():
    th = thresholds()
    one, run, low = th["one reading"], th["running sum"], th["leak below"]
    res = lambda lo, hi, ticks: dict(label="what the room did minus what the model said", col="residual_c", color=S2,
                                     h=130, range=(lo, hi), ticks=ticks, fmt=lambda v: f"{v:+.1f}°" if v else "0", tip="residual")
    r1 = res(-3.6, 3.6, (-3, 0, 3))
    r1["lines"] = [(one, f"one-reading threshold ±{one:.2f}°"), (-one, "")]
    rs = lambda top, ticks, note=None: dict(label="running sum of the residual, slack 0.2", col="running_sum", color=S2, stroke=2,
                                            h=110, range=(0, top), ticks=ticks, fmt=lambda v: f"{v:.0f}", tip="running sum",
                                            lines=[(run, f"threshold {run:.1f}")], clip_note=note)
    day_chart("heater-monitor-ordinary",
              "An ordinary day: the residual and the running sum",
              "240 readings with nothing broken and the door open for 62 of them. The residual jumps by up to "
              "three degrees in the step after the door opens or shuts and sits within a few tenths otherwise. "
              "The running sum climbs at each door and drains away again, peaking at 3.1 against a threshold of 8.3.",
              [r1, rs(12, (0, 4, 8, 12))])
    d = read("heater-monitor-dead")
    r2 = res(-3.6, 3.6, (-3, 0, 3))
    day_chart("heater-monitor-dead",
              "A dead heater: the residual and the running sum",
              "The heater stops heating at step 120. From then on the thermostat keeps it switched on and every "
              "reading is about a degree below the model's prediction. The running sum leaves zero and passes the "
              "threshold 17 steps later, where an ordinary day never got above 3.",
              [r2, rs(12, (0, 4, 8, 12), "off the top from step %d, reaching %.0f by step 240" % (next(i + 1 for i, r in enumerate(d) if float(r["running_sum"]) > 12), max(float(r["running_sum"]) for r in d)))],
              fault_at=th["fault_at"] + 0.5, rings=int(th["dead_rings"]))
    r3 = res(-3.6, 3.6, (-3, 0, 3))
    leak = dict(label="the leak the model believes in, averaged over twenty steps", col="leak_20", color=S3, stroke=2,
                h=130, range=(-0.04, 0.22), ticks=(0, 0.07, 0.14, 0.21), fmt=lambda v: f"{v:.2f}", tip="leak",
                lines=[(low, f"threshold {low:.3f}")])
    day_chart("heater-monitor-stuck",
              "A stuck heater: the residual settles and the leak gives it away",
              "The heater sticks on at step 120 and the thermostat switches it off. The residual is positive for a "
              "few steps and then settles back to zero, because the model's field moves to a level where the room "
              "barely leaks. The leak the model believes in, averaged over twenty steps, falls below anything an "
              "ordinary day produced and the monitor rings 8 steps after the fault.",
              [r3, leak], fault_at=th["fault_at"] + 0.5, rings=int(th["stuck_rings"]))
    print("wrote heater-monitor-ordinary.svg, heater-monitor-dead.svg, heater-monitor-stuck.svg")


if __name__ == "__main__":
    main()
