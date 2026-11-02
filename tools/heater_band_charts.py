#!/usr/bin/env python3
"""Draw the charts for the world models 108 post from the CSVs that
`tools/heater_band.ts csv` writes. Same look and helpers as heater_charts.py.

    node --no-warnings tools/heater_band.ts csv
    python3 tools/heater_band_charts.py

The real room is S1 and the band is the physics-shaped model's green, S3, as a
translucent ribbon with its mean as a line. The width grid is one hue, the site
blue, dark for narrow and light for wide (steps 600 to 100 of the sequential ramp,
validated as an ordinal set against the surface).
"""
from __future__ import annotations

from heater_charts import (GRID, INK, INK2, MUTED, S1, S3, SURFACE, W, esc, read,
                           svg_open, text, write)

RAMP = ["#184f95", "#256abf", "#5598e7", "#9ec5f4", "#cde2fb"]   # narrow -> wide


def ramp(t: float) -> str:
    t = max(0.0, min(1.0, t)) * (len(RAMP) - 1)
    i = min(int(t), len(RAMP) - 2)
    f = t - i
    a, b = RAMP[i], RAMP[i + 1]
    rgb = [round(int(a[k:k + 2], 16) * (1 - f) + int(b[k:k + 2], 16) * f) for k in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(rgb)


def chart_room():
    rows = read("heater-band-room")
    steps = [int(r["step"]) for r in rows]
    H = 400
    L, R, T, B = 48, 110, 40, 40
    pw, ph = W - L - R, H - T - B
    vmin, vmax = 0, 24

    def X(s):
        return L + s / steps[-1] * pw

    def Y(v):
        return T + ph - (v - vmin) / (vmax - vmin) * ph

    o = [svg_open(H, "The test room at minus four with the band of eight models over it",
                  "The real room from 22 degrees with the heater on and the door open from step 10 to 20. "
                  "A thin ribbon is the eight models' one-step-ahead predictions, and three wider fans are "
                  "their five-step forecasts from steps 5, 13 and 23. The room sits inside the band at step 5, "
                  "just above it at step 13, and well above it at step 23.")]
    o.append(f'<rect x="{X(10):.1f}" y="{T}" width="{X(20) - X(10):.1f}" height="{ph}" fill="{GRID}" opacity="0.5"/>')
    o.append(text((X(10) + X(20)) / 2, T + 13, "door open", 10, INK2, "middle"))
    for tv in range(vmin, vmax + 1, 4):
        y = Y(tv)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(text(L - 8, y + 4, f"{tv}°", 11, MUTED, "end"))
    for s in (0, 5, 10, 13, 20, 23, 30):
        o.append(text(X(s), T + ph + 16, str(s), 11, MUTED, "middle"))
    o.append(text(L + pw, T + ph + 32, "step", 11, MUTED, "end"))
    # one step ahead: a thin ribbon
    one = [(int(r["step"]), float(r["one_step_lo_c"]), float(r["one_step_hi_c"])) for r in rows if r["one_step_lo_c"]]
    top = " ".join(f"{X(s):.1f},{Y(hi):.1f}" for s, _, hi in one)
    bot = " ".join(f"{X(s):.1f},{Y(lo):.1f}" for s, lo, _ in reversed(one))
    o.append(f'<polygon points="{top} {bot}" fill="{S3}" opacity="0.35"/>')
    # the three fans
    for frm in (5, 13, 23):
        fan = [(int(r["step"]), float(r["fan_lo_c"]), float(r["fan_hi_c"]), float(r["fan_mean_c"])) for r in rows if r["fan_from"] == str(frm)]
        start = float(rows[frm]["real_c"])
        pts_top = [f"{X(frm):.1f},{Y(start):.1f}"] + [f"{X(s):.1f},{Y(hi):.1f}" for s, _, hi, _ in fan]
        pts_bot = [f"{X(s):.1f},{Y(lo):.1f}" for s, lo, _, _ in reversed(fan)]
        o.append(f'<polygon points="{" ".join(pts_top + pts_bot)}" fill="{S3}" opacity="0.3"/>')
        mean_pts = [f"{X(frm):.1f},{Y(start):.1f}"] + [f"{X(s):.1f},{Y(m):.1f}" for s, _, _, m in fan]
        o.append(f'<polyline points="{" ".join(mean_pts)}" fill="none" stroke="{S3}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        s, lo, hi, m = fan[-1]
        o.append(f'<circle cx="{X(s):.1f}" cy="{Y(m):.1f}" r="3.5" fill="{S3}" stroke="{SURFACE}" stroke-width="2"/>')
        o.append(text(X(s) + 8, Y(lo) + 16 if frm == 23 else Y(hi) - 2, f"{lo:.1f} to {hi:.1f}°", 11, INK, "start", 600))
    real = [float(r["real_c"]) for r in rows]
    o.append(f'<polyline points="{" ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in zip(steps, real))}" fill="none" stroke="{S1}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    o.append(f'<circle cx="{X(steps[-1]):.1f}" cy="{Y(real[-1]):.1f}" r="3.5" fill="{S1}" stroke="{SURFACE}" stroke-width="2"/>')
    o.append(text(X(steps[-1]) + 8, Y(real[-1]) - 4, f"real {real[-1]:.1f}°", 11, INK, "start", 600))
    for r in rows:
        s = int(r["step"])
        parts = [f"step {s}: real {float(r['real_c']):.2f}°C"]
        if r["one_step_lo_c"]:
            parts.append(f"one step ahead {float(r['one_step_lo_c']):.2f} to {float(r['one_step_hi_c']):.2f}")
        if r["fan_from"]:
            parts.append(f"forecast from step {r['fan_from']}: {float(r['fan_lo_c']):.2f} to {float(r['fan_hi_c']):.2f}")
        if r["door"] == "open":
            parts.append("door open")
        o.append(f'<circle cx="{X(s):.1f}" cy="{Y(float(r["real_c"])):.1f}" r="7" fill="transparent"><title>{esc(", ".join(parts))}</title></circle>')
    lx = L
    for label, c, kind in (("Real room", S1, "line"), ("Eight models, one step ahead", S3, "thin"), ("Five steps blind, coldest to warmest of the eight", S3, "band")):
        if kind == "line":
            o.append(f'<line x1="{lx}" y1="14" x2="{lx + 22}" y2="14" stroke="{c}" stroke-width="2"/>')
        elif kind == "thin":
            o.append(f'<rect x="{lx}" y="11" width="22" height="6" fill="{c}" opacity="0.35"/>')
        else:
            o.append(f'<rect x="{lx}" y="8" width="22" height="12" fill="{c}" opacity="0.3"/>')
            o.append(f'<line x1="{lx}" y1="14" x2="{lx + 22}" y2="14" stroke="{c}" stroke-width="2"/>')
        o.append(text(lx + 28, 18, label, 12, INK2))
        lx += 28 + 6.6 * len(label) + 20
    write("heater-band-room", "".join(o))


def chart_grid():
    rows = read("heater-band-grid")
    outs = sorted({int(r["outside_c"]) for r in rows})
    rooms = sorted({int(r["room_c"]) for r in rows})
    wmax = max(float(r["width_c"]) for r in rows)
    H = 420
    T, B = 44, 70
    L0, gap, Rw = 50, 44, 20
    panel = (W - L0 - gap - Rw) / 2
    ph = H - T - B
    cw, ch = panel / len(outs), ph / len(rooms)
    o = [svg_open(H, "How far the eight models disagree, five steps ahead, over room and outside temperature",
                  "Two grids, door shut and door open, of the width of the band after five steps with the heater on. "
                  "Dark cells are narrow and light cells wide. The band widens with the gap between room and outside, "
                  "not with distance from the log: dots mark cells where the log has twenty or more rows within a degree, "
                  f"and widths run from {min(float(r['width_c']) for r in rows):.2f} to {wmax:.2f} degrees.")]
    for p, door in enumerate(("shut", "open")):
        L = L0 + p * (panel + gap)
        o.append(text(L, T - 12, f"door {door}", 12, INK, "start", 600))
        for r in rows:
            if r["door"] != door:
                continue
            i, j = outs.index(int(r["outside_c"])), rooms.index(int(r["room_c"]))
            wdt = float(r["width_c"])
            x, y = L + i * cw, T + ph - (j + 1) * ch
            o.append(f'<rect x="{x + 1:.1f}" y="{y + 1:.1f}" width="{cw - 2:.1f}" height="{ch - 2:.1f}" rx="2" fill="{ramp(wdt / wmax)}">'
                     f'<title>{esc(f"outside {r['outside_c']}°, room {r['room_c']}°, door {door}: band {wdt:.2f}° wide after five steps, {r['log_rows']} log rows nearby")}</title></rect>')
            if int(r["log_rows"]) >= 20:
                o.append(f'<circle cx="{x + cw / 2:.1f}" cy="{y + ch / 2:.1f}" r="2.2" fill="{SURFACE}" pointer-events="none"/>')
        for i, ov in enumerate(outs):
            if ov % 4 == 2 or ov in (-10, 16):
                o.append(text(L + (i + 0.5) * cw, T + ph + 16, f"{ov}°", 10, MUTED, "middle"))
        o.append(text(L + panel / 2, T + ph + 34, "outside", 11, MUTED, "middle"))
        if p == 0:
            for j, rv in enumerate(rooms):
                if rv % 4 == 0:
                    o.append(text(L - 8, T + ph - (j + 0.5) * ch + 4, f"{rv}°", 10, MUTED, "end"))
            o.append(f'<text x="{L - 36}" y="{T + ph / 2}" font-size="11" fill="{MUTED}" text-anchor="middle" transform="rotate(-90 {L - 36} {T + ph / 2})">room</text>')
    # the ramp
    lx, ly, lw = L0, H - 22, 160
    for k in range(40):
        o.append(f'<rect x="{lx + k * lw / 40:.1f}" y="{ly}" width="{lw / 40 + 0.5:.1f}" height="10" fill="{ramp(k / 39)}"/>')
    o.append(text(lx, ly - 4, "band width after five steps", 11, INK2))
    o.append(text(lx, ly + 22, "0°", 10, MUTED))
    o.append(text(lx + lw, ly + 22, f"{wmax:.1f}°", 10, MUTED, "end"))
    o.append(f'<circle cx="{lx + lw + 40}" cy="{ly + 5}" r="2.2" fill="{INK2}"/>')
    o.append(text(lx + lw + 48, ly + 9, "twenty or more log rows within a degree", 11, INK2))
    write("heater-band-grid", "".join(o))


if __name__ == "__main__":
    chart_room()
    chart_grid()
    print("wrote heater-band-room.svg, heater-band-grid.svg")
