#!/usr/bin/env python3
"""Draw the charts for the world models 103 and 104 posts from the CSVs that
tools/heater_fit.ts and tools/heater_door_fit.ts write. Pure standard library.

    node --no-warnings tools/heater_fit.ts
    node --no-warnings tools/heater_door_fit.ts
    python3 tools/heater_charts.py
"""
from __future__ import annotations

import csv
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHARTS = ROOT / "static" / "charts"
DATA = ROOT / "static" / "data"

SURFACE = "#0d1117"
INK = "#e4e9f2"
INK2 = "#a7b0c0"
MUTED = "#6f7a8c"
GRID = "#1c2430"
AXIS = "#2c3542"
S1, S2, S3, S4 = "#3987e5", "#d95926", "#199e70", "#8f6fe6"  # validated against SURFACE, as a set of four
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
W = 720


def esc(s):
    return html.escape(str(s), quote=True)


def text(x, y, s, size=12, fill=INK2, anchor="start", weight=400, extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" {extra}>{esc(s)}</text>')


def svg_open(height, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" '
            f'width="{W}" height="{height}" role="img" aria-labelledby="t d" '
            f'font-family="{FONT}" style="max-width:100%;height:auto;display:block">'
            f'<title id="t">{esc(title)}</title><desc id="d">{esc(desc)}</desc>'
            f'<rect width="{W}" height="{height}" fill="{SURFACE}"/>')


def legend(x, y, items):
    out, cx = [], x
    for label, color in items:
        out.append(f'<rect x="{cx}" y="{y - 9}" width="12" height="12" rx="2" fill="{color}"/>')
        out.append(text(cx + 18, y + 1, label, 12, INK2))
        cx += 18 + 7 * len(label) + 24
    return "".join(out)


def read(name):
    with (DATA / f"{name}.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(name, body):
    CHARTS.mkdir(parents=True, exist_ok=True)
    (CHARTS / f"{name}.svg").write_text(body + "</svg>\n", encoding="utf-8")


def chart_rollout():
    rows = read("heater-rollout")
    steps = [int(r["step"]) for r in rows]
    series = [
        ("Real room", [float(r["real_room_c"]) for r in rows], S1),
        ("Learned model", [float(r["learned_model_c"]) for r in rows], S2),
        ("Hand-written guess", [float(r["hand_written_guess_c"]) for r in rows], S3),
    ]
    H = 380
    L, R, T, B = 48, 150, 36, 48
    pw, ph = W - L - R, H - T - B
    vmin, vmax = 0, 24

    def X(s):
        return L + s / steps[-1] * pw

    def Y(v):
        return T + ph - (v - vmin) / (vmax - vmin) * ph

    o = [svg_open(H, "Thirty steps in a room the model never saw: outside at minus four, starting at 22 degrees",
                  "Three lines. The real room and the learned model fall together from 22 to about 9 degrees over ten steps with the heater off, climb to 11 over ten steps with it on, then fall to about 4. The hand-written guess drops much faster, reaching 5 by step ten and under 1 by step thirty.")]
    # heater-on band, steps 10 to 20
    o.append(f'<rect x="{X(10):.1f}" y="{T}" width="{X(20) - X(10):.1f}" height="{ph}" fill="{GRID}" opacity="0.5"/>')
    o.append(text((X(10) + X(20)) / 2, T + 16, "heater on", 11, INK2, "middle"))
    for tv in range(vmin, vmax + 1, 4):
        y = Y(tv)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(text(L - 8, y + 4, f"{tv}°", 11, MUTED, "end"))
    for s in range(0, steps[-1] + 1, 5):
        o.append(text(X(s), T + ph + 20, str(s), 11, MUTED, "middle"))
    o.append(text(L + pw, T + ph + 36, "step", 11, MUTED, "end"))
    o.append(f'<line x1="{L}" y1="{Y(vmin):.1f}" x2="{L + pw}" y2="{Y(vmin):.1f}" stroke="{AXIS}" stroke-width="1"/>')
    for label, vals, c in series:
        pts = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in zip(steps, vals))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for s, v in zip(steps, vals):
            o.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="4" fill="{c}" stroke="{SURFACE}" stroke-width="2"><title>{esc(f"{label}, step {s}: {v:.1f}°C")}</title></circle>')
    # end labels; real and learned end within half a degree, so stagger them
    xe = X(steps[-1])
    real, learned, guess = series[0][1][-1], series[1][1][-1], series[2][1][-1]
    o.append(text(xe + 12, Y(learned) - 6, f"learned {learned:.1f}°", 12, INK, "start", 600))
    o.append(text(xe + 12, Y(real) + 14, f"real {real:.1f}°", 12, INK, "start", 600))
    o.append(text(xe + 12, Y(guess) + 4, f"guess {guess:.1f}°", 12, INK, "start", 600))
    o.append(legend(L, 16, [(l, c) for l, _, c in series]))
    write("heater-rollout", "".join(o))


def chart_loss():
    rows = read("heater-loss")
    epochs = [int(r["epoch"]) for r in rows]
    loss = [float(r["mean_squared_error"]) for r in rows]
    H = 300
    L, R, T, B = 56, 40, 28, 48
    pw, ph = W - L - R, H - T - B
    vmax = 0.32
    floor = 0.01

    def X(e):
        return L + e / epochs[-1] * pw

    def Y(v):
        return T + ph - v / vmax * ph

    o = [svg_open(H, "Mean squared error during training, ten thousand epochs",
                  "One line falling from 0.30 at the first epoch to 0.10 by epoch one thousand and flattening at 0.01, the level of the noise in the logs, from about epoch five thousand.")]
    for tv in (0, 0.1, 0.2, 0.3):
        y = Y(tv)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(text(L - 8, y + 4, f"{tv:g}", 11, MUTED, "end"))
    for e in range(0, epochs[-1] + 1, 2000):
        o.append(text(X(e), T + ph + 20, f"{e:,}", 11, MUTED, "middle"))
    o.append(text(L + pw, T + ph + 36, "epoch", 11, MUTED, "end"))
    o.append(f'<line x1="{L}" y1="{Y(0):.1f}" x2="{L + pw}" y2="{Y(0):.1f}" stroke="{AXIS}" stroke-width="1"/>')
    o.append(f'<line x1="{L}" y1="{Y(floor):.1f}" x2="{L + pw}" y2="{Y(floor):.1f}" stroke="{AXIS}" stroke-width="1" stroke-dasharray="4 4"/>')
    o.append(text(L + pw / 2, Y(floor) - 8, "noise floor, 0.01", 11, MUTED, "middle"))
    pts = " ".join(f"{X(e):.1f},{Y(v):.1f}" for e, v in zip(epochs, loss))
    o.append(f'<polyline points="{pts}" fill="none" stroke="{S1}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    for e, v in zip(epochs, loss):
        if e in (1, 1001, 5001, epochs[-1]):
            o.append(f'<circle cx="{X(e):.1f}" cy="{Y(v):.1f}" r="4" fill="{S1}" stroke="{SURFACE}" stroke-width="2"><title>{esc(f"epoch {e}: {v:.4f}")}</title></circle>')
    o.append(text(X(1) + 8, Y(loss[0]) + 4, f"{loss[0]:.2f}", 12, INK, "start", 600))
    o.append(text(X(epochs[-1]) - 8, Y(loss[-1]) - 8, f"{loss[-1]:.3f}", 12, INK, "end", 600))
    write("heater-loss", "".join(o))


def chart_door_unseen():
    """The 104 room: real temperature with the door band, three five-step
    forecasts from each model, and the private field underneath."""
    rows = read("heater-door-unseen")
    fcs = read("heater-door-forecast")
    steps = [int(r["step"]) for r in rows]
    real = [float(r["real_room_c"]) for r in rows]
    field = [float(r["private_field"]) for r in rows]
    door = [int(r["door_open"]) for r in rows]
    H = 470
    L, R, T, B = 48, 150, 36, 48
    strip_h, gap = 80, 34
    pw = W - L - R
    ph = H - T - B - strip_h - gap
    vmin, vmax = 0, 24

    def X(s):
        return L + s / steps[-1] * pw

    def Y(v):
        return T + ph - (v - vmin) / (vmax - vmin) * ph

    sT = T + ph + gap
    fmin, fmax = -0.3, 1.0

    def FY(v):
        return sT + strip_h - (v - fmin) / (fmax - fmin) * strip_h

    o = [svg_open(H, "Thirty steps with the heater on at minus four outside, the door open from step 10 to 20",
                  "Top: the real room falls from 22 to 17 over ten steps, drops to 2 while the door is open, then climbs back to 8. At steps 5, 13 and 23 two five-step forecasts branch off: the four-weight model drifts high and, from step 13, forecasts a rise while the room is falling; the latent model follows the real room each time. Bottom: the private field sits near zero while the door is closed, jumps to 0.9 the step after it opens, and falls back the step after it closes.")]
    # door-open band across both panels, steps 10 to 20 (the door opened before step 11's transition)
    o.append(f'<rect x="{X(10):.1f}" y="{T}" width="{X(20) - X(10):.1f}" height="{ph}" fill="{GRID}" opacity="0.5"/>')
    o.append(f'<rect x="{X(10):.1f}" y="{sT}" width="{X(20) - X(10):.1f}" height="{strip_h}" fill="{GRID}" opacity="0.5"/>')
    o.append(text((X(10) + X(20)) / 2, T + 16, "door open", 11, INK2, "middle"))
    for tv in range(vmin, vmax + 1, 4):
        y = Y(tv)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(text(L - 8, y + 4, f"{tv}°", 11, MUTED, "end"))
    o.append(f'<line x1="{L}" y1="{Y(vmin):.1f}" x2="{L + pw}" y2="{Y(vmin):.1f}" stroke="{AXIS}" stroke-width="1"/>')
    # real room
    pts = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in zip(steps, real))
    o.append(f'<polyline points="{pts}" fill="none" stroke="{S1}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    for s, v in zip(steps, real):
        o.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="3.5" fill="{S1}" stroke="{SURFACE}" stroke-width="2"><title>{esc(f"Real room, step {s}: {v:.1f}°C")}</title></circle>')
    # forecast fans
    by_from = {}
    for r in fcs:
        by_from.setdefault(int(r["from_step"]), []).append(r)
    for frm, rs in sorted(by_from.items()):
        rs.sort(key=lambda r: int(r["steps_ahead"]))
        for col, key, label in ((S2, "four_weight_model_c", "four-weight model"), (S4, "latent_model_c", "latent model")):
            vals = [float(r[key]) for r in rs]
            pts = " ".join(f"{X(frm + i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
            o.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
            o.append(f'<circle cx="{X(frm + 5):.1f}" cy="{Y(vals[-1]):.1f}" r="4" fill="{col}" stroke="{SURFACE}" stroke-width="2"><title>{esc(f"{label}, forecast from step {frm}, step {frm + 5}: {vals[-1]:.1f}°C")}</title></circle>')
        if frm == 13:
            o.append(text(X(frm) - 8, Y(float(rs[0]["real_room_c"])) + 4, f"from {frm}", 10, INK2, "end"))
        else:
            o.append(text(X(frm), Y(float(rs[0]["real_room_c"])) - 10, f"from {frm}", 10, INK2, "middle"))
    # labels on the step-13 fan, where the two models disagree most
    rs = by_from[13]
    f4, fl, rr = float(rs[5]["four_weight_model_c"]), float(rs[5]["latent_model_c"]), float(rs[5]["real_room_c"])
    o.append(text(X(18) + 8, Y(f4) + 4, f"four-weight {f4:.1f}°", 11, INK, "start", 600))
    o.append(text(X(18) + 8, Y(fl) + 12, f"latent {fl:.1f}°, real {rr:.1f}°", 11, INK, "start", 600))
    o.append(legend(L, 16, [("Real room", S1), ("Four-weight model, 5-step forecast", S2), ("Latent model, 5-step forecast", S4)]))
    # the strip: the private field against the real door
    o.append(text(L, sT - 8, "the private field, and the door it never saw", 11, INK2))
    for tv in (0, 1):
        o.append(f'<line x1="{L}" y1="{FY(tv):.1f}" x2="{L + pw}" y2="{FY(tv):.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(text(L - 8, FY(tv) + 4, str(tv), 11, MUTED, "end"))
    dpts = " ".join(f"{X(s):.1f},{FY(d):.1f}" for s, d in zip(steps, door))
    o.append(f'<polyline points="{dpts}" fill="none" stroke="{INK2}" stroke-width="1.5" stroke-dasharray="4 3"/>')
    fpts = " ".join(f"{X(s):.1f},{FY(v):.1f}" for s, v in zip(steps, field))
    o.append(f'<polyline points="{fpts}" fill="none" stroke="{S4}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    for s, v in zip(steps, field):
        o.append(f'<circle cx="{X(s):.1f}" cy="{FY(v):.1f}" r="3" fill="{S4}" stroke="{SURFACE}" stroke-width="1.5"><title>{esc(f"Private field, step {s}: {v:.2f}")}</title></circle>')
    o.append(text(L + pw + 8, FY(1) + 4, "door (dashed)", 11, INK2))
    o.append(text(L + pw + 8, FY(field[-1]) + 12, "field", 11, INK2))
    for s in range(0, steps[-1] + 1, 5):
        o.append(text(X(s), sT + strip_h + 20, str(s), 11, MUTED, "middle"))
    o.append(text(L + pw, sT + strip_h + 36, "step", 11, MUTED, "end"))
    write("heater-door-unseen", "".join(o))


def chart_door_loss():
    """The latent model's training loss against three reference lines."""
    rows = read("heater-door-loss")
    epochs = [int(r["epoch"]) for r in rows]
    loss = [float(r["latent_model_mse"]) for r in rows]
    four = float(rows[0]["four_weight_model_final_mse"])
    late = float(rows[0]["floor_door_logged_one_step_late"])
    logged = float(rows[0]["floor_door_logged"])
    H = 320
    L, R, T, B = 56, 40, 28, 48
    pw, ph = W - L - R, H - T - B
    vmax = 0.5

    def X(e):
        return L + e / epochs[-1] * pw

    def Y(v):
        return T + ph - v / vmax * ph

    o = [svg_open(H, "Mean squared error while training the latent model, three thousand epochs",
                  f"One line falling from 0.48 to {loss[-1]:.2f}, settling on a dashed line at {late:.2f}, the floor for a model that learns of the door one step late. Two more dashed lines: {four:.2f}, where the four-weight model stopped, and {logged:.2f}, the floor for a model that could see the door.")]
    for tv in (0, 0.1, 0.2, 0.3, 0.4, 0.5):
        y = Y(tv)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(text(L - 8, y + 4, f"{tv:g}", 11, MUTED, "end"))
    for e in range(0, epochs[-1] + 1, 500):
        o.append(text(X(e), T + ph + 20, f"{e:,}", 11, MUTED, "middle"))
    o.append(text(L + pw, T + ph + 36, "epoch", 11, MUTED, "end"))
    o.append(f'<line x1="{L}" y1="{Y(0):.1f}" x2="{L + pw}" y2="{Y(0):.1f}" stroke="{AXIS}" stroke-width="1"/>')
    for v, label, dy in ((four, f"four-weight model, {four:.2f}", -8), (late, f"door logged one step late, {late:.2f}", 14), (logged, f"door logged, {logged:.2f}", -8)):
        o.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{L + pw}" y2="{Y(v):.1f}" stroke="{AXIS}" stroke-width="1" stroke-dasharray="4 4"/>')
        o.append(text(L + pw - 4, Y(v) + dy, label, 11, MUTED, "end"))
    pts = " ".join(f"{X(e):.1f},{Y(v):.1f}" for e, v in zip(epochs, loss))
    o.append(f'<polyline points="{pts}" fill="none" stroke="{S4}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    for e, v in zip(epochs, loss):
        if e in (1, 496, 1006, epochs[-1]):
            o.append(f'<circle cx="{X(e):.1f}" cy="{Y(v):.1f}" r="4" fill="{S4}" stroke="{SURFACE}" stroke-width="2"><title>{esc(f"epoch {e}: {v:.4f}")}</title></circle>')
    o.append(text(X(1) + 8, Y(loss[0]) + 4, f"{loss[0]:.2f}", 12, INK, "start", 600))
    o.append(text(X(epochs[-1]) - 8, Y(loss[-1]) - 10, f"{loss[-1]:.3f}", 12, INK, "end", 600))
    write("heater-door-loss", "".join(o))


if __name__ == "__main__":
    chart_rollout()
    chart_loss()
    chart_door_unseen()
    chart_door_loss()
    print("wrote", sorted(p.name for p in CHARTS.glob("heater-*.svg")))
