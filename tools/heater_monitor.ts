// The code from the world models 109 post: the physics-shaped model from 107 run
// alongside a room for a day, the gap between what it predicted and what the room
// did, and three ways to turn that gap (and the model's private field) into an alarm
// that rings for a dead heater, a stuck one or an open window and not for the door.
// Run from the repo root with Node 23 or newer:
//
//     node --no-warnings tools/heater_monitor.ts           # prints every number in the post
//     node --no-warnings tools/heater_monitor.ts csv       # also writes static/data/heater-monitor-*.csv
//     python3 tools/heater_monitor_charts.py               # draws static/charts/heater-monitor-*.svg
//
// The room, the log and the physics-shaped model are tools/heater_plan.ts with the
// same seeds, so the model's weights match the 107 post.
import { writeFileSync, mkdirSync } from "node:fs";
type State = { roomTemp: number; outsideTemp: number; heaterOn: boolean };
type Action = "turnHeaterOn" | "turnHeaterOff" | "wait";
type Transition = { state: State; action: Action; nextState: State };

let seed = 20261102;
function rand(): number {
  seed = (seed + 0x6D2B79F5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
function gauss(): number {
  const u = 1 - rand(), v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function heaterAfter(state: State, action: Action): boolean {
  return action === "turnHeaterOn" ? true : action === "turnHeaterOff" ? false : state.heaterOn;
}
const ms = (t0: number) => (performance.now() - t0).toFixed(0) + " ms";
const q = (a: number[], p: number) => { const b = [...a].sort((x, y) => x - y); return b[Math.min(b.length - 1, Math.floor(p * b.length))]; };
const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;

// ---- the room with a door (104) -----------------------------------------------
const LEAK_CLOSED = 0.07, LEAK_OPEN = 0.25, REAL_HEAT = 1.2, NOISE = 0.1;
let doorOpen = false;
function someone() {
  if (doorOpen) { if (rand() < 1 / 8) doorOpen = false; }
  else if (rand() < 1 / 25) doorOpen = true;
}
function realRoom(state: State, action: Action): State {
  const heaterOn = heaterAfter(state, action);
  const leak = (state.outsideTemp - state.roomTemp) * (doorOpen ? LEAK_OPEN : LEAK_CLOSED);
  return { roomTemp: state.roomTemp + leak + (heaterOn ? REAL_HEAT : 0) + gauss() * NOISE, outsideTemp: state.outsideTemp, heaterOn };
}
function thermostat(state: State, setpoint: number): Action {
  if (rand() < 0.2) return (["turnHeaterOn", "turnHeaterOff", "wait"] as Action[])[Math.floor(rand() * 3)];
  if (state.roomTemp < setpoint - 0.5 && !state.heaterOn) return "turnHeaterOn";
  if (state.roomTemp > setpoint + 0.5 && state.heaterOn) return "turnHeaterOff";
  return "wait";
}
function record(episodes: number, steps: number): Transition[][] {
  const log: Transition[][] = [];
  for (let e = 0; e < episodes; e++) {
    let state: State = { roomTemp: 10 + rand() * 12, outsideTemp: 2 + rand() * 10, heaterOn: rand() < 0.5 };
    const setpoint = 18 + rand() * 3;
    doorOpen = false;
    const episode: Transition[] = [];
    for (let t = 0; t < steps; t++) {
      someone();
      const action = thermostat(state, setpoint);
      const nextState = realRoom(state, action);
      episode.push({ state, action, nextState });
      state = nextState;
    }
    log.push(episode);
  }
  return log;
}

// ---- the physics-shaped model (107) ----------------------------------------------
function inputs(state: State, action: Action) {
  const heaterOn = heaterAfter(state, action);
  return { room: (state.roomTemp - 17) / 5, outside: (state.outsideTemp - 7) / 5, heater: heaterOn ? 1 : 0, heaterOn };
}
const physicsKeys = ["leak", "doorLeak", "heat", "uField", "uRoom", "uOutside", "uHeater", "uChange", "uBias"] as const;
type PhysicsWeights = Record<typeof physicsKeys[number], number>;

class PhysicsModel {
  #field = 0;
  w: PhysicsWeights;
  constructor(w: PhysicsWeights) { this.w = w; }
  reset() { this.#field = 0; }
  get field() { return this.#field; }
  set field(f: number) { this.#field = f; }
  act(state: State, action: Action): State {
    const w = this.w, heaterOn = heaterAfter(state, action);
    const gap = state.outsideTemp - state.roomTemp;
    const change = (w.leak + w.doorLeak * this.#field) * gap + (heaterOn ? w.heat : 0);
    return { roomTemp: state.roomTemp + change, outsideTemp: state.outsideTemp, heaterOn };
  }
  observe(state: State, action: Action, nextState: State) {
    const f = inputs(state, action), w = this.w;
    const change = (nextState.roomTemp - state.roomTemp) / 5;
    this.#field = Math.tanh(
      w.uField * this.#field + w.uRoom * f.room + w.uOutside * f.outside +
      w.uHeater * f.heater + w.uChange * change + w.uBias);
  }
}

function trainPhysics(log: Transition[][], epochs: number, from?: PhysicsWeights) {
  const w = {} as PhysicsWeights;
  for (const k of physicsKeys) w[k] = from ? from[k] : k.startsWith("u") ? (rand() - 0.5) * 0.2 : 0;
  const n = log.flat().length;
  let loss = 0;
  for (let epoch = 0; epoch < epochs; epoch++) {
    const g = {} as PhysicsWeights;
    for (const k of physicsKeys) g[k] = 0;
    loss = 0;
    for (const episode of log) {
      const steps: { f: ReturnType<typeof inputs>; gap: number; change: number; field: number; err: number; next: number }[] = [];
      let field = 0;
      for (const t of episode) {
        const f = inputs(t.state, t.action), gap = (t.state.outsideTemp - t.state.roomTemp) / 10;
        const target = t.nextState.roomTemp - t.state.roomTemp;
        const err = (w.leak + w.doorLeak * field) * gap * 10 + w.heat * f.heater - target;
        loss += err * err;
        const change = target / 5;
        const next = Math.tanh(w.uField * field + w.uRoom * f.room + w.uOutside * f.outside + w.uHeater * f.heater + w.uChange * change + w.uBias);
        steps.push({ f, gap, change, field, err, next });
        field = next;
      }
      let dField = 0;
      for (let i = steps.length - 1; i >= 0; i--) {
        const s = steps[i], dPre = dField * (1 - s.next * s.next);
        g.uField += dPre * s.field; g.uRoom += dPre * s.f.room; g.uOutside += dPre * s.f.outside;
        g.uHeater += dPre * s.f.heater; g.uChange += dPre * s.change; g.uBias += dPre;
        g.leak += s.err * s.gap; g.doorLeak += s.err * s.field * s.gap; g.heat += s.err * s.f.heater;
        dField = s.err * w.doorLeak * s.gap * 10 + dPre * w.uField;
      }
    }
    for (const k of physicsKeys) w[k] -= (k === "leak" || k === "doorLeak" ? 0.03 : 0.3) * g[k] / n;
  }
  return { w, loss: loss / n };
}

// ---- the log and the single model, as in 107 ---------------------------------------
const log = record(80, 60);
seed = 99;
let t0 = performance.now();
const single = trainPhysics(log, 3000);
console.log("physics-shaped model loss", single.loss.toFixed(4), "(0.179 in the 107 post) in", ms(t0));
const pw = single.w;


// ---- a day with something going wrong halfway -----------------------------------
// 240 readings under a plain thermostat, someone using the door as in the log, and
// at reading 120 one of three things breaks. The model watches every step: it
// predicts the next reading, sees the real one, and updates its field.
type Fault = "none" | "dead" | "stuck" | "window";
const faults: Fault[] = ["dead", "stuck", "window"];
const faultName: Record<Fault, string> = { none: "nothing", dead: "heater dead", stuck: "heater stuck on", window: "window open" };
const STEPS = 240, AT = 120, WARMUP = 10, WINDOW_LEAK = 0.18, FALLING = -0.15;
function makeDay(k: number) {
  seed = k; doorOpen = false;
  const outside = 2 + rand() * 10, setpoint = 18 + rand() * 3, start = setpoint - 1 + rand() * 2;
  const noise: number[] = [], door: boolean[] = [];
  for (let i = 0; i < STEPS; i++) { someone(); door.push(doorOpen); noise.push(gauss() * NOISE); }
  return { outside, setpoint, start, noise, door };
}
type Watched = { real: number[]; predicted: number[]; residual: number[]; leak: number[]; heater: boolean[]; door: boolean[]; outside: number; setpoint: number };
// frozen: a model that never updates its field, held where it sits with the door shut
function watchDay(k: number, fault: Fault, frozen = false, windowLeak = WINDOW_LEAK): Watched {
  const d = makeDay(k), m = new PhysicsModel(pw);
  if (frozen) m.field = SHUT_FIELD;
  let s: State = { roomTemp: d.start, outsideTemp: d.outside, heaterOn: false };
  const w: Watched = { real: [], predicted: [], residual: [], leak: [], heater: [], door: d.door, outside: d.outside, setpoint: d.setpoint };
  for (let i = 0; i < STEPS; i++) {
    let on = s.heaterOn;
    if (s.roomTemp < d.setpoint - 0.5) on = true; else if (s.roomTemp > d.setpoint + 0.5) on = false;
    const action: Action = on ? "turnHeaterOn" : "turnHeaterOff";
    const broken = i >= AT;
    const heating = broken && fault === "dead" ? false : broken && fault === "stuck" ? true : on;
    const leak = (d.door[i] ? LEAK_OPEN : LEAK_CLOSED) + (broken && fault === "window" ? windowLeak : 0);
    const next: State = { roomTemp: s.roomTemp + (s.outsideTemp - s.roomTemp) * leak + (heating ? REAL_HEAT : 0) + d.noise[i], outsideTemp: s.outsideTemp, heaterOn: on };
    const predicted = m.act(s, action).roomTemp;
    if (!frozen) m.observe(s, action, next);
    w.real.push(next.roomTemp); w.predicted.push(predicted); w.residual.push(next.roomTemp - predicted);
    w.leak.push(m.w.leak + m.w.doorLeak * m.field); w.heater.push(on);
    s = next;
  }
  return w;
}

// The field in its own units means nothing, so everything below reads it as the leak
// it gives: leak + doorLeak * field, which can be checked against the room.
let SHUT_FIELD = 0;
{
  const shut: number[] = [], open: number[] = [];
  for (let k = 0; k < 200; k++) { const w = watchDay(100000 + k, "none"); w.leak.forEach((l, i) => (w.door[i] ? open : shut).push(l)); }
  SHUT_FIELD = (mean(shut) - pw.leak) / pw.doorLeak;
  console.log(`\nthe leak the model believes in on ordinary days: door shut ${mean(shut).toFixed(3)} (true 0.07), open ${mean(open).toFixed(3)} (true 0.25); door-shut range ${q(shut, 0).toFixed(3)} to ${q(shut, 1).toFixed(3)}`);
}

// ---- the residual on ordinary days ------------------------------------------------
const CALIBRATE = 1000, CHECK = 1000, BROKEN = 200;
const calm = Array.from({ length: CALIBRATE }, (_, k) => watchDay(100000 + k, "none"));
{
  const all = calm.flatMap(w => w.residual.slice(WARMUP));
  const shut = calm.flatMap(w => w.residual.slice(WARMUP).filter((_, j) => !w.door[j + WARMUP] && !w.door[j + WARMUP - 1]));
  const rms = (a: number[]) => Math.sqrt(mean(a.map(x => x * x)));
  console.log(`\n== residuals over ${CALIBRATE} ordinary days ==`);
  console.log(`all steps: rms ${rms(all).toFixed(3)}, 1st percentile ${q(all, 0.01).toFixed(2)}, 99th ${q(all, 0.99).toFixed(2)}, lowest ${q(all, 0).toFixed(2)}, highest ${q(all, 1).toFixed(2)}`);
  console.log(`door shut this step and last: rms ${rms(shut).toFixed(3)}, ${(100 * shut.length / all.length).toFixed(0)}% of steps`);
  const big = calm.flatMap(w => w.residual.map((r, i) => ({ r, i, w })).filter(x => x.i >= WARMUP && Math.abs(x.r) > 1));
  const atDoor = big.filter(x => x.w.door[x.i] !== x.w.door[x.i - 1] || x.w.door[x.i] !== (x.w.door[x.i + 1] ?? x.w.door[x.i])).length;
  console.log(`residuals beyond a degree: ${big.length}, ${(100 * atDoor / big.length).toFixed(0)}% within a step of the door opening or closing`);
}

// ---- detectors ------------------------------------------------------------------
// Each detector turns a watched day into a number per step and rings when it passes a
// threshold. The threshold is set on 1,000 ordinary days so that ten of them ring,
// then checked on 1,000 different ordinary days and 200 days of each fault.
type Detector = (w: Watched) => number[];
const oneReading: Detector = w => w.residual.map(Math.abs);
const cusum = (slack: number): Detector => w => {
  let up = 0, down = 0;
  return w.residual.map(r => { up = Math.max(0, up + r - slack); down = Math.max(0, down - r - slack); return Math.max(up, down); });
};
const average = (n: number) => (a: number[]) => a.map((_, i) => mean(a.slice(Math.max(0, i - n + 1), i + 1)));
// how far the leak, averaged over twenty steps, has fallen below 0.1
const leakTooLow: Detector = w => average(20)(w.leak).map(l => 0.1 - l);
const watched = (d: Detector): Detector => w => d(w).map((v, i) => i < WARMUP ? -Infinity : v);
function calibrate(d: Detector, frozen = false, rate = 0.01) {
  const days = frozen ? calm.map((_, k) => watchDay(100000 + k, "none", true)) : calm;
  const peaks = days.map(w => Math.max(...watched(d)(w)));
  return q(peaks, 1 - rate);
}
const either = (a: Detector, ta: number, b: Detector, tb: number): Detector => w => { const x = a(w), y = b(w); return x.map((v, i) => Math.max(v / ta, y[i] / tb)); };
type Result = { name: string; threshold: number; falseDays: number; found: Record<string, { found: number; early: number; median: number; delays: number[] }> };
function evaluate(name: string, d: Detector, threshold: number, frozen = false, windowLeak = WINDOW_LEAK): Result {
  const rings = (w: Watched) => watched(d)(w).findIndex(v => v > threshold);
  let falseDays = 0;
  for (let k = 0; k < CHECK; k++) if (rings(watchDay(300000 + k, "none", frozen)) >= 0) falseDays++;
  const found: Result["found"] = {};
  for (const f of faults) {
    const delays: number[] = []; let early = 0;
    for (let k = 0; k < BROKEN; k++) {
      const at = rings(watchDay(200000 + k, f, frozen, windowLeak));
      if (at >= 0 && at < AT) early++; else if (at >= AT) delays.push(at - AT + 1);
    }
    found[f] = { found: delays.length, early, median: delays.length ? q(delays, 0.5) : NaN, delays };
  }
  const line = faults.map(f => `${faultName[f]} ${found[f].found}/${BROKEN}${found[f].found ? `, median ${found[f].median} steps` : ""}`).join(" | ");
  console.log(`${name.padEnd(36)} threshold ${threshold.toFixed(2).padStart(6)} | ordinary days that rang ${String(falseDays).padStart(2)}/${CHECK} | ${line}`);
  return { name, threshold, falseDays, found };
}

console.log(`\n== detectors, each set so 1 in 100 ordinary days rings; fault at step ${AT} of ${STEPS} ==`);
const results: Result[] = [];
results.push(evaluate("one reading", oneReading, calibrate(oneReading)));
for (const slack of [0.1, 0.2, 0.4]) results.push(evaluate(`running sum, slack ${slack}`, cusum(slack), calibrate(cusum(slack))));
results.push(evaluate("running sum, slack 0.2, field frozen", cusum(0.2), calibrate(cusum(0.2), true), true));
results.push(evaluate("leak too low", leakTooLow, calibrate(leakTooLow)));
const tSum = calibrate(cusum(0.2), false, 0.005), tLow = calibrate(leakTooLow, false, 0.005);
const monitor = either(cusum(0.2), tSum, leakTooLow, tLow);
results.push(evaluate("running sum or leak too low", monitor, 1));
console.log(`the two thresholds at half a percent each: running sum ${tSum.toFixed(2)}, leak below ${(0.1 - tLow).toFixed(3)} for twenty steps`);
for (const wl of [0.06, 0.12, 0.18, 0.25]) {
  const r = evaluate(`  monitor, window adding ${wl}`, monitor, 1, false, wl);
  void r;
}
{
  // the stuck heaters the monitor misses: how cold was it outside?
  const hit: number[] = [], miss: number[] = [];
  for (let k = 0; k < BROKEN; k++) {
    const w = watchDay(200000 + k, "stuck");
    (watched(monitor)(w).some(v => v > 1) ? hit : miss).push(w.outside);
    }
  console.log(`stuck heater: outside on days it was found ${mean(hit).toFixed(1)}° mean, on days it was missed ${mean(miss).toFixed(1)}° mean (warmest ${q(miss, 1).toFixed(1)}°)`);
}

// ---- naming the alarm ---------------------------------------------------------------
// When the monitor rings, say what it looks like. Heat arriving from nowhere is a
// heater stuck on. Heat going missing is a dead heater if the room is still falling
// with the heater on, and an open window if the heater is holding it at a lower level.
function name(w: Watched, at: number, falling = FALLING): Fault {
  if (leakTooLow(w)[at] > tLow) return "stuck";
  const from = Math.max(WARMUP, at - 19);
  if (w.residual.slice(from, at + 1).reduce((a, b) => a + b, 0) > 0) return "stuck";
  const recent = Array.from({ length: 5 }, (_, j) => at - j).filter(i => w.heater[i]);
  if (!recent.length) return "window";
  const change = mean(recent.map(i => w.real[i] - (i ? w.real[i - 1] : w.real[i])));
  return change < falling ? "dead" : "window";
}
for (const falling of [-0.05, -0.1, -0.15, -0.2, -0.3]) {
  const right = faults.map(f => { let n = 0; for (let k = 0; k < BROKEN; k++) { const w = watchDay(200000 + k, f), at = watched(monitor)(w).findIndex(v => v > 1); if (at >= AT && name(w, at, falling) === f) n++; } return n; });
  console.log(`naming with the room falling faster than ${-falling} a step: right for ${right.join(", ")}`);
}
console.log("\n== naming what the monitor found ==");
const naming: Record<string, Record<string, number>> = {};
for (const f of faults) {
  const counts: Record<string, number> = { dead: 0, stuck: 0, window: 0 };
  for (let k = 0; k < BROKEN; k++) {
    const w = watchDay(200000 + k, f), at = watched(monitor)(w).findIndex(v => v > 1);
    if (at >= AT) counts[name(w, at)]++;
  }
  naming[f] = counts;
  console.log(`${faultName[f].padEnd(16)} named ${faults.map(g => `${faultName[g]} ${counts[g]}`).join(", ")}`);
}

// ---- the days in the charts ---------------------------------------------------------
const ordinary = watchDay(300041, "none"), stuck = watchDay(200006, "stuck"), dead = watchDay(200000, "dead");
for (const [label, w, f] of [["ordinary", ordinary, "none"], ["stuck", stuck, "stuck"], ["dead", dead, "dead"]] as [string, Watched, Fault][]) {
  const s = cusum(0.2)(w), low = average(20)(w.leak), at = watched(monitor)(w).findIndex(v => v > 1);
  console.log(`${label} day: outside ${w.outside.toFixed(1)}, setpoint ${w.setpoint.toFixed(1)}, door open ${w.door.filter(Boolean).length} steps, biggest residual ${Math.max(...w.residual.slice(WARMUP).map(Math.abs)).toFixed(2)}, running sum peak ${Math.max(...s.slice(WARMUP)).toFixed(2)}, leak low ${Math.min(...low.slice(WARMUP)).toFixed(3)}, rings at ${at}${at >= 0 && f !== "none" ? ` (${at - AT + 1} steps after) named ${faultName[name(w, at)]}` : ""}`);
}

if (process.argv[2] === "csv") {
  mkdirSync("static/data", { recursive: true });
  const dayCsv = (w: Watched) => {
    const s = cusum(0.2)(w), low = average(20)(w.leak);
    return "step,door,heater,real_c,predicted_c,residual_c,running_sum,leak,leak_20\n" +
      w.real.map((r, i) => `${i + 1},${w.door[i] ? "open" : "shut"},${w.heater[i] ? "on" : "off"},${r.toFixed(2)},${w.predicted[i].toFixed(2)},${w.residual[i].toFixed(3)},${s[i].toFixed(3)},${w.leak[i].toFixed(4)},${low[i].toFixed(4)}`).join("\n") + "\n";
  };
  writeFileSync("static/data/heater-monitor-ordinary.csv", dayCsv(ordinary));
  writeFileSync("static/data/heater-monitor-stuck.csv", dayCsv(stuck));
  writeFileSync("static/data/heater-monitor-dead.csv", dayCsv(dead));
  writeFileSync("static/data/heater-monitor-detectors.csv",
    "detector,threshold,ordinary_days_rang,dead_found,dead_median_steps,stuck_found,stuck_median_steps,window_found,window_median_steps\n" +
    results.map(r => `"${r.name}",${r.threshold.toFixed(3)},${r.falseDays},` + faults.map(f => `${r.found[f].found},${isNaN(r.found[f].median) ? "" : r.found[f].median}`).join(",")).join("\n") + "\n");
  writeFileSync("static/data/heater-monitor-thresholds.csv",
    `detector,threshold\none reading,${calibrate(oneReading).toFixed(3)}\nrunning sum,${tSum.toFixed(3)}\nleak below,${(0.1 - tLow).toFixed(4)}\n` +
    `fault_at,${AT}\nordinary_rings,${watched(monitor)(ordinary).findIndex(v => v > 1)}\nstuck_rings,${watched(monitor)(stuck).findIndex(v => v > 1) + 1}\ndead_rings,${watched(monitor)(dead).findIndex(v => v > 1) + 1}\n`);
  console.log("wrote static/data/heater-monitor-*.csv");
}
