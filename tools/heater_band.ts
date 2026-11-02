// The code from the world models 108 post: eight physics-shaped models fitted on
// resampled logs, the spread of their predictions as a band, and 107's planner
// given the cold edge of the band instead of one number. Run from the repo root
// with Node 23 or newer:
//
//     node --no-warnings tools/heater_band.ts           # prints every number in the post
//     node --no-warnings tools/heater_band.ts csv       # also writes static/data/heater-band-*.csv
//     python3 tools/heater_band_charts.py               # draws static/charts/heater-band-*.svg
//
// The room, the log and the physics-shaped model are tools/heater_plan.ts with the
// same seeds, so the single model's weights match the 107 post. QUICK=1 runs fewer
// evenings.
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
const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
const EVENINGS = process.env.QUICK ? 40 : 200;

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

// ---- the ensemble: the same model fitted eight times on resampled logs -------------
// Each member gets its own log: eighty episodes drawn from the eighty with replacement,
// so about a third of the episodes are left out of each and others appear twice.
const K = 8;
function resample<T>(items: T[]): T[] {
  return Array.from({ length: items.length }, () => items[Math.floor(rand() * items.length)]);
}
seed = 108; t0 = performance.now();
const members = Array.from({ length: K }, () => trainPhysics(resample(log), 3000));
console.log(`\n== ${K} members on resampled logs, ${ms(t0)} ==`);
for (const [i, m] of members.entries()) {
  console.log(`member ${i + 1}: loss ${m.loss.toFixed(4)}, leak ${m.w.leak.toFixed(3)}, doorLeak ${m.w.doorLeak.toFixed(3)}, heat ${m.w.heat.toFixed(3)}, uField ${m.w.uField.toFixed(2)}, uChange ${m.w.uChange.toFixed(2)}`);
}
// a warm-started, shorter fit, the version the browser runs
seed = 108; t0 = performance.now();
const warm = Array.from({ length: 5 }, () => trainPhysics(resample(log), 600, pw));
console.log(`5 warm-started members, 600 epochs each, ${ms(t0)}: leaks`, warm.map(m => m.w.leak.toFixed(3)).join(" "), "heats", warm.map(m => m.w.heat.toFixed(3)).join(" "));

// The band: every member watches the same room, and a forecast is every member's forecast.
class Band {
  models: PhysicsModel[];
  constructor(ws: PhysicsWeights[]) { this.models = ws.map(w => new PhysicsModel(w)); }
  observe(s: State, a: Action, n: State) { for (const m of this.models) m.observe(s, a, n); }
  // one row per member, one column per step
  forecasts(s: State, heater: boolean[]): number[][] { return this.models.map(m => rollout(m, s, heater)); }
  edges(s: State, heater: boolean[]) {
    const f = this.forecasts(s, heater);
    return heater.map((_, i) => { const col = f.map(r => r[i]); return { lo: Math.min(...col), hi: Math.max(...col), mean: mean(col) }; });
  }
  low(s: State, heater: boolean[]): number[] {
    const f = this.forecasts(s, heater);
    return heater.map((_, i) => Math.min(...f.map(r => r[i])));
  }
}
function rollout(m: { act(s: State, a: Action): State }, s: State, heater: boolean[]): number[] {
  const out: number[] = [];
  for (const on of heater) { s = m.act(s, on ? "turnHeaterOn" : "turnHeaterOff"); out.push(s.roomTemp); }
  return out;
}

// ---- the 104 test room: 22 inside, minus four outside, door open from step 10 to 20 ----
seed = 7; doorOpen = false;
const start: State = { roomTemp: 22, outsideTemp: -4, heaterOn: false };
const script: Action[] = ["turnHeaterOn", ...Array(29).fill("wait") as Action[]];
const truth: State[] = [start];
const doors: boolean[] = [];
for (let i = 0; i < 30; i++) { doorOpen = i >= 10 && i < 20; doors.push(doorOpen); truth.push(realRoom(truth[i], script[i])); }

function bandAt(ws: PhysicsWeights[], from: number) {
  const b = new Band(ws);
  for (let i = 0; i < from; i++) b.observe(truth[i], script[i], truth[i + 1]);
  return b;
}
const memberWs = members.map(m => m.w), warmWs = warm.map(m => m.w);
console.log("\n== the test room, five steps blind ==");
const fanRows: Record<number, { lo: number; hi: number; mean: number }[]> = {};
for (const from of [5, 13, 23]) {
  const edges = bandAt(memberWs, from).edges(truth[from], Array(5).fill(true));
  const wedges = bandAt(warmWs, from).edges(truth[from], Array(5).fill(true));
  const one = rollout(bandAt([pw], from).models[0], truth[from], Array(5).fill(true));
  fanRows[from] = edges;
  const e = edges[4], real = truth[from + 5].roomTemp;
  console.log(`from step ${from} (${doors[from] ? "door open" : "door shut"}): real ${real.toFixed(2)}, single ${one[4].toFixed(2)}, band ${e.lo.toFixed(2)} to ${e.hi.toFixed(2)} (width ${(e.hi - e.lo).toFixed(2)}, mean ${e.mean.toFixed(2)}); warm five: width ${(wedges[4].hi - wedges[4].lo).toFixed(2)}`);
}
// the band one step ahead through the whole room, for the chart
const oneStep: { lo: number; hi: number }[] = [];
{
  const b = new Band(memberWs);
  for (let i = 0; i < 30; i++) {
    const col = b.models.map(m => m.act(truth[i], script[i]).roomTemp);
    oneStep.push({ lo: Math.min(...col), hi: Math.max(...col) });
    b.observe(truth[i], script[i], truth[i + 1]);
  }
  const widths = oneStep.map(e => e.hi - e.lo);
  console.log(`one step ahead, band width: door shut mean ${mean(widths.filter((_, i) => !doors[i])).toFixed(2)}, door open mean ${mean(widths.filter((_, i) => doors[i])).toFixed(2)}`);
}

function quietRoom(s: State, on: boolean, door: boolean): State {
  return { roomTemp: s.roomTemp + (s.outsideTemp - s.roomTemp) * (door ? LEAK_OPEN : LEAK_CLOSED) + (on ? REAL_HEAT : 0), outsideTemp: s.outsideTemp, heaterOn: on };
}

// ---- where the models know, on a grid --------------------------------------------
// Five steps blind with the heater on and the door shut, from every room and outside
// temperature on a grid, each member's field held where the test room's shut-door steps
// leave it (its own level, in its own units). Next to each cell, how many rows of the
// log fall within a degree of it.
console.log("\n== five-step band width over room and outside temperature, door shut ==");
const shutField = memberWs.map((_, m) => mean(Array.from({ length: 10 }, (_, i) => bandAt(memberWs, i + 1).models[m].field)));
const openField = memberWs.map((_, m) => mean(Array.from({ length: 9 }, (_, i) => bandAt(memberWs, i + 12).models[m].field)));
const gridCells = ["outside_c,room_c,door,width_c,log_rows"];
const rows = log.flat();
for (const door of [false, true]) {
  for (let outside = -10; outside <= 16; outside += 2) {
    for (let room = 4; room <= 28; room += 2) {
      const b = new Band(memberWs);
      b.models.forEach((m, i) => { m.field = door ? openField[i] : shutField[i]; });
      const e = b.edges({ roomTemp: room, outsideTemp: outside, heaterOn: false }, Array(5).fill(true))[4];
      const dense = rows.filter(t => Math.abs(t.state.outsideTemp - outside) <= 1 && Math.abs(t.state.roomTemp - room) <= 1).length;
      gridCells.push(`${outside},${room},${door ? "open" : "shut"},${(e.hi - e.lo).toFixed(2)},${dense}`);
    }
  }
}
{
  const cell = (o: number, r: number, d = "shut") => gridCells.find(c => c.startsWith(`${o},${r},${d},`))!.split(",");
  for (const [o, r] of [[8, 18], [4, 14], [10, 20], [-4, 18], [8, 26], [-10, 8], [12, 6]]) console.log(`outside ${o}, room ${r}, shut: width ${cell(o, r)[3]}, ${cell(o, r)[4]} log rows nearby; open: width ${cell(o, r, "open")[3]}`);
  const inLog = gridCells.slice(1).map(c => c.split(",")).filter(c => c[2] === "shut" && +c[4] >= 20), outLog = gridCells.slice(1).map(c => c.split(",")).filter(c => c[2] === "shut" && +c[4] === 0);
  console.log(`cells with 20 or more log rows nearby: ${inLog.length}, mean width ${mean(inLog.map(c => +c[3])).toFixed(2)}; cells with none: ${outLog.length}, mean width ${mean(outLog.map(c => +c[3])).toFixed(2)}, max ${Math.max(...outLog.map(c => +c[3])).toFixed(2)}`);
}

// ---- planning --------------------------------------------------------------------
type Plan = { heater: boolean[]; cost: number };
function cost(temps: number[], heater: boolean[], target: number[]) {
  let c = 0, slack = Infinity;
  temps.forEach((t, i) => {
    if (heater[i]) c += 1;
    if (t < target[i]) c += 100 * (target[i] - t);
    slack = Math.min(slack, t - target[i]);
  });
  return c - 0.01 * Math.min(1, Math.max(0, slack));
}
function exhaustive(predict: (h: boolean[]) => number[], target: number[]): Plan {
  const steps = target.length;
  let best: Plan = { heater: [], cost: Infinity };
  for (let mask = 0; mask < 2 ** steps; mask++) {
    const heater = Array.from({ length: steps }, (_, i) => (mask >> (steps - 1 - i) & 1) === 1);
    const c = cost(predict(heater), heater, target);
    if (c < best.cost) best = { heater, cost: c };
  }
  return best;
}
function cem(predict: (h: boolean[]) => number[], target: number[], samples = 200, rounds = 6): Plan {
  let odds = target.map(() => 0.5);
  let best: Plan = { heater: [], cost: Infinity };
  for (let r = 0; r < rounds; r++) {
    const tried: Plan[] = [];
    for (let k = 0; k < samples; k++) {
      const heater = odds.map(p => rand() < p);
      tried.push({ heater, cost: cost(predict(heater), heater, target) });
    }
    tried.sort((a, b) => a.cost - b.cost);
    if (tried[0].cost < best.cost) best = tried[0];
    const elite = tried.slice(0, samples / 10);
    odds = odds.map((_, i) => elite.filter(e => e.heater[i]).length / elite.length);
  }
  return best;
}
const bits = (h: boolean[]) => h.map(b => b ? "1" : "0").join("");
const ons = (h: boolean[]) => h.filter(Boolean).length;
function realCheck(heater: boolean[], from: State = truth[5]) {
  const lows: number[] = [];
  for (let k = 0; k < 200; k++) {
    seed = 1000 + k; doorOpen = false;
    let s = from, low = Infinity;
    for (const on of heater) { s = realRoom(s, on ? "turnHeaterOn" : "turnHeaterOff"); low = Math.min(low, s.roomTemp); }
    lows.push(low);
  }
  return { low: mean(lows), ok: lows.filter(l => l >= 12).length };
}

// The 106 question again: from the step 5 reading, every reading from 6 to 25 at or above 12.
console.log("\n== the 106 question ==");
const T20 = Array(20).fill(12);
const trueRoom = { act: (s: State, a: Action) => { const on = heaterAfter(s, a); return { roomTemp: s.roomTemp + (s.outsideTemp - s.roomTemp) * LEAK_CLOSED + (on ? REAL_HEAT : 0), outsideTemp: s.outsideTemp, heaterOn: on }; } };
const questionRows = ["model,heater_steps,schedule,model_low_c,real_low_c,safe_in_200"];
{
  const single5 = bandAt([pw], 5).models[0], band5 = bandAt(memberWs, 5);
  const cases: [string, (h: boolean[]) => number[]][] = [
    ["true constants", h => rollout(trueRoom, truth[5], h)],
    ["physics-shaped, one model", h => rollout(single5, truth[5], h)],
    ["band, coldest of eight", h => band5.low(truth[5], h)],
    ["band, mean of eight", h => band5.edges(truth[5], h).map(e => e.mean)],
  ];
  for (const [name, predict] of cases) {
    t0 = performance.now();
    const b = exhaustive(predict, T20);
    const took = performance.now() - t0;
    const rc = realCheck(b.heater);
    const low = Math.min(...predict(b.heater));
    console.log(`${name}: ${bits(b.heater)} ${ons(b.heater)} on, model low ${low.toFixed(2)}, in ${took.toFixed(0)} ms; real lowest ${rc.low.toFixed(2)}, ok in ${rc.ok} of 200`);
    questionRows.push(`"${name}",${ons(b.heater)},${bits(b.heater)},${low.toFixed(2)},${rc.low.toFixed(2)},${rc.ok}`);
  }
  // what the band says about the true fifteen-step plan and the single model's sixteen
  const fifteen = exhaustive(h => rollout(trueRoom, truth[5], h), T20).heater;
  const e15 = band5.edges(truth[5], fifteen);
  const i15 = e15.map(e => e.lo).indexOf(Math.min(...e15.map(e => e.lo)));
  console.log(`the true fifteen-step plan, through the band: lowest edge ${e15[i15].lo.toFixed(2)} to ${e15[i15].hi.toFixed(2)} at step ${6 + i15}, single model says ${Math.min(...rollout(single5, truth[5], fifteen)).toFixed(2)}`);
}

// ---- evenings, as in 107 ----------------------------------------------------
type Day = { outside: number; start: number; arrive: number; steps: number; noise: number[]; door: boolean[]; name: string };
function evening(k: number, outside: number, withDoor: boolean): Day {
  seed = 5000 + k; doorOpen = false;
  const noise: number[] = [], door: boolean[] = [];
  for (let i = 0; i < 40; i++) { if (withDoor) someone(); door.push(doorOpen); noise.push(gauss() * NOISE); }
  return { outside, start: 14, arrive: 25, steps: 40, noise, door, name: "evening" };
}
const wantAt = (d: Day, reading: number) => reading >= d.arrive ? 18 : 12;

type Model = { observe(s: State, a: Action, next: State): void; predict(s: State, heater: boolean[]): number[] };
type Controller = (d: Day, step: number, s: State, model: Model) => boolean;
const singleModel = (): Model => {
  const m = new PhysicsModel(pw);
  return { observe: (s, a, n) => m.observe(s, a, n), predict: (s, h) => rollout(m, s, h) };
};
const bandModel = (ws = memberWs): Model => {
  const b = new Band(ws);
  return { observe: (s, a, n) => b.observe(s, a, n), predict: (s, h) => b.low(s, h) };
};
const marginModel = (margin: number): Model => {
  const m = new PhysicsModel(pw);
  return { observe: (s, a, n) => m.observe(s, a, n), predict: (s, h) => rollout(m, s, h).map(t => t - margin) };
};

function runDay(d: Day, ctl: Controller, model: Model) {
  let s: State = { roomTemp: d.start, outsideTemp: d.outside, heaterOn: false };
  const temps: number[] = [], heater: boolean[] = [];
  for (let i = 0; i < d.steps; i++) {
    const on = ctl(d, i, s, model);
    const leak = d.door[i] ? LEAK_OPEN : LEAK_CLOSED;
    const next: State = { roomTemp: s.roomTemp + (s.outsideTemp - s.roomTemp) * leak + (on ? REAL_HEAT : 0) + d.noise[i], outsideTemp: s.outsideTemp, heaterOn: on };
    model.observe(s, on ? "turnHeaterOn" : "turnHeaterOff", next);
    temps.push(next.roomTemp); heater.push(on); s = next;
  }
  let short = 0, misses = 0;
  temps.forEach((t, i) => { const want = wantAt(d, i + 1); if (t < want) { short += want - t; misses++; } });
  return { temps, heater, on: ons(heater), short, misses };
}
const thermostatWithLead = (lead: number): Controller => (d, i, s) => {
  const want = i + 1 + lead >= d.arrive ? 18 : 12;
  return s.roomTemp < want + 1 ? true : s.roomTemp > want + 1.7 ? false : s.heaterOn;
};
const replan: Controller = (d, i, s, model) => {
  const target = Array.from({ length: Math.min(20, d.steps - i) }, (_, j) => wantAt(d, i + 1 + j));
  return cem(h => model.predict(s, h), target).heater[0];
};
function days(make: (k: number) => Day, ctl: Controller, model: () => Model) {
  seed = 42;
  const rs = Array.from({ length: EVENINGS }, (_, k) => runDay(make(k), ctl, model()));
  return { on: mean(rs.map(r => r.on)), short: mean(rs.map(r => r.short)), misses: mean(rs.map(r => r.misses)), runs: rs };
}
const fmt = (r: { on: number; misses: number; short: number }) => `heater ${r.on.toFixed(1)}, readings short ${r.misses.toFixed(2)}, degree-steps short ${r.short.toFixed(2)}`;
const leads: Record<number, number> = { 3: 18, 5: 11, 8: 7 };   // tuned in 107

const dayRows = ["day,outside_c,door,controller,heater_steps,readings_short,degree_steps_short"];
const ctls: [string, Controller, () => Model][] = [
  ["planner, one model", replan, singleModel],
  ["planner, half a degree of margin", replan, () => marginModel(0.5)],
  ["planner, the band", replan, () => bandModel()],
];
console.log("\n== evenings, as in 107 ==");
t0 = performance.now();
for (const outside of [3, 5, 8]) {
  for (const withDoor of [false, true]) {
    const all: typeof ctls = [[`thermostat, lead tuned for ${outside}`, thermostatWithLead(leads[outside]), singleModel], ...ctls];
    for (const [name, ctl, model] of all) {
      const r = days(k => evening(k, outside, withDoor), ctl, model);
      console.log(`evening ${outside}° ${withDoor ? "door  " : "sealed"} | ${name.padEnd(34)} ${fmt(r)}`);
      dayRows.push(`evening,${outside},${withDoor ? "yes" : "no"},"${name}",${r.on.toFixed(1)},${r.misses.toFixed(2)},${r.short.toFixed(2)}`);
    }
  }
}
{
  // the warm-started five, the browser's version, on the 3 degree evening
  const r = days(k => evening(k, 3, false), replan, () => bandModel(warmWs));
  console.log(`evening 3° sealed | ${"planner, warm five (browser)".padEnd(34)} ${fmt(r)}`);
}
console.log("evenings took", ms(t0));

// The same question at seven outside temperatures, from minus four to eight. The room
// starts at 22 with the heater on and the door shut, every model watches five quiet
// steps, and the plan runs from there: twenty readings at or above 12, fewest heater steps.
console.log("\n== the question at seven temperatures ==");
const gridRows = ["outside_c,start_c,model,heater_steps,model_low_c,real_low_c,safe_in_200"];
const gridMs = { one: 0, band: 0 };
for (let outside = -4; outside <= 8; outside += 2) {
  const b = new Band(memberWs), one = new PhysicsModel(pw);
  let s: State = { roomTemp: 22, outsideTemp: outside, heaterOn: false };
  for (let i = 0; i < 5; i++) { const n = quietRoom(s, true, false); b.observe(s, "turnHeaterOn", n); one.observe(s, "turnHeaterOn", n); s = n; }
  const from = s;
  const cases: [string, (h: boolean[]) => number[]][] = [
    ["true constants", h => rollout(trueRoom, from, h)],
    ["one model", h => rollout(one, from, h)],
    ["the band", h => b.low(from, h)],
  ];
  const line: string[] = [];
  for (const [name, predict] of cases) {
    t0 = performance.now();
    const p = exhaustive(predict, T20);
    const took = performance.now() - t0;
    if (name === "one model") gridMs.one += took; if (name === "the band") gridMs.band += took;
    const rc = realCheck(p.heater, from);
    line.push(`${name}: ${ons(p.heater)} on, low ${Math.min(...predict(p.heater)).toFixed(2)}, real low ${rc.low.toFixed(2)}, safe ${rc.ok}`);
    gridRows.push(`${outside},${from.roomTemp.toFixed(2)},"${name}",${ons(p.heater)},${Math.min(...predict(p.heater)).toFixed(2)},${rc.low.toFixed(2)},${rc.ok}`);
  }
  console.log(`outside ${outside}° from ${from.roomTemp.toFixed(2)} | ` + line.join(" | "));
}
console.log(`exhaustive search per question: one model ${(gridMs.one / 7).toFixed(0)} ms, the band ${(gridMs.band / 7).toFixed(0)} ms`);
{
  seed = 42; t0 = performance.now(); const m = bandModel(); const d = evening(0, 3, false);
  for (let k = 0; k < 200; k++) replan(d, 0, { roomTemp: 14, outsideTemp: 3, heaterOn: false }, m);
  console.log("one planner decision with the band:", ((performance.now() - t0) / 200).toFixed(2), "ms");
}

if (process.argv[2] === "csv") {
  mkdirSync("static/data", { recursive: true });
  writeFileSync("static/data/heater-band-room.csv",
    "step,door,real_c,one_step_lo_c,one_step_hi_c,fan_from,fan_lo_c,fan_hi_c,fan_mean_c\n" +
    truth.map((s, i) => {
      const o = i ? oneStep[i - 1] : null;
      let fan = ",,,";
      for (const from of [5, 13, 23]) if (i > from && i <= from + 5) { const e = fanRows[from][i - from - 1]; fan = `${from},${e.lo.toFixed(2)},${e.hi.toFixed(2)},${e.mean.toFixed(2)}`; }
      return `${i},${i && doors[i - 1] ? "open" : "shut"},${s.roomTemp.toFixed(2)},${o ? o.lo.toFixed(2) : ""},${o ? o.hi.toFixed(2) : ""},${fan}`;
    }).join("\n") + "\n");
  writeFileSync("static/data/heater-band-grid.csv", gridCells.join("\n") + "\n");
  writeFileSync("static/data/heater-band-question.csv", questionRows.join("\n") + "\n");
  writeFileSync("static/data/heater-band-days.csv", dayRows.join("\n") + "\n");
  writeFileSync("static/data/heater-band-questions.csv", gridRows.join("\n") + "\n");
  console.log("wrote static/data/heater-band-*.csv");
}
