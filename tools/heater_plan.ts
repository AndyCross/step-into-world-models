// The code from the world models 107 post: the physics-shaped model from 106 in
// full, then planning against it. Run from the repo root with Node 23 or newer:
//
//     node --no-warnings tools/heater_plan.ts           # prints every number in the post
//     node --no-warnings tools/heater_plan.ts csv       # also writes static/data/heater-plan-*.csv
//     python3 tools/heater_plan_charts.py               # draws static/charts/heater-plan-*.svg
//
// The room, the log and the 104 model are tools/heater_door_fit.ts with the same seed,
// and the physics-shaped model is the one in tools/heater_llm.ts, so the weights match
// both posts.
import { writeFileSync, mkdirSync } from "node:fs";
type State = { roomTemp: number; outsideTemp: number; heaterOn: boolean };
type Action = "turnHeaterOn" | "turnHeaterOff" | "wait";
type Transition = { state: State; action: Action; nextState: State };

// seeded PRNG so the numbers in the post are reproducible
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
function record(episodes: number, steps: number, outLo = 2): Transition[][] {
  const log: Transition[][] = [];
  for (let e = 0; e < episodes; e++) {
    let state: State = { roomTemp: 10 + rand() * 12, outsideTemp: outLo + rand() * (12 - outLo), heaterOn: rand() < 0.5 };
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

// ---- the 104 model, thirteen weights ----------------------------------------------
function inputs(state: State, action: Action) {
  const heaterOn = heaterAfter(state, action);
  return { room: (state.roomTemp - 17) / 5, outside: (state.outsideTemp - 7) / 5, heater: heaterOn ? 1 : 0, heaterOn };
}
const keys104 = ["room", "outside", "heater", "bias", "fieldRoom", "fieldOutside", "fieldBias", "uField", "uRoom", "uOutside", "uHeater", "uChange", "uBias"] as const;
type W104 = Record<typeof keys104[number], number>;
function act104(w: W104, field: number, state: State, action: Action): State {
  const f = inputs(state, action);
  const base = w.room * f.room + w.outside * f.outside + w.heater * f.heater + w.bias;
  const extra = w.fieldRoom * f.room + w.fieldOutside * f.outside + w.fieldBias;
  return { roomTemp: state.roomTemp + base + field * extra, outsideTemp: state.outsideTemp, heaterOn: f.heaterOn };
}
function observe104(w: W104, field: number, state: State, action: Action, nextState: State) {
  const f = inputs(state, action), change = (nextState.roomTemp - state.roomTemp) / 5;
  return Math.tanh(w.uField * field + w.uRoom * f.room + w.uOutside * f.outside + w.uHeater * f.heater + w.uChange * change + w.uBias);
}
function train104(log: Transition[][], epochs: number, lr: number) {
  const w = {} as W104;
  for (const k of keys104) w[k] = k.startsWith("u") ? (rand() - 0.5) * 0.2 : 0;
  const n = log.flat().length;
  let loss = 0;
  for (let epoch = 0; epoch < epochs; epoch++) {
    const g = {} as W104;
    for (const k of keys104) g[k] = 0;
    loss = 0;
    for (const episode of log) {
      const steps: any[] = [];
      let field = 0;
      for (const t of episode) {
        const f = inputs(t.state, t.action), target = t.nextState.roomTemp - t.state.roomTemp;
        const base = w.room * f.room + w.outside * f.outside + w.heater * f.heater + w.bias;
        const extra = w.fieldRoom * f.room + w.fieldOutside * f.outside + w.fieldBias;
        const err = base + field * extra - target;
        loss += err * err;
        const change = target / 5;
        const next = Math.tanh(w.uField * field + w.uRoom * f.room + w.uOutside * f.outside + w.uHeater * f.heater + w.uChange * change + w.uBias);
        steps.push({ f, change, field, extra, err, next });
        field = next;
      }
      let dField = 0;
      for (let i = steps.length - 1; i >= 0; i--) {
        const s = steps[i], dPre = dField * (1 - s.next * s.next);
        g.uField += dPre * s.field; g.uRoom += dPre * s.f.room; g.uOutside += dPre * s.f.outside;
        g.uHeater += dPre * s.f.heater; g.uChange += dPre * s.change; g.uBias += dPre;
        g.room += s.err * s.f.room; g.outside += s.err * s.f.outside; g.heater += s.err * s.f.heater; g.bias += s.err;
        g.fieldRoom += s.err * s.field * s.f.room; g.fieldOutside += s.err * s.field * s.f.outside; g.fieldBias += s.err * s.field;
        dField = s.err * s.extra + dPre * w.uField;
      }
    }
    for (const k of keys104) w[k] -= lr * g[k] / n;
  }
  return { w, loss: loss / n };
}

// ---- the physics-shaped model (106, round two) ----------------------------------
// act:     change = (leak + doorLeak * field) * (outside - room) + heat * heater
// observe: the 104 one, unchanged
const physicsKeys = ["leak", "doorLeak", "heat", "bias", "uField", "uRoom", "uOutside", "uHeater", "uChange", "uBias"] as const;
type PhysicsWeights = Record<typeof physicsKeys[number], number>;

class PhysicsModel {
  #field = 0;
  w: PhysicsWeights;
  constructor(w: PhysicsWeights) { this.w = w; }

  reset() { this.#field = 0; }
  get field() { return this.#field; }

  act(state: State, action: Action): State {
    const w = this.w, heaterOn = heaterAfter(state, action);
    const gap = state.outsideTemp - state.roomTemp;
    const change = (w.leak + w.doorLeak * this.#field) * gap + (heaterOn ? w.heat : 0) + w.bias;
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

// Same forwards-and-backwards walk as 104. The gap is divided by ten inside the loop so
// the leak gradients are on the same scale as the heater's; the two leak weights get a
// tenth of the learning rate for the same reason. With withBias false the bias stays at zero.
function trainPhysics(log: Transition[][], epochs: number, withBias = false) {
  const w = {} as PhysicsWeights;
  for (const k of physicsKeys) w[k] = k.startsWith("u") ? (rand() - 0.5) * 0.2 : 0;
  const n = log.flat().length;
  const losses: number[] = [];
  for (let epoch = 0; epoch < epochs; epoch++) {
    const g = {} as PhysicsWeights;
    for (const k of physicsKeys) g[k] = 0;
    let loss = 0;
    for (const episode of log) {
      const steps: { f: ReturnType<typeof inputs>; gap: number; change: number; field: number; err: number; next: number }[] = [];
      let field = 0;
      for (const t of episode) {
        const f = inputs(t.state, t.action), gap = (t.state.outsideTemp - t.state.roomTemp) / 10;
        const target = t.nextState.roomTemp - t.state.roomTemp;
        const err = (w.leak + w.doorLeak * field) * gap * 10 + w.heat * f.heater + w.bias - target;
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
        g.leak += s.err * s.gap; g.doorLeak += s.err * s.field * s.gap; g.heat += s.err * s.f.heater; g.bias += s.err;
        dField = s.err * w.doorLeak * s.gap * 10 + dPre * w.uField;
      }
    }
    for (const k of physicsKeys) {
      if (k === "bias" && !withBias) continue;
      w[k] -= (k === "leak" || k === "doorLeak" ? 0.03 : 0.3) * g[k] / n;
    }
    losses.push(loss / n);
  }
  return { w, losses };
}

// ---- fit everything ------------------------------------------------------------------
const log = record(80, 60);
let t0 = performance.now();
const m104 = train104(log, 3000, 0.3);
console.log("104 model loss", m104.loss.toFixed(4), "(0.1717 in the 104 post) in", ms(t0));
seed = 99; t0 = performance.now();
const physics = trainPhysics(log, 3000);
const physicsMs = performance.now() - t0;
console.log("physics-shaped model loss", physics.losses.at(-1)!.toFixed(4), "in", physicsMs.toFixed(0), "ms");
seed = 99;
const withBias = trainPhysics(log, 3000, true);
console.log("with a bias, loss", withBias.losses.at(-1)!.toFixed(4));

// ---- the 104 test room: same seed, door open from step 10 to step 20 ------------------
seed = 7;
const start: State = { roomTemp: 22, outsideTemp: -4, heaterOn: false };
const script: Action[] = ["turnHeaterOn", ...Array(29).fill("wait") as Action[]];
const truth: State[] = [start];
const doors: boolean[] = [];
for (let i = 0; i < 30; i++) { doorOpen = i >= 10 && i < 20; doors.push(doorOpen); truth.push(realRoom(truth[i], script[i])); }

// a model that has watched the test room up to step `from`
function physicsAt(w: PhysicsWeights, from: number) {
  const m = new PhysicsModel(w);
  for (let i = 0; i < from; i++) m.observe(truth[i], script[i], truth[i + 1]);
  return m;
}
// roll a heater schedule forward blind from a state, with the field held where it is
function rollout(m: { act(s: State, a: Action): State }, s: State, heater: boolean[]): number[] {
  const out: number[] = [];
  for (const on of heater) { s = m.act(s, on ? "turnHeaterOn" : "turnHeaterOff"); out.push(s.roomTemp); }
  return out;
}
const as104 = (w: W104, field: number) => ({ act: (s: State, a: Action) => act104(w, field, s, a) });
function field104At(w: W104, from: number) {
  let field = 0;
  for (let i = 0; i < from; i++) field = observe104(w, field, truth[i], script[i], truth[i + 1]);
  return field;
}

const pw = physics.w;
console.log("\n== the physics-shaped model ==");
console.log("weights", JSON.stringify(Object.fromEntries(Object.entries(pw).map(([k, v]) => [k, +v.toFixed(3)]))));
{
  // the field on the test room, step by step, with one-step errors
  const m = new PhysicsModel(pw);
  let f104 = 0;
  const rows: string[] = [];
  const err = { p: [] as number[], l: [] as number[] };
  for (let i = 0; i < 30; i++) {
    const p = m.act(truth[i], script[i]).roomTemp, l = act104(m104.w, f104, truth[i], script[i]).roomTemp;
    err.p.push(p - truth[i + 1].roomTemp); err.l.push(l - truth[i + 1].roomTemp);
    m.observe(truth[i], script[i], truth[i + 1]);
    f104 = observe104(m104.w, f104, truth[i], script[i], truth[i + 1]);
    rows.push(`${i + 1} door ${doors[i] ? "open  " : "closed"} real ${truth[i + 1].roomTemp.toFixed(2)} | physics ${p.toFixed(2)} field ${m.field.toFixed(2)} leak ${(pw.leak + pw.doorLeak * m.field).toFixed(3)} | 104 ${l.toFixed(2)} field ${f104.toFixed(2)}`);
  }
  console.log(rows.join("\n"));
  const rms = (a: number[], skip = new Set<number>()) => Math.sqrt(mean(a.filter((_, i) => !skip.has(i)).map(v => v * v)));
  const skip = new Set([10, 20]);
  console.log("one step ahead over 30 steps: physics", rms(err.p).toFixed(2), "104", rms(err.l).toFixed(2), "| without the two door steps: physics", rms(err.p, skip).toFixed(2), "104", rms(err.l, skip).toFixed(2));
  // effective leak with the door shut: field over steps 1-10; open: steps 12-20
  const fields: number[] = []; const m2 = new PhysicsModel(pw);
  for (let i = 0; i < 30; i++) { m2.observe(truth[i], script[i], truth[i + 1]); fields.push(m2.field); }
  const shut = mean(fields.slice(0, 10)), open = mean(fields.slice(11, 20));
  console.log("field shut (mean steps 1-10)", shut.toFixed(3), "-> leak", (pw.leak + pw.doorLeak * shut).toFixed(3), "| open (12-20)", open.toFixed(3), "-> leak", (pw.leak + pw.doorLeak * open).toFixed(3), "| max", Math.max(...fields).toFixed(2), "| leak at field 0", pw.leak.toFixed(3), "at field 1", (pw.leak + pw.doorLeak).toFixed(3));
  for (const from of [5, 13, 23]) {
    const real = truth[from + 5].roomTemp;
    const p = rollout(physicsAt(pw, from), truth[from], Array(5).fill(true)).at(-1)!;
    const l = rollout(as104(m104.w, field104At(m104.w, from)), truth[from], Array(5).fill(true)).at(-1)!;
    console.log(`forecast from step ${from}: real ${real.toFixed(2)}, physics ${p.toFixed(2)}, 104 ${l.toFixed(2)}`);
  }
  const realF = truth.slice(14, 19).map(s => s.roomTemp);
  const pf = rollout(physicsAt(pw, 13), truth[13], Array(5).fill(true));
  const lf = rollout(as104(m104.w, field104At(m104.w, 13)), truth[13], Array(5).fill(true));
  const rmse = (f: number[]) => Math.sqrt(mean(f.map((v, i) => (v - realF[i]) ** 2)));
  console.log("door forecast rmse: physics", rmse(pf).toFixed(2), "104", rmse(lf).toFixed(2));
}

// heater off for twenty steps from step 5, door shut: the real room averaged over 200 draughts
function realMean(from: State, heater: string, draws = 200) {
  const sums = Array(heater.length).fill(0);
  for (let k = 0; k < draws; k++) {
    seed = 1000 + k; doorOpen = false; let s = from;
    [...heater].forEach((c, i) => { s = realRoom(s, c === "1" ? "turnHeaterOn" : "turnHeaterOff"); sums[i] += s.roomTemp; });
  }
  return sums.map(v => v / draws);
}
const off20 = Array(20).fill(false);
const offRuns = {
  real: realMean(truth[5], "0".repeat(20)),
  physics: rollout(physicsAt(pw, 5), truth[5], off20),
  bias: rollout(physicsAt(withBias.w, 5), truth[5], off20),
  m104: rollout(as104(m104.w, field104At(m104.w, 5)), truth[5], off20),
  cold: [] as number[],
};
{
  // the thirteen weights refitted on logs that go down to minus six outside
  seed = 20261102;
  const cold = record(80, 60, -6);
  const c = train104(cold, 3000, 0.3);
  offRuns.cold = rollout(as104(c.w, field104At(c.w, 5)), truth[5], off20);
  console.log("104 refitted on logs from -6 to 12 outside: loss", c.loss.toFixed(4));
}
console.log("\n== heater off for twenty steps from step 5, step 25 ==");
console.log(Object.entries(offRuns).map(([k, v]) => `${k} ${v.at(-1)!.toFixed(2)}`).join(" | "),
  `| bias fit: bias ${withBias.w.bias.toFixed(3)}, leak ${withBias.w.leak.toFixed(3)}, heat ${withBias.w.heat.toFixed(2)}, field at step 5 ${physicsAt(withBias.w, 5).field.toFixed(3)} so leak ${(withBias.w.leak + withBias.w.doorLeak * physicsAt(withBias.w, 5).field).toFixed(3)}, settles ${(withBias.w.bias / (withBias.w.leak + withBias.w.doorLeak * physicsAt(withBias.w, 5).field)).toFixed(1)} from outside`);

// ---- planning --------------------------------------------------------------------
// The 106 question: from the step 5 reading, keep every reading from step 6 to 25 at or
// above 12, with as few heater steps as possible.
type Plan = { heater: boolean[]; cost: number };
function cost(temps: number[], heater: boolean[], target: number[]) {
  let c = 0, slack = Infinity;
  temps.forEach((t, i) => {
    if (heater[i]) c += 1;                                  // one per heater step
    if (t < target[i]) c += 100 * (target[i] - t);          // a hundred per degree short
    slack = Math.min(slack, t - target[i]);
  });
  return c - 0.01 * Math.min(1, Math.max(0, slack));        // and, between equals, the one with more room to spare
}

// 1. try every schedule
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

// 2. random shooting: try n random schedules, keep the cheapest
function shoot(predict: (h: boolean[]) => number[], target: number[], n: number): Plan {
  let best: Plan = { heater: [], cost: Infinity };
  for (let k = 0; k < n; k++) {
    const heater = target.map(() => rand() < 0.5);
    const c = cost(predict(heater), heater, target);
    if (c < best.cost) best = { heater, cost: c };
  }
  return best;
}

// 3. the cross-entropy method: sample from per-step odds, keep the best tenth,
//    set the odds to how often the best tenth had the heater on, and go again
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
function realCheck(heater: boolean[]) {
  const lows: number[] = [];
  for (let k = 0; k < 200; k++) {
    seed = 1000 + k; doorOpen = false;
    let s = truth[5], low = Infinity;
    for (const on of heater) { s = realRoom(s, on ? "turnHeaterOn" : "turnHeaterOff"); low = Math.min(low, s.roomTemp); }
    lows.push(low);
  }
  return { low: mean(lows), ok: lows.filter(l => l >= 12).length };
}

console.log("\n== the 106 question ==");
const T20 = Array(20).fill(12);
const mPhys5 = physicsAt(pw, 5), f104at5 = field104At(m104.w, 5);
const predictPhysics = (h: boolean[]) => rollout(mPhys5, truth[5], h);
const predict104 = (h: boolean[]) => rollout(as104(m104.w, f104at5), truth[5], h);
const trueRoom = { act: (s: State, a: Action) => { const on = heaterAfter(s, a); return { roomTemp: s.roomTemp + (s.outsideTemp - s.roomTemp) * LEAK_CLOSED + (on ? REAL_HEAT : 0), outsideTemp: s.outsideTemp, heaterOn: on }; } };
const predictTrue = (h: boolean[]) => rollout(trueRoom, truth[5], h);
const searchRows = ["method,setting,schedules_tried,feasible_in_100,found_16_in_100,ms_per_plan"];
let exhaustiveMs = 0;
for (const [name, predict] of [["true constants", predictTrue], ["104 model", predict104], ["physics-shaped", predictPhysics]] as const) {
  t0 = performance.now();
  const b = exhaustive(predict, T20);
  const took = performance.now() - t0;
  if (name === "physics-shaped") exhaustiveMs = took;
  const rc = realCheck(b.heater);
  console.log(`exhaustive on ${name}: ${bits(b.heater)} ${ons(b.heater)} on, model low ${Math.min(...predict(b.heater)).toFixed(2)}, in ${took.toFixed(0)} ms; real lowest ${rc.low.toFixed(2)}, ok in ${rc.ok} of 200`);
}
searchRows.push(`every schedule,,1048576,100,100,${exhaustiveMs.toFixed(0)}`);
for (const n of (process.env.QUICK ? [1000] : [100, 1000, 10000, 100000])) {
  seed = 500; const res: Plan[] = []; t0 = performance.now();
  for (let k = 0; k < 100; k++) res.push(shoot(predictPhysics, T20, n));
  const took = (performance.now() - t0) / 100;
  const counts = res.map(r => Math.min(...predictPhysics(r.heater)) >= 12 ? ons(r.heater) : NaN);
  const feasible = counts.filter(c => !isNaN(c));
  console.log(`random shooting, ${n}: feasible ${feasible.length} of 100, found 16 in ${feasible.filter(c => c === 16).length}, heater steps of feasible ${Math.min(...feasible)}-${Math.max(...feasible)} mean ${mean(feasible).toFixed(1)}; ${took.toFixed(1)} ms each`);
  searchRows.push(`random shooting,${n} schedules,${n},${feasible.length},${feasible.filter(c => c === 16).length},${took.toFixed(1)}`);
}
for (const [samples, rounds] of [[100, 6], [200, 6], [200, 10]]) {
  seed = 600; const res: Plan[] = []; t0 = performance.now();
  for (let k = 0; k < 100; k++) res.push(cem(predictPhysics, T20, samples, rounds));
  const took = (performance.now() - t0) / 100;
  const counts = res.map(r => Math.min(...predictPhysics(r.heater)) >= 12 ? ons(r.heater) : NaN);
  const hist: Record<string, number> = {}; for (const c of counts) hist[String(c)] = (hist[String(c)] ?? 0) + 1;
  const plans: Record<string, number> = {}; for (const r of res) plans[bits(r.heater)] = (plans[bits(r.heater)] ?? 0) + 1;
  console.log(`CEM ${samples} x ${rounds}: heater steps ${JSON.stringify(hist)}, distinct plans ${JSON.stringify(plans)}; ${took.toFixed(2)} ms each`);
  searchRows.push(`cross-entropy,${samples} x ${rounds} rounds,${samples * rounds},${counts.filter(c => !isNaN(c)).length},${counts.filter(c => c === 16).length},${took.toFixed(2)}`);
}
{
  const h = Array(20).fill(true); let k = 0; t0 = performance.now();
  while (performance.now() - t0 < 500) { predictPhysics(h); k++; }
  console.log("twenty-step rollouts per second:", Math.round(k * 2).toLocaleString());
}

// ---- replanning every step against a thermostat --------------------------------------
// An evening of 40 steps. Someone wants the room at 18 from step 25 and only cares that it
// stays at 12 before then. Every controller gets the same draughts, and, on evenings with
// the door, the same door, run by run.
type Evening = { outside: number; start: number; arrive: number; steps: number; noise: number[]; door: boolean[] };
function evening(k: number, outside: number, withDoor: boolean): Evening {
  seed = 5000 + k; doorOpen = false;
  const noise: number[] = [], door: boolean[] = [];
  for (let i = 0; i < 40; i++) { if (withDoor) someone(); door.push(doorOpen); noise.push(gauss() * NOISE); }
  return { outside, start: 14, arrive: 25, steps: 40, noise, door };
}
const wantAt = (ev: Evening, reading: number) => reading >= ev.arrive ? 18 : 12;

// What a controller sees: the reading now, the step, and a model that has watched every step so far.
type Model = { observe(s: State, a: Action, next: State): void; predict(s: State, heater: boolean[]): number[] };
type Controller = (ev: Evening, step: number, s: State, model: Model) => boolean;
const physicsModel = (): Model => {
  const m = new PhysicsModel(pw);
  return { observe: (s, a, n) => m.observe(s, a, n), predict: (s, h) => rollout(m, s, h) };
};
const model104 = (): Model => {
  let field = 0;
  return { observe: (s, a, n) => { field = observe104(m104.w, field, s, a, n); }, predict: (s, h) => rollout(as104(m104.w, field), s, h) };
};

function runEvening(ev: Evening, ctl: Controller, model: Model) {
  let s: State = { roomTemp: ev.start, outsideTemp: ev.outside, heaterOn: false };
  const temps: number[] = [], heater: boolean[] = [];
  for (let i = 0; i < ev.steps; i++) {
    const on = ctl(ev, i, s, model);
    const leak = ev.door[i] ? LEAK_OPEN : LEAK_CLOSED;
    const next: State = { roomTemp: s.roomTemp + (s.outsideTemp - s.roomTemp) * leak + (on ? REAL_HEAT : 0) + ev.noise[i], outsideTemp: s.outsideTemp, heaterOn: on };
    model.observe(s, on ? "turnHeaterOn" : "turnHeaterOff", next);
    temps.push(next.roomTemp); heater.push(on); s = next;
  }
  let short = 0, misses = 0;
  temps.forEach((t, i) => { const want = wantAt(ev, i + 1); if (t < want) { short += want - t; misses++; } });
  return { temps, heater, on: ons(heater), short, misses };
}

// A thermostat that switches on a degree above what is wanted and off 0.7 above that,
// and moves to the evening setting `lead` steps before the person arrives.
const thermostatWithLead = (lead: number): Controller => (ev, i, s) => {
  const want = i + 1 + lead >= ev.arrive ? 18 : 12;
  return s.roomTemp < want + 1 ? true : s.roomTemp > want + 1.7 ? false : s.heaterOn;
};
// The planner: twenty steps ahead (or to the end of the evening), plan with CEM, do the first step.
const replan: Controller = (ev, i, s, model) => {
  const target = Array.from({ length: Math.min(20, ev.steps - i) }, (_, j) => wantAt(ev, i + 1 + j));
  return cem(h => model.predict(s, h), target).heater[0];
};

function evenings(outside: number, withDoor: boolean, ctl: Controller, model: () => Model) {
  seed = 42;
  const rs = Array.from({ length: 200 }, (_, k) => runEvening(evening(k, outside, withDoor), ctl, model()));
  return { on: mean(rs.map(r => r.on)), short: mean(rs.map(r => r.short)), misses: mean(rs.map(r => r.misses)) };
}
const fmt = (r: { on: number; misses: number; short: number }) => `heater ${r.on.toFixed(1)}, readings short ${r.misses.toFixed(1)}, degree-steps short ${r.short.toFixed(2)}`;

console.log("\n== an evening ==");
// the lead a thermostat needs, by outside temperature: the cheapest with fewer than a quarter of a reading short
const bestLead: Record<number, number> = {};
for (const outside of [3, 5, 8]) {
  const line: string[] = [];
  let best = { lead: -1, on: Infinity };
  for (let lead = 0; lead <= 24; lead++) {
    const r = evenings(outside, false, thermostatWithLead(lead), physicsModel);
    line.push(`${lead}:${r.on.toFixed(1)}/${r.misses.toFixed(2)}`);
    if (r.misses < 0.25 && r.on < best.on) best = { lead, on: r.on };
  }
  bestLead[outside] = best.lead;
  console.log(`outside ${outside}: best lead ${best.lead} | ` + line.join(" "));
}
t0 = performance.now();
const evRows = ["outside_c,door,controller,heater_steps,readings_short,degree_steps_short"];
for (const outside of [3, 5, 8]) {
  for (const withDoor of [false, true]) {
    const ctls: [string, Controller, () => Model][] = [
      ["thermostat, no lead", thermostatWithLead(0), physicsModel],
      ["thermostat, lead tuned for 5", thermostatWithLead(bestLead[5]), physicsModel],
      ...(outside === 5 ? [] : [[`thermostat, lead tuned for ${outside}`, thermostatWithLead(bestLead[outside]), physicsModel] as [string, Controller, () => Model]]),
      ["planner, physics-shaped model", replan, physicsModel],
      ["planner, 104 model", replan, model104],
    ];
    for (const [name, ctl, model] of ctls) {
      const r = evenings(outside, withDoor, ctl, model);
      console.log(`outside ${outside} ${withDoor ? "door  " : "sealed"} | ${name.padEnd(32)} ${fmt(r)}`);
      evRows.push(`${outside},${withDoor ? "yes" : "no"},"${name}",${r.on.toFixed(1)},${r.misses.toFixed(2)},${r.short.toFixed(2)}`);
    }
  }
}
console.log("evenings took", ms(t0));
{
  seed = 42; t0 = performance.now(); const m = physicsModel(); const ev = evening(0, 5, false);
  for (let k = 0; k < 1000; k++) replan(ev, 0, { roomTemp: 14, outsideTemp: 5, heaterOn: false }, m);
  console.log("one decision:", ((performance.now() - t0) / 1000).toFixed(2), "ms");
}

// one evening at 3 outside for the chart, the planner against the thermostat tuned for 5:
// the evening whose results sit closest to the means over all 200
const at3 = { p: [] as ReturnType<typeof runEvening>[], t: [] as ReturnType<typeof runEvening>[] };
for (let k = 0; k < 200; k++) {
  seed = 42; at3.p.push(runEvening(evening(k, 3, false), replan, physicsModel()));
  at3.t.push(runEvening(evening(k, 3, false), thermostatWithLead(bestLead[5]), physicsModel()));
}
const avg = (rs: typeof at3.p) => ({ on: mean(rs.map(r => r.on)), misses: mean(rs.map(r => r.misses)) });
const ap = avg(at3.p), at = avg(at3.t);
let chartK = 0, bestD = Infinity;
for (let k = 0; k < 200; k++) {
  const d = Math.abs(at3.p[k].misses - ap.misses) + Math.abs(at3.t[k].misses - at.misses) + Math.abs(at3.p[k].on - ap.on) / 2 + Math.abs(at3.t[k].on - at.on) / 2;
  if (d < bestD) { bestD = d; chartK = k; }
}
const ev0 = evening(chartK, 3, false), pRun = at3.p[chartK], tRun = at3.t[chartK];
console.log(`chart evening ${chartK}, outside 3: planner ${pRun.on} on, ${pRun.misses} short; thermostat ${tRun.on} on, ${tRun.misses} short (means ${ap.on.toFixed(1)}/${ap.misses.toFixed(1)} and ${at.on.toFixed(1)}/${at.misses.toFixed(1)})`);
console.log("planner    ", bits(pRun.heater), "\nthermostat ", bits(tRun.heater));

if (process.argv[2] === "csv") {
  mkdirSync("static/data", { recursive: true });
  writeFileSync("static/data/heater-plan-heater-off.csv",
    "step,real_mean_c,physics_shaped_c,physics_with_bias_c,model_104_c,model_104_cold_logs_c\n" +
    `5,${[1, 2, 3, 4, 5].map(() => truth[5].roomTemp.toFixed(2)).join(",")}\n` +
    offRuns.real.map((_, i) => [i + 6, offRuns.real[i], offRuns.physics[i], offRuns.bias[i], offRuns.m104[i], offRuns.cold[i]].map((v, j) => j ? v.toFixed(2) : v).join(",")).join("\n") + "\n");
  writeFileSync("static/data/heater-plan-search.csv", searchRows.join("\n") + "\n");
  writeFileSync("static/data/heater-plan-evenings.csv", evRows.join("\n") + "\n");
  writeFileSync("static/data/heater-plan-evening.csv",
    "step,target_c,planner_c,planner_heater,thermostat_c,thermostat_heater\n" +
    `0,12,${ev0.start.toFixed(2)},,${ev0.start.toFixed(2)},\n` +
    pRun.temps.map((t, i) => `${i + 1},${wantAt(ev0, i + 1)},${t.toFixed(2)},${pRun.heater[i] ? 1 : 0},${tRun.temps[i].toFixed(2)},${tRun.heater[i] ? 1 : 0}`).join("\n") + "\n");
  console.log("wrote static/data/heater-plan-*.csv");
}
