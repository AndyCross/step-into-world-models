// The code from the world models 104 post, end to end, plus the CSV output
// for its charts. Run from the repo root with Node 23 or newer:
//
//     node --no-warnings tools/heater_door_fit.ts
//
// Writes static/data/heater-door-loss.csv, heater-door-unseen.csv and
// heater-door-forecast.csv. Then run tools/heater_charts.py to draw them.
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

// ---- the room, now with a door --------------------------------------------
const LEAK_CLOSED = 0.07;   // the 103 room
const LEAK_OPEN = Number(process.env.LEAK_OPEN ?? 0.25);   // three and a half times leakier with the door open
const REAL_HEAT = 1.2;
const NOISE = 0.1;

let doorOpen = false;       // the room's secret; never written to the log

function someone() {
  // opens the door about once every 25 steps, leaves it open about 8
  if (doorOpen) { if (rand() < 1 / 8) doorOpen = false; }
  else if (rand() < 1 / 25) doorOpen = true;
}

function realRoom(state: State, action: Action): State {
  const heaterOn = heaterAfter(state, action);
  const leak = (state.outsideTemp - state.roomTemp) * (doorOpen ? LEAK_OPEN : LEAK_CLOSED);
  const heat = heaterOn ? REAL_HEAT : 0;
  const draught = gauss() * NOISE;
  return { roomTemp: state.roomTemp + leak + heat + draught, outsideTemp: state.outsideTemp, heaterOn };
}

// ---- recording, one array per episode ----------------------------------------
function thermostat(state: State, setpoint: number): Action {
  if (rand() < 0.2) return (["turnHeaterOn", "turnHeaterOff", "wait"] as Action[])[Math.floor(rand() * 3)];
  if (state.roomTemp < setpoint - 0.5 && !state.heaterOn) return "turnHeaterOn";
  if (state.roomTemp > setpoint + 0.5 && state.heaterOn) return "turnHeaterOff";
  return "wait";
}

const doorTrace: boolean[] = [];   // kept outside the log, only for marking the charts

function record(episodes: number, steps: number): Transition[][] {
  const log: Transition[][] = [];
  for (let e = 0; e < episodes; e++) {
    let state: State = {
      roomTemp: 10 + rand() * 12,
      outsideTemp: 2 + rand() * 10,
      heaterOn: rand() < 0.5,
    };
    const setpoint = 18 + rand() * 3;
    doorOpen = false;
    const episode: Transition[] = [];
    for (let t = 0; t < steps; t++) {
      someone();
      const action = thermostat(state, setpoint);
      const nextState = realRoom(state, action);
      episode.push({ state, action, nextState });
      doorTrace.push(doorOpen);
      state = nextState;
    }
    log.push(episode);
  }
  return log;
}

const log = record(80, 60);
const rows = log.flat();
console.log("transitions:", rows.length, "door open on", (doorTrace.filter(Boolean).length / rows.length * 100).toFixed(0) + "% of them");

// ---- the 103 model, refitted to the door log -------------------------------------
type Weights4 = { room: number; outside: number; heater: number; bias: number };

function features(state: State, action: Action) {
  const heaterOn = heaterAfter(state, action);
  return { room: state.roomTemp, outside: state.outsideTemp, heater: heaterOn ? 1 : 0, bias: 1, heaterOn };
}

function train4(log: Transition[], epochs: number, lr: number) {
  const w: Weights4 = { room: 0, outside: 0, heater: 0, bias: 0 };
  const losses: number[] = [];
  for (let epoch = 0; epoch < epochs; epoch++) {
    const g: Weights4 = { room: 0, outside: 0, heater: 0, bias: 0 };
    let loss = 0;
    for (const t of log) {
      const f = features(t.state, t.action);
      const target = t.nextState.roomTemp - t.state.roomTemp;
      const guess = w.room * f.room + w.outside * f.outside + w.heater * f.heater + w.bias * f.bias;
      const err = guess - target;
      loss += err * err;
      g.room += err * f.room; g.outside += err * f.outside; g.heater += err * f.heater; g.bias += err * f.bias;
    }
    const n = log.length;
    w.room -= lr * g.room / n; w.outside -= lr * g.outside / n; w.heater -= lr * g.heater / n; w.bias -= lr * g.bias / n;
    losses.push(loss / n);
  }
  return { w, losses };
}

let t0 = Date.now();
const four = train4(rows, 10000, 0.003);
console.log("103 model refitted to the door log:", JSON.stringify(Object.fromEntries(Object.entries(four.w).map(([k, v]) => [k, +v.toFixed(3)]))),
  "loss", four.losses[four.losses.length - 1].toFixed(4), "in", Date.now() - t0, "ms");
function predict4(w: Weights4, state: State, action: Action): State {
  const f = features(state, action);
  const delta = w.room * f.room + w.outside * f.outside + w.heater * f.heater + w.bias;
  return { roomTemp: state.roomTemp + delta, outsideTemp: state.outsideTemp, heaterOn: f.heaterOn };
}
// and the weights 103 found in the sealed room, for comparison
const sealed: Weights4 = { room: -0.074, outside: 0.069, heater: 1.19, bias: 0.077 };

// ---- the model with a private field ---------------------------------------------
// Inputs are centred and scaled this time (see the post), so the loop converges in
// thousands of epochs rather than tens of thousands.
function inputs(state: State, action: Action) {
  const heaterOn = heaterAfter(state, action);
  return { room: (state.roomTemp - 17) / 5, outside: (state.outsideTemp - 7) / 5, heater: heaterOn ? 1 : 0, heaterOn };
}

type Weights = {
  // act: change in room temperature = base + field * extra
  room: number; outside: number; heater: number; bias: number;
  fieldRoom: number; fieldOutside: number; fieldBias: number;
  // observe: next field = tanh(a weighted sum of the field and what just happened)
  uField: number; uRoom: number; uOutside: number; uHeater: number; uChange: number; uBias: number;
};
const keys = Object.keys({
  room: 0, outside: 0, heater: 0, bias: 0, fieldRoom: 0, fieldOutside: 0, fieldBias: 0,
  uField: 0, uRoom: 0, uOutside: 0, uHeater: 0, uChange: 0, uBias: 0,
} satisfies Weights) as (keyof Weights)[];

class WorldModel {
  #field = 0;              // s_t lives here, and nowhere in the logs
  w: Weights;
  constructor(w: Weights) { this.w = w; }

  reset() { this.#field = 0; }
  get field() { return this.#field; }

  act(state: State, action: Action): State {
    const f = inputs(state, action), w = this.w;
    const base = w.room * f.room + w.outside * f.outside + w.heater * f.heater + w.bias;
    const extra = w.fieldRoom * f.room + w.fieldOutside * f.outside + w.fieldBias;
    return { roomTemp: state.roomTemp + base + this.#field * extra, outsideTemp: state.outsideTemp, heaterOn: f.heaterOn };
  }

  observe(state: State, action: Action, nextState: State) {
    const f = inputs(state, action), w = this.w;
    const change = (nextState.roomTemp - state.roomTemp) / 5;
    this.#field = Math.tanh(
      w.uField * this.#field + w.uRoom * f.room + w.uOutside * f.outside +
      w.uHeater * f.heater + w.uChange * change + w.uBias);
  }
}

function train(log: Transition[][], epochs: number, lr: number) {
  const w = {} as Weights;
  for (const k of keys) w[k] = k.startsWith("u") ? (rand() - 0.5) * 0.2 : 0;
  const losses: number[] = [];
  const n = log.flat().length;
  for (let epoch = 0; epoch < epochs; epoch++) {
    const g = {} as Weights;
    for (const k of keys) g[k] = 0;
    let loss = 0;
    for (const episode of log) {
      // forwards through the episode, remembering what the field was at every step
      const steps: { f: ReturnType<typeof inputs>; change: number; field: number; extra: number; err: number; next: number }[] = [];
      let field = 0;
      for (const t of episode) {
        const f = inputs(t.state, t.action);
        const target = t.nextState.roomTemp - t.state.roomTemp;
        const base = w.room * f.room + w.outside * f.outside + w.heater * f.heater + w.bias;
        const extra = w.fieldRoom * f.room + w.fieldOutside * f.outside + w.fieldBias;
        const err = base + field * extra - target;
        loss += err * err;
        const change = target / 5;
        const next = Math.tanh(w.uField * field + w.uRoom * f.room + w.uOutside * f.outside + w.uHeater * f.heater + w.uChange * change + w.uBias);
        steps.push({ f, change, field, extra, err, next });
        field = next;
      }
      // backwards through the episode: how much did each step's field matter to everything after it?
      let dField = 0;   // d loss / d (field after this step)
      for (let i = steps.length - 1; i >= 0; i--) {
        const s = steps[i];
        const dPre = dField * (1 - s.next * s.next);           // through the tanh
        g.uField += dPre * s.field; g.uRoom += dPre * s.f.room; g.uOutside += dPre * s.f.outside;
        g.uHeater += dPre * s.f.heater; g.uChange += dPre * s.change; g.uBias += dPre;
        g.room += s.err * s.f.room; g.outside += s.err * s.f.outside; g.heater += s.err * s.f.heater; g.bias += s.err;
        g.fieldRoom += s.err * s.field * s.f.room; g.fieldOutside += s.err * s.field * s.f.outside; g.fieldBias += s.err * s.field;
        dField = s.err * s.extra + dPre * w.uField;            // on to the step before
      }
    }
    for (const k of keys) w[k] -= lr * g[k] / n;
    losses.push(loss / n);
  }
  return { w, losses };
}

t0 = Date.now();
const EPOCHS = 3000, LR = 0.3;
const latent = train(log, EPOCHS, LR);
console.log("latent model:", JSON.stringify(Object.fromEntries(Object.entries(latent.w).map(([k, v]) => [k, +v.toFixed(3)]))),
  "loss", latent.losses[EPOCHS - 1].toFixed(4), "in", Date.now() - t0, "ms");
console.log("loss trace", [0, 99, 499, 999, 1999, 2999].map(i => `${i + 1}:${latent.losses[i].toFixed(4)}`).join(" "));

// ---- the two floors ------------------------------------------------------------
// A model that sees the door, and one that sees it a step late. Closed form, so it is
// exact; the chart calls the second the "one step late" floor.
{
  const feats = (t: Transition, d: number) => [t.state.roomTemp, t.state.outsideTemp, heaterAfter(t.state, t.action) ? 1 : 0, 1, d * t.state.roomTemp, d * t.state.outsideTemp, d];
  const solve = (A: number[][], b: number[]) => { const n = b.length; const M = A.map((r, i) => [...r, b[i]]); for (let c = 0; c < n; c++) { let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r; [M[c], M[p]] = [M[p], M[c]]; for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; } } return M.map((r, i) => r[n] / r[i]); };
  const A = Array.from({ length: 7 }, () => Array(7).fill(0)), b = Array(7).fill(0);
  rows.forEach((t, i) => { const x = feats(t, doorTrace[i] ? 1 : 0), y = t.nextState.roomTemp - t.state.roomTemp; for (let a = 0; a < 7; a++) { b[a] += x[a] * y; for (let c = 0; c < 7; c++) A[a][c] += x[a] * x[c]; } });
  const w = solve(A, b);
  let now = 0, late = 0, i = 0;
  for (const episode of log) { let prev = 0; for (const t of episode) { const d = doorTrace[i++] ? 1 : 0; const y = t.nextState.roomTemp - t.state.roomTemp; const p = (dd: number) => feats(t, dd).reduce((s, v, j) => s + v * w[j], 0); now += (p(d) - y) ** 2; late += (p(prev) - y) ** 2; prev = d; } }
  console.log("floor if the door were logged:", (now / rows.length).toFixed(4), " floor if it were logged one step late:", (late / rows.length).toFixed(4));
  console.log("  (the logged-door fit: closed leak", (-w[0]).toFixed(3), "open adds", (-w[4]).toFixed(3) + ")");
  // The one-step-late line is a reference, not a floor: a model that knows the door a step
  // late can hedge for the chance it has moved since (1 in 8 while open, 1 in 25 while shut),
  // or be refitted on the late door state, and both score lower.
  let hedged = 0, surprise = 0, moves = 0; i = 0;
  const A2 = Array.from({ length: 7 }, () => Array(7).fill(0)), b2 = Array(7).fill(0);
  for (const episode of log) { let prev = 0; for (const t of episode) { const d = doorTrace[i++] ? 1 : 0; const y = t.nextState.roomTemp - t.state.roomTemp; const x = feats(t, prev); hedged += (feats(t, prev ? 7 / 8 : 1 / 25).reduce((s, v, j) => s + v * w[j], 0) - y) ** 2; if (d !== prev) { moves++; surprise += (x.reduce((s, v, j) => s + v * w[j], 0) - y) ** 2; } for (let a = 0; a < 7; a++) { b2[a] += x[a] * y; for (let c = 0; c < 7; c++) A2[a][c] += x[a] * x[c]; } prev = d; } }
  const w2 = solve(A2, b2); let refit = 0; i = 0;
  for (const episode of log) { let prev = 0; for (const t of episode) { const d = doorTrace[i++] ? 1 : 0; refit += (feats(t, prev).reduce((s, v, j) => s + v * w2[j], 0) - (t.nextState.roomTemp - t.state.roomTemp)) ** 2; prev = d; } }
  console.log("  one step late, hedged:", (hedged / rows.length).toFixed(4), " refitted on the late door:", (refit / rows.length).toFixed(4), ` the ${moves} steps the door moved on:`, (surprise / rows.length).toFixed(4));
  (globalThis as any).floors = { now: now / rows.length, late: late / rows.length };
}

// ---- does the field track the door it never saw? --------------------------------
{
  const m = new WorldModel(latent.w);
  let i = 0, sumOpen = 0, nOpen = 0, sumClosed = 0, nClosed = 0;
  const fields: number[] = [];
  for (const episode of log) { m.reset(); for (const t of episode) { const d = m.field; fields.push(d); if (doorTrace[i++]) { sumOpen += d; nOpen++; } else { sumClosed += d; nClosed++; } m.observe(t.state, t.action, t.nextState); } }
  const mid = (sumOpen / nOpen + sumClosed / nClosed) / 2, sign = sumOpen / nOpen > sumClosed / nClosed;
  let agree = 0; fields.forEach((d, j) => { if ((d > mid) === (doorTrace[j] === sign)) agree++; });
  console.log("field before each step: mean", (sumOpen / nOpen).toFixed(2), "when the door was open,", (sumClosed / nClosed).toFixed(2), "when closed; on the right side of the midpoint", (agree / fields.length * 100).toFixed(1) + "% of the time");
}

// ---- a room it has never seen, one step at a time ---------------------------------
seed = 7;
const start: State = { roomTemp: 22, outsideTemp: -4, heaterOn: false };
const script: Action[] = ["turnHeaterOn", ...Array(29).fill("wait") as Action[]];
const truth: State[] = [start];
const doors: boolean[] = [];
for (let i = 0; i < 30; i++) {
  doorOpen = i >= 10 && i < 20;   // open for ten steps in the middle
  doors.push(doorOpen);
  truth.push(realRoom(truth[i], script[i]));
}

const m = new WorldModel(latent.w);
m.reset();
const oneStep4: number[] = [start.roomTemp], oneStepSealed: number[] = [start.roomTemp], oneStepLatent: number[] = [start.roomTemp], fieldTrace: number[] = [m.field];
for (let i = 0; i < 30; i++) {
  oneStep4.push(predict4(four.w, truth[i], script[i]).roomTemp);
  oneStepSealed.push(predict4(sealed, truth[i], script[i]).roomTemp);
  oneStepLatent.push(m.act(truth[i], script[i]).roomTemp);
  m.observe(truth[i], script[i], truth[i + 1]);
  fieldTrace.push(m.field);
}
const rmse = (p: number[]) => Math.sqrt(truth.slice(1).reduce((a, s, i) => a + (s.roomTemp - p[i + 1]) ** 2, 0) / 30);
console.log("one step ahead, rmse over 30 steps: refitted 103 model", rmse(oneStep4).toFixed(2), " sealed-room 103 weights", rmse(oneStepSealed).toFixed(2), " latent model", rmse(oneStepLatent).toFixed(2));
const rmseBut = (p: number[]) => Math.sqrt(truth.slice(1).reduce((a, s, i) => (i === 10 || i === 20) ? a : a + (s.roomTemp - p[i + 1]) ** 2, 0) / 28);
console.log("  leaving out the two steps the door moved on: refitted 103 model", rmseBut(oneStep4).toFixed(2), " latent model", rmseBut(oneStepLatent).toFixed(2));
console.log("step,door,real,four,sealed,latent,field");
for (let i = 0; i <= 30; i++) console.log(`${i},${i ? (doors[i - 1] ? 1 : 0) : 0},${truth[i].roomTemp.toFixed(2)},${oneStep4[i].toFixed(2)},${oneStepSealed[i].toFixed(2)},${oneStepLatent[i].toFixed(2)},${fieldTrace[i].toFixed(2)}`);

// ---- what if: a five step forecast from inside the open-door stretch ----------------
function forecast(from: number, steps: number) {
  // the latent model has observed everything up to `from`; now it acts without observing
  const mm = new WorldModel(latent.w); mm.reset();
  for (let i = 0; i < from; i++) mm.observe(truth[i], script[i], truth[i + 1]);
  const lat: number[] = [truth[from].roomTemp], four_: number[] = [truth[from].roomTemp];
  let sl = truth[from], s4 = truth[from];
  for (let i = from; i < from + steps; i++) { sl = mm.act(sl, script[i]); s4 = predict4(four.w, s4, script[i]); lat.push(sl.roomTemp); four_.push(s4.roomTemp); }
  return { lat, four: four_, real: truth.slice(from, from + steps + 1).map(s => s.roomTemp), field: mm.field };
}
const forecasts = [5, 13, 23].map(from => ({ from, ...forecast(from, 5) }));
for (const fc of forecasts) {
  console.log(`forecast from step ${fc.from} (field ${fc.field.toFixed(2)}): real ${fc.real.map(v => v.toFixed(1)).join(" ")} | 103 ${fc.four.map(v => v.toFixed(1)).join(" ")} | latent ${fc.lat.map(v => v.toFixed(1)).join(" ")}`);
}

// ---- CSVs for the charts ------------------------------------------------------------
mkdirSync("static/data", { recursive: true });
const floors = (globalThis as any).floors as { now: number; late: number };
const lossRows: string[] = [];
const fourFinal = four.losses[four.losses.length - 1];
for (let i = 0; i < EPOCHS; i++) if (i % 15 === 0 || i === EPOCHS - 1) lossRows.push(`${i + 1},${latent.losses[i].toFixed(5)},${fourFinal.toFixed(4)},${floors.late.toFixed(4)},${floors.now.toFixed(4)}`);
writeFileSync("static/data/heater-door-loss.csv",
  "epoch,latent_model_mse,four_weight_model_final_mse,floor_door_logged_one_step_late,floor_door_logged\n" + lossRows.join("\n") + "\n");
writeFileSync("static/data/heater-door-unseen.csv",
  "step,door_open,real_room_c,four_weight_one_step_c,latent_one_step_c,private_field\n" +
  truth.map((s, i) => `${i},${i ? (doors[i - 1] ? 1 : 0) : 0},${s.roomTemp.toFixed(2)},${oneStep4[i].toFixed(2)},${oneStepLatent[i].toFixed(2)},${fieldTrace[i].toFixed(3)}`).join("\n") + "\n");
writeFileSync("static/data/heater-door-forecast.csv",
  "from_step,steps_ahead,real_room_c,four_weight_model_c,latent_model_c\n" +
  forecasts.flatMap(fc => fc.real.map((v, i) => `${fc.from},${i},${v.toFixed(2)},${fc.four[i].toFixed(2)},${fc.lat[i].toFixed(2)}`)).join("\n") + "\n");
console.log("wrote static/data/heater-door-*.csv");
