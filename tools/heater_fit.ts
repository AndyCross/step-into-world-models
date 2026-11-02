// The code from the world models 103 post, end to end, plus the CSV output
// for its two charts. Run from the repo root with Node 23 or newer:
//
//     node --no-warnings tools/heater_fit.ts
//
// Writes static/data/heater-rollout.csv and static/data/heater-loss.csv.
// Then run tools/heater_charts.py to draw them.
import { writeFileSync, mkdirSync } from "node:fs";


type State = { roomTemp: number; outsideTemp: number; heaterOn: boolean };
type Action = "turnHeaterOn" | "turnHeaterOff" | "wait";
type Transition = { state: State; action: Action; nextState: State };

// ---- the 101 guess -------------------------------------------------------
function predictGuess(state: State, action: Action): State {
  const heaterOn =
    action === "turnHeaterOn" ? true :
    action === "turnHeaterOff" ? false :
    state.heaterOn;
  const leak = (state.outsideTemp - state.roomTemp) * 0.1;
  const heat = heaterOn ? 1.5 : 0;
  return { roomTemp: state.roomTemp + leak + heat, outsideTemp: state.outsideTemp, heaterOn };
}

// ---- the "real" room, standing in for hardware ---------------------------
// seeded PRNG so the numbers in the post are reproducible
let seed = 20261208;
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

const REAL_LEAK = 0.07;   // the room is better insulated than I guessed
const REAL_HEAT = 1.2;    // and the heater is weaker
const NOISE = 0.1;

function realRoom(state: State, action: Action): State {
  const heaterOn =
    action === "turnHeaterOn" ? true :
    action === "turnHeaterOff" ? false :
    state.heaterOn;
  const leak = (state.outsideTemp - state.roomTemp) * REAL_LEAK;
  const heat = heaterOn ? REAL_HEAT : 0;
  const draught = gauss() * NOISE;
  return { roomTemp: state.roomTemp + leak + heat + draught, outsideTemp: state.outsideTemp, heaterOn };
}

// ---- record logs ---------------------------------------------------------
function thermostat(state: State, setpoint: number): Action {
  if (rand() < 0.2) return (["turnHeaterOn", "turnHeaterOff", "wait"] as Action[])[Math.floor(rand() * 3)];
  if (state.roomTemp < setpoint - 0.5 && !state.heaterOn) return "turnHeaterOn";
  if (state.roomTemp > setpoint + 0.5 && state.heaterOn) return "turnHeaterOff";
  return "wait";
}

function record(episodes: number, steps: number): Transition[] {
  const log: Transition[] = [];
  for (let e = 0; e < episodes; e++) {
    let state: State = {
      roomTemp: 10 + rand() * 12,       // 10 to 22
      outsideTemp: 2 + rand() * 10,     // 2 to 12
      heaterOn: rand() < 0.5,
    };
    const setpoint = 18 + rand() * 3;
    for (let t = 0; t < steps; t++) {
      const action = thermostat(state, setpoint);
      const nextState = realRoom(state, action);
      log.push({ state, action, nextState });
      state = nextState;
    }
  }
  return log;
}

const log = record(80, 60);
console.log("transitions:", log.length);
console.log(JSON.stringify(log.slice(0, 2), null, 1));

// ---- the WHERE clause from 102 -------------------------------------------
const probe = log[1234];
const matches = log.filter(t =>
  t.state.roomTemp === probe.state.roomTemp &&
  t.state.outsideTemp === probe.state.outsideTemp &&
  t.state.heaterOn === probe.state.heaterOn &&
  t.action === probe.action);
console.log("exact matches for row 1234:", matches.length, "(itself)");
const near = log.filter(t =>
  Math.abs(t.state.roomTemp - probe.state.roomTemp) < 0.05 &&
  Math.abs(t.state.outsideTemp - probe.state.outsideTemp) < 0.05 &&
  t.state.heaterOn === probe.state.heaterOn &&
  t.action === probe.action);
console.log("within 0.05 degrees:", near.length, "probe:", JSON.stringify(probe));

// ---- the model: four numbers ---------------------------------------------
type Weights = { room: number; outside: number; heater: number; bias: number };

function features(state: State, action: Action) {
  const heaterOn =
    action === "turnHeaterOn" ? true :
    action === "turnHeaterOff" ? false :
    state.heaterOn;
  return { room: state.roomTemp, outside: state.outsideTemp, heater: heaterOn ? 1 : 0, bias: 1, heaterOn };
}

function predictWith(w: Weights, state: State, action: Action): State {
  const f = features(state, action);
  const delta = w.room * f.room + w.outside * f.outside + w.heater * f.heater + w.bias * f.bias;
  return { roomTemp: state.roomTemp + delta, outsideTemp: state.outsideTemp, heaterOn: f.heaterOn };
}

function train(log: Transition[], epochs: number, lr: number) {
  const w: Weights = { room: 0, outside: 0, heater: 0, bias: 0 };
  const losses: number[] = [];
  for (let epoch = 0; epoch < epochs; epoch++) {
    const g: Weights = { room: 0, outside: 0, heater: 0, bias: 0 };
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

const t0 = Date.now();
const EP = 10000, LR = 0.003;
const { w, losses } = train(log, EP, LR);
console.log("trained in ms:", Date.now() - t0);
console.log("weights:", w);
console.log("loss:", [0,9,99,999,EP/2-1,EP-1].map(i => losses[i].toFixed(4)));

// ---- a room it has never seen -------------------------------------------
const unseen: State = { roomTemp: 22, outsideTemp: -4, heaterOn: false };
const script: Action[] = ["wait", "wait", "wait", "wait", "wait", "wait", "wait", "wait", "wait", "wait",
  "turnHeaterOn", "wait", "wait", "wait", "wait", "wait", "wait", "wait", "wait", "wait",
  "turnHeaterOff", "wait", "wait", "wait", "wait", "wait", "wait", "wait", "wait", "wait"];

function rollout(start: State, actions: Action[], step: (s: State, a: Action) => State): State[] {
  const out = [start];
  for (const a of actions) out.push(step(out[out.length - 1], a));
  return out;
}
seed = 1;
const truth = rollout(unseen, script, realRoom);
const learned = rollout(unseen, script, (s, a) => predictWith(w, s, a));
const guessed = rollout(unseen, script, predictGuess);
console.log("step,truth,learned,guess101");
const rows = truth.map((s, i) => `${i},${s.roomTemp.toFixed(2)},${learned[i].roomTemp.toFixed(2)},${guessed[i].roomTemp.toFixed(2)}`);
rows.forEach(r => console.log(r));
mkdirSync("static/data", { recursive: true });
writeFileSync("static/data/heater-rollout.csv", "step,real_room_c,learned_model_c,hand_written_guess_c\n" + rows.join("\n") + "\n");
writeFileSync("static/data/heater-loss.csv", "epoch,mean_squared_error\n" + losses.map((l, i) => `${i + 1},${l.toFixed(5)}`).filter((_, i) => i % 50 === 0 || i === losses.length - 1).join("\n") + "\n");
const err = (a: State[], b: State[]) => Math.sqrt(a.reduce((acc, s, i) => acc + (s.roomTemp - b[i].roomTemp) ** 2, 0) / a.length);
console.log("rmse learned:", err(truth, learned).toFixed(3), "rmse guess:", err(truth, guessed).toFixed(3));
