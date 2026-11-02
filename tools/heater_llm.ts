// The experiment from the world models 106 post: Claude on its own, Claude with
// the logs pasted in, and Claude with the 104 model as a tool, on the 104 test room.
// Needs the Anthropic SDK and an API key. From the repo root, Node 23 or newer:
//
//     npm install --no-save @anthropic-ai/sdk@1.3.0
//     node --no-warnings tools/heater_llm.ts run <dir> [runs] [condition]   # calls the API, writes <dir>/out-*.json
//     node --no-warnings tools/heater_llm.ts score <dir>        # scores them, writes static/data/heater-llm-*.csv
//     python3 tools/heater_llm_charts.py                        # draws static/charts/heater-llm-*.svg
//
// The model and room below are tools/heater_door_fit.ts with the same seed, so the
// thirteen weights are the ones in the 104 post. Round two adds a model with the same
// field and the physics Claude proposed. A full run is eighteen requests and
// cost about three and a half dollars at Claude Opus 5.5 list prices in October 2026.
import { writeFileSync, readFileSync, mkdirSync } from "node:fs";
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

const EPOCHS = 3000, LR = 0.3;
const latent = train(log, EPOCHS, LR);
console.log("latent model loss", latent.losses[EPOCHS - 1].toFixed(4), "(0.1717 in the 104 post)");


// ---- the 104 test room: same seed, door open from step 10 to step 20 ------------
seed = 7;
const start: State = { roomTemp: 22, outsideTemp: -4, heaterOn: false };
const script: Action[] = ["turnHeaterOn", ...Array(29).fill("wait") as Action[]];
const truth: State[] = [start];
for (let i = 0; i < 30; i++) { doorOpen = i >= 10 && i < 20; truth.push(realRoom(truth[i], script[i])); }

// the 104 model, having watched the room up to step `from`, run blind under a heater schedule
function simulate(from: number, heater: boolean[]): number[] {
  const m = new WorldModel(latent.w); m.reset();
  for (let i = 0; i < from; i++) m.observe(truth[i], script[i], truth[i + 1]);
  let s = truth[from]; const out: number[] = [];
  for (const on of heater) { s = m.act(s, on ? "turnHeaterOn" : "turnHeaterOff"); out.push(+s.roomTemp.toFixed(2)); }
  return out;
}

// ---- round two: the same private field, with act in the shape Claude proposed ------
// change = (leak + doorLeak * field) * (outside - room) + heat * heater
// No bias: the physics has none, and with one the fit trades it off against the leak.
// observe and the backwards walk are the 104 ones; only the prediction line changed.
type PhysicsWeights = { leak: number; doorLeak: number; heat: number; uField: number; uRoom: number; uOutside: number; uHeater: number; uChange: number; uBias: number };
const physicsKeys = ["leak", "doorLeak", "heat", "uField", "uRoom", "uOutside", "uHeater", "uChange", "uBias"] as const;

function physicsInputs(state: State, action: Action) {
  return { ...inputs(state, action), gap: (state.outsideTemp - state.roomTemp) / 10 };
}
function physicsObserve(w: PhysicsWeights, field: number, f: ReturnType<typeof physicsInputs>, change: number) {
  return Math.tanh(w.uField * field + w.uRoom * f.room + w.uOutside * f.outside + w.uHeater * f.heater + w.uChange * change + w.uBias);
}

function trainPhysics(log: Transition[][], epochs: number) {
  const w = {} as PhysicsWeights;
  for (const k of physicsKeys) w[k] = k.startsWith("u") ? (rand() - 0.5) * 0.2 : 0;
  const n = log.flat().length;
  let loss = 0;
  for (let epoch = 0; epoch < epochs; epoch++) {
    const g = {} as PhysicsWeights;
    for (const k of physicsKeys) g[k] = 0;
    loss = 0;
    for (const episode of log) {
      const steps: { f: ReturnType<typeof physicsInputs>; change: number; field: number; err: number; next: number }[] = [];
      let field = 0;
      for (const t of episode) {
        const f = physicsInputs(t.state, t.action);
        const target = t.nextState.roomTemp - t.state.roomTemp;
        const err = (w.leak + w.doorLeak * field) * f.gap * 10 + w.heat * f.heater - target;
        loss += err * err;
        const change = target / 5, next = physicsObserve(w, field, f, change);
        steps.push({ f, change, field, err, next });
        field = next;
      }
      let dField = 0;
      for (let i = steps.length - 1; i >= 0; i--) {
        const s = steps[i], dPre = dField * (1 - s.next * s.next);
        g.uField += dPre * s.field; g.uRoom += dPre * s.f.room; g.uOutside += dPre * s.f.outside;
        g.uHeater += dPre * s.f.heater; g.uChange += dPre * s.change; g.uBias += dPre;
        g.leak += s.err * s.f.gap; g.doorLeak += s.err * s.field * s.f.gap; g.heat += s.err * s.f.heater;
        dField = s.err * w.doorLeak * s.f.gap * 10 + dPre * w.uField;
      }
    }
    for (const k of physicsKeys) w[k] -= (k === "leak" || k === "doorLeak" ? 0.03 : 0.3) * g[k] / n;
  }
  return { w, loss: loss / n };
}

seed = 99;
const physics = trainPhysics(log, 3000);
console.log("physics-shaped model loss", physics.loss.toFixed(4));

function simulatePhysics(from: number, heater: boolean[]): number[] {
  const w = physics.w;
  let field = 0;
  for (let i = 0; i < from; i++) field = physicsObserve(w, field, physicsInputs(truth[i], script[i]), (truth[i + 1].roomTemp - truth[i].roomTemp) / 5);
  let s = truth[from]; const out: number[] = [];
  for (const on of heater) {
    const f = physicsInputs(s, on ? "turnHeaterOn" : "turnHeaterOff");
    s = { roomTemp: s.roomTemp + (w.leak + w.doorLeak * field) * f.gap * 10 + w.heat * f.heater, outsideTemp: s.outsideTemp, heaterOn: on };
    out.push(+s.roomTemp.toFixed(2));
  }
  return out;
}

// ---- the prompts ---------------------------------------------------------------
const history = (to: number) =>
  "step,room_c,outside_c,heater_until_next_step\n" +
  truth.slice(0, to + 1).map((s, i) => `${i},${s.roomTemp.toFixed(2)},${s.outsideTemp},${i === to ? "" : (truth[i + 1].heaterOn ? "on" : "off")}`).join("\n");

const logCsv = () =>
  "episode,room_c,outside_c,heater_until_next_step,next_room_c\n" +
  log.flatMap((ep, e) => ep.map(t => `${e},${t.state.roomTemp.toFixed(2)},${t.state.outsideTemp.toFixed(2)},${t.nextState.heaterOn ? "on" : "off"},${t.nextState.roomTemp.toFixed(2)}`)).join("\n");

const withLog = () => `Here is a log of 4,800 earlier steps from the same room, 80 episodes of 60 steps, recorded under a thermostat:\n\n${logCsv()}\n\n`;

const FORECAST_Q = (extra: string) => `${extra}A room with a heater. One reading per step. Here is what has happened so far:

${history(13)}

The heater stays on from step 13 to step 17. Forecast the room temperature at steps 14 to 18. Explain briefly, then end with one line of JSON: {"forecast": [five numbers]}`;

const PLAN_Q = (tool: boolean) => `A room with a heater, one reading per step, ${tool ? "and a model of the room you can run with the simulate_room tool" : "and no tools"}. It is -4 °C outside.

${history(5)}

I want the room at or above 12 °C at every reading from step 6 to step 25, with the heater on for as few steps as possible. Which of steps 5 to 24 should the heater be on for, and what is the lowest the room gets? And could the heater hold the room at 16 °C or above for those 20 steps if it ran the whole time?

End with one line of JSON: {"heater": "a string of 20 characters, 1 for on and 0 for off, for the settings at steps 5 to 24", "lowest_c": number, "hold_16": true or false}`;

const simTool = {
  name: "simulate_room",
  description: "Run the room's fitted world model forward from the current reading (step 5) under a heater schedule, without touching the real heater. Returns the predicted reading at each following step.",
  input_schema: {
    type: "object" as const,
    properties: { heater: { type: "array", items: { type: "string", enum: ["on", "off"] }, description: "Heater setting from step 5 onwards, one entry per step. Entry i is the setting from step 5+i to step 6+i." } },
    required: ["heater"],
    additionalProperties: false,
  },
  strict: true,
};

const CONDITIONS = ["forecast", "forecast-log", "plan", "plan-log", "plan-tool", "plan-tool-physics"] as const;

// ---- run: ask Claude ------------------------------------------------------------
async function run(dir: string, runs: number, only?: string) {
  const { default: Anthropic } = await import("@anthropic-ai/sdk");
  const client = new Anthropic();

  async function ask(content: string, tools?: (typeof simTool)[], sim = simulate) {
    const messages: any[] = [{ role: "user", content }];
    const calls: { heater: string; temps: number[] }[] = [];
    let inTok = 0, outTok = 0; const t0 = Date.now();
    while (true) {
      const r = await client.messages.create({ model: "claude-opus-5-5", max_tokens: 16000, messages, ...(tools ? { tools } : {}) });
      inTok += r.usage.input_tokens; outTok += r.usage.output_tokens;
      if (r.stop_reason === "refusal") throw new Error("refused");
      messages.push({ role: "assistant", content: r.content });
      const uses = r.content.filter((b: any) => b.type === "tool_use") as any[];
      const text = r.content.filter((b: any) => b.type === "text").map((b: any) => b.text).join("\n");
      if (r.stop_reason !== "tool_use" || !uses.length) return { text, calls, inTok, outTok, ms: Date.now() - t0 };
      messages.push({ role: "user", content: uses.map(u => {
        const temps = sim(5, (u.input.heater as string[]).map(h => h === "on"));
        calls.push({ heater: (u.input.heater as string[]).map(h => h === "on" ? 1 : 0).join(""), temps });
        return { type: "tool_result", tool_use_id: u.id, content: JSON.stringify({ predicted_room_c: temps }) };
      }) });
    }
  }

  mkdirSync(dir, { recursive: true });
  for (const which of CONDITIONS) {
    if (only && which !== only) continue;
    const out = [];
    for (let i = 0; i < runs; i++) {
      const r =
        which === "forecast" ? await ask(FORECAST_Q("")) :
        which === "forecast-log" ? await ask(FORECAST_Q(withLog())) :
        which === "plan" ? await ask(PLAN_Q(false)) :
        which === "plan-log" ? await ask(withLog() + PLAN_Q(false)) :
        which === "plan-tool" ? await ask(PLAN_Q(true), [simTool]) :
        await ask(PLAN_Q(true), [simTool], simulatePhysics);
      const m = [...r.text.matchAll(/\{[^{}]*\}/g)];
      out.push({ ...r, json: m.length ? JSON.parse(m[m.length - 1][0]) : null });
      console.log(which, i + 1, JSON.stringify(out[i].json));
    }
    writeFileSync(`${dir}/out-${which}.json`, JSON.stringify(out, null, 2));
  }
}

// ---- score: check the answers against the room -----------------------------------
// The real room with the door shut, from the step 5 reading, over many draughts.
function realLows(heater: string, draws: number) {
  const lows: number[] = [];
  for (let k = 0; k < draws; k++) {
    seed = 1000 + k; doorOpen = false;
    let s = truth[5], low = Infinity;
    for (const c of heater) { s = realRoom(s, c === "1" ? "turnHeaterOn" : "turnHeaterOff"); low = Math.min(low, s.roomTemp); }
    lows.push(low);
  }
  return lows;
}

function score(dir: string) {
  const real = truth.slice(14, 19).map(s => s.roomTemp);
  const rmse = (f: number[]) => Math.sqrt(f.reduce((a, v, i) => a + (v - real[i]) ** 2, 0) / 5);
  const lat = simulate(13, Array(5).fill(true));
  console.log("real", real.map(v => v.toFixed(2)).join(" "), "| 104 model", lat.join(" "), "rmse", rmse(lat).toFixed(2));

  const phys = simulatePhysics(13, Array(5).fill(true));
  console.log("physics-shaped model", phys.join(" "), "rmse", rmse(phys).toFixed(2));
  console.log("heater off for twenty steps from step 5: 104 model", simulate(5, Array(20).fill(false))[19], "| physics-shaped", simulatePhysics(5, Array(20).fill(false))[19]);
  console.log("physics-shaped weights", JSON.stringify(Object.fromEntries(Object.entries(physics.w).map(([k, v]) => [k, +(k.endsWith("eak") ? v * 10 : v).toFixed(3)]))));

  // the fewest heater steps on each model, by trying all 2^20 schedules
  for (const [name, sim] of [["104 model", simulate], ["physics-shaped model", simulatePhysics]] as const) {
    let t0 = Date.now(), best = { n: 99, s: "", low: 0 };
    for (let mask = 0; mask < 1 << 20; mask++) {
      const s = mask.toString(2).padStart(20, "0"), n = [...s].filter(c => c === "1").length;
      if (n > best.n) continue;
      const low = Math.min(...sim(5, [...s].map(c => c === "1")));
      if (low >= 12 && (n < best.n || low > best.low)) best = { n, s, low };
    }
    const lows = realLows(best.s, 200);
    console.log(`fewest on the ${name}:`, best, "after trying every schedule in", Date.now() - t0, "ms; real lowest", (lows.reduce((a, b) => a + b) / 200).toFixed(2), "ok in", lows.filter(l => l >= 12).length, "of 200");
  }

  const rows = ["condition,run,answer,score,input_tokens,output_tokens,seconds,tool_calls,usd_at_list_price"];
  for (const which of CONDITIONS) {
    const out = JSON.parse(readFileSync(`${dir}/out-${which}.json`, "utf8"));
    out.forEach((r: any, i: number) => {
      let answer: string, sc: string;
      if (which.startsWith("forecast")) { answer = r.json.forecast.join(" "); sc = `rmse ${rmse(r.json.forecast).toFixed(2)}`; }
      else {
        const lows = realLows(r.json.heater, 200);
        answer = `${r.json.heater} (${[...r.json.heater].filter((c: string) => c === "1").length} on; says lowest ${r.json.lowest_c})`;
        sc = `real lowest ${(lows.reduce((a, b) => a + b) / lows.length).toFixed(2)}; stays at 12 or above in ${lows.filter(l => l >= 12).length} of 200`;
      }
      const usd = (r.inTok * 4 + r.outTok * 20) / 1e6;
      rows.push([which, i + 1, answer, sc, r.inTok, r.outTok, (r.ms / 1000).toFixed(1), r.calls.length, usd.toFixed(3)].map(v => `"${v}"`).join(","));
    });
  }
  mkdirSync("static/data", { recursive: true });
  writeFileSync("static/data/heater-llm-runs.csv", rows.join("\n") + "\n");

  // chart data: every forecast's error, each model's plan as predicted and as it went, and the heater left off
  const fRows = ["who,run,rmse", `104 model,1,${rmse(lat).toFixed(3)}`, `physics-shaped model,1,${rmse(phys).toFixed(3)}`];
  for (const [which, who] of [["forecast", "Claude, readings only"], ["forecast-log", "Claude, with the log"]]) {
    JSON.parse(readFileSync(`${dir}/out-${which}.json`, "utf8")).forEach((r: any, i: number) => fRows.push(`"${who}",${i + 1},${rmse(r.json.forecast).toFixed(3)}`));
  }
  writeFileSync("static/data/heater-llm-forecast-errors.csv", fRows.join("\n") + "\n");

  const realMean = (heater: string) => {
    const sums = Array(20).fill(0);
    for (let k = 0; k < 200; k++) {
      seed = 1000 + k; doorOpen = false; let st = truth[5];
      [...heater].forEach((c, i) => { st = realRoom(st, c === "1" ? "turnHeaterOn" : "turnHeaterOff"); sums[i] += st.roomTemp; });
    }
    return sums.map(v => v / 200);
  };
  const plan104 = JSON.parse(readFileSync(`${dir}/out-plan-tool.json`, "utf8"))[1].json.heater as string;
  const planPhys = JSON.parse(readFileSync(`${dir}/out-plan-tool-physics.json`, "utf8"))[0].json.heater as string;
  const p104 = simulate(5, [...plan104].map(c => c === "1")), r104 = realMean(plan104);
  const pPhys = simulatePhysics(5, [...planPhys].map(c => c === "1")), rPhys = realMean(planPhys);
  writeFileSync("static/data/heater-llm-plans.csv",
    `step,plan_104_heater,plan_104_predicted_c,plan_104_real_mean_c,plan_physics_heater,plan_physics_predicted_c,plan_physics_real_mean_c\n5,,${truth[5].roomTemp.toFixed(2)},${truth[5].roomTemp.toFixed(2)},,${truth[5].roomTemp.toFixed(2)},${truth[5].roomTemp.toFixed(2)}\n` +
    p104.map((_, i) => `${i + 6},${plan104[i]},${p104[i].toFixed(2)},${r104[i].toFixed(2)},${planPhys[i]},${pPhys[i].toFixed(2)},${rPhys[i].toFixed(2)}`).join("\n") + "\n");

  const off = "0".repeat(20), offReal = realMean(off), off104 = simulate(5, Array(20).fill(false)), offPhys = simulatePhysics(5, Array(20).fill(false));
  writeFileSync("static/data/heater-llm-heater-off.csv",
    `step,real_mean_c,model_104_c,physics_shaped_c\n5,${truth[5].roomTemp.toFixed(2)},${truth[5].roomTemp.toFixed(2)},${truth[5].roomTemp.toFixed(2)}\n` +
    offReal.map((v, i) => `${i + 6},${v.toFixed(2)},${off104[i].toFixed(2)},${offPhys[i].toFixed(2)}`).join("\n") + "\n");
  console.log(rows.join("\n"));
}

const [mode, dir, runs] = process.argv.slice(2);
if (mode === "run") await run(dir, Number(runs ?? 3), process.argv[5]);
else if (mode === "score") score(dir);
else console.log("usage: heater_llm.ts run|score <dir> [runs]");
