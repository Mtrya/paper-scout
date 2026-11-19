// REA fast-slow coupling probe: silent failures, gate comparison, judge noise, async staleness.
// Uses REA's own runtime (runRoutedEpisode / runAsyncEpisode) with a stochastic drawer fixture.
import { buildRouteTable, createRouteMemoryRouter, routeTableSha256 } from "../REA/src/framework/route-memory.mjs";
import { runRoutedEpisode } from "../REA/src/framework/routed-runtime.mjs";
import { runAsyncEpisode } from "../REA/src/framework/async-runtime.mjs";
import { createAsyncSystemDemo } from "../REA/examples/async-system-demo.mjs";

// ---------- deterministic RNG ----------
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------- Probe A: routed drawer with silent failures ----------
// Drawer resistance r ~ U[0,1): "fast" program opens fully iff r < 0.5, otherwise it
// returns normally after 3 blocks with the drawer only half open (SILENT failure).
// "slow" program always opens fully in 8 blocks. Judge reads open% with noise:
//   P(says done | not done) = alpha (false done -> forfeit), P(says not_done | done) = beta.
function makeDrawerEpisode({ seed, alpha = 0, beta = 0, gate = "judge", first = null, ladder = null, maxSteps = 60 }) {
  const rng = mulberry32(seed * 7919 + 13);
  const r = rng();
  const jammed = r >= 0.5;
  let open = 0;
  let active = null;
  let left = 0;
  const programs = {
    fast: { per: jammed ? 0.02 : 0.34, blocks: 3 },
    slow: { per: 0.13, blocks: 8 },
  };
  const observation = () => ({
    id: `drawer-${open.toFixed(2)}`, timestampMs: Date.now(), actions: [{ name: "continue_program", arguments: {} }],
    text: `The drawer is ${Math.round(open * 100)}% open.`,
    state: { open: Number(open.toFixed(3)) },
  });
  const environment = {
    id: "probe-drawer", timing: "turn-based", resumable: true,
    actions: [{ name: "continue_program", arguments: {} }],
    async reset() { open = 0; active = null; left = 0; return { observation: observation() }; },
    async step(action, { reasoning }) {
      const candidateId = reasoning.route.candidateId;
      if (active !== candidateId) { active = candidateId; left = programs[candidateId].blocks; }
      open = Math.min(1, open + programs[candidateId].per);
      left -= 1;
      const status = left === 0 ? "returned" : "running";
      if (status === "returned") active = null;
      return { observation: observation(), info: { capability: { candidateId, status } } };
    },
    async stop() {},
    async score() { return { success: open >= 0.99, open }; },
  };
  const program = id => ({ id, kind: "program", policy: { id, async decide() { return { action: { name: "continue_program", arguments: {} } }; } } });
  const devEpisodes = [
    { task: "drawer", candidateId: "fast", success: true, steps: 3 },
    { task: "drawer", candidateId: "fast", success: true, steps: 3 },
    { task: "drawer", candidateId: "slow", success: true, steps: 8 },
    { task: "drawer", candidateId: "slow", success: true, steps: 8 },
  ];
  const table = buildRouteTable(devEpisodes, { candidates: ["fast", "slow"], createdAt: "2026-11-20T00:00:00.000Z" });
  const routerOpts = { table };
  if (gate === "exception") routerOpts.recoverOn = "exception";
  else if (gate === "always") routerOpts.recoverOn = "always";
  else if (gate === "never") routerOpts.recoverOn = "never";
  else routerOpts.recoverOn = "judge";
  if (first) {
    // single-policy baseline: fixed ladder, no escalation
    table.tasks["drawer"] = { evidence: {}, ranked: [first], choice: first, ladder: [first] };
  }
  if (ladder) table.tasks["drawer"].ladder = ladder;
  table.sha256 = routeTableSha256(table);
  const system1 = createRouteMemoryRouter(routerOpts);
  const system2 = gate === "judge" ? {
    id: "noisy-judge",
    async judge({ observation: obs }) {
      const percent = Number(/(\d+)% open/u.exec(obs.text)?.[1] ?? 0);
      const done = percent >= 99;
      const u = rng();
      let verdict;
      if (done) verdict = u < beta ? "not_done" : "done";
      else verdict = u < alpha ? "done" : "not_done";
      return { verdict, done: verdict === "done", confidence: 0.9, rationale: obs.text };
    },
  } : null;
  return {
    environment, system0: [program("fast"), program("slow")], system1, system2,
    task: "drawer", seed, instruction: "Open the drawer fully.",
    maxSteps, maxDurationMs: 60000, maxRuns: 3,
  };
}

async function probeA() {
  const N = 400;
  const configs = [];
  for (const gate of ["never-fast", "never-slow", "exception", "always", "always-same", "judge"]) {
    for (const alpha of [0, 0.05, 0.1, 0.2, 0.3]) {
      for (const beta of [0, 0.2, 0.5]) {
        if (gate !== "judge" && (alpha !== 0 || beta !== 0)) continue;
        configs.push({ gate, alpha, beta });
      }
    }
  }
  const results = [];
  for (const cfg of configs) {
    let succ = 0, steps = 0, judgeCalls = 0, escalations = 0;
    for (let seed = 1; seed <= N; seed++) {
      const first = cfg.gate === "never-fast" ? "fast" : cfg.gate === "never-slow" ? "slow" : null;
      const gate = cfg.gate === "never-fast" || cfg.gate === "never-slow" ? "never"
        : cfg.gate === "always-same" ? "always" : cfg.gate;
      const ladder = cfg.gate === "always-same" ? ["fast", "fast"] : null;
      const ep = makeDrawerEpisode({ seed, alpha: cfg.alpha, beta: cfg.beta, gate, first, ladder });
      let jc = 0, runs = 0;
      const summary = await runRoutedEpisode({ ...ep, onEvent: e => { if (e.type === "judge" && e.status === "started") jc++; if (e.type === "run" && e.status === "started") runs++; } });
      if (summary.success) succ++;
      steps += summary.steps;
      judgeCalls += jc;
      escalations += Math.max(0, runs - 1);
    }
    results.push({ ...cfg, success: +(succ / N).toFixed(4), stepsPerEp: +(steps / N).toFixed(2), judgeCallsPerEp: +(judgeCalls / N).toFixed(3), escalationsPerEp: +(escalations / N).toFixed(3) });
  }
  return results;
}

// ---------- Probe B: async coupling sweep ----------
async function probeB() {
  const rows = [];
  for (const brainMs of [100, 500, 1350, 3000, 6000]) {
    for (const maxAge of [0, 1, 3, 6, 12, 24]) {
      let succ = 0, steps = 0, wall = 0, overlap = 0, applied = 0, stale = 0, corrections = 0;
      const trials = 5;
      for (let seed = 0; seed < trials; seed++) {
        const fixture = createAsyncSystemDemo({ brainMs, maxAdviceAgeSteps: maxAge, thinkEverySteps: 1, onEvent: e => { if (e.type === "thinking" && e.status === "stale") stale++; } });
        const summary = await runAsyncEpisode(fixture);
        if (summary.success) succ++;
        steps += summary.steps;
        wall += summary.durationMs;
        overlap += summary.asyncMetrics.overlapMs;
        applied += summary.asyncMetrics.adviceApplied;
        corrections += summary.asyncMetrics.corrections;
      }
      rows.push({ brainMs, maxAdviceAgeSteps: maxAge, success: +(succ / trials).toFixed(2), steps: +(steps / trials).toFixed(1), wallMs: Math.round(wall / trials), overlapMs: Math.round(overlap / trials), adviceApplied: +(applied / trials).toFixed(1), corrections: +(corrections / trials).toFixed(1), staleEvents: +(stale / trials).toFixed(1) });
    }
  }
  // sync vs async comparison
  const modes = [];
  for (const mode of ["sync", "async"]) {
    for (const brainMs of [1350, 3000]) {
      const fixture = createAsyncSystemDemo({ mode, brainMs, maxAdviceAgeSteps: 24, thinkEverySteps: 5 });
      const summary = await runAsyncEpisode(fixture);
      modes.push({ mode, brainMs, success: summary.success, steps: summary.steps, wallMs: Math.round(summary.durationMs), overlapMs: Math.round(summary.asyncMetrics.overlapMs), blockedThinkingMs: Math.round(summary.asyncMetrics.blockedThinkingMs), adviceApplied: summary.asyncMetrics.adviceApplied, system2Calls: summary.asyncMetrics.system2Calls });
    }
  }
  return { sweep: rows, modes };
}

const mode = process.argv[2] ?? "all";
const outDir = new URL(".", import.meta.url).pathname;
if (mode === "all" || mode === "a") {
  const a = await probeA();
  const { writeFileSync } = await import("node:fs");
  writeFileSync(outDir + "results_a.json", JSON.stringify(a, null, 1));
  process.stdout.write(`PROBE_A_DONE n=${a.length}\n`);
}
if (mode === "all" || mode === "b") {
  const b = await probeB();
  const { writeFileSync } = await import("node:fs");
  writeFileSync(outDir + "results_b.json", JSON.stringify(b, null, 1));
  process.stdout.write(`PROBE_B_DONE sweep=${b.sweep.length} modes=${b.modes.length}\n`);
}
