// fl.js — a 64→32→10 MLP and a Federated Averaging simulation, from scratch, no dependencies.
// No DOM in here: the same file runs in the page, in bench(), and under node.
(function (root) {
  'use strict';

  // ---------- model layout: one flat vector so averaging is a single loop ----------
  const D = 64, H = 32, C = 10;
  const oW1 = 0, ob1 = D * H, oW2 = ob1 + H, ob2 = oW2 + H * C;
  const P = ob2 + C;                 // 2,410 parameters
  const BYTES_PER_MODEL = P * 4;     // sent as float32 on the wire

  // ---------- seeded RNG (mulberry32) ----------
  function rng(seed) {
    let a = seed >>> 0;
    const f = () => {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    f.normal = () => {
      let u = 0; while (u === 0) u = f();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * f());
    };
    f.int = (n) => Math.floor(f() * n);
    return f;
  }
  function shuffle(arr, r) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = r.int(i + 1); const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  }

  // ---------- MLP ----------
  function initParams(r) {
    const p = new Float64Array(P);
    const s1 = Math.sqrt(2 / D), s2 = Math.sqrt(2 / H);
    for (let i = 0; i < D * H; i++) p[oW1 + i] = r.normal() * s1;
    for (let i = 0; i < H * C; i++) p[oW2 + i] = r.normal() * s2;
    return p;
  }

  // Forward pass for one sample; fills h (hidden ReLU) and z (softmax probs). Returns argmax.
  function forward(p, X, off, h, z) {
    for (let j = 0; j < H; j++) {
      let s = p[ob1 + j];
      const w = oW1 + j * D;
      for (let i = 0; i < D; i++) s += p[w + i] * X[off + i];
      h[j] = s > 0 ? s : 0;
    }
    let mx = -Infinity, arg = 0;
    for (let k = 0; k < C; k++) {
      let s = p[ob2 + k];
      const w = oW2 + k * H;
      for (let j = 0; j < H; j++) s += p[w + j] * h[j];
      z[k] = s;
      if (s > mx) { mx = s; arg = k; }
    }
    let sum = 0;
    for (let k = 0; k < C; k++) { z[k] = Math.exp(z[k] - mx); sum += z[k]; }
    for (let k = 0; k < C; k++) z[k] /= sum;
    return arg;
  }

  // Mini-batch SGD with softmax cross-entropy. idx = sample indices this learner may use.
  function train(p, X, Y, idx, epochs, lr, batch, r) {
    if (!idx.length) return;
    const g = new Float64Array(P), h = new Float64Array(H), z = new Float64Array(C), dh = new Float64Array(H);
    const order = Array.from(idx);
    for (let e = 0; e < epochs; e++) {
      shuffle(order, r);
      for (let b0 = 0; b0 < order.length; b0 += batch) {
        const b1 = Math.min(order.length, b0 + batch);
        g.fill(0);
        for (let b = b0; b < b1; b++) {
          const n = order[b], off = n * D;
          forward(p, X, off, h, z);
          z[Y[n]] -= 1;                         // dL/dlogits
          dh.fill(0);
          for (let k = 0; k < C; k++) {
            const dz = z[k], w = oW2 + k * H;
            g[ob2 + k] += dz;
            for (let j = 0; j < H; j++) { g[w + j] += dz * h[j]; dh[j] += dz * p[w + j]; }
          }
          for (let j = 0; j < H; j++) {
            if (h[j] <= 0) continue;              // ReLU gate
            const dj = dh[j], w = oW1 + j * D;
            g[ob1 + j] += dj;
            for (let i = 0; i < D; i++) g[w + i] += dj * X[off + i];
          }
        }
        const s = lr / (b1 - b0);
        for (let i = 0; i < P; i++) p[i] -= s * g[i];
      }
    }
  }

  function accuracy(p, X, Y) {
    const h = new Float64Array(H), z = new Float64Array(C);
    let ok = 0;
    for (let n = 0; n < Y.length; n++) if (forward(p, X, n * D, h, z) === Y[n]) ok++;
    return ok / Y.length;
  }

  // ---------- data ----------
  function decode(split, maxValue) {
    const n = split.y.length, X = new Float64Array(n * D);
    for (let s = 0; s < n; s++) {
      const str = split.x[s];
      for (let i = 0; i < D; i++) X[s * D + i] = (str.charCodeAt(i) - 97) / maxValue;
    }
    return { X, Y: Int32Array.from(split.y), n };
  }
  function prepare(json) {
    return { train: decode(json.train, json.maxValue), test: decode(json.test, json.maxValue) };
  }

  // ---------- partition: label skew in [0,1] ----------
  // Client k "owns" digits 2k mod 10 and 2k+1 mod 10. With probability `skew` a sample goes
  // to a random owner of its digit; otherwise to a uniformly random client.
  // skew 0 → IID; skew 1 → every client sees exactly two digits.
  const NCLIENTS = 8;
  function homeDigits(k) { return [(2 * k) % 10, (2 * k + 1) % 10]; }
  function partition(Y, skew, seed, nClients = NCLIENTS) {
    const r = rng(seed * 7919 + 13);
    const owners = Array.from({ length: 10 }, () => []);
    for (let k = 0; k < nClients; k++) homeDigits(k).forEach((d) => owners[d].push(k));
    const clients = Array.from({ length: nClients }, () => []);
    for (let n = 0; n < Y.length; n++) {
      const own = owners[Y[n]];
      const k = (r() < skew && own.length) ? own[r.int(own.length)] : r.int(nClients);
      clients[k].push(n);
    }
    return clients.map((idx, k) => {
      const hist = new Array(10).fill(0);
      idx.forEach((n) => hist[Y[n]]++);
      return { id: k, idx: Int32Array.from(idx), n: idx.length, hist, home: homeDigits(k) };
    });
  }

  // ---------- the simulation ----------
  // Each round: sample m clients, each trains a copy of the global model for E local epochs,
  // server replaces global with the sample-count-weighted mean (FedAvg).
  // Baselines advance in lockstep: a centralized model trained E epochs on pooled data, and
  // eight local-only models trained E epochs on their own slice.
  function createSim(data, opts) {
    const o = Object.assign({ skew: 0, seed: 1, epochs: 2, perRound: 8, lr: 0.1, batch: 16 }, opts);
    const { train: tr, test: te } = data;
    const clients = partition(tr.Y, o.skew, o.seed);
    const init = initParams(rng(o.seed));
    const r = rng(o.seed * 31 + 7);
    const allIdx = Int32Array.from({ length: tr.n }, (_, i) => i);

    const sim = {
      opts: o, clients, round: 0, bytes: 0,
      global: Float64Array.from(init),
      central: Float64Array.from(init),
      locals: clients.map(() => Float64Array.from(init)),
      history: [],
      rawBytes: tr.n * D,     // 8-bit pixels that never leave the devices
      params: P, bytesPerModel: BYTES_PER_MODEL,
    };

    function evaluate(participants) {
      const localAcc = sim.locals.map((p) => accuracy(p, te.X, te.Y));
      const rec = {
        round: sim.round,
        fedavg: accuracy(sim.global, te.X, te.Y),
        central: accuracy(sim.central, te.X, te.Y),
        local: localAcc.reduce((a, b) => a + b, 0) / localAcc.length,
        localAcc, participants: participants || [], bytes: sim.bytes,
      };
      sim.history.push(rec);
      return rec;
    }

    sim.step = function () {
      const { epochs, lr, batch } = o;
      const m = Math.max(1, Math.min(clients.length, o.perRound | 0));
      const chosen = shuffle(clients.map((c) => c.id), r).slice(0, m).sort((a, b) => a - b);

      // FedAvg
      const acc = new Float64Array(P);
      let total = 0;
      for (const k of chosen) {
        const c = clients[k];
        if (!c.n) continue;
        const w = Float64Array.from(sim.global);        // download
        train(w, tr.X, tr.Y, c.idx, epochs, lr, batch, r);
        for (let i = 0; i < P; i++) acc[i] += c.n * w[i]; // upload, weighted by n_k
        total += c.n;
      }
      if (total > 0) for (let i = 0; i < P; i++) sim.global[i] = acc[i] / total;
      sim.bytes += chosen.length * 2 * BYTES_PER_MODEL;

      // baselines
      train(sim.central, tr.X, tr.Y, allIdx, epochs, lr, batch, r);
      clients.forEach((c, k) => train(sim.locals[k], tr.X, tr.Y, c.idx, epochs, lr, batch, r));

      sim.round++;
      return evaluate(chosen);
    };
    sim.set = function (patch) { Object.assign(o, patch); };

    evaluate([]);
    return sim;
  }

  // ---------- bench: fixed protocol, no animation ----------
  function bench(data, cfg) {
    const c = Object.assign({ skews: [0, 1], seeds: [1, 2, 3], rounds: 30, epochs: 2, perRound: 8, lr: 0.1 }, cfg);
    const t0 = Date.now();
    const rows = [];
    for (const skew of c.skews) {
      const runs = c.seeds.map((seed) => {
        const sim = createSim(data, { skew, seed, epochs: c.epochs, perRound: c.perRound, lr: c.lr });
        for (let i = 0; i < c.rounds; i++) sim.step();
        const last = sim.history[sim.history.length - 1];
        return { seed, fedavg: last.fedavg, central: last.central, local: last.local, bytes: sim.bytes, rawBytes: sim.rawBytes };
      });
      const mean = (k) => runs.reduce((a, b) => a + b[k], 0) / runs.length;
      const range = (k) => [Math.min(...runs.map((x) => x[k])), Math.max(...runs.map((x) => x[k]))];
      rows.push({
        skew, runs,
        fedavg: mean('fedavg'), central: mean('central'), local: mean('local'),
        fedavgRange: range('fedavg'), centralRange: range('central'), localRange: range('local'),
        bytes: runs[0].bytes, rawBytes: runs[0].rawBytes,
      });
    }
    return { config: c, params: P, bytesPerModel: BYTES_PER_MODEL, rows, ms: Date.now() - t0 };
  }

  root.FL = { P, D, H, C, BYTES_PER_MODEL, NCLIENTS, rng, initParams, train, accuracy, prepare, partition, createSim, bench, homeDigits };
  if (typeof module !== 'undefined') module.exports = root.FL;
})(typeof globalThis !== 'undefined' ? globalThis : this);
