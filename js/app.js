// app.js — UI for the FedAvg playground: network diagram, controls, live chart, bench().
(function () {
  'use strict';
  const FL = window.FL;
  const NS = 'http://www.w3.org/2000/svg';
  const $ = (s) => document.querySelector(s);
  const HORIZON = 30;
  const reduceMotion = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---------- theme ----------
  const themeBtn = $('#theme');
  function effectiveTheme() {
    const t = document.documentElement.dataset.theme;
    if (t) return t;
    return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  themeBtn.addEventListener('click', () => {
    const next = effectiveTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('theme', next); } catch (e) { /* storage off */ }
  });

  // ---------- state ----------
  const cfg = { skew: 0, epochs: 2, perRound: 8, lr: 0.1, speed: 1, seed: 1 };
  let data = null, sim = null, running = false, busy = false;

  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  const pct = (v) => (v * 100).toFixed(1);
  const kb = (b) => b / 1024;
  function fmtKB(b) {
    return Math.round(kb(b)).toLocaleString('en-US') + ' KB';
  }

  // ---------- network diagram ----------
  const net = $('#net');
  const W = 640, Hh = 600, CX = 320, CY = 300, RX = 236, RY = 218, CW = 150, CHh = 100;
  const nodes = [];
  let packetLayer, serverAcc, serverRound;

  function clientPos(k) {
    const a = -Math.PI / 2 + (k * 2 * Math.PI) / FL.NCLIENTS;
    return { x: CX + RX * Math.cos(a), y: CY + RY * Math.sin(a) };
  }

  function buildNet() {
    net.setAttribute('viewBox', `0 0 ${W} ${Hh}`);
    net.textContent = '';
    const edges = el('g', {}, net);
    packetLayer = el('g', {}, net);
    const cl = el('g', {}, net);
    nodes.length = 0;
    for (let k = 0; k < FL.NCLIENTS; k++) {
      const p = clientPos(k);
      const edge = el('line', { x1: CX, y1: CY, x2: p.x, y2: p.y, class: 'edge' }, edges);
      const g = el('g', { class: 'client', transform: `translate(${p.x - CW / 2},${p.y - CHh / 2})` }, cl);
      el('rect', { class: 'node', width: CW, height: CHh, rx: 10 }, g);
      const name = el('text', { class: 't-name', x: 11, y: 20 }, g); name.textContent = `client ${k + 1}`;
      const n = el('text', { class: 't-small', x: CW - 11, y: 20, 'text-anchor': 'end' }, g);
      const bars = [];
      const bw = (CW - 20) / 10;
      for (let d = 0; d < 10; d++) {
        bars.push(el('rect', { class: 'bar', x: 10 + d * bw + 1, y: 62, width: bw - 2.5, height: 0, rx: 1 }, g));
        const t = el('text', { class: 't-digit', x: 10 + d * bw + bw / 2 - 0.25, y: 74 }, g); t.textContent = d;
      }
      const acc = el('text', { class: 't-small', x: 11, y: 91 }, g);
      nodes.push({ g, edge, n, bars, acc, pos: p });
    }
    // server
    const sg = el('g', { class: 'server' }, net);
    el('rect', { class: 'node', x: CX - 70, y: CY - 46, width: 140, height: 92, rx: 12 }, sg);
    const t1 = el('text', { class: 't-name', x: CX - 56, y: CY - 22 }, sg); t1.textContent = 'server';
    serverRound = el('text', { class: 't-small', x: CX + 56, y: CY - 24, 'text-anchor': 'end' }, sg);
    serverAcc = el('text', { class: 't-big', x: CX - 57, y: CY + 12 }, sg);
    const t3 = el('text', { class: 't-small', x: CX - 56, y: CY + 32 }, sg); t3.textContent = 'global model acc';
  }

  function renderClients(rec) {
    const chosen = new Set(rec ? rec.participants : []);
    sim.clients.forEach((c, k) => {
      const nd = nodes[k];
      nd.n.textContent = `n=${c.n}`;
      const max = Math.max(1, ...c.hist);
      c.hist.forEach((v, d) => {
        const h = (v / max) * 32;
        nd.bars[d].setAttribute('y', 62 - h);
        nd.bars[d].setAttribute('height', h);
        nd.bars[d].setAttribute('class', 'bar' + (sim.opts.skew > 0 && c.home.includes(d) ? ' home' : ''));
      });
      const la = rec ? rec.localAcc[k] : null;
      nd.acc.textContent = la == null ? '' : `local-only ${pct(la)}%`;
      const partial = rec && rec.round > 0 && rec.participants.length < FL.NCLIENTS;
      nd.g.setAttribute('class', 'client' + (partial ? (chosen.has(k) ? ' on' : ' off') : ''));
      nd.edge.setAttribute('class', 'edge' + (partial && chosen.has(k) ? ' on' : ''));
    });
  }

  function setChosen(chosen, training) {
    const set = new Set(chosen);
    nodes.forEach((nd, k) => {
      const on = set.has(k);
      nd.g.setAttribute('class', 'client ' + (on ? (training ? 'on training' : 'on') : 'off'));
      nd.edge.setAttribute('class', 'edge' + (on ? ' on' : ''));
    });
  }

  // Packets travel along the server↔client edges; shared-clock animation, resolves when done.
  function animatePackets(chosen, dir, ms) {
    if (ms <= 0) return Promise.resolve();
    packetLayer.textContent = '';
    const pk = chosen.map((k) => ({ k, c: el('circle', { r: 5.5, class: dir === 'down' ? 'pk-down' : 'pk-up' }, packetLayer), d: Math.random() * 0.18 }));
    return new Promise((res) => {
      const t0 = performance.now();
      function frame(now) {
        const t = (now - t0) / ms;
        for (const p of pk) {
          let u = Math.min(1, Math.max(0, (t - p.d) / (1 - 0.18)));
          u = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
          const { x, y } = nodes[p.k].pos;
          // stop at the edge of the cards / server box, not their centres
          const a = dir === 'down' ? 0.24 + 0.58 * u : 0.82 - 0.58 * u;
          p.c.setAttribute('cx', CX + (x - CX) * a);
          p.c.setAttribute('cy', CY + (y - CY) * a);
          p.c.setAttribute('opacity', t < p.d ? 0 : 1);
        }
        if (t < 1) requestAnimationFrame(frame);
        else { packetLayer.textContent = ''; res(); }
      }
      requestAnimationFrame(frame);
    });
  }
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));

  // ---------- chart ----------
  const chart = $('#chart'), tip = $('#tip');
  const CWd = 560, CHt = 260, M = { l: 34, r: 70, t: 10, b: 26 };
  const iw = CWd - M.l - M.r, ih = CHt - M.t - M.b;
  const SERIES = [
    { key: 'fedavg', cls: 's-fed', name: 'fedavg' },
    { key: 'central', cls: 's-cen', name: 'centralized' },
    { key: 'local', cls: 's-loc', name: 'local-only' },
  ];
  let chartEls = null;
  function xMax() { return Math.max(HORIZON, sim ? sim.round : 0); }
  const sx = (r) => M.l + (r / xMax()) * iw;
  const sy = (v) => M.t + (1 - v) * ih;

  function buildChart() {
    chart.setAttribute('viewBox', `0 0 ${CWd} ${CHt}`);
    chart.textContent = '';
    const axis = el('g', { class: 'axis' }, chart);
    const xt = el('g', { class: 'axis' }, chart);
    const lines = SERIES.map((s) => el('path', { class: s.cls }, chart));
    const labels = SERIES.map(() => el('text', { class: 'lbl strong' }, chart));
    const cross = el('line', { class: 'cross', y1: M.t, y2: M.t + ih, opacity: 0 }, chart);
    const dots = SERIES.map((s) => el('circle', { r: 4, opacity: 0, fill: s.key === 'fedavg' ? 'var(--accent)' : s.key === 'central' ? 'var(--ink)' : 'var(--ink-3)', stroke: 'var(--card)', 'stroke-width': 2 }, chart));
    const hit = el('rect', { x: M.l, y: M.t, width: iw, height: ih, fill: 'transparent' }, chart);
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
      el('line', { class: 'grid', x1: M.l, x2: M.l + iw, y1: sy(v), y2: sy(v) }, axis);
      const t = el('text', { x: M.l - 8, y: sy(v) + 3, 'text-anchor': 'end' }, axis); t.textContent = v * 100 + '%';
    });
    chartEls = { lines, labels, xt, cross, dots, hit };
    hit.addEventListener('pointermove', hover);
    hit.addEventListener('pointerleave', () => { tip.style.opacity = 0; cross.setAttribute('opacity', 0); dots.forEach((d) => d.setAttribute('opacity', 0)); });
  }

  function renderChart() {
    const H = sim.history, { lines, labels, xt } = chartEls;
    xt.textContent = '';
    const step = xMax() > 40 ? 20 : 10;
    for (let r = 0; r <= xMax(); r += step) {
      const t = el('text', { x: sx(r), y: CHt - 6, 'text-anchor': 'middle' }, xt); t.textContent = r;
    }
    SERIES.forEach((s, i) => {
      lines[i].setAttribute('d', H.map((h, j) => (j ? 'L' : 'M') + sx(h.round).toFixed(1) + ',' + sy(h[s.key]).toFixed(1)).join(''));
    });
    // direct labels at line ends, nudged apart so they never overlap
    const last = H[H.length - 1];
    const lab = SERIES.map((s, i) => ({ i, y: sy(last[s.key]), text: `${s.name === 'centralized' ? 'central' : s.name} ${pct(last[s.key])}` }));
    lab.sort((a, b) => a.y - b.y);
    for (let j = 1; j < lab.length; j++) if (lab[j].y - lab[j - 1].y < 12) lab[j].y = lab[j - 1].y + 12;
    const over = lab[lab.length - 1].y - (M.t + ih + 4);
    if (over > 0) lab.forEach((l) => (l.y -= over));
    lab.forEach((l) => {
      const t = labels[l.i];
      t.setAttribute('x', sx(last.round) + 8);
      t.setAttribute('y', l.y + 3.5);
      t.textContent = l.text;
    });
  }

  function hover(ev) {
    const H = sim.history;
    const rect = chart.getBoundingClientRect();
    const x = ((ev.clientX - rect.left) / rect.width) * CWd;
    const r = Math.round(((x - M.l) / iw) * xMax());
    const h = H[Math.max(0, Math.min(H.length - 1, r))];
    if (!h) return;
    const px = sx(h.round);
    chartEls.cross.setAttribute('x1', px); chartEls.cross.setAttribute('x2', px); chartEls.cross.setAttribute('opacity', 1);
    SERIES.forEach((s, i) => { const d = chartEls.dots[i]; d.setAttribute('cx', px); d.setAttribute('cy', sy(h[s.key])); d.setAttribute('opacity', 1); });
    tip.innerHTML = `<div class="row"><b>round ${h.round}</b><span>${fmtKB(h.bytes)}</span></div>` +
      SERIES.map((s) => `<div class="row"><span><i class="k-sw sw-${s.cls.slice(2)}"></i>${s.name}</span><b>${pct(h[s.key])}%</b></div>`).join('');
    const wrap = chart.parentElement.getBoundingClientRect();
    const left = (px / CWd) * rect.width + (rect.left - wrap.left);
    tip.style.opacity = 1;
    const tw = tip.offsetWidth;
    tip.style.left = Math.max(0, Math.min(wrap.width - tw, left + (left > wrap.width / 2 ? -tw - 12 : 12))) + 'px';
    tip.style.top = '8px';
  }

  // ---------- stats ----------
  function renderStats() {
    const h = sim.history[sim.history.length - 1];
    $('#s-round').innerHTML = `${sim.round}<small>/${HORIZON}</small>`;
    $('#s-fed').textContent = pct(h.fedavg) + '%';
    $('#s-cen').textContent = pct(h.central) + '%';
    $('#s-loc').textContent = pct(h.local) + '%';
    $('#s-bytes').textContent = fmtKB(sim.bytes);
    $('#s-raw').textContent = fmtKB(sim.rawBytes);
    $('#s-ratio').textContent = sim.bytes ? `${(sim.bytes / sim.rawBytes).toFixed(1)}× the raw pixels` : 'nothing sent yet';
    $('#s-gap').textContent = `gap to central ${((h.central - h.fedavg) * 100).toFixed(1)} pts`;
    serverAcc.textContent = pct(h.fedavg) + '%';
    serverRound.textContent = `round ${sim.round}`;
  }
  function renderAll() {
    const h = sim.history[sim.history.length - 1];
    renderClients(h);
    renderChart();
    renderStats();
  }

  // ---------- simulation control ----------
  function reset() {
    running = false;
    updateRunBtn();
    sim = FL.createSim(data, { skew: cfg.skew, seed: cfg.seed, epochs: cfg.epochs, perRound: cfg.perRound, lr: cfg.lr });
    renderAll();
  }

  async function oneRound() {
    if (busy) return;
    busy = true;
    sim.set({ epochs: cfg.epochs, perRound: cfg.perRound, lr: cfg.lr });
    const animate = cfg.speed > 0 && !reduceMotion && !document.hidden;
    const base = animate ? 620 / cfg.speed : 0;
    // FedAvg samples its clients inside step(); peek by running it first, then replay the
    // exchange visually before revealing the new numbers.
    const rec = sim.step();
    if (animate) {
      setChosen(rec.participants, false);
      await animatePackets(rec.participants, 'down', base);
      setChosen(rec.participants, true);
      await wait(base * 0.45);
      setChosen(rec.participants, false);
      await animatePackets(rec.participants, 'up', base);
    } else {
      await wait(16);
    }
    renderAll();
    busy = false;
  }

  async function loop() {
    while (running) {
      if (sim.round >= HORIZON) { running = false; break; }
      await oneRound();
    }
    updateRunBtn();
  }

  const runBtn = $('#run');
  function updateRunBtn() {
    runBtn.textContent = running ? 'pause' : sim && sim.round >= HORIZON ? 'done' : sim && sim.round > 0 ? 'resume' : 'run';
    runBtn.disabled = !running && sim && sim.round >= HORIZON;
    runBtn.setAttribute('aria-pressed', running ? 'true' : 'false');
  }
  runBtn.addEventListener('click', () => {
    if (!sim) return;
    running = !running;
    updateRunBtn();
    if (running) loop();
  });
  $('#step').addEventListener('click', () => { if (sim && !running) oneRound().then(updateRunBtn); });
  $('#reset').addEventListener('click', () => { if (data) { running = false; waitIdle().then(reset); } });
  async function waitIdle() { while (busy) await wait(20); }

  // ---------- controls ----------
  const skew = $('#skew');
  function skewText(v) {
    if (v === 0) return '0.00 · iid';
    if (v === 1) return '1.00 · 2 digits each';
    return v.toFixed(2);
  }
  skew.addEventListener('input', () => {
    cfg.skew = +skew.value;
    $('#skew-out').textContent = skewText(cfg.skew);
    if (!data) return;
    running = false;
    waitIdle().then(reset);   // a new skew is a new partition: start over
  });
  const per = $('#per');
  per.addEventListener('input', () => { cfg.perRound = +per.value; $('#per-out').textContent = `${cfg.perRound} / 8`; });
  $('#lr').addEventListener('change', (e) => { cfg.lr = +e.target.value; });
  function seg(id, key, after) {
    const box = $(id);
    box.addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return;
      box.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', x === b ? 'true' : 'false'));
      cfg[key] = +b.dataset.v;
      if (after) after();
    });
  }
  seg('#epochs', 'epochs');
  seg('#speed', 'speed');

  // ---------- bench(): fixed protocol, used for the numbers in the write-up ----------
  // 30 rounds, 2 local epochs, 8/8 clients, lr 0.1, seeds 1-3, skew 0 and 1.
  window.bench = async function (opts) {
    if (!data) data = FL.prepare(await loadData());
    const o = Object.assign({ skews: [0, 1], seeds: [1, 2, 3], rounds: 30, epochs: 2, perRound: 8, lr: 0.1 }, opts);
    const rows = [];
    const t0 = performance.now();
    for (const s of o.skews) {
      await wait(0); // let the page breathe between configurations
      rows.push(FL.bench(data, Object.assign({}, o, { skews: [s] })).rows[0]);
    }
    const out = { config: o, params: FL.P, bytesPerModel: FL.BYTES_PER_MODEL, rows, ms: Math.round(performance.now() - t0), ua: navigator.userAgent };
    console.log('bench', out);
    return out;
  };

  const benchBtn = $('#bench-btn');
  benchBtn.addEventListener('click', async () => {
    benchBtn.disabled = true;
    $('#bench-status').textContent = 'running 6 × 30 rounds… the tab may stall for a few seconds';
    await wait(50);
    const r = await window.bench();
    r.rows.forEach((row) => {
      const tr = document.querySelector(`[data-skew="${row.skew}"]`);
      if (!tr) return;
      tr.querySelector('.b-fed').textContent = pct(row.fedavg) + '%';
      tr.querySelector('.b-cen').textContent = pct(row.central) + '%';
      tr.querySelector('.b-loc').textContent = pct(row.local) + '%';
    });
    $('#bench-status').textContent = `done in ${(r.ms / 1000).toFixed(1)} s in your browser — table updated with your numbers`;
    benchBtn.disabled = false;
  });

  // ---------- boot ----------
  function loadData() {
    return fetch('data/digits.json').then((r) => { if (!r.ok) throw new Error('digits.json ' + r.status); return r.json(); });
  }
  buildNet();
  buildChart();
  loadData().then(async (json) => {
    data = FL.prepare(json);
    reset();
    const q = new URLSearchParams(location.search);
    if (q.has('bench')) {
      const out = await window.bench();
      const pre = document.createElement('pre');
      pre.id = 'bench-out'; pre.hidden = true; pre.textContent = JSON.stringify(out);
      document.body.appendChild(pre);
    } else if (q.has('rounds')) {
      // used for screenshots: fast-forward N rounds with the current controls
      const n = Math.min(200, +q.get('rounds') || 0);
      if (q.has('skew')) { cfg.skew = +q.get('skew'); skew.value = cfg.skew; $('#skew-out').textContent = skewText(cfg.skew); reset(); }
      for (let i = 0; i < n; i++) sim.step();
      renderAll();
      updateRunBtn();
    }
  }).catch((e) => {
    $('#status').textContent = 'could not load data/digits.json — serve this folder over http (see README).';
    console.error(e);
  });
})();
