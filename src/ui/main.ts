// 应用主控：Worker 调度、候选确认、补配测量、渲染交互
import type {
  Dataset, FracturePatch, MatchCandidate, MeasureResult, Mesh, Shard, WorkerResponse,
} from '../core/types';
import { SurfaceLabel } from '../core/types';
import { deserializeDataset, type SerializedDataset } from '../core/serialize';
import { analyzeShard, subPatchesOf } from '../core/fracture';
import { greedyAssembly, DEFAULT_OPTIONS } from '../core/matcher';
import { globalICP } from '../core/global-icp';
import { verifyGaps, type GapReport } from '../core/gap';
import { measureVessel, exportLineSVG, exportMeasureJSON } from '../core/measure';
import { fillMissingBySymmetry, mergeFillMesh } from '../core/fill';
import { parseOBJ } from '../core/obj-loader';
import { identityRigid, applyRigid, v3 } from '../core/math3d';
import type { RigidTransform } from '../core/types';
import { Renderer, shardColor, type RenderableShard } from '../webgl/renderer';

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const btn = (id: string): HTMLButtonElement => document.getElementById(id) as HTMLButtonElement;

interface AppState {
  dataset: Dataset | null;
  renderables: RenderableShard[];
  candidates: MatchCandidate[];
  accepted: MatchCandidate[];
  measure: MeasureResult | null;
  gap: GapReport | null;
  fillMesh: Mesh | null;
  exploded: boolean;
  worker: Worker | null;
}

const state: AppState = {
  dataset: null,
  renderables: [],
  candidates: [],
  accepted: [],
  measure: null,
  gap: null,
  fillMesh: null,
  exploded: false,
  worker: null,
};

let renderer: Renderer;

function initRenderer(): void {
  renderer = new Renderer($<HTMLCanvasElement>('glcanvas'));
  bindCamera();
  requestAnimationFrame(renderLoop);
}

function renderLoop(): void {
  const linePoints = state.measure && $<HTMLInputElement>('chk-outline').checked
    ? outlineToWorld(state.measure)
    : null;
  renderer.render(state.renderables, linePoints);
  updateLabels();
  requestAnimationFrame(renderLoop);
}

// ---------------- 数据载入 ----------------

async function loadDemo(): Promise<void> {
  const resp = await fetch('./datasets/pot30.json');
  const json = (await resp.json()) as SerializedDataset;
  loadDataset(deserializeDataset(json));
}

function loadDataset(ds: Dataset): void {
  state.dataset = ds;
  state.candidates = [];
  state.accepted = [];
  state.measure = null;
  state.gap = null;
  state.fillMesh = null;
  renderer.setFillMesh(null);
  state.renderables = ds.shards.map((s) => ({
    id: s.id,
    code: s.code,
    mesh: s.mesh,
    transform: identityRigid(),
    color: shardColor(s.id),
    visible: true,
  }));
  renderer.setShards(state.renderables);
  frameCameraToContent();
  renderShardList();
  renderCandidates();
  btn('btn-analyze').disabled = false;
  btn('btn-match').disabled = true;
  btn('btn-fill').disabled = true;
  btn('btn-measure').disabled = true;
  btn('btn-export-svg').disabled = true;
  btn('btn-export-json').disabled = true;
  $('match-metrics').innerHTML = `<div>已载入 <b>${ds.shards.length}</b> 片碎片${ds.missingIds?.length ? `，缺口 ${ds.missingIds.length} 处待补配` : ''}</div>`;
  $('measure-output').innerHTML = '';
  $('gap-output').innerHTML = '';
}

function frameCameraToContent(): void {
  let mn: [number, number, number] = [Infinity, Infinity, Infinity];
  let mx: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (const r of state.renderables) {
    for (let i = 0; i < r.mesh.positions.length / 3; i++) {
      const p = applyRigid(r.transform, [
        r.mesh.positions[i * 3], r.mesh.positions[i * 3 + 1], r.mesh.positions[i * 3 + 2],
      ]);
      for (let k = 0; k < 3; k++) {
        if (p[k] < mn[k]) mn[k] = p[k];
        if (p[k] > mx[k]) mx[k] = p[k];
      }
    }
  }
  const center: [number, number, number] = [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2];
  const span = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]);
  renderer.camera.target = center;
  renderer.camera.distance = span * 1.9 + 60;
}

// ---------------- Worker 流程 ----------------

function getWorker(): Worker {
  if (!state.worker) state.worker = new Worker(new URL('../worker/matcher.worker.ts', import.meta.url), { type: 'module' });
  return state.worker;
}

function setProgress(show: boolean, done?: number, total?: number, text?: string): void {
  $('progress-wrap').hidden = !show;
  if (show) {
    const pct = total ? Math.round((done! / total) * 100) : 0;
    $('progress-bar').style.width = `${pct}%`;
    $('progress-text').textContent = text ?? '';
  }
}

async function runAnalyze(): Promise<void> {
  if (!state.dataset) return;
  btn('btn-analyze').disabled = true;
  setProgress(true, 0, state.dataset.shards.length, '断口分析中…');
  const t0 = performance.now();
  // 分析在 Worker 中执行
  const fractures = await callWorker<Array<[number, FracturePatch]>>({
    kind: 'analyze',
    shards: state.dataset.shards.map((s) => ({ id: s.id, mesh: s.mesh })),
  }, (done, total, current) => setProgress(true, done, total, `断口分析：${current}`));
  const fmap = new Map(fractures);
  for (const s of state.dataset.shards) s.fracture = fmap.get(s.id);
  const withFrac = state.dataset.shards.filter((s) => s.fracture).length;
  setProgress(false);
  btn('btn-match').disabled = false;
  $('match-metrics').innerHTML =
    `<div>断口分析完成：<b>${withFrac}</b>/${state.dataset.shards.length} 片识别到边缝</div>` +
    `<div>用时 <b>${(performance.now() - t0).toFixed(0)} ms</b>（Worker 线程）</div>`;
}

async function runMatch(): Promise<void> {
  if (!state.dataset) return;
  const ready = state.dataset.shards.filter((s) => s.fracture);
  if (!ready.length) return;
  btn('btn-match').disabled = true;
  setProgress(true, 0, (ready.length * (ready.length - 1)) / 2, '全自动配对中…');
  const t0 = performance.now();
  const resp = await callWorker<{ candidates: MatchCandidate[]; elapsedMs: number }>({
    kind: 'matchAll',
    shards: ready.map((s) => ({ id: s.id, fracture: s.fracture! })),
    options: DEFAULT_OPTIONS,
  }, (done, total, current) => setProgress(true, done, total, `配对：${current}`));
  state.candidates = resp.candidates;
  state.accepted = [];
  setProgress(false);
  renderCandidates();
  const gt = state.dataset.groundTruthAdjacency;
  let recallTxt = '';
  if (gt) {
    const present = new Set(state.dataset.shards.map((s) => s.id));
    const gtSet = new Set(gt.filter(([a, b]) => present.has(a) && present.has(b)).map(([a, b]) => a < b ? `${a}-${b}` : `${b}-${a}`));
    const hit = state.candidates.filter((c) => gtSet.has(c.a < c.b ? `${c.a}-${c.b}` : `${c.b}-${c.a}`)).length;
    recallTxt = `<div>真值邻接召回：<b class="${hit / gtSet.size >= 0.85 ? 'pass' : 'fail'}">${(hit / gtSet.size * 100).toFixed(1)}%</b>（${hit}/${gtSet.size}）</div>`;
  }
  $('match-metrics').innerHTML =
    `<div>配对耗时 <b class="${resp.elapsedMs <= 20000 ? 'pass' : 'fail'}">${resp.elapsedMs.toFixed(0)} ms</b> / 20000 ms</div>` +
    `<div>产出候选 <b>${state.candidates.length}</b> 条</div>` + recallTxt;
  btn('btn-match').disabled = false;
  btn('btn-fill').disabled = false;
  btn('btn-measure').disabled = false;
}

function callWorker<T>(req: unknown, onProgress?: (done: number, total: number, current: string) => void): Promise<T> {
  const worker = getWorker();
  return new Promise((resolve, reject) => {
    const handler = (ev: MessageEvent<WorkerResponse>) => {
      const msg = ev.data;
      if (msg.kind === 'progress') {
        onProgress?.(msg.done, msg.total, msg.current);
      } else if (msg.kind === 'error') {
        worker.removeEventListener('message', handler);
        reject(new Error(msg.message));
      } else if (msg.kind.endsWith(':done')) {
        worker.removeEventListener('message', handler);
        if (msg.kind === 'analyze:done') resolve(msg.fractures as T);
        else if (msg.kind === 'matchAll:done') resolve({ candidates: msg.candidates, elapsedMs: msg.elapsedMs } as T);
        else resolve(null as T);
      }
    };
    worker.addEventListener('message', handler);
    worker.postMessage(req);
  });
}

// ---------------- 候选确认与拼合 ----------------

function renderCandidates(): void {
  const el = $('candidate-list');
  if (!state.candidates.length) {
    el.innerHTML = '<div class="empty">尚无候选</div>';
    return;
  }
  const acceptedKeys = new Set(state.accepted.map((c) => keyOf(c)));
  el.innerHTML = '';
  state.candidates.slice(0, 60).forEach((c, idx) => {
    const accepted = acceptedKeys.has(keyOf(c));
    const div = document.createElement('div');
    div.className = 'cand' + (accepted ? ' accepted' : '');
    div.innerHTML =
      `<div class="line1"><span class="pair">#${c.a} ↔ #${c.b}</span><span class="score">得分 ${c.score.toFixed(3)}</span></div>` +
      `<div class="detail">
        <span>平均缝隙 <b>${c.gap.toFixed(3)} mm</b></span>
        <span>重叠点 ${c.overlap}</span>
        <span>厚度一致性 ${c.thicknessConsistency.toFixed(2)}</span>
        <span>排名 ${idx + 1}</span>
      </div>` +
      (accepted
        ? '<div class="actions"><button class="btn small ok" disabled>已接受</button></div>'
        : '<div class="actions"><button class="btn small ok" data-act="accept">接受并拼合</button><button class="btn small skip" data-act="skip">跳过</button></div>');
    div.querySelector('[data-act="accept"]')?.addEventListener('click', () => acceptCandidate(c));
    div.querySelector('[data-act="skip"]')?.addEventListener('click', () => { div.style.display = 'none'; });
    el.appendChild(div);
  });
}

function keyOf(c: MatchCandidate): string {
  return c.a < c.b ? `${c.a}-${c.b}` : `${c.b}-${c.a}`;
}

/** 接受单条候选：把 b 并入 a 所在组，整组变换联动 */
function acceptCandidate(c: MatchCandidate): void {
  if (!state.dataset) return;
  if (state.accepted.some((x) => keyOf(x) === keyOf(c))) return;
  state.accepted.push(c);
  applyAssembly(state.accepted);
  renderCandidates();
  updateGapReport();
}

/** 一键自动拼合全部候选 */
function autoAssemble(): void {
  if (!state.dataset || !state.candidates.length) return;
  const ids = state.dataset.shards.map((s) => s.id);
  const centroids = new Map(state.dataset.shards.map((s) => [s.id, s.centroid]));
  state.accepted = greedyAssembly(ids, state.candidates, 4, centroids).accepted;
  applyAssembly(state.accepted);
  renderCandidates();
  updateGapReport();
}

/** 先应用并查集拼合，再跑全局 ICP（直接改写 shard.transform），最后同步渲染对象 */
function applyAssembly(edges: MatchCandidate[]): void {
  if (!state.dataset) return;
  const ids = state.dataset.shards.map((s) => s.id);
  const centroids = new Map(state.dataset.shards.map((s) => [s.id, s.centroid]));
  const assembly = greedyAssembly(ids, edges, 4, centroids);
  const byId = new Map(state.dataset.shards.map((s) => [s.id, s]));
  for (const r of state.renderables) {
    const shard = byId.get(r.id);
    if (shard) {
      shard.transform = assembly.transforms.get(r.id) ?? identityRigid();
      r.transform = shard.transform;
    }
  }
  globalICP(state.dataset.shards.filter((s) => s.fracture), edges, 3);
  for (const r of state.renderables) {
    const shard = byId.get(r.id);
    if (shard) r.transform = shard.transform;
  }
}

function updateGapReport(): void {
  if (!state.dataset || !state.accepted.length) return;
  const adj = state.accepted.map((c) => [c.a, c.b] as [number, number]);
  state.gap = verifyGaps(state.dataset.shards, adj);
  const g = state.gap;
  $('gap-output').innerHTML =
    `<div>可测邻接边 <b>${g.edgeCount}</b> 条</div>` +
    `<div>平均缝隙 <b class="${g.meanGap <= 0.3 ? 'ok' : 'bad'}">${g.meanGap.toFixed(3)} mm</b> / 0.3 mm</div>` +
    `<div>合格边占比 <b class="${g.passRatio >= 0.9 ? 'ok' : 'bad'}">${(g.passRatio * 100).toFixed(1)}%</b></div>` +
    g.perEdge.slice(0, 6).map((e) =>
      `<div>#${e.a}-#${e.b}：${e.meanGap.toFixed(3)} mm ${e.meanGap <= 0.3 ? '✓' : '✗'}</div>`).join('');
}

// ---------------- 补配与测量 ----------------

function runFill(): void {
  if (!state.dataset) return;
  const missing = state.dataset.missingIds ?? [];
  if (!missing.length) {
    alert('当前数据集没有标记缺口');
    return;
  }
  const presentWorld = state.dataset.shards.map((s) => ({
    positions: transformAll(s.mesh.positions, s.transform),
    labels: s.mesh.labels,
  }));
  // 缺口方位：用编号对应的原始分布（示例数据集保留了规则网格，按角度均分）
  const columns = 6;
  const regions = missing.map((id, k) => {
    const col = id % columns;
    const row = Math.floor(id / columns);
    const theta = (col + 0.5) / columns * Math.PI * 2;
    const rowCuts = [26, 52, 80, 106];
    const z0 = row === 0 ? 0 : rowCuts[row - 1];
    const z1 = row === 4 ? state.dataset!.profile.height : rowCuts[row];
    return { theta, thetaWidth: (Math.PI * 2) / columns, z0, z1 };
  });
  const fills = fillMissingBySymmetry(presentWorld, state.dataset.profile, regions);
  let merged: Mesh = mergeAllShards();
  for (const f of fills) merged = mergeFillMesh(merged, f);
  state.fillMesh = merged;
  // 渲染端只显示补配增量：单独抽出 Fill 部分
  const fillOnly = extractLabel(merged, SurfaceLabel.Fill);
  renderer.setFillMesh(fillOnly);
  $('gap-output').innerHTML += `<div class="ok">已按器形对称补配 ${fills.length} 处（青蓝显示）</div>`;
}

function runMeasure(): void {
  if (!state.dataset) return;
  const merged = state.fillMesh ?? mergeAllShards();
  state.measure = measureVessel(merged);
  const m = state.measure;
  const truth = state.dataset.profile;
  $('measure-output').innerHTML =
    `<div>器形库匹配：<b>${m.bestMatch.type}</b>（库距 ${m.bestMatch.distance.toFixed(3)}）</div>` +
    `<div>口径 <b>${m.mouthDiameter.toFixed(1)}</b> mm${truth ? `　设计值 ${(truth.outer[truth.outer.length - 1][0] * 2).toFixed(0)}` : ''}</div>` +
    `<div>腹径 <b>${m.bellyDiameter.toFixed(1)}</b> mm${truth ? `　设计值 ${(Math.max(...truth.outer.map((p) => p[0])) * 2).toFixed(0)}` : ''}</div>` +
    `<div>底径 <b>${m.baseDiameter.toFixed(1)}</b> mm${truth ? `　设计值 ${(truth.outer[0][0] * 2).toFixed(0)}` : ''}</div>` +
    `<div>通高 <b>${m.height.toFixed(1)}</b> mm${truth ? `　设计值 ${truth.height}` : ''}</div>`;
  btn('btn-export-svg').disabled = false;
  btn('btn-export-json').disabled = false;
}

function mergeAllShards(): Mesh {
  const ds = state.dataset!;
  let positions = new Float32Array(0);
  let labels = new Uint8Array(0);
  let indices = new Uint32Array(0);
  for (const s of ds.shards) {
    const tp = transformAll(s.mesh.positions, s.transform);
    const np = new Float32Array(positions.length + tp.length);
    np.set(positions); np.set(tp, positions.length);
    const nl = new Uint8Array(labels.length + s.mesh.labels.length);
    nl.set(labels); nl.set(s.mesh.labels, labels.length);
    const base = positions.length / 3;
    const ni = new Uint32Array(indices.length + s.mesh.indices.length);
    ni.set(indices);
    for (let i = 0; i < s.mesh.indices.length; i++) ni[indices.length + i] = s.mesh.indices[i] + base;
    positions = np; labels = nl; indices = ni;
  }
  return { positions, labels, indices };
}

function extractLabel(mesh: Mesh, label: SurfaceLabel): Mesh {
  const keep: number[] = [];
  for (let i = 0; i < mesh.positions.length / 3; i++) if (mesh.labels[i] === label) keep.push(i);
  const remap = new Map<number, number>();
  keep.forEach((oldV, newV) => remap.set(oldV, newV));
  const positions = new Float32Array(keep.length * 3);
  const labels = new Uint8Array(keep.length);
  keep.forEach((oldV, newV) => {
    positions[newV * 3] = mesh.positions[oldV * 3];
    positions[newV * 3 + 1] = mesh.positions[oldV * 3 + 1];
    positions[newV * 3 + 2] = mesh.positions[oldV * 3 + 2];
    labels[newV] = mesh.labels[oldV];
  });
  const idx: number[] = [];
  for (let i = 0; i < mesh.indices.length; i += 3) {
    const a = mesh.indices[i], b = mesh.indices[i + 1], c = mesh.indices[i + 2];
    if (remap.has(a) && remap.has(b) && remap.has(c)) {
      idx.push(remap.get(a)!, remap.get(b)!, remap.get(c)!);
    }
  }
  return { positions, labels, indices: Uint32Array.from(idx) };
}

function transformAll(positions: Float32Array, tf: RigidTransform): Float32Array {
  const out = new Float32Array(positions.length);
  for (let i = 0; i < positions.length / 3; i++) {
    const w = applyRigid(tf, [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]]);
    out[i * 3] = w[0]; out[i * 3 + 1] = w[1]; out[i * 3 + 2] = w[2];
  }
  return out;
}

function outlineToWorld(m: MeasureResult): Float32Array {
  const { origin, dir } = m.axis;
  const helper: [number, number, number] = Math.abs(dir[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const ex = v3.norm(v3.cross(helper, dir));
  const ey = v3.cross(dir, ex);
  const z0 = m.outline[0][1];
  const pts: number[] = [];
  for (const [r, z] of m.outline) {
    const p: [number, number, number] = [
      origin[0] + ex[0] * r + ey[0] * 0 + dir[0] * (z - z0),
      origin[1] + ex[1] * r + ey[1] * 0 + dir[1] * (z - z0),
      origin[2] + ex[2] * r + ey[2] * 0 + dir[2] * (z - z0),
    ];
    pts.push(...p);
  }
  return Float32Array.from(pts);
}

// ---------------- 碎片清单与标注 ----------------

function renderShardList(): void {
  const el = $('shard-list');
  el.innerHTML = '';
  for (const r of state.renderables) {
    const div = document.createElement('div');
    div.className = 'shard-item' + (r.id === renderer.selectedId ? ' selected' : '');
    div.innerHTML =
      `<span class="dot" style="background:rgb(${r.color.map((c) => Math.round(c * 255)).join(',')})"></span>` +
      `<span class="code">${r.code}</span>` +
      `<span class="eye">${r.visible ? '👁' : '—'}</span>`;
    div.addEventListener('click', () => {
      renderer.selectedId = renderer.selectedId === r.id ? -1 : r.id;
      renderShardList();
    });
    el.appendChild(div);
  }
}

function updateLabels(): void {
  const layer = $('label-layer');
  if (!$<HTMLInputElement>('chk-labels').checked) {
    layer.innerHTML = '';
    return;
  }
  const existing = new Map<string, HTMLDivElement>();
  layer.querySelectorAll<HTMLDivElement>('.shard-label').forEach((d) => existing.set(d.dataset.id!, d));
  const seen = new Set<string>();
  for (const r of state.renderables) {
    if (!r.visible) continue;
    // 质心变换到世界坐标
    const wc = applyRigid(r.transform, centroidOfMesh(r.mesh, r.transform));
    const sp = renderer.worldToScreen(wc);
    if (sp.depth > 1) continue;
    seen.add(String(r.id));
    let tag = existing.get(String(r.id));
    if (!tag) {
      tag = document.createElement('div');
      tag.className = 'shard-label';
      tag.dataset.id = String(r.id);
      tag.textContent = r.code;
      layer.appendChild(tag);
    }
    tag.style.left = `${sp.x}px`;
    tag.style.top = `${sp.y}px`;
  }
  existing.forEach((tag, id) => { if (!seen.has(id)) tag.remove(); });
}

function centroidOfMesh(mesh: Mesh, _tf: RigidTransform): [number, number, number] {
  let x = 0, y = 0, z = 0;
  const n = mesh.positions.length / 3;
  for (let i = 0; i < n; i++) {
    x += mesh.positions[i * 3]; y += mesh.positions[i * 3 + 1]; z += mesh.positions[i * 3 + 2];
  }
  return [x / n, y / n, z / n];
}

// ---------------- 相机与点选 ----------------

function bindCamera(): void {
  const canvas = $<HTMLCanvasElement>('glcanvas');
  let dragging: 'rotate' | 'pan' | null = null;
  let lastX = 0, lastY = 0;
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('mousedown', (e) => {
    dragging = e.button === 2 ? 'pan' : 'rotate';
    lastX = e.clientX; lastY = e.clientY;
  });
  window.addEventListener('mouseup', () => { dragging = null; });
  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    const c = renderer.camera;
    if (dragging === 'rotate') {
      c.yaw += dx * 0.008;
      c.pitch = Math.max(-1.45, Math.min(1.45, c.pitch + dy * 0.008));
    } else {
      const scale = c.distance * 0.0012;
      const right: [number, number, number] = [Math.cos(c.yaw), 0, -Math.sin(c.yaw)];
      const up: [number, number, number] = [0, 1, 0];
      c.target = [
        c.target[0] - (right[0] * dx - up[0] * dy) * scale,
        c.target[1] - (right[1] * dx - up[1] * dy) * scale,
        c.target[2] - (right[2] * dx - up[2] * dy) * scale,
      ];
    }
  });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    renderer.camera.distance = Math.max(30, Math.min(1500, renderer.camera.distance * (1 + Math.sign(e.deltaY) * 0.1)));
  }, { passive: false });
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const id = renderer.pick(state.renderables, e.clientX - rect.left, e.clientY - rect.top);
    renderer.selectedId = id;
    renderShardList();
    showTooltip(e, id);
  });
}

function showTooltip(e: MouseEvent, id: number): void {
  const tip = $<HTMLDivElement>('shard-tooltip');
  if (id < 0 || !state.dataset) { tip.hidden = true; return; }
  const shard = state.dataset.shards.find((s) => s.id === id);
  if (!shard) { tip.hidden = true; return; }
  const subs = shard.fracture ? subPatchesOf(shard.fracture).length : 0;
  tip.innerHTML =
    `<b>${shard.code}</b><br>碎片序号 #${id}<br>面积 ${shard.area.toFixed(0)} mm²<br>边缝子块 ${subs} 条`;
  tip.style.left = `${e.clientX + 14}px`;
  tip.style.top = `${e.clientY + 14}px`;
  tip.hidden = false;
}

// ---------------- 事件绑定 ----------------

function bindEvents(): void {
  $('btn-load-demo').addEventListener('click', () => loadDemo().catch(console.error));
  $('btn-analyze').addEventListener('click', () => runAnalyze().catch(console.error));
  $('btn-match').addEventListener('click', () => runMatch().catch(console.error));
  $('btn-fill').addEventListener('click', runFill);
  $('btn-measure').addEventListener('click', runMeasure);
  $('btn-export-svg').addEventListener('click', () => {
    if (!state.measure) return;
    download('vessel-outline.svg', exportLineSVG(state.measure), 'image/svg+xml');
  });
  $('btn-export-json').addEventListener('click', () => {
    if (!state.measure) return;
    download('measure.json', exportMeasureJSON(state.measure), 'application/json');
  });
  $('chk-fill').addEventListener('change', (e) => { renderer.showFill = (e.target as HTMLInputElement).checked; });
  $('chk-outline').addEventListener('change', () => { /* renderLoop 内读取 */ });
  $('sel-color').addEventListener('change', (e) => {
    renderer.shardMode = Number((e.target as HTMLSelectElement).value) as 0 | 2;
  });
  $('btn-reset-view').addEventListener('click', () => {
    renderer.camera = { yaw: 0.6, pitch: 0.35, distance: renderer.camera.distance, target: renderer.camera.target };
    frameCameraToContent();
  });
  $('btn-explode').addEventListener('click', () => {
    state.exploded = !state.exploded;
    const center = renderer.camera.target;
    for (const r of state.renderables) {
      if (state.exploded) {
        const c = applyRigid(r.transform, centroidOfMesh(r.mesh, r.transform));
        const dir = v3.norm(v3.sub(c, center));
        r.transform = { ...r.transform, t: [r.transform.t[0] + dir[0] * 45, r.transform.t[1] + dir[1] * 45, r.transform.t[2] + dir[2] * 45] };
      } else if (state.dataset) {
        const shard = state.dataset.shards.find((s) => s.id === r.id);
        if (shard) r.transform = shard.transform;
      }
    }
  });
  $('file-obj').addEventListener('change', async (e) => {
    const input = e.target as HTMLInputElement;
    if (!input.files?.length) return;
    const shards: Shard[] = [];
    for (let i = 0; i < input.files.length; i++) {
      const file = input.files[i];
      const text = await file.text();
      const mesh = parseOBJ(text);
      shards.push({
        id: i,
        code: file.name.replace(/\.obj$/i, ''),
        mesh,
        transform: identityRigid(),
        area: 0,
        centroid: centroidOfMesh(mesh, identityRigid()),
      });
    }
    loadDataset({
      name: 'OBJ 导入', unit: 'mm',
      profile: { height: 0, outer: [], inner: [] },
      shards,
    });
  });
}

function download(name: string, content: string, mime: string): void {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}

// 自动拼合按钮（候选区顶部快捷操作）
function addAutoButton(): void {
  const btn = document.createElement('button');
  btn.className = 'btn primary';
  btn.style.width = '100%';
  btn.style.marginTop = '8px';
  btn.textContent = '一键自动拼合全部候选';
  btn.addEventListener('click', autoAssemble);
  $('candidate-list').before(btn);
}

initRenderer();
bindEvents();
addAutoButton();
loadDemo().catch(console.error);
