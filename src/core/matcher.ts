import type { FracturePatch, MatchCandidate, MatchOptions, RigidTransform, Vec3 } from './types';
import { GridHash, applyRigid, identityRigid, invertRigid, mulRigid, rigidFromCorrespondences, v3 } from './math3d';
import { descDistance } from './descriptors';
import { subPatchesOf } from './fracture';

export const DEFAULT_OPTIONS: MatchOptions = {
  iterationsFactor: 1.0,
  gapThreshold: 1.5,
  minThicknessConsistency: 0.25,
};

function mean(x: Float32Array): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i];
  return s / (x.length || 1);
}

/** 两个子断口粗筛：规模与平均壁厚要接近 */
export function pairPreFilter(a: FracturePatch, b: FracturePatch): boolean {
  const na = a.points.length / 3;
  const nb = b.points.length / 3;
  const ratio = Math.min(na, nb) / Math.max(na, nb);
  if (ratio < 0.2) return false;
  const ta = mean(a.thickness), tb = mean(b.thickness);
  if (Math.abs(ta - tb) / Math.max(ta, tb, 1e-6) > 0.6) return false;
  return true;
}

interface Hypothesis {
  tf: RigidTransform;
  inliers: number;
  gap: number;
}

/**
 * 两个互补断口子块的配准。
 * 几何事实：同一条缝两侧点云位置几乎重合、法向相反。
 * 策略：RANSAC 三点求刚体（点位置对应）+ 反向点面 ICP 精修。
 */
function matchSubPatch(
  a: FracturePatch,
  b: FracturePatch,
  options: MatchOptions,
): (Omit<MatchCandidate, 'a' | 'b'> & { transformBtoA: RigidTransform }) | null {
  if (!pairPreFilter(a, b)) return null;

  const na = a.points.length / 3;
  const nb = b.points.length / 3;
  const dim = a.descDim;
  const inlierDist = options.gapThreshold;

  // FPFH 双向最近邻 + 互逆种子
  const nearestAtoB = new Int32Array(na);
  const nearestBtoA = new Int32Array(nb);
  for (let i = 0; i < na; i++) {
    let best = 0, bd = Infinity;
    for (let j = 0; j < nb; j++) {
      const d = descDistance(a.desc, i, b.desc, j, dim);
      if (d < bd) { bd = d; best = j; }
    }
    nearestAtoB[i] = best;
  }
  for (let j = 0; j < nb; j++) {
    let best = 0, bd = Infinity;
    for (let i = 0; i < na; i++) {
      const d = descDistance(a.desc, i, b.desc, j, dim);
      if (d < bd) { bd = d; best = i; }
    }
    nearestBtoA[j] = best;
  }
  const mutual: Array<[number, number]> = [];
  for (let i = 0; i < na; i++) {
    const j = nearestAtoB[i];
    if (nearestBtoA[j] === i) mutual.push([i, j]);
  }

  const gridA = GridHash.build(a.points, inlierDist);
  const iterations = Math.round(40 * options.iterationsFactor);
  let best: Hypothesis | null = null;

  // 零位移先验：同坐标系扫描件中，同一条缝两侧点云初始即贴合
  const identityHyp = countInliers(a, b, identityRigid(), gridA, inlierDist);
  if (identityHyp) best = identityHyp;

  const tried = new Set<number>();
  let trial = 0;
  let guard = 0;

  while (trial < iterations && guard < iterations * 10) {
    guard++;
    let ia: number, ib: number, ic: number;
    let ja: number, jb: number, jc: number;
    if (mutual.length >= 3 && trial < iterations * 0.75) {
      const pick = sampleThree(mutual.length);
      [ia, ja] = mutual[pick[0]];
      [ib, jb] = mutual[pick[1]];
      [ic, jc] = mutual[pick[2]];
    } else {
      ia = randInt(na); ib = randInt(na); ic = randInt(na);
      ja = nearestAtoB[ia]; jb = nearestAtoB[ib]; jc = nearestAtoB[ic];
    }
    if (ia === ib || ib === ic || ia === ic) continue;
    if (ja === jb || jb === jc || ja === jc) continue;
    const key = (ia * 1000000 + ib * 1000 + ic) | 0;
    if (tried.has(key)) continue;
    tried.add(key);

    const d01 = ptDist(a.points, ia, a.points, ib);
    const d12 = ptDist(a.points, ib, a.points, ic);
    const d02 = ptDist(a.points, ia, a.points, ic);
    const e01 = ptDist(b.points, ja, b.points, jb);
    const e12 = ptDist(b.points, jb, b.points, jc);
    const e02 = ptDist(b.points, ja, b.points, jc);
    if (d01 < 0.8 || d12 < 0.8 || d02 < 1.4) continue;
    if (!distCompatible(d01, e01) || !distCompatible(d12, e12) || !distCompatible(d02, e02)) continue;

    const tf = rigidFromCorrespondences(
      [ptOf(a.points, ia), ptOf(a.points, ib), ptOf(a.points, ic)],
      [ptOf(b.points, ja), ptOf(b.points, jb), ptOf(b.points, jc)],
    );
    const hyp = countInliers(a, b, tf, gridA, inlierDist);
    if (hyp && (!best || hyp.inliers > best.inliers ||
      (hyp.inliers === best.inliers && hyp.gap < best.gap))) {
      best = hyp;
    }
    trial++;
  }

  if (!best || best.inliers < Math.max(6, Math.min(na, nb) * 0.2)) return null;

  // 反向点面 ICP（互补面法向相反）
  let tf = best.tf;
  let icpGap = best.gap;
  let icpInliers = best.inliers;
  for (let iter = 0; iter < 10; iter++) {
    const refined = icpStepOpposing(a, b, tf, inlierDist);
    if (!refined) break;
    tf = refined.tf;
    if (Math.abs(refined.gap - icpGap) < 1e-5) { icpGap = refined.gap; icpInliers = refined.inliers; break; }
    icpGap = refined.gap;
    icpInliers = refined.inliers;
  }

  // 厚度梯度：扫描件断口法向在全局坐标下同向，
  // 同缝两侧壁厚变化方向一致（都指向口沿增厚/减薄方向），故取同向点积
  const thicknessConsistency = gradientConsistencySame(a, b, tf, inlierDist, gridA);
  if (thicknessConsistency < options.minThicknessConsistency) return null;

  const thicknessMatch = thicknessCheck(a, b, tf, inlierDist, gridA);
  if (thicknessMatch < 0.4) return null;

  // 釉面防反：两片外壁朝向应同侧（同坐标系扫描件天然成立，翻面会破坏它）
  if (!orientationCheck(a, b, tf)) return null;

  const overlapRatio = icpInliers / Math.min(na, nb);
  if (overlapRatio < 0.3) return null;

  const gapScore = clamp01(1 - icpGap / options.gapThreshold);
  const score = clamp01(0.38 * gapScore + 0.27 * overlapRatio + 0.2 * thicknessConsistency + 0.15 * thicknessMatch);

  return { score, gap: icpGap, transformBtoA: tf, overlap: icpInliers, thicknessConsistency };
}

/**
 * 两片碎片配对：枚举各自的边缝子块，取最佳组合。
 * 子块对是否成立由 RANSAC 内点率 + ICP 缝隙 + 厚度/釉面校验决定，
 * 不做质心距硬过滤（同缝两侧质心可相距一个胎壁以上）。
 */
export function matchPair(
  fa: FracturePatch,
  fb: FracturePatch,
  options: MatchOptions = DEFAULT_OPTIONS,
): (Omit<MatchCandidate, 'a' | 'b'> & { transformBtoA: RigidTransform }) | null {
  const subsA = subPatchesOf(fa);
  const subsB = subPatchesOf(fb);
  let best: (Omit<MatchCandidate, 'a' | 'b'> & { transformBtoA: RigidTransform }) | null = null;
  for (const sa of subsA) {
    for (const sb of subsB) {
      const r = matchSubPatch(sa, sb, options);
      if (r && (!best || r.score > best.score)) best = r;
    }
  }
  return best;
}

function countInliers(
  a: FracturePatch,
  b: FracturePatch,
  tf: RigidTransform,
  gridA: GridHash,
  inlierDist: number,
): Hypothesis | null {
  const nb = b.points.length / 3;
  let inliers = 0;
  let gapSum = 0;
  for (let j = 0; j < nb; j++) {
    const p = applyRigid(tf, ptOf(b.points, j));
    const ia = gridA.nearest(p[0], p[1], p[2], inlierDist);
    if (ia < 0) continue;
    const d = pointPlane(p, a.points, a.normals, ia);
    if (Math.abs(d) > inlierDist) continue;
    gapSum += Math.abs(d);
    inliers++;
  }
  if (inliers < 5) return null;
  return { tf, inliers, gap: gapSum / inliers };
}

/** 反向点面 ICP：目标残差 r = n_a·(q - p_a)，配准后 q≈p_a 且 n_b≈-n_a */
function icpStepOpposing(
  a: FracturePatch,
  b: FracturePatch,
  tf: RigidTransform,
  inlierDist: number,
): { tf: RigidTransform; gap: number; inliers: number } | null {
  const gridA = GridHash.build(a.points, inlierDist);
  const nb = b.points.length / 3;
  const JtJ = new Float64Array(36);
  const Jtr = new Float64Array(6);
  let inliers = 0;
  let gapSum = 0;
  for (let j = 0; j < nb; j++) {
    const q = applyRigid(tf, ptOf(b.points, j));
    const ia = gridA.nearest(q[0], q[1], q[2], inlierDist);
    if (ia < 0) continue;
    const nx = a.normals[ia * 3], ny = a.normals[ia * 3 + 1], nz = a.normals[ia * 3 + 2];
    const d = pointPlane(q, a.points, a.normals, ia);
    if (Math.abs(d) > inlierDist) continue;
    const cx = q[1] * nz - q[2] * ny;
    const cy = q[2] * nx - q[0] * nz;
    const cz = q[0] * ny - q[1] * nx;
    const row = [cx, cy, cz, nx, ny, nz];
    for (let u = 0; u < 6; u++) {
      Jtr[u] -= row[u] * d;
      for (let w = 0; w < 6; w++) JtJ[u * 6 + w] += row[u] * row[w];
    }
    gapSum += Math.abs(d);
    inliers++;
  }
  if (inliers < 6) return null;
  for (let i = 0; i < 6; i++) JtJ[i * 6 + i] *= 1 + 1e-5;
  const delta = solve6(JtJ, Jtr);
  if (!delta) return { tf, gap: gapSum / inliers, inliers };
  const dT = deltaToRigid(delta);
  const newTf = mulRigid(dT, tf);
  const check = countInliers(a, b, newTf, gridA, inlierDist);
  if (!check || check.gap > gapSum / inliers + 0.2) {
    return { tf, gap: gapSum / inliers, inliers };
  }
  return { tf: newTf, gap: check.gap, inliers: check.inliers };
}

function solve6(A: Float64Array, b: Float64Array): number[] | null {
  const m = new Float64Array(36);
  m.set(A);
  const y = new Float64Array(6);
  y.set(b);
  for (let col = 0; col < 6; col++) {
    let piv = col;
    for (let r = col + 1; r < 6; r++) {
      if (Math.abs(m[r * 6 + col]) > Math.abs(m[piv * 6 + col])) piv = r;
    }
    if (Math.abs(m[piv * 6 + col]) < 1e-12) return null;
    if (piv !== col) {
      for (let k = 0; k < 6; k++) {
        const t = m[col * 6 + k]; m[col * 6 + k] = m[piv * 6 + k]; m[piv * 6 + k] = t;
      }
      const t = y[col]; y[col] = y[piv]; y[piv] = t;
    }
    for (let r = 0; r < 6; r++) {
      if (r === col) continue;
      const f = m[r * 6 + col] / m[col * 6 + col];
      for (let k = col; k < 6; k++) m[r * 6 + k] -= f * m[col * 6 + k];
      y[r] -= f * y[col];
    }
  }
  const x = new Array<number>(6);
  for (let i = 0; i < 6; i++) x[i] = y[i] / m[i * 6 + i];
  return x;
}

function deltaToRigid(d: number[]): RigidTransform {
  const [wx, wy, wz, tx, ty, tz] = d;
  const angle = Math.hypot(wx, wy, wz);
  let r: number[];
  if (angle < 1e-9) {
    r = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  } else {
    const kx = wx / angle, ky = wy / angle, kz = wz / angle;
    const c = Math.cos(angle), s = Math.sin(angle), C = 1 - c;
    r = [
      c + kx * kx * C, kx * ky * C - kz * s, kx * kz * C + ky * s,
      ky * kx * C + kz * s, c + ky * ky * C, ky * kz * C - kx * s,
      kz * kx * C - ky * s, kz * ky * C + kx * s, c + kz * kz * C,
    ];
  }
  return { r, t: [tx, ty, tz] };
}

/** 厚度梯度一致性：同坐标系下两侧壁厚变化同向 */
function gradientConsistencySame(
  a: FracturePatch,
  b: FracturePatch,
  tf: RigidTransform,
  inlierDist: number,
  gridA: GridHash,
): number {
  const nb = b.points.length / 3;
  let sum = 0, hits = 0;
  const meanA = mean(a.thickness);
  for (let j = 0; j < nb; j++) {
    const p = applyRigid(tf, ptOf(b.points, j));
    const ia = gridA.nearest(p[0], p[1], p[2], inlierDist);
    if (ia < 0) continue;
    const db = applyRigidVec(tf, [b.thickDir[j * 3], b.thickDir[j * 3 + 1], b.thickDir[j * 3 + 2]]);
    const da: Vec3 = [a.thickDir[ia * 3], a.thickDir[ia * 3 + 1], a.thickDir[ia * 3 + 2]];
    const dot = v3.dot(da, db);
    const w = Math.min(1, Math.abs(a.thickness[ia] - meanA) + 0.15);
    sum += dot * w;
    hits += w;
  }
  if (!hits) return 0.5; // 梯度不显著时中性给分，不误杀
  return clamp01(0.5 + 0.5 * (sum / hits));
}

function thicknessCheck(
  a: FracturePatch,
  b: FracturePatch,
  tf: RigidTransform,
  inlierDist: number,
  gridA: GridHash,
): number {
  const nb = b.points.length / 3;
  let good = 0, hits = 0;
  // 厚度估计本身有系统性偏差，用相对误差（35%）而非绝对差
  for (let j = 0; j < nb; j++) {
    const p = applyRigid(tf, ptOf(b.points, j));
    const ia = gridA.nearest(p[0], p[1], p[2], inlierDist);
    if (ia < 0) continue;
    hits++;
    const ta = a.thickness[ia], tb = b.thickness[j];
    if (Math.abs(ta - tb) / Math.max(ta, tb, 1e-6) < 0.45) good++;
  }
  return hits ? good / hits : 0.5;
}

function orientationCheck(a: FracturePatch, b: FracturePatch, tf: RigidTransform): boolean {
  const bOuter = applyRigidVec(tf, b.outerDir);
  const bInner = applyRigidVec(tf, b.innerDir);
  // 同坐标系扫描件：外壁方向应同侧；innerDir 估计较弱不作为硬否决
  const outerDot = v3.dot(a.outerDir, bOuter);
  if (outerDot <= -0.5) return false;
  const innerDot = v3.dot(a.innerDir, b.innerDir);
  if (innerDot <= -0.7) return false;
  return true;
}

function applyRigidVec(tf: RigidTransform, v: Vec3): Vec3 {
  const r = tf.r;
  return [
    r[0] * v[0] + r[1] * v[1] + r[2] * v[2],
    r[3] * v[0] + r[4] * v[1] + r[5] * v[2],
    r[6] * v[0] + r[7] * v[1] + r[8] * v[2],
  ];
}

function pointPlane(p: Vec3, pts: Float32Array, normals: Float32Array, ia: number): number {
  const dx = p[0] - pts[ia * 3];
  const dy = p[1] - pts[ia * 3 + 1];
  const dz = p[2] - pts[ia * 3 + 2];
  return dx * normals[ia * 3] + dy * normals[ia * 3 + 1] + dz * normals[ia * 3 + 2];
}

function distCompatible(d1: number, d2: number, tol = 0.9): boolean {
  return Math.abs(d1 - d2) <= tol * Math.max(d1, d2) + 0.35;
}

function ptDist(pa: Float32Array, i: number, pb: Float32Array, j: number): number {
  return Math.hypot(
    pa[i * 3] - pb[j * 3],
    pa[i * 3 + 1] - pb[j * 3 + 1],
    pa[i * 3 + 2] - pb[j * 3 + 2],
  );
}

function ptOf(p: Float32Array, i: number): Vec3 {
  return [p[i * 3], p[i * 3 + 1], p[i * 3 + 2]];
}

function sampleThree(n: number): [number, number, number] {
  const a = randInt(n);
  let b = randInt(n - 1);
  if (b >= a) b++;
  let c = randInt(n - 2);
  if (c >= Math.min(a, b)) c++;
  if (c >= Math.max(a, b)) c++;
  return [a, b, c];
}

function randInt(n: number): number {
  return Math.floor(Math.random() * n);
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

// ---------------- 全量配对与贪心拼接 ----------------

export interface MatchAllResult {
  candidates: MatchCandidate[];
  elapsedMs: number;
}

export function matchAll(
  shards: Array<{ id: number; fracture: FracturePatch }>,
  options: MatchOptions = DEFAULT_OPTIONS,
  onProgress?: (done: number, total: number, current: string) => void,
): MatchAllResult {
  const t0 = now();
  const n = shards.length;
  const total = (n * (n - 1)) / 2;
  const candidates: MatchCandidate[] = [];
  let done = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const r = matchPair(shards[i].fracture, shards[j].fracture, options);
      if (r) candidates.push({ a: shards[i].id, b: shards[j].id, ...r });
      done++;
      if (onProgress && (done % 6 === 0 || done === total)) {
        onProgress(done, total, `${shards[i].id}×${shards[j].id}`);
      }
    }
  }
  candidates.sort((x, y) => y.score - x.score);
  return { candidates, elapsedMs: now() - t0 };
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

/**
 * 贪心并查集拼接：按得分从高到低接受候选，
 * 整组刚体变换串联——"接上一片整体跟着动"。
 * 全局空间一致性：候选给出的 b->a 变换若把 b 质心搬到远离 a 质心的位置
 * （超过两片在扫描坐标下原距离的 2 倍），判为局部贴合的假边。
 */
export interface AssemblyResult {
  transforms: Map<number, RigidTransform>;
  accepted: MatchCandidate[];
}

export function greedyAssembly(
  shardIds: number[],
  candidates: MatchCandidate[],
  maxDegree = 4,
  centroids?: Map<number, Vec3>,
): AssemblyResult {
  const parent = new Map<number, number>();
  const toRoot = new Map<number, RigidTransform>();
  for (const id of shardIds) {
    parent.set(id, id);
    toRoot.set(id, identityRigid());
  }
  const degree = new Map<number, number>(shardIds.map((id) => [id, 0]));
  const accepted: MatchCandidate[] = [];
  const find = (x: number): number => {
    const p = parent.get(x)!;
    if (p === x) return x;
    const root = find(p);
    if (root !== p) {
      toRoot.set(x, mulRigid(toRoot.get(p)!, toRoot.get(x)!));
      parent.set(x, root);
    }
    return root;
  };

  const ranked = [...candidates].sort((x, y) => y.score - x.score);
  for (const c of ranked) {
    const ra = find(c.a);
    const rb = find(c.b);
    if (ra === rb) continue;
    if ((degree.get(c.a) ?? 0) >= maxDegree || (degree.get(c.b) ?? 0) >= maxDegree) continue;

    // 全局空间一致性：变换后 b 质心与 a 质心距离应合理
    if (centroids && centroids.has(c.a) && centroids.has(c.b)) {
      const ca = centroids.get(c.a)!;
      const cb = centroids.get(c.b)!;
      const originalDist = v3.dist(ca, cb);
      const moved = applyRigid(c.transformBtoA, cb);
      const movedDist = v3.dist(moved, ca);
      // 仅当原始距离很远（>80mm，非邻片）且变换后仍保持远距时才判假边
      if (originalDist > 80 && movedDist > originalDist * 0.85) continue;
      if (movedDist > 140) continue;
    }

    const aToRoot = toRoot.get(c.a)!;
    const bToRoot = toRoot.get(c.b)!;
    // b 点先由自身坐标到 b 组旧根（bToRoot），再由旧根映射到新根：
    // 新 toRoot = bNewToRoot · inv(bToRoot) · 旧 toRoot
    const bNewToRoot = mulRigid(aToRoot, c.transformBtoA);
    const rbOffset = mulRigid(bNewToRoot, invertRigid(bToRoot));
    // 先一次性收集 rb 组成员，再统一改变换与父指针（不能边 find 边改）
    const members: number[] = [];
    for (const id of shardIds) if (find(id) === rb) members.push(id);
    for (const id of members) {
      toRoot.set(id, mulRigid(rbOffset, toRoot.get(id)!));
      parent.set(id, ra);
    }
    degree.set(c.a, (degree.get(c.a) ?? 0) + 1);
    degree.set(c.b, (degree.get(c.b) ?? 0) + 1);
    accepted.push(c);
  }
  for (const id of shardIds) find(id);
  return { transforms: toRoot, accepted };
}
