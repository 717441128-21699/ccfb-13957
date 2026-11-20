// 全局拼合精修：对已接受的每条邻接边做点面 ICP，增量修正整组位姿
import type { FracturePatch, MatchCandidate, RigidTransform, Shard } from './types';
import { GridHash, applyRigid, mulRigid } from './math3d';
import { collectFracturePoints } from './fracture';

export interface RefineStats {
  iterations: number;
  maxDelta: number;
}

/**
 * 对拼合结果做全局 ICP（Gauss-Seidel 沿生成树传播）：
 * 每轮遍历候选边，把 b 的断口点在当前整体位姿下向 a 配准，
 * 求得小增量刚体后施加到生成树中 b 侧的全部碎片——"接上一片整体跟着动"。
 */
export function globalICP(
  shards: Shard[],
  edges: MatchCandidate[],
  maxIter = 3,
): RefineStats {
  const byId = new Map(shards.map((s) => [s.id, s]));
  const adj = new Map<number, number[]>();
  for (const s of shards) adj.set(s.id, []);
  for (const e of edges) {
    adj.get(e.a)?.push(e.b);
    adj.get(e.b)?.push(e.a);
  }
  let maxDelta = 0;
  let iterations = 0;

  for (let iter = 0; iter < maxIter; iter++) {
    iterations++;
    let iterMax = 0;
    const ordered = [...edges].sort((x, y) => y.score - x.score);
    for (const edge of ordered) {
      const a = byId.get(edge.a);
      const b = byId.get(edge.b);
      if (!a?.fracture || !b?.fracture) continue;
      const delta = edgeIncrement(a, b);
      if (!delta) continue;
      const side = bSideNodes(adj, edge.a, edge.b);
      for (const id of side) {
        const s = byId.get(id);
        if (s) s.transform = mulRigid(delta, s.transform);
      }
      const dt = Math.hypot(delta.t[0], delta.t[1], delta.t[2]);
      if (dt > iterMax) iterMax = dt;
    }
    maxDelta = iterMax;
    if (iterMax < 0.02) break;
  }
  return { iterations, maxDelta };
}

/** 去掉 a-b 边后，生成树中包含 b 的一侧节点 */
function bSideNodes(adj: Map<number, number[]>, a: number, b: number): number[] {
  const out: number[] = [];
  const seen = new Set<number>([a]);
  const stack = [b];
  seen.add(b);
  while (stack.length) {
    const x = stack.pop()!;
    out.push(x);
    for (const n of adj.get(x) ?? []) {
      if (!seen.has(n)) {
        seen.add(n);
        stack.push(n);
      }
    }
  }
  return out;
}

/** 求 b 相对 a 在当前世界位姿下的 ICP 增量（返回作用于 b 局部坐标的刚体） */
function edgeIncrement(a: Shard, b: Shard): RigidTransform | null {
  const fa = a.fracture!;
  const fb = b.fracture!;

  // a 全部边缝子块的世界坐标
  const aCol = collectFracturePoints(fa);
  const worldA = new Float32Array(aCol.points.length);
  for (let i = 0; i < aCol.points.length / 3; i++) {
    const w = applyRigid(a.transform, [aCol.points[i * 3], aCol.points[i * 3 + 1], aCol.points[i * 3 + 2]]);
    worldA[i * 3] = w[0]; worldA[i * 3 + 1] = w[1]; worldA[i * 3 + 2] = w[2];
  }
  const worldAN = new Float32Array(aCol.normals.length);
  for (let i = 0; i < aCol.normals.length / 3; i++) {
    const n = rotVec(a.transform, [aCol.normals[i * 3], aCol.normals[i * 3 + 1], aCol.normals[i * 3 + 2]]);
    worldAN[i * 3] = n[0]; worldAN[i * 3 + 1] = n[1]; worldAN[i * 3 + 2] = n[2];
  }
  const grid = GridHash.build(worldA, 1.5);

  // 6×6 最小二乘（世界系下的小增量）
  const JtJ = new Float64Array(36);
  const Jtr = new Float64Array(6);
  let inliers = 0;
  const bCol = collectFracturePoints(fb);
  for (let j = 0; j < bCol.points.length / 3; j++) {
    const q = applyRigid(b.transform, [bCol.points[j * 3], bCol.points[j * 3 + 1], bCol.points[j * 3 + 2]]);
    const ia = grid.nearest(q[0], q[1], q[2], 1.5);
    if (ia < 0) continue;
    const nx = worldAN[ia * 3], ny = worldAN[ia * 3 + 1], nz = worldAN[ia * 3 + 2];
    const dx = q[0] - worldA[ia * 3], dy = q[1] - worldA[ia * 3 + 1], dz = q[2] - worldA[ia * 3 + 2];
    const d = dx * nx + dy * ny + dz * nz;
    if (Math.abs(d) > 1.5) continue;
    const cx = q[1] * nz - q[2] * ny;
    const cy = q[2] * nx - q[0] * nz;
    const cz = q[0] * ny - q[1] * nx;
    const row = [cx, cy, cz, nx, ny, nz];
    for (let u = 0; u < 6; u++) {
      Jtr[u] -= row[u] * d;
      for (let w = 0; w < 6; w++) JtJ[u * 6 + w] += row[u] * row[w];
    }
    inliers++;
  }
  if (inliers < 8) return null;
  for (let i = 0; i < 6; i++) JtJ[i * 6 + i] *= 1 + 1e-4;
  const delta = solve6(JtJ, Jtr);
  if (!delta) return null;
  const worldDelta = deltaToRigid(delta);
  // 步长保护：单次增量平移不超过 2mm、转角不超过 3°，防止假最近邻带飞
  const stepT = Math.hypot(delta[3], delta[4], delta[5]);
  const stepR = Math.hypot(delta[0], delta[1], delta[2]);
  if (stepT > 2 || stepR > 0.052) return null;
  // 只在能降低当前平均缝隙时接受
  const before = meanPointPlane(worldA, worldAN, grid, bCol, b.transform);
  const trialTf = mulRigid(worldDelta, b.transform);
  const after = meanPointPlane(worldA, worldAN, grid, bCol, trialTf);
  if (!(after < before)) return null;
  // 世界系增量直接以 worldDelta·b.transform 施加（调用方对 b 侧子树统一处理）
  return worldDelta;
}

/** 当前 b 断口点到 a 面的平均点面距离 */
function meanPointPlane(
  worldA: Float32Array,
  worldAN: Float32Array,
  grid: GridHash,
  bCol: { points: Float32Array },
  bTf: RigidTransform,
): number {
  let sum = 0, n = 0;
  for (let j = 0; j < bCol.points.length / 3; j++) {
    const q = applyRigid(bTf, [bCol.points[j * 3], bCol.points[j * 3 + 1], bCol.points[j * 3 + 2]]);
    const ia = grid.nearest(q[0], q[1], q[2], 1.5);
    if (ia < 0) continue;
    const d =
      (q[0] - worldA[ia * 3]) * worldAN[ia * 3] +
      (q[1] - worldA[ia * 3 + 1]) * worldAN[ia * 3 + 1] +
      (q[2] - worldA[ia * 3 + 2]) * worldAN[ia * 3 + 2];
    sum += Math.abs(d);
    n++;
  }
  return n ? sum / n : Infinity;
}

function rotVec(tf: RigidTransform, v: [number, number, number]): [number, number, number] {
  const r = tf.r;
  return [
    r[0] * v[0] + r[1] * v[1] + r[2] * v[2],
    r[3] * v[0] + r[4] * v[1] + r[5] * v[2],
    r[6] * v[0] + r[7] * v[1] + r[8] * v[2],
  ];
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
