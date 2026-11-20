import type { RigidTransform, Shard } from './types';
import { SurfaceLabel } from './types';
import { GridHash, applyRigid } from './math3d';
import { collectFracturePoints } from './fracture';

export interface GapReport {
  edgeCount: number;
  /** 每条邻接边的平均缝隙 (mm) */
  perEdge: Array<{ a: number; b: number; meanGap: number; maxGap: number; samples: number }>;
  /** 全局平均缝隙 */
  meanGap: number;
  /** 满足 0.3mm 要求的边占比 */
  passRatio: number;
  passed: boolean;
}

/**
 * 对已拼合碎片，逐对测量断口贴合缝隙：
 * 把 b 的断口点变到整体坐标，找 a 断口最近点，统计点到平面距离。
 */
export function verifyGaps(shards: Shard[], adjacency: Array<[number, number]>): GapReport {
  const byId = new Map(shards.map((s) => [s.id, s]));
  const perEdge: GapReport['perEdge'] = [];
  let allSum = 0, allN = 0, pass = 0;

  for (const [ia, ib] of adjacency) {
    const a = byId.get(ia), b = byId.get(ib);
    if (!a || !b || !a.fracture || !b.fracture) continue;

    // a 全部边缝子块的断口点变换到整体坐标
    const aCol = collectFracturePoints(a.fracture);
    const ap = aCol.points;
    const an = aCol.normals;
    const worldA = new Float32Array(ap.length);
    const worldAN = new Float32Array(ap.length);
    for (let i = 0; i < ap.length / 3; i++) {
      const w = applyRigid(a.transform, [ap[i * 3], ap[i * 3 + 1], ap[i * 3 + 2]]);
      worldA[i * 3] = w[0]; worldA[i * 3 + 1] = w[1]; worldA[i * 3 + 2] = w[2];
      const nn = rotOnly(a.transform, [an[i * 3], an[i * 3 + 1], an[i * 3 + 2]]);
      worldAN[i * 3] = nn[0]; worldAN[i * 3 + 1] = nn[1]; worldAN[i * 3 + 2] = nn[2];
    }
    const grid = GridHash.build(worldA, 1.5);

    let sum = 0, max = 0, samples = 0;
    const bCol = collectFracturePoints(b.fracture);
    const bp = bCol.points;
    for (let j = 0; j < bp.length / 3; j++) {
      const w = applyRigid(b.transform, [bp[j * 3], bp[j * 3 + 1], bp[j * 3 + 2]]);
      const hit = grid.nearest(w[0], w[1], w[2], 1.5);
      if (hit < 0) continue;
      const dx = w[0] - worldA[hit * 3];
      const dy = w[1] - worldA[hit * 3 + 1];
      const dz = w[2] - worldA[hit * 3 + 2];
      const d = Math.abs(dx * worldAN[hit * 3] + dy * worldAN[hit * 3 + 1] + dz * worldAN[hit * 3 + 2]);
      sum += d;
      if (d > max) max = d;
      samples++;
    }
    if (!samples) continue;
    const meanGap = sum / samples;
    perEdge.push({ a: ia, b: ib, meanGap, maxGap: max, samples });
    allSum += sum;
    allN += samples;
    if (meanGap <= 0.3) pass++;
  }

  const meanGap = allN ? allSum / allN : 0;
  return {
    edgeCount: perEdge.length,
    perEdge,
    meanGap,
    passRatio: perEdge.length ? pass / perEdge.length : 0,
    passed: meanGap <= 0.3,
  };
}

/** 只统计断口标签点的工具（给 UI 高亮用） */
export function fractureWorldPoints(shard: Shard): Float32Array | null {
  if (!shard.fracture) return null;
  const { points } = collectFracturePoints(shard.fracture);
  const out = new Float32Array(points.length);
  for (let i = 0; i < points.length / 3; i++) {
    const w = applyRigid(shard.transform, [points[i * 3], points[i * 3 + 1], points[i * 3 + 2]]);
    out[i * 3] = w[0]; out[i * 3 + 1] = w[1]; out[i * 3 + 2] = w[2];
  }
  return out;
}

function rotOnly(tf: RigidTransform, v: [number, number, number]): [number, number, number] {
  const r = tf.r;
  return [
    r[0] * v[0] + r[1] * v[1] + r[2] * v[2],
    r[3] * v[0] + r[4] * v[1] + r[5] * v[2],
    r[6] * v[0] + r[7] * v[1] + r[8] * v[2],
  ];
}

export function surfaceLabelName(l: SurfaceLabel): string {
  return ['釉面', '胎面', '断口', '底足', '补配'][l] || '未知';
}
