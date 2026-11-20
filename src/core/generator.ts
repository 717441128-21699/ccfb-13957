// 合成扫描数据生成器：构造一个破成 N 片的陶瓷罐
// 结构：回转体器形 + 水平断口环 + 垂直锯齿断口 + 每顶点扫描噪声
import type { Mesh, PotProfile, RigidTransform, Shard, Vec3 } from './types';
import { SurfaceLabel } from './types';
import { identityRigid } from './math3d';

/** 可复现随机数（mulberry32） */
export class Rng {
  private s: number;
  constructor(seed: number) {
    this.s = seed >>> 0;
  }
  next(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  range(a: number, b: number): number {
    return a + (b - a) * this.next();
  }
}

/** 罐器形：外轮廓 (r,z)，自底足到口沿 */
export function defaultProfile(): PotProfile {
  const outer: Array<[number, number]> = [
    [52, 0], [55, 4], [62, 12], [72, 28], [80, 48],
    [82, 62], [78, 78], [66, 96], [52, 112], [46, 124], [48, 130],
  ];
  const wall = (z: number): number => {
    // 胎壁由厚到薄：底足 7mm -> 口沿 3.2mm
    return 7.0 - (3.8 * z) / 130;
  };
  const inner = outer.map(([r, z]) => [Math.max(8, r - wall(z)), z] as [number, number]);
  return { height: 130, outer, inner };
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function profileRadius(profile: Array<[number, number]>, z: number): number {
  if (z <= profile[0][1]) return profile[0][0];
  for (let i = 0; i < profile.length - 1; i++) {
    const [r0, z0] = profile[i];
    const [r1, z1] = profile[i + 1];
    if (z >= z0 && z <= z1) {
      return lerp(r0, r1, (z - z0) / (z1 - z0 || 1));
    }
  }
  return profile[profile.length - 1][0];
}

export interface GenerateOptions {
  columns?: number; // 垂直切缝条数
  rowCuts?: number[]; // 水平断口环高度
  seed?: number;
  noiseMm?: number; // 扫描噪声幅值
  missingIds?: number[]; // 挖掉的碎片（补配演示）
  thetaSteps?: number; // 每片角度方向网格
  zStep?: number; // 高度方向网格尺寸 (mm)
}

interface CutFns {
  vertical: Array<(z: number) => number>; // 每条垂直缝的角度偏移
  horizontal: Array<(th: number) => number>; // 每条水平缝的高度偏移
  rng: Rng;
}

function makeCuts(columns: number, rowCuts: number[], rng: Rng): CutFns {
  const vertical = [];
  for (let k = 0; k < columns; k++) {
    const base = (k / columns) * Math.PI * 2;
    const p1 = rng.range(0, Math.PI * 2), p2 = rng.range(0, Math.PI * 2);
    const a1 = rng.range(0.012, 0.03), a2 = rng.range(0.006, 0.016);
    vertical.push((z: number) => base + a1 * Math.sin(z * 0.11 + p1) + a2 * Math.sin(z * 0.27 + p2));
  }
  const horizontal = rowCuts.map((base) => {
    const p1 = rng.range(0, Math.PI * 2), p2 = rng.range(0, Math.PI * 2);
    const a1 = rng.range(1.6, 3.0), a2 = rng.range(0.7, 1.4);
    return (th: number) => base + a1 * Math.sin(3 * th + p1) + a2 * Math.sin(7 * th + p2);
  });
  return { vertical, horizontal, rng };
}

export interface GeneratedDataset {
  shards: Shard[];
  profile: PotProfile;
  adjacency: Array<[number, number]>;
  missingIds: number[];
  bounds: Array<{ col: number; row: number; th0: number; th1: number; z0: number; z1: number }>;
  cuts: CutFns;
}

export function generatePot(opts: GenerateOptions = {}): GeneratedDataset {
  const columns = opts.columns ?? 6;
  const rowCuts = opts.rowCuts ?? [26, 52, 80, 106];
  const rng = new Rng(opts.seed ?? 20261120);
  const noise = opts.noiseMm ?? 0.08;
  const profile = defaultProfile();
  const cuts = makeCuts(columns, rowCuts, rng);

  const rows = rowCuts.length + 1;
  const zLevels = [0, ...rowCuts, profile.height];
  const bounds: GeneratedDataset['bounds'] = [];
  let id = 0;
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < columns; col++) {
      // 估算该片的角度/高度范围（用缝函数在区间内取极值）
      let thMin = Infinity, thMax = -Infinity;
      const zLo = zLevels[row], zHi = zLevels[row + 1];
      for (let z = zLo; z <= zHi + 0.01; z += 2) {
        const t0 = wrapAngle(cuts.vertical[col](z));
        const t1 = wrapAngle(cuts.vertical[(col + 1) % columns](z));
        thMin = Math.min(thMin, t0, t1);
        thMax = Math.max(thMax, t0, t1);
      }
      let hMin = Infinity, hMax = -Infinity;
      const thBase = (col + 0.5) / columns * Math.PI * 2;
      for (let dth = -0.6; dth <= 0.6; dth += 0.15) {
        const th = wrapAngle(thBase + dth);
        if (row > 0) hMin = Math.min(hMin, cuts.horizontal[row - 1](th));
        if (row < rows - 1) hMax = Math.max(hMax, cuts.horizontal[row](th));
      }
      bounds.push({
        col, row,
        th0: thMin - 0.05,
        th1: thMax + 0.05,
        z0: row === 0 ? 0 : hMin - 1,
        z1: row === rows - 1 ? profile.height : hMax + 1,
      });
      id++;
    }
  }

  const shards: Shard[] = [];
  for (let sid = 0; sid < bounds.length; sid++) {
    const mesh = buildShardMesh(sid, bounds[sid], profile, cuts, columns, rows, zLevels, rng, noise, opts);
    if (!mesh) continue;
    const centroid = avgCentroid(mesh.positions);
    shards.push({
      id: sid,
      code: `T01-H07-${String(sid + 1).padStart(3, '0')}`,
      mesh,
      transform: identityRigid(),
      area: surfaceArea(mesh),
      centroid,
    });
  }

  const adjacency = buildAdjacencyGT(bounds);
  return {
    shards,
    profile,
    adjacency,
    missingIds: opts.missingIds ?? [],
    bounds,
    cuts,
  };
}

function wrapAngle(a: number): number {
  const t = Math.PI * 2;
  return ((a % t) + t) % t;
}

/** 判断 (theta,z) 是否属于某片 */
function insideShard(
  sid: number,
  b: GeneratedDataset['bounds'][number],
  cuts: CutFns,
  columns: number,
  rows: number,
  theta: number,
  z: number,
): boolean {
  const th = wrapAngle(theta);
  // 角度：位于两条垂直缝之间（同一柱跨 0 角时需展开）
  const aLeft = cuts.vertical[b.col](z);
  const aRight = cuts.vertical[(b.col + 1) % columns](z);
  if (!angleBetween(th, aLeft, aRight)) return false;
  // 高度：位于两条水平缝之间
  if (b.row > 0 && z < cuts.horizontal[b.row - 1](th)) return false;
  if (b.row < rows - 1 && z > cuts.horizontal[b.row](th)) return false;
  void sid;
  return true;
}

function angleBetween(th: number, left: number, right: number): boolean {
  const t = Math.PI * 2;
  let l = wrapAngle(left), r = wrapAngle(right);
  if (r <= l) r += t;
  let x = th;
  if (x < l) x += t;
  return x >= l && x <= r;
}

function buildShardMesh(
  sid: number,
  b: GeneratedDataset['bounds'][number],
  profile: PotProfile,
  cuts: CutFns,
  columns: number,
  rows: number,
  zLevels: number[],
  rng: Rng,
  noise: number,
  opts: GenerateOptions,
): Mesh | null {
  const positions: number[] = [];
  const labels: number[] = [];
  const indices: number[] = [];
  const nTheta = opts.thetaSteps ?? 14;
  const zStep = opts.zStep ?? 3.2;

  const addV = (x: number, y: number, z: number, label: SurfaceLabel): number => {
    // 扫描噪声：各方向独立小扰动
    positions.push(
      x + rng.range(-noise, noise),
      y + rng.range(-noise, noise),
      z + rng.range(-noise, noise),
    );
    labels.push(label);
    return positions.length / 3 - 1;
  };

  // ---------- 外壁 / 内壁参数网格 ----------
  const th0 = Math.max(0, b.th0), th1 = Math.min(Math.PI * 2, b.th1);
  const z0 = Math.max(0, b.z0), z1 = Math.min(profile.height, b.z1);
  const nt = nTheta;
  const nz = Math.max(4, Math.ceil((z1 - z0) / zStep));

  const grid = (isOuter: boolean): number[][] => {
    const g: number[][] = [];
    for (let iz = 0; iz <= nz; iz++) {
      const z = lerp(z0, z1, iz / nz);
      const row: number[] = [];
      for (let it = 0; it <= nt; it++) {
        const th = lerp(th0, th1, it / nt);
        if (!insideShard(sid, b, cuts, columns, rows, th, z)) {
          row.push(-1);
          continue;
        }
        const rOut = profileRadius(profile.outer, z);
        const rIn = profileRadius(profile.inner, z);
        const r = isOuter ? rOut : rIn;
        row.push(addV(r * Math.cos(th), r * Math.sin(th), z, isOuter ? SurfaceLabel.Outer : SurfaceLabel.Inner));
      }
      g.push(row);
    }
    return g;
  };

  const gOut = grid(true);
  const gIn = grid(false);
  for (let iz = 0; iz < nz; iz++) {
    for (let it = 0; it < nt; it++) {
      const q = [gOut[iz][it], gOut[iz][it + 1], gOut[iz + 1][it + 1], gOut[iz + 1][it]];
      if (q.every((v) => v >= 0)) indices.push(q[0], q[1], q[2], q[0], q[2], q[3]);
      const p = [gIn[iz][it], gIn[iz + 1][it], gIn[iz + 1][it + 1], gIn[iz][it + 1]];
      if (p.every((v) => v >= 0)) indices.push(p[0], p[1], p[2], p[0], p[2], p[3]);
    }
  }

  // ---------- 垂直断口条带（左右两条角缝） ----------
  for (const side of [b.col, (b.col + 1) % columns]) {
    addVerticalFracture(side, b, profile, cuts, rows, zLevels, rng, noise, addV, indices);
  }
  // ---------- 水平断口条带（上下两条环缝） ----------
  if (b.row > 0) addHorizontalFracture(b.row - 1, b, profile, cuts, rng, noise, addV, indices, true);
  if (b.row < rows - 1) addHorizontalFracture(b.row, b, profile, cuts, rng, noise, addV, indices, false);

  // ---------- 底足平面（最底排碎片） ----------
  if (b.row === 0) {
    addBaseCap(b, profile, cuts, columns, rows, sid, addV, indices);
  }
  // ---------- 口沿环面（最顶排碎片） ----------
  if (b.row === rows - 1) {
    addRimCap(b, profile, cuts, columns, rows, sid, addV, indices);
  }

  if (positions.length < 9) return null;
  return {
    positions: Float32Array.from(positions),
    indices: Uint32Array.from(indices),
    labels: Uint8Array.from(labels),
  };
}

type AddV = (x: number, y: number, z: number, label: SurfaceLabel) => number;

function addVerticalFracture(
  side: number,
  b: GeneratedDataset['bounds'][number],
  profile: PotProfile,
  cuts: CutFns,
  rows: number,
  zLevels: number[],
  rng: Rng,
  noise: number,
  addV: AddV,
  indices: number[],
): void {
  const zLo = zLevels[b.row];
  const zHi = zLevels[b.row + 1];
  const segs = 14;
  const across = 4; // 壁厚方向层数
  const ring: number[][] = [];
  for (let i = 0; i <= segs; i++) {
    const z = lerp(zLo, zHi, i / segs);
    const th = cuts.vertical[side](z);
    const rOut = profileRadius(profile.outer, z);
    const rIn = profileRadius(profile.inner, z);
    const rowV: number[] = [];
    for (let k = 0; k <= across; k++) {
      const t = k / across;
      // 断口高低起伏：壁厚方向加正弦起伏 + 噪声
      const jag = 0.35 * Math.sin(th * 5 + z * 0.2 + k) + rng.range(-noise, noise);
      const r = lerp(rOut, rIn, t) + jag;
      rowV.push(addV(r * Math.cos(th), r * Math.sin(th), z + rng.range(-noise, noise), SurfaceLabel.Fracture));
    }
    ring.push(rowV);
  }
  for (let i = 0; i < segs; i++) {
    for (let k = 0; k < across; k++) {
      const a = ring[i][k], bb = ring[i][k + 1], c = ring[i + 1][k + 1], d = ring[i + 1][k];
      indices.push(a, bb, c, a, c, d);
    }
  }
  void rows;
}

function addHorizontalFracture(
  cutIndex: number,
  b: GeneratedDataset['bounds'][number],
  profile: PotProfile,
  cuts: CutFns,
  rng: Rng,
  noise: number,
  addV: AddV,
  indices: number[],
  _isLower: boolean,
): void {
  const th0 = Math.max(0, b.th0), th1 = Math.min(Math.PI * 2, b.th1);
  const segs = 14;
  const across = 4;
  const ring: number[][] = [];
  for (let i = 0; i <= segs; i++) {
    const th = lerp(th0, th1, i / segs);
    const z = cuts.horizontal[cutIndex](th);
    const rOut = profileRadius(profile.outer, z);
    const rIn = profileRadius(profile.inner, z);
    const rowV: number[] = [];
    for (let k = 0; k <= across; k++) {
      const t = k / across;
      const jag = 0.35 * Math.sin(th * 6 + k * 1.7) + rng.range(-noise, noise);
      const r = lerp(rOut, rIn, t) + jag;
      rowV.push(addV(r * Math.cos(th), r * Math.sin(th), z + rng.range(-noise, noise), SurfaceLabel.Fracture));
    }
    ring.push(rowV);
  }
  for (let i = 0; i < segs; i++) {
    for (let k = 0; k < across; k++) {
      const a = ring[i][k], bb = ring[i][k + 1], c = ring[i + 1][k + 1], d = ring[i + 1][k];
      indices.push(a, bb, c, a, c, d);
    }
  }
}

function addBaseCap(
  b: GeneratedDataset['bounds'][number],
  profile: PotProfile,
  cuts: CutFns,
  columns: number,
  rows: number,
  sid: number,
  addV: AddV,
  indices: number[],
): void {
  const segs = 14;
  const th0 = Math.max(0, b.th0), th1 = Math.min(Math.PI * 2, b.th1);
  const rOut = profileRadius(profile.outer, 0);
  const rIn = profileRadius(profile.inner, 0);
  const ring: number[][] = [];
  for (let i = 0; i <= segs; i++) {
    const th = lerp(th0, th1, i / segs);
    if (!insideShard(sid, b, cuts, columns, rows, th, 0.5)) continue;
    const o = addV(rOut * Math.cos(th), rOut * Math.sin(th), 0, SurfaceLabel.Base);
    const inn = addV(rIn * Math.cos(th), rIn * Math.sin(th), 0, SurfaceLabel.Base);
    ring.push([o, inn]);
  }
  for (let i = 0; i < ring.length - 1; i++) {
    indices.push(ring[i][0], ring[i + 1][0], ring[i + 1][1], ring[i][0], ring[i + 1][1], ring[i][1]);
  }
}

function addRimCap(
  b: GeneratedDataset['bounds'][number],
  profile: PotProfile,
  cuts: CutFns,
  columns: number,
  rows: number,
  sid: number,
  addV: AddV,
  indices: number[],
): void {
  const segs = 14;
  const th0 = Math.max(0, b.th0), th1 = Math.min(Math.PI * 2, b.th1);
  const z = profile.height;
  const rOut = profileRadius(profile.outer, z);
  const rIn = profileRadius(profile.inner, z);
  const ring: number[][] = [];
  for (let i = 0; i <= segs; i++) {
    const th = lerp(th0, th1, i / segs);
    if (!insideShard(sid, b, cuts, columns, rows, th, z - 0.5)) continue;
    const o = addV(rOut * Math.cos(th), rOut * Math.sin(th), z, SurfaceLabel.Outer);
    const inn = addV(rIn * Math.cos(th), rIn * Math.sin(th), z, SurfaceLabel.Inner);
    ring.push([o, inn]);
  }
  for (let i = 0; i < ring.length - 1; i++) {
    indices.push(ring[i][0], ring[i][1], ring[i + 1][1], ring[i][0], ring[i + 1][1], ring[i + 1][0]);
  }
}

function avgCentroid(p: Float32Array): Vec3 {
  let x = 0, y = 0, z = 0;
  const n = p.length / 3;
  for (let i = 0; i < n; i++) {
    x += p[i * 3]; y += p[i * 3 + 1]; z += p[i * 3 + 2];
  }
  return [x / n, y / n, z / n];
}

function surfaceArea(mesh: Mesh): number {
  const p = mesh.positions;
  let area = 0;
  for (let i = 0; i < mesh.indices.length; i += 3) {
    const ia = mesh.indices[i] * 3, ib = mesh.indices[i + 1] * 3, ic = mesh.indices[i + 2] * 3;
    const ux = p[ib] - p[ia], uy = p[ib + 1] - p[ia + 1], uz = p[ib + 2] - p[ia + 2];
    const vx = p[ic] - p[ia], vy = p[ic + 1] - p[ia + 1], vz = p[ic + 2] - p[ia + 2];
    area += 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
  }
  return area;
}

/** 真值邻接：共享垂直缝或水平缝的两片 */
function buildAdjacencyGT(bounds: GeneratedDataset['bounds']): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  for (let i = 0; i < bounds.length; i++) {
    for (let j = i + 1; j < bounds.length; j++) {
      const a = bounds[i], c = bounds[j];
      const zOverlap = Math.min(a.z1, c.z1) - Math.max(a.z0, c.z0);
      const thOverlap = Math.min(a.th1, c.th1) - Math.max(a.th0, c.th0);
      // 同排相邻柱
      if (a.row === c.row && zOverlap > 2 && thOverlap > -0.2) {
        const diff = Math.abs(a.col - c.col);
        if (diff === 1 || diff === bounds.filter((x) => x.row === a.row).length - 1) {
          out.push([i, j]);
          continue;
        }
      }
      // 同柱相邻排
      if (a.col === c.col && Math.abs(a.row - c.row) === 1 && thOverlap > -0.2) {
        out.push([i, j]);
      }
    }
  }
  return out;
}

export function datasetToJSON(ds: GeneratedDataset, name: string) {
  return {
    name,
    unit: 'mm',
    profile: ds.profile,
    groundTruthAdjacency: ds.adjacency,
    missingIds: ds.missingIds,
    shards: ds.shards.map((s) => ({
      id: s.id,
      code: s.code,
      area: s.area,
      centroid: s.centroid,
      mesh: {
        positions: Array.from(s.mesh.positions),
        indices: Array.from(s.mesh.indices),
        labels: Array.from(s.mesh.labels),
      },
    })),
  };
}

export function identity(): RigidTransform {
  return identityRigid();
}
