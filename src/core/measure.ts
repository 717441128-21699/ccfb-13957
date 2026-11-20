import type { MeasureResult, Mesh, PotProfile, Vec3 } from './types';
import { SurfaceLabel } from './types';
import { pca } from './geometry';
import { v3 } from './math3d';

/** 内置器形库（轮廓归一化到高度 1），用于同类比对 */
export interface VesselType {
  type: string;
  dynasty: string;
  /** 归一化轮廓 (r/h, z/h)，底到口 */
  outline: Array<[number, number]>;
}

export const VESSEL_LIBRARY: VesselType[] = [
  {
    type: '罐',
    dynasty: '通用',
    outline: [
      [0.40, 0], [0.43, 0.05], [0.50, 0.15], [0.58, 0.32], [0.62, 0.48],
      [0.60, 0.62], [0.52, 0.78], [0.40, 0.9], [0.36, 0.97], [0.37, 1.0],
    ],
  },
  {
    type: '瓶',
    dynasty: '通用',
    outline: [
      [0.30, 0], [0.34, 0.08], [0.46, 0.3], [0.5, 0.45], [0.42, 0.62],
      [0.26, 0.78], [0.16, 0.9], [0.15, 0.97], [0.18, 1.0],
    ],
  },
  {
    type: '钵',
    dynasty: '通用',
    outline: [
      [0.34, 0], [0.42, 0.1], [0.55, 0.35], [0.66, 0.65], [0.72, 0.9], [0.74, 1.0],
    ],
  },
  {
    type: '碗',
    dynasty: '通用',
    outline: [
      [0.26, 0], [0.36, 0.15], [0.52, 0.45], [0.68, 0.8], [0.76, 1.0],
    ],
  },
];

/**
 * 从拼合后的整体网格测量器形。
 * 轴：外壁点 PCA 的最小特征方向；投影后按 z 分带量半径。
 */
export function measureVessel(mesh: Mesh, profile?: PotProfile): MeasureResult {
  const outerPts: number[] = [];
  const n = mesh.positions.length / 3;
  for (let i = 0; i < n; i++) {
    if (mesh.labels[i] === SurfaceLabel.Outer || mesh.labels[i] === SurfaceLabel.Fill) {
      outerPts.push(i);
    }
  }
  const packed = new Float32Array(outerPts.length * 3);
  for (let i = 0; i < outerPts.length; i++) {
    packed[i * 3] = mesh.positions[outerPts[i] * 3];
    packed[i * 3 + 1] = mesh.positions[outerPts[i] * 3 + 1];
    packed[i * 3 + 2] = mesh.positions[outerPts[i] * 3 + 2];
  }
  let sx = 0, sy = 0, sz = 0;
  for (let i = 0; i < packed.length / 3; i++) {
    sx += packed[i * 3]; sy += packed[i * 3 + 1]; sz += packed[i * 3 + 2];
  }
  const cnt = packed.length / 3 || 1;
  const origin: Vec3 = [sx / cnt, sy / cnt, sz / cnt];

  const { vectors } = pca(packed);
  // 用回转对称度选轴：真轴在各高度带内的径向距离方差最小
  const symmetryScore: Array<{ dir: Vec3; score: number }> = [];
  for (let v = 0; v < 3; v++) {
    const dir: Vec3 = [vectors[v * 3], vectors[v * 3 + 1], vectors[v * 3 + 2]];
    const projZ: number[] = [];
    const projR: number[] = [];
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < packed.length / 3; i++) {
      const d: Vec3 = [packed[i * 3] - origin[0], packed[i * 3 + 1] - origin[1], packed[i * 3 + 2] - origin[2]];
      const z = v3.dot(d, dir);
      projZ.push(z);
      projR.push(Math.sqrt(Math.max(0, v3.dot(d, d) - z * z)));
      if (z < mn) mn = z;
      if (z > mx) mx = z;
    }
    const bands = 20;
    const span = mx - mn || 1;
    const bandLists: number[][] = Array.from({ length: bands }, () => []);
    for (let i = 0; i < projZ.length; i++) {
      const b = Math.min(bands - 1, Math.floor(((projZ[i] - mn) / span) * bands));
      bandLists[b].push(projR[i]);
    }
    let varSum = 0, used = 0;
    for (const list of bandLists) {
      if (list.length < 3) continue;
      const m = list.reduce((a, b) => a + b, 0) / list.length;
      varSum += list.reduce((a, b) => a + (b - m) ** 2, 0) / list.length;
      used++;
    }
    symmetryScore.push({ dir, score: used ? varSum / used : Infinity });
  }
  symmetryScore.sort((a, b) => a.score - b.score);
  let axis: Vec3 = refineAxis(packed, origin, symmetryScore[0].dir);
  // 让轴指向口沿（z 增大方向）：用平均投影符号校正
  let ps = 0;
  for (let i = 0; i < outerPts.length; i++) {
    ps += v3.dot([packed[i * 3] - origin[0], packed[i * 3 + 1] - origin[1], packed[i * 3 + 2] - origin[2]], axis);
  }
  if (ps < 0) axis = [-axis[0], -axis[1], -axis[2]];

  // 投影到轴坐标
  const proj = new Float32Array(outerPts.length);
  const radial = new Float32Array(outerPts.length);
  let zMin = Infinity, zMax = -Infinity;
  for (let i = 0; i < outerPts.length; i++) {
    const d: Vec3 = [packed[i * 3] - origin[0], packed[i * 3 + 1] - origin[1], packed[i * 3 + 2] - origin[2]];
    const z = v3.dot(d, axis);
    proj[i] = z;
    radial[i] = Math.sqrt(Math.max(0, v3.dot(d, d) - z * z));
    if (z < zMin) zMin = z;
    if (z > zMax) zMax = z;
  }
  const height = zMax - zMin;

  // 分带统计半径：用 90 分位数抗噪声，避免单点离群把口径/底径撑大
  const bands = 40;
  const bandLists: number[][] = Array.from({ length: bands }, () => []);
  for (let i = 0; i < outerPts.length; i++) {
    const b = Math.max(0, Math.min(bands - 1, Math.floor(((proj[i] - zMin) / (height || 1)) * bands)));
    if (Number.isFinite(b)) bandLists[b].push(radial[i]);
  }
  const bandR = bandLists.map((list) => percentile(list, 0.75));
  const outline: Array<[number, number]> = bandR.map((r, i) => [r, zMin + ((i + 0.5) / bands) * height]);

  // 口沿/底足：取投影 z 最靠端部的点环（最高 1.5% / 最低 1.5%），用中位数抗离群
  const endRing = (top: boolean): number => {
    const zThresh = top ? zMax - (zMax - zMin) * 0.015 : zMin + (zMax - zMin) * 0.015;
    const vals: number[] = [];
    for (let i = 0; i < outerPts.length; i++) {
      if (top ? proj[i] >= zThresh : proj[i] <= zThresh) vals.push(radial[i]);
    }
    return vals.length ? percentile(vals, 0.5) : percentile(bandR.filter((r) => r > 0), top ? 1 : 0);
  };
  const mouthDiameter = 2 * endRing(true);
  const baseDiameter = 2 * endRing(false);
  const bellyDiameter = 2 * Math.max(...bandR.filter((r) => r > 0));

  // 与器形库比对：归一化轮廓 L2 距离
  const normalized: Array<[number, number]> = outline.map(([r, z]) => [
    r / (height || 1),
    (z - zMin) / (height || 1),
  ]);
  let bestMatch = { type: '未知', distance: Infinity };
  for (const v of VESSEL_LIBRARY) {
    const d = outlineDistance(normalized, v.outline);
    if (d < bestMatch.distance) bestMatch = { type: v.type, distance: d };
  }

  return {
    mouthDiameter,
    bellyDiameter,
    baseDiameter,
    height,
    axis: { origin, dir: axis },
    outline,
    bestMatch,
  };
}

/** 归一化轮廓间距离：重采样到相同 z 栅格后比较 r */
function outlineDistance(a: Array<[number, number]>, b: Array<[number, number]>): number {
  const N = 24;
  let sum = 0;
  for (let i = 0; i < N; i++) {
    const z = i / (N - 1);
    const ra = resampleRadius(a, z);
    const rb = resampleRadius(b, z);
    sum += (ra - rb) ** 2;
  }
  return Math.sqrt(sum / N);
}

function resampleRadius(outline: Array<[number, number]>, z: number): number {
  if (z <= outline[0][1]) return outline[0][0];
  for (let i = 0; i < outline.length - 1; i++) {
    const [r0, z0] = outline[i];
    const [r1, z1] = outline[i + 1];
    if (z >= z0 && z <= z1) return r0 + ((r1 - r0) * (z - z0)) / (z1 - z0 || 1);
  }
  return outline[outline.length - 1][0];
}

/** 导出考古线图：SVG 半剖轮廓 + 测量标注 */
export function exportLineSVG(m: MeasureResult): string {
  const pad = 60;
  const w = 520, h = 620;
  const scale = Math.min((w - 2 * pad) / (m.bellyDiameter + 40), (h - 2 * pad) / m.height);
  const cx = w / 2;
  const toX = (r: number) => cx - r * scale;
  const toY = (z: number) => h - pad - (z - m.outline[0][1]) * scale;

  const path = m.outline
    .map(([r, z], i) => `${i === 0 ? 'M' : 'L'}${toX(r).toFixed(1)},${toY(z).toFixed(1)}`)
    .join(' ');
  const mirror = m.outline
    .map(([r, z], i) => `${i === 0 ? 'M' : 'L'}${(2 * cx - toX(r)).toFixed(1)},${toY(z).toFixed(1)}`)
    .join(' ');

  const dims = `
    <text x="${pad}" y="28" font-size="18" font-family="serif">器形：${m.bestMatch.type}（库距 ${m.bestMatch.distance.toFixed(3)}）</text>
    <text x="${pad}" y="50" font-size="14" font-family="serif">口径 ${m.mouthDiameter.toFixed(1)} mm　腹径 ${m.bellyDiameter.toFixed(1)} mm</text>
    <text x="${pad}" y="70" font-size="14" font-family="serif">底径 ${m.baseDiameter.toFixed(1)} mm　高 ${m.height.toFixed(1)} mm</text>`;

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
  <style>text{fill:#222} .ol{fill:none;stroke:#111;stroke-width:1.6} .dim{stroke:#888;stroke-width:0.8;stroke-dasharray:4 3}</style>
  <rect width="${w}" height="${h}" fill="#fff"/>
  ${dims}
  <path class="ol" d="${path}"/>
  <path class="ol" d="${mirror}"/>
  <line class="dim" x1="${cx}" y1="${pad + 20}" x2="${cx}" y2="${h - pad}"/>
</svg>`;
}

/** 导出测量数据 JSON（供报告归档） */
export function exportMeasureJSON(m: MeasureResult): string {
  return JSON.stringify(
    {
      mouthDiameter: round(m.mouthDiameter),
      bellyDiameter: round(m.bellyDiameter),
      baseDiameter: round(m.baseDiameter),
      height: round(m.height),
      bestMatch: m.bestMatch,
      outline: m.outline.map(([r, z]) => [round(r), round(z)]),
    },
    null,
    2,
  );
}

function round(x: number): number {
  return Math.round(x * 100) / 100;
}

/** 在初始轴附近做小角度网格搜索，最小化各高度带径向方差（精修倾斜） */
function refineAxis(packed: Float32Array, origin: Vec3, initial: Vec3): Vec3 {
  const scoreOf = (dir: Vec3): number => {
    const bands = 20;
    const lists: number[][] = Array.from({ length: bands }, () => []);
    const zs: number[] = [];
    for (let i = 0; i < packed.length / 3; i++) {
      const d: Vec3 = [packed[i * 3] - origin[0], packed[i * 3 + 1] - origin[1], packed[i * 3 + 2] - origin[2]];
      zs.push(v3.dot(d, dir));
    }
    let mn = Infinity, mx = -Infinity;
    for (const z of zs) {
      if (z < mn) mn = z;
      if (z > mx) mx = z;
    }
    const span = mx - mn || 1;
    for (let i = 0; i < zs.length; i++) {
      const d: Vec3 = [packed[i * 3] - origin[0], packed[i * 3 + 1] - origin[1], packed[i * 3 + 2] - origin[2]];
      const z = zs[i];
      const r = Math.sqrt(Math.max(0, v3.dot(d, d) - z * z));
      const b = Math.min(bands - 1, Math.floor(((z - mn) / span) * bands));
      lists[b].push(r);
    }
    let sum = 0, used = 0;
    for (const list of lists) {
      if (list.length < 3) continue;
      const m = list.reduce((a, b) => a + b, 0) / list.length;
      sum += list.reduce((a, b) => a + (b - m) ** 2, 0) / list.length;
      used++;
    }
    return used ? sum / used : Infinity;
  };

  let best = initial;
  let bestScore = scoreOf(best);
  // 构造垂直基
  const helper: Vec3 = Math.abs(initial[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const e1 = v3.norm(v3.cross(helper, initial));
  const e2 = v3.cross(initial, e1);
  for (const ang of [0.12, 0.06, 0.03]) {
    for (const [a1, a2] of [[ang, 0], [-ang, 0], [0, ang], [0, -ang], [ang, ang], [-ang, -ang]]) {
      const cand = v3.norm([
        initial[0] + a1 * e1[0] + a2 * e2[0],
        initial[1] + a1 * e1[1] + a2 * e2[1],
        initial[2] + a1 * e1[2] + a2 * e2[2],
      ]);
      const s = scoreOf(cand);
      if (s < bestScore) {
        bestScore = s;
        best = cand;
      }
    }
  }
  return best;
}

function percentile(x: number[], q: number): number {
  if (!x.length) return 0;
  const s = Float32Array.from(x).sort();
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
}

function mean(x: number[]): number {
  if (!x.length) return 0;
  return x.reduce((a, b) => a + b, 0) / x.length;
}
