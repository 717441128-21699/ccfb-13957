import type { FracturePatch, Mesh, Vec3 } from './types';
import { SurfaceLabel } from './types';
import {
  buildAdjacency,
  centroidOf,
  computeVertexNormals,
  estimateFractureThickness,
} from './geometry';
import { GridHash, v3 } from './math3d';
import { computeFPFH } from './descriptors';

/**
 * 无标签扫描件的表面分类兜底：
 * - 高曲率/法向突变且连接两个光滑区域的顶点 -> 断口
 * - 光滑区域按到整体质心距离区分外壁（远）/内壁（近）
 * 合成数据自带标签时不调用。
 */
export function classifySurfaces(mesh: Mesh): Uint8Array {
  const n = mesh.positions.length / 3;
  const labels = new Uint8Array(n);
  const normals = computeVertexNormals(mesh);
  const adj = buildAdjacency(mesh);
  const c = centroidOf(mesh.positions);
  const curvature = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    let dotSum = 0;
    for (const j of adj[i]) {
      dotSum +=
        normals[i * 3] * normals[j * 3] +
        normals[i * 3 + 1] * normals[j * 3 + 1] +
        normals[i * 3 + 2] * normals[j * 3 + 2];
    }
    const deg = adj[i].length || 1;
    curvature[i] = 1 - dotSum / deg; // 0 光滑，越大越尖锐
  }
  // Otsu 式阈值：取曲率 60 分位作为断口门槛
  const sorted = Float32Array.from(curvature).sort();
  const threshold = sorted[Math.floor(n * 0.6)] * 1.4;

  for (let i = 0; i < n; i++) {
    if (curvature[i] > threshold && threshold > 1e-4) {
      labels[i] = SurfaceLabel.Fracture;
    } else {
      const d = v3.dist(
        [mesh.positions[i * 3], mesh.positions[i * 3 + 1], mesh.positions[i * 3 + 2]],
        c,
      );
      // 沿法向再走半步：外壁沿法向远离质心
      const along =
        (normals[i * 3] * (mesh.positions[i * 3] - c[0]) +
          normals[i * 3 + 1] * (mesh.positions[i * 3 + 1] - c[1]) +
          normals[i * 3 + 2] * (mesh.positions[i * 3 + 2] - c[2])) /
        (d || 1);
      labels[i] = along > 0 ? SurfaceLabel.Outer : SurfaceLabel.Inner;
    }
  }
  return labels;
}

/** 空间均匀下采样，控制配对计算规模 */
function subsample(points: Float32Array, maxPoints: number): number[] {
  const n = points.length / 3;
  if (n <= maxPoints) return Array.from({ length: n }, (_, i) => i);
  const cell = new Map<number, number>();
  // 包围盒
  let mnx = Infinity, mny = Infinity, mnz = Infinity, mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = points[i * 3], y = points[i * 3 + 1], z = points[i * 3 + 2];
    if (x < mnx) mnx = x; if (y < mny) mny = y; if (z < mnz) mnz = z;
    if (x > mxx) mxx = x; if (y > mxy) mxy = y; if (z > mxz) mxz = z;
  }
  const span = Math.max(mxx - mnx, mxy - mny, mxz - mnz) || 1;
  const gridN = Math.max(8, Math.ceil(Math.cbrt(maxPoints)));
  const inv = gridN / span;
  for (let i = 0; i < n; i++) {
    const gx = Math.floor((points[i * 3] - mnx) * inv);
    const gy = Math.floor((points[i * 3 + 1] - mny) * inv);
    const gz = Math.floor((points[i * 3 + 2] - mnz) * inv);
    const key = ((gx * 73856093) ^ (gy * 19349663) ^ (gz * 83492791)) | 0;
    if (!cell.has(key)) cell.set(key, i);
  }
  return [...cell.values()];
}

export interface AnalyzeOptions {
  maxFracturePoints?: number;
  fpfhRadius?: number;
}

/**
 * 断口连通分割：一片碎片通常含 2~4 条边缝，
 * 按顶点邻接做连通域，得到每条边缝的子断口。
 */
export function splitFractureComponents(mesh: Mesh, fractureIdx: number[]): number[][] {
  const isFrac = new Set(fractureIdx);
  const adj = buildAdjacency(mesh);
  const seen = new Set<number>();
  const comps: number[][] = [];
  for (const start of fractureIdx) {
    if (seen.has(start)) continue;
    const stack = [start];
    const comp: number[] = [];
    seen.add(start);
    while (stack.length) {
      const v = stack.pop()!;
      comp.push(v);
      for (const nb of adj[v]) {
        if (isFrac.has(nb) && !seen.has(nb)) {
          seen.add(nb);
          stack.push(nb);
        }
      }
    }
    if (comp.length >= 8) comps.push(comp);
  }
  comps.sort((a, b) => b.length - a.length);
  return comps;
}

/** 分析单片碎片：定位断口、按边缝拆分、厚度、梯度、FPFH 与釉/胎朝向 */
export function analyzeShard(mesh: Mesh, opts: AnalyzeOptions = {}): FracturePatch | null {
  const maxPts = opts.maxFracturePoints ?? 420;
  const labels = mesh.labels.length ? mesh.labels : classifySurfaces(mesh);
  if (!mesh.labels.length) mesh.labels = labels;

  const vertCount = mesh.positions.length / 3;
  const allFracture: number[] = [];
  for (let i = 0; i < vertCount; i++) {
    if (labels[i] === SurfaceLabel.Fracture) allFracture.push(i);
  }
  if (allFracture.length < 12) return null;

  const normalsAll = computeVertexNormals(mesh);
  const { thickness, thickDir } = estimateFractureThickness(mesh, allFracture, normalsAll);
  const thickMap = new Map<number, number>();
  const tdirMap = new Map<number, [number, number, number]>();
  allFracture.forEach((vi, k) => {
    thickMap.set(vi, thickness[k]);
    tdirMap.set(vi, [thickDir[k * 3], thickDir[k * 3 + 1], thickDir[k * 3 + 2]]);
  });

  // 按边缝拆成连通子块，逐块建 patch
  const components = splitFractureComponents(mesh, allFracture);
  if (!components.length) return null;

  // 汇总所有子块（保留顺序：面积大的在前）
  const subPatches: FracturePatch[] = [];
  for (const comp of components) {
    const sub = buildSubPatch(mesh, comp, normalsAll, thickMap, tdirMap, labels, maxPts, opts.fpfhRadius ?? 2.5);
    if (sub) subPatches.push(sub);
  }
  if (!subPatches.length) return null;

  // 主 patch 取最大子块，同时把全部子块挂在 subPatches 上
  const main = subPatches[0];
  (main as FracturePatch & { subPatches?: FracturePatch[] }).subPatches = subPatches;
  return main;
}

/** 取分析结果中的全部边缝子块（兼容旧调用） */
export function subPatchesOf(f: FracturePatch): FracturePatch[] {
  return (f as FracturePatch & { subPatches?: FracturePatch[] }).subPatches ?? [f];
}

/** 汇总全部边缝子块的点与法向（碎片局部坐标） */
export function collectFracturePoints(f: FracturePatch): { points: Float32Array; normals: Float32Array } {
  const subs = subPatchesOf(f);
  if (subs.length === 1) return { points: f.points, normals: f.normals };
  const pn = subs.reduce((n, s) => n + s.points.length, 0);
  const points = new Float32Array(pn);
  const normals = new Float32Array(pn);
  let off = 0;
  for (const s of subs) {
    points.set(s.points, off);
    normals.set(s.normals, off);
    off += s.points.length;
  }
  return { points, normals };
}

function buildSubPatch(
  mesh: Mesh,
  comp: number[],
  normalsAll: Float32Array,
  thickMap: Map<number, number>,
  tdirMap: Map<number, [number, number, number]>,
  labels: Uint8Array,
  maxPts: number,
  fpfhRadius: number,
): FracturePatch | null {
  // 下采样
  const fullPts = new Float32Array(comp.length * 3);
  for (let k = 0; k < comp.length; k++) {
    const vi = comp[k] * 3;
    fullPts[k * 3] = mesh.positions[vi];
    fullPts[k * 3 + 1] = mesh.positions[vi + 1];
    fullPts[k * 3 + 2] = mesh.positions[vi + 2];
  }
  const keep = subsample(fullPts, maxPts);
  const m = keep.length;
  if (m < 8) return null;
  const points = new Float32Array(m * 3);
  const normals = new Float32Array(m * 3);
  const thick = new Float32Array(m);
  const tdir = new Float32Array(m * 3);
  for (let k = 0; k < m; k++) {
    const vi = comp[keep[k]];
    points[k * 3] = mesh.positions[vi * 3];
    points[k * 3 + 1] = mesh.positions[vi * 3 + 1];
    points[k * 3 + 2] = mesh.positions[vi * 3 + 2];
    normals[k * 3] = normalsAll[vi * 3];
    normals[k * 3 + 1] = normalsAll[vi * 3 + 1];
    normals[k * 3 + 2] = normalsAll[vi * 3 + 2];
    thick[k] = thickMap.get(vi) ?? 6;
    const td = tdirMap.get(vi) ?? [0, 0, 0];
    tdir[k * 3] = td[0]; tdir[k * 3 + 1] = td[1]; tdir[k * 3 + 2] = td[2];
  }
  const desc = computeFPFH(points, normals, fpfhRadius);
  const centroid = centroidOf(points);
  const outerDir = surfaceDir(mesh, labels, centroid, SurfaceLabel.Outer);
  const innerDir = surfaceDir(mesh, labels, centroid, SurfaceLabel.Inner);
  return {
    points, normals, thickness: thick, thickDir: tdir,
    desc, descDim: 33, outerDir, innerDir, centroid,
  };
}

function surfaceDir(mesh: Mesh, labels: Uint8Array, from: Vec3, label: SurfaceLabel): Vec3 {
  const pts: number[] = [];
  const n = mesh.positions.length / 3;
  for (let i = 0; i < n; i++) {
    if (labels[i] === label) pts.push(i);
  }
  if (!pts.length) return [0, 0, 0];
  const packed = new Float32Array(pts.length * 3);
  for (let i = 0; i < pts.length; i++) {
    packed[i * 3] = mesh.positions[pts[i] * 3];
    packed[i * 3 + 1] = mesh.positions[pts[i] * 3 + 1];
    packed[i * 3 + 2] = mesh.positions[pts[i] * 3 + 2];
  }
  const grid = GridHash.build(packed, 3);
  let sx = 0, sy = 0, sz = 0, hits = 0;
  // 对若干断口邻域采样求平均方向，避免单点偏差
  const step = Math.max(1, Math.floor(pts.length / 40));
  for (let i = 0; i < pts.length; i += step) {
    const j = grid.nearest(packed[i * 3], packed[i * 3 + 1], packed[i * 3 + 2], 20);
    if (j < 0) continue;
    sx += packed[j * 3] - from[0];
    sy += packed[j * 3 + 1] - from[1];
    sz += packed[j * 3 + 2] - from[2];
    hits++;
  }
  if (!hits) return [0, 0, 0];
  return v3.norm([sx / hits, sy / hits, sz / hits]);
}
