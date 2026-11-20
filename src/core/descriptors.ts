import { GridHash } from './math3d';

/**
 * 简化版 FPFH（Fast Point Feature Histograms）。
 * 每个点 33 维：α/φ/θ 三个角度各 11 个 bin。
 * 先算 SPFH，再加权一次邻居重加权得到 FPFH。
 */
export function computeFPFH(
  points: Float32Array,
  normals: Float32Array,
  radius: number,
  bins = 11,
): Float32Array {
  const n = points.length / 3;
  const dim = bins * 3;
  const grid = GridHash.build(points, radius);
  const spfh = new Float32Array(n * dim);
  const neigh: number[] = [];

  for (let i = 0; i < n; i++) {
    const px = points[i * 3], py = points[i * 3 + 1], pz = points[i * 3 + 2];
    const ux = normals[i * 3], uy = normals[i * 3 + 1], uz = normals[i * 3 + 2];
    grid.radius(px, py, pz, radius, neigh);
    let weightSum = 0;
    const h = spfh.subarray(i * dim, (i + 1) * dim);
    for (const j of neigh) {
      if (j === i) continue;
      let dx = points[j * 3] - px;
      let dy = points[j * 3 + 1] - py;
      let dz = points[j * 3 + 2] - pz;
      const d = Math.hypot(dx, dy, dz);
      if (d < 1e-6) continue;
      dx /= d; dy /= d; dz /= d;
      // v = u × dline，保证符号确定
      let vx = uy * dz - uz * dy;
      let vy = uz * dx - ux * dz;
      let vz = ux * dy - uy * dx;
      const vl = Math.hypot(vx, vy, vz) || 1;
      vx /= vl; vy /= vl; vz /= vl;
      // w = u × v
      const wx = uy * vz - uz * vy;
      const wy = uz * vx - ux * vz;
      const wz = ux * vy - uy * vx;
      const nx = normals[j * 3], ny = normals[j * 3 + 1], nz = normals[j * 3 + 2];
      const alpha = vx * nx + vy * ny + vz * nz; // [-1,1]
      const phi = ux * dx + uy * dy + uz * dz; // [-1,1]
      const theta = Math.atan2(wx * nx + wy * ny + wz * nz, ux * nx + uy * ny + uz * nz); // [-π,π]
      const ba = Math.min(bins - 1, Math.floor(((alpha + 1) / 2) * bins));
      const bp = Math.min(bins - 1, Math.floor(((phi + 1) / 2) * bins));
      const bt = Math.min(bins - 1, Math.floor(((theta + Math.PI) / (2 * Math.PI)) * bins));
      const w = 1 / d;
      h[ba] += w;
      h[bins + bp] += w;
      h[2 * bins + bt] += w;
      weightSum += w;
    }
    if (weightSum > 0) {
      for (let k = 0; k < dim; k++) h[k] /= weightSum;
    }
  }

  // 加权重加权：FPFH(p) = SPFH(p) + 1/k Σ (1/||p-pk||) SPFH(pk)
  const out = new Float32Array(n * dim);
  for (let i = 0; i < n; i++) {
    const px = points[i * 3], py = points[i * 3 + 1], pz = points[i * 3 + 2];
    grid.radius(px, py, pz, radius, neigh);
    const dst = out.subarray(i * dim, (i + 1) * dim);
    dst.set(spfh.subarray(i * dim, (i + 1) * dim));
    let wsum = 0;
    for (const j of neigh) {
      if (j === i) continue;
      const dx = points[j * 3] - px, dy = points[j * 3 + 1] - py, dz = points[j * 3 + 2] - pz;
      const w = 1 / (Math.hypot(dx, dy, dz) || 1);
      wsum += w;
      const src = spfh.subarray(j * dim, (j + 1) * dim);
      for (let k = 0; k < dim; k++) dst[k] += w * src[k];
    }
    if (wsum > 0) {
      for (let k = 0; k < dim; k++) dst[k] /= 1 + wsum;
    }
  }
  return out;
}

/** 两个描述子的 L1 距离（越小越相似） */
export function descDistance(a: Float32Array, ai: number, b: Float32Array, bj: number, dim: number): number {
  let s = 0;
  const oa = ai * dim, ob = bj * dim;
  for (let k = 0; k < dim; k++) s += Math.abs(a[oa + k] - b[ob + k]);
  return s;
}
