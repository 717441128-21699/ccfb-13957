import type { Mesh, Vec3 } from './types';
import { SurfaceLabel } from './types';
import { GridHash } from './math3d';

/** 计算每顶点法向（面积加权） */
export function computeVertexNormals(mesh: Mesh): Float32Array {
  const { positions, indices } = mesh;
  const n = positions.length / 3;
  const normals = new Float32Array(n * 3);
  for (let i = 0; i < indices.length; i += 3) {
    const ia = indices[i] * 3, ib = indices[i + 1] * 3, ic = indices[i + 2] * 3;
    const ux = positions[ib] - positions[ia], uy = positions[ib + 1] - positions[ia + 1], uz = positions[ib + 2] - positions[ia + 2];
    const vx = positions[ic] - positions[ia], vy = positions[ic + 1] - positions[ia + 1], vz = positions[ic + 2] - positions[ia + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    normals[ia] += nx; normals[ia + 1] += ny; normals[ia + 2] += nz;
    normals[ib] += nx; normals[ib + 1] += ny; normals[ib + 2] += nz;
    normals[ic] += nx; normals[ic + 1] += ny; normals[ic + 2] += nz;
  }
  for (let i = 0; i < n; i++) {
    const x = normals[i * 3], y = normals[i * 3 + 1], z = normals[i * 3 + 2];
    const l = Math.hypot(x, y, z) || 1;
    normals[i * 3] = x / l;
    normals[i * 3 + 1] = y / l;
    normals[i * 3 + 2] = z / l;
  }
  return normals;
}

/** 顶点邻接表 */
export function buildAdjacency(mesh: Mesh): number[][] {
  const n = mesh.positions.length / 3;
  const adj: number[][] = Array.from({ length: n }, () => []);
  const { indices } = mesh;
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i], b = indices[i + 1], c = indices[i + 2];
    adj[a].push(b, c);
    adj[b].push(a, c);
    adj[c].push(a, b);
  }
  for (const list of adj) {
    list.sort((x, y) => x - y);
    let w = 0;
    for (let i = 0; i < list.length; i++) {
      if (i === 0 || list[i] !== list[i - 1]) list[w++] = list[i];
    }
    list.length = w;
  }
  return adj;
}

/** 三角形面积总和（按标签可选） */
export function meshArea(mesh: Mesh, label?: SurfaceLabel): number {
  const p = mesh.positions;
  let area = 0;
  for (let i = 0; i < mesh.indices.length; i += 3) {
    if (label !== undefined) {
      const a = mesh.indices[i];
      if (mesh.labels[a] !== label) continue;
    }
    const ia = mesh.indices[i] * 3, ib = mesh.indices[i + 1] * 3, ic = mesh.indices[i + 2] * 3;
    const ux = p[ib] - p[ia], uy = p[ib + 1] - p[ia + 1], uz = p[ib + 2] - p[ia + 2];
    const vx = p[ic] - p[ia], vy = p[ic + 1] - p[ia + 1], vz = p[ic + 2] - p[ia + 2];
    area += 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
  }
  return area;
}

export function centroidOf(positions: Float32Array, indices?: Uint32Array): Vec3 {
  let sx = 0, sy = 0, sz = 0;
  if (indices) {
    for (let i = 0; i < indices.length; i++) {
      const j = indices[i] * 3;
      sx += positions[j]; sy += positions[j + 1]; sz += positions[j + 2];
    }
    const k = indices.length || 1;
    return [sx / k, sy / k, sz / k];
  }
  const n = positions.length / 3;
  for (let i = 0; i < n; i++) {
    sx += positions[i * 3]; sy += positions[i * 3 + 1]; sz += positions[i * 3 + 2];
  }
  return [sx / n, sy / n, sz / n];
}

/**
 * PCA：返回特征值（升序）与对应特征向量（行主序 3×3，最后一行是主方向）
 */
export function pca(points: Float32Array): { values: [number, number, number]; vectors: number[] } {
  const c = centroidOf(points);
  let sxx = 0, syy = 0, szz = 0, sxy = 0, sxz = 0, syz = 0;
  const n = points.length / 3;
  for (let i = 0; i < n; i++) {
    const x = points[i * 3] - c[0];
    const y = points[i * 3 + 1] - c[1];
    const z = points[i * 3 + 2] - c[2];
    sxx += x * x; syy += y * y; szz += z * z;
    sxy += x * y; sxz += x * z; syz += y * z;
  }
  const inv = 1 / n;
  return eigSym3(
    sxx * inv, sxy * inv, sxz * inv,
    syy * inv, syz * inv, szz * inv,
  );
}

/** 3×3 实对称矩阵特征分解，特征值升序 */
function eigSym3(a: number, b: number, c: number, d: number, e: number, f: number):
  { values: [number, number, number]; vectors: number[] } {
  // 经典 Jacobi 旋转
  const m = [
    [a, b, c],
    [b, d, e],
    [c, e, f],
  ];
  const v = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let iter = 0; iter < 50; iter++) {
    let off = 0;
    for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) off += m[i][j] * m[i][j];
    if (off < 1e-14) break;
    for (let p = 0; p < 3; p++) {
      for (let q = p + 1; q < 3; q++) {
        if (Math.abs(m[p][q]) < 1e-15) continue;
        const tau = (m[q][q] - m[p][p]) / (2 * m[p][q]);
        const t = Math.sign(tau || 1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const cs = 1 / Math.sqrt(1 + t * t);
        const sn = t * cs;
        for (let k = 0; k < 3; k++) {
          const mkp = m[k][p], mkq = m[k][q];
          m[k][p] = cs * mkp - sn * mkq;
          m[k][q] = sn * mkp + cs * mkq;
        }
        for (let k = 0; k < 3; k++) {
          const mpk = m[p][k], mqk = m[q][k];
          m[p][k] = cs * mpk - sn * mqk;
          m[q][k] = sn * mpk + cs * mqk;
        }
        for (let k = 0; k < 3; k++) {
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = cs * vkp - sn * vkq;
          v[k][q] = sn * vkp + cs * vkq;
        }
      }
    }
  }
  const idx = [0, 1, 2].sort((i, j) => m[i][i] - m[j][j]);
  const values: [number, number, number] = [m[idx[0]][idx[0]], m[idx[1]][idx[1]], m[idx[2]][idx[2]]];
  // 每行一个特征向量（升序对应特征值）
  const vecs = [
    v[0][idx[0]], v[1][idx[0]], v[2][idx[0]],
    v[0][idx[1]], v[1][idx[1]], v[2][idx[1]],
    v[0][idx[2]], v[1][idx[2]], v[2][idx[2]],
  ];
  return { values, vectors: vecs };
}

/**
 * 胎壁厚度估计（碎片级，不依赖邻片）：
 * 断口点位于壁厚内部，分别找本片外壁、内壁的最近点，
 * dOut + dIn 即该处壁厚；增厚方向取内外壁最近点连线。
 */
export function estimateFractureThickness(
  mesh: Mesh,
  fractureIdx: number[],
  normals: Float32Array,
): { thickness: Float32Array; thickDir: Float32Array } {
  const n = fractureIdx.length;
  const thickness = new Float32Array(n);
  const thickDir = new Float32Array(n * 3);

  // 收集本片外壁/内壁点
  const outer: number[] = [];
  const inner: number[] = [];
  const vertCount = mesh.positions.length / 3;
  for (let i = 0; i < vertCount; i++) {
    if (mesh.labels[i] === SurfaceLabel.Outer) outer.push(i);
    else if (mesh.labels[i] === SurfaceLabel.Inner) inner.push(i);
  }
  const outerPts = packPoints(mesh.positions, outer);
  const innerPts = packPoints(mesh.positions, inner);
  const cellSize = 1.5;
  const go = outerPts.length ? GridHash.build(outerPts, cellSize) : null;
  const gi = innerPts.length ? GridHash.build(innerPts, cellSize) : null;
  const maxRay = 14;

  // 同时记录外壁方向，供厚度梯度兜底
  const outerVec = new Float32Array(n * 3);

  for (let k = 0; k < n; k++) {
    const vi = fractureIdx[k];
    const x = mesh.positions[vi * 3];
    const y = mesh.positions[vi * 3 + 1];
    const z = mesh.positions[vi * 3 + 2];
    let dOut = maxRay, dIn = maxRay;
    let ox = 0, oy = 0, oz = 0;
    if (go) {
      const j = go.nearest(x, y, z, maxRay);
      if (j >= 0) {
        dOut = dist3(mesh.positions, vi, outerPts, j);
        ox = outerPts[j * 3] - x; oy = outerPts[j * 3 + 1] - y; oz = outerPts[j * 3 + 2] - z;
      }
    }
    if (gi) {
      const j = gi.nearest(x, y, z, maxRay);
      if (j >= 0) dIn = dist3(mesh.positions, vi, innerPts, j);
    }
    // 断口横贯壁厚：最近壁面距离近似局部半壁厚，整体壁厚取 2×min；
    // 断口中部点两壁距离接近，求和也成立
    const nearest = Math.min(dOut, dIn);
    thickness[k] = Math.min(maxRay, nearest * 2 + Math.abs(dOut - dIn) * 0.25);
    const ol = Math.hypot(ox, oy, oz) || 1;
    outerVec[k * 3] = ox / ol; outerVec[k * 3 + 1] = oy / ol; outerVec[k * 3 + 2] = oz / ol;
  }

  // 厚度梯度：在断口邻域内差分，再投影到断口切平面
  const adj = buildAdjacency(mesh);
  const pos = mesh.positions;
  const indexOfFracture = new Map<number, number>();
  for (let k = 0; k < n; k++) indexOfFracture.set(fractureIdx[k], k);
  for (let k = 0; k < n; k++) {
    const vi = fractureIdx[k];
    let ax = 0, ay = 0, az = 0;
    let cnt = 0;
    for (const nb of adj[vi]) {
      const k2 = indexOfFracture.get(nb);
      if (k2 === undefined) continue;
      const dx = pos[nb * 3] - pos[vi * 3];
      const dy = pos[nb * 3 + 1] - pos[vi * 3 + 1];
      const dz = pos[nb * 3 + 2] - pos[vi * 3 + 2];
      const dl = Math.hypot(dx, dy, dz) || 1;
      const dt = thickness[k2] - thickness[k];
      ax += (dt * dx) / (dl * dl);
      ay += (dt * dy) / (dl * dl);
      az += (dt * dz) / (dl * dl);
      cnt++;
    }
    // 投影到切平面（去掉法向分量）
    const nx = normals[vi * 3], ny = normals[vi * 3 + 1], nz = normals[vi * 3 + 2];
    const dot = ax * nx + ay * ny + az * nz;
    ax -= dot * nx; ay -= dot * ny; az -= dot * nz;
    const l = Math.hypot(ax, ay, az);
    if (cnt > 0 && l > 1e-6) {
      thickDir[k * 3] = ax / l;
      thickDir[k * 3 + 1] = ay / l;
      thickDir[k * 3 + 2] = az / l;
    } else {
      // 梯度不显著时沿外壁方向在切平面内的投影
      const ox = outerVec[k * 3], oy = outerVec[k * 3 + 1], oz = outerVec[k * 3 + 2];
      const od = ox * nx + oy * ny + oz * nz;
      let px = ox - od * nx, py = oy - od * ny, pz = oz - od * nz;
      const pl = Math.hypot(px, py, pz) || 1;
      thickDir[k * 3] = px / pl;
      thickDir[k * 3 + 1] = py / pl;
      thickDir[k * 3 + 2] = pz / pl;
    }
  }
  return { thickness, thickDir };
}

function packPoints(positions: Float32Array, idx: number[]): Float32Array {
  const out = new Float32Array(idx.length * 3);
  for (let i = 0; i < idx.length; i++) {
    out[i * 3] = positions[idx[i] * 3];
    out[i * 3 + 1] = positions[idx[i] * 3 + 1];
    out[i * 3 + 2] = positions[idx[i] * 3 + 2];
  }
  return out;
}

function dist3(a: Float32Array, ia: number, b: Float32Array, j: number): number {
  const i3 = ia * 3;
  return Math.hypot(a[i3] - b[j * 3], a[i3 + 1] - b[j * 3 + 1], a[i3 + 2] - b[j * 3 + 2]);
}
