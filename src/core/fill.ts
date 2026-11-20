// 缺口补配：按器形轴对称 + 轮廓线重建缺失碎片
import type { Mesh, PotProfile, RigidTransform, Vec3 } from './types';
import { SurfaceLabel } from './types';
import { applyRigid, invertRigid, mulRigid } from './math3d';
import { pca } from './geometry';

export interface FillPatch {
  mesh: Mesh;
  /** 补配占据的角度区间与高度区间（整体坐标） */
  thetaRange: [number, number];
  zRange: [number, number];
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function profileRadius(profile: Array<[number, number]>, z: number): number {
  if (z <= profile[0][1]) return profile[0][0];
  for (let i = 0; i < profile.length - 1; i++) {
    const [r0, z0] = profile[i];
    const [r1, z1] = profile[i + 1];
    if (z >= z0 && z <= z1) return lerp(r0, r1, (z - z0) / (z1 - z0 || 1));
  }
  return profile[profile.length - 1][0];
}

/**
 * 由现存碎片估计器物中轴（PCA），再把缺失区域按轴对称轮廓补出。
 * missingShardBounds：缺失片在整体坐标下的大致方位（由其邻片断口位置推得）。
 */
export function fillMissingBySymmetry(
  presentShapes: Array<{ positions: Float32Array; labels: Uint8Array }>,
  profile: PotProfile,
  missingRegions: Array<{ theta: number; thetaWidth: number; z0: number; z1: number }>,
): FillPatch[] {
  // 用现存外壁点估计中轴
  const pts: number[] = [];
  for (const s of presentShapes) {
    const n = s.positions.length / 3;
    for (let i = 0; i < n; i++) {
      if (s.labels[i] === SurfaceLabel.Outer) pts.push(i);
    }
  }
  const packed = new Float32Array(pts.length * 3);
  let k = 0;
  for (const s of presentShapes) {
    const n = s.positions.length / 3;
    for (let i = 0; i < n; i++) {
      if (s.labels[i] === SurfaceLabel.Outer) {
        packed[k * 3] = s.positions[i * 3];
        packed[k * 3 + 1] = s.positions[i * 3 + 1];
        packed[k * 3 + 2] = s.positions[i * 3 + 2];
        k++;
      }
    }
  }
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < packed.length / 3; i++) {
    cx += packed[i * 3]; cy += packed[i * 3 + 1]; cz += packed[i * 3 + 2];
  }
  const cnt = packed.length / 3 || 1;
  const { vectors } = pca(packed);
  // 与测量模块一致：用回转对称度（各高度带径向方差最小）选轴
  let axis: Vec3 = [vectors[6], vectors[7], vectors[8]];
  let bestVar = Infinity;
  const oc: Vec3 = [cx / cnt, cy / cnt, cz / cnt];
  for (let v = 0; v < 3; v++) {
    const dir: Vec3 = [vectors[v * 3], vectors[v * 3 + 1], vectors[v * 3 + 2]];
    const zs: number[] = [];
    const rs: number[] = [];
    for (let i = 0; i < packed.length / 3; i++) {
      const d: Vec3 = [packed[i * 3] - oc[0], packed[i * 3 + 1] - oc[1], packed[i * 3 + 2] - oc[2]];
      const z = dot(d, dir);
      zs.push(z);
      rs.push(Math.sqrt(Math.max(0, dot(d, d) - z * z)));
    }
    let mn = Infinity, mx = -Infinity;
    for (const z of zs) { if (z < mn) mn = z; if (z > mx) mx = z; }
    const bands = 16, span = mx - mn || 1;
    const lists: number[][] = Array.from({ length: bands }, () => []);
    for (let i = 0; i < zs.length; i++) {
      lists[Math.min(bands - 1, Math.floor(((zs[i] - mn) / span) * bands))].push(rs[i]);
    }
    let vsum = 0, used = 0;
    for (const list of lists) {
      if (list.length < 3) continue;
      const m = list.reduce((a, b) => a + b, 0) / list.length;
      vsum += list.reduce((a, b) => a + (b - m) ** 2, 0) / list.length;
      used++;
    }
    const score = used ? vsum / used : Infinity;
    if (score < bestVar) { bestVar = score; axis = dir; }
  }
  const center: Vec3 = oc;

  // 构造轴对齐坐标系：eZ = 轴，eX/eY 取垂直平面
  const eZ = norm(axis);
  const helper: Vec3 = Math.abs(eZ[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const eX = norm(cross(helper, eZ));
  const eY = cross(eZ, eX);

  return missingRegions.map((region) => buildFillPatch(region, profile, center, eX, eY, eZ));
}

function buildFillPatch(
  region: { theta: number; thetaWidth: number; z0: number; z1: number },
  profile: PotProfile,
  center: Vec3,
  eX: Vec3,
  eY: Vec3,
  eZ: Vec3,
): FillPatch {
  const positions: number[] = [];
  const labels: number[] = [];
  const indices: number[] = [];
  const nt = 16, nz = Math.max(6, Math.ceil((region.z1 - region.z0) / 3));
  // 中心在轴上的投影高度：世界点 = center + eX·r·cosθ + eY·r·sinθ + eZ·(z-zBase)
  const zBase = dot(center, eZ);

  const add = (r: number, th: number, z: number, label: SurfaceLabel): number => {
    const h = z - zBase;
    const x = center[0] + eX[0] * r * Math.cos(th) + eY[0] * r * Math.sin(th) + eZ[0] * h;
    const y = center[1] + eX[1] * r * Math.cos(th) + eY[1] * r * Math.sin(th) + eZ[1] * h;
    const zz = center[2] + eX[2] * r * Math.cos(th) + eY[2] * r * Math.sin(th) + eZ[2] * h;
    positions.push(x, y, zz);
    labels.push(label);
    return positions.length / 3 - 1;
  };

  const thA = region.theta - region.thetaWidth / 2;
  const thB = region.theta + region.thetaWidth / 2;
  const gOut: number[][] = [];
  const gIn: number[][] = [];
  for (let iz = 0; iz <= nz; iz++) {
    const z = lerp(region.z0, region.z1, iz / nz);
    const rOut = profileRadius(profile.outer, z);
    const rIn = profileRadius(profile.inner, z);
    const rowO: number[] = [], rowI: number[] = [];
    for (let it = 0; it <= nt; it++) {
      const th = lerp(thA, thB, it / nt);
      rowO.push(add(rOut, th, z, SurfaceLabel.Fill));
      rowI.push(add(rIn, th, z, SurfaceLabel.Fill));
    }
    gOut.push(rowO);
    gIn.push(rowI);
  }
  for (let iz = 0; iz < nz; iz++) {
    for (let it = 0; it < nt; it++) {
      const q = [gOut[iz][it], gOut[iz][it + 1], gOut[iz + 1][it + 1], gOut[iz + 1][it]];
      indices.push(q[0], q[1], q[2], q[0], q[2], q[3]);
      const p = [gIn[iz][it], gIn[iz + 1][it], gIn[iz + 1][it + 1], gIn[iz][it + 1]];
      indices.push(p[0], p[1], p[2], p[0], p[2], p[3]);
    }
  }
  // 四周补封边（断口色由渲染端按 Fill 统一换色）
  capEdge(gOut, gIn, 0, indices);
  capEdge(gOut, gIn, nt, indices);
  capRing(gOut[0], gIn[0], indices);
  capRing(gOut[nz], gIn[nz], indices);

  return {
    mesh: {
      positions: Float32Array.from(positions),
      indices: Uint32Array.from(indices),
      labels: Uint8Array.from(labels),
    },
    thetaRange: [thA, thB],
    zRange: [region.z0, region.z1],
  };
}

function capEdge(gOut: number[][], gIn: number[][], it: number, indices: number[]): void {
  for (let iz = 0; iz < gOut.length - 1; iz++) {
    const a = gOut[iz][it], b = gIn[iz][it], c = gIn[iz + 1][it], d = gOut[iz + 1][it];
    indices.push(a, b, c, a, c, d);
  }
}

function capRing(rowO: number[], rowI: number[], indices: number[]): void {
  for (let it = 0; it < rowO.length - 1; it++) {
    indices.push(rowO[it], rowO[it + 1], rowI[it + 1], rowO[it], rowI[it + 1], rowI[it]);
  }
}

function norm(v: Vec3): Vec3 {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** 由缺失片邻片的断口位置推估缺失区域方位 */
export function inferMissingRegions(
  missing: Array<{ id: number; centroid: Vec3 }>,
  presentWorld: Array<{ id: number; transform: RigidTransform; centroid: Vec3 }>,
  axisCenter: Vec3,
  axisDir: Vec3,
): Array<{ theta: number; thetaWidth: number; z0: number; z1: number }> {
  return missing.map((m) => {
    // 用缺失片原始质心（罐坐标）在轴系中的柱坐标定位
    const d: Vec3 = [m.centroid[0] - axisCenter[0], m.centroid[1] - axisCenter[1], m.centroid[2] - axisCenter[2]];
    const z = dot(d, axisDir);
    const radialVec: Vec3 = [d[0] - axisDir[0] * z, d[1] - axisDir[1] * z, d[2] - axisDir[2] * z];
    const theta = Math.atan2(dot(radialVec, perpY(axisDir)), dot(radialVec, perpX(axisDir)));
    return { theta, thetaWidth: (Math.PI * 2) / 6, z0: z - 12, z1: z + 12 };
  });
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function perpX(axis: Vec3): Vec3 {
  const helper: Vec3 = Math.abs(axis[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  return norm(cross(helper, axis));
}
function perpY(axis: Vec3): Vec3 {
  const helper: Vec3 = Math.abs(axis[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  return cross(axis, norm(cross(helper, axis)));
}

/** 把补配网格并入整体网格（换色由 label=Fill 决定） */
export function mergeFillMesh(target: Mesh, patch: FillPatch): Mesh {
  const base = target.positions.length / 3;
  const positions = new Float32Array(target.positions.length + patch.mesh.positions.length);
  positions.set(target.positions, 0);
  positions.set(patch.mesh.positions, target.positions.length);
  const labels = new Uint8Array(target.labels.length + patch.mesh.labels.length);
  labels.set(target.labels, 0);
  labels.set(patch.mesh.labels, target.labels.length);
  const indices = new Uint32Array(target.indices.length + patch.mesh.indices.length);
  indices.set(target.indices, 0);
  for (let i = 0; i < patch.mesh.indices.length; i++) {
    indices[target.indices.length + i] = patch.mesh.indices[i] + base;
  }
  return { positions, labels, indices };
}

export { applyRigid, invertRigid, mulRigid };
