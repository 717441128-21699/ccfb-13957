import type { Vec3, RigidTransform } from './types';

export const v3 = {
  add: (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  scale: (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a: Vec3, b: Vec3): Vec3 => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ],
  len: (a: Vec3): number => Math.hypot(a[0], a[1], a[2]),
  norm: (a: Vec3): Vec3 => {
    const l = Math.hypot(a[0], a[1], a[2]) || 1;
    return [a[0] / l, a[1] / l, a[2] / l];
  },
  dist: (a: Vec3, b: Vec3): number => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]),
  lerp: (a: Vec3, b: Vec3, t: number): Vec3 => [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ],
  zero: (): Vec3 => [0, 0, 0],
};

/** 4×4 列主序矩阵（WebGL 友好） */
export type Mat4 = Float32Array;

export function mat4Identity(): Mat4 {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

export function mat4Mul(a: Mat4, b: Mat4): Mat4 {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return o;
}

export function mat4Translation(t: Vec3): Mat4 {
  const m = mat4Identity();
  m[12] = t[0];
  m[13] = t[1];
  m[14] = t[2];
  return m;
}

export function mat4Perspective(fovy: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovy / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) / (near - far);
  m[11] = -1;
  m[14] = (2 * far * near) / (near - far);
  return m;
}

export function mat4LookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
  const z = v3.norm(v3.sub(eye, target));
  const x = v3.norm(v3.cross(up, z));
  const y = v3.cross(z, x);
  const m = mat4Identity();
  m[0] = x[0]; m[4] = x[1]; m[8] = x[2];
  m[1] = y[0]; m[5] = y[1]; m[9] = y[2];
  m[2] = z[0]; m[6] = z[1]; m[10] = z[2];
  m[12] = -v3.dot(x, eye);
  m[13] = -v3.dot(y, eye);
  m[14] = -v3.dot(z, eye);
  return m;
}

/** 由 3×3 旋转 + 平移构造列主序 4×4 */
export function rigidToMat4(tf: RigidTransform): Mat4 {
  const m = mat4Identity();
  const r = tf.r;
  // r 为行主序：m[列*4+行]
  m[0] = r[0]; m[4] = r[1]; m[8] = r[2];
  m[1] = r[3]; m[5] = r[4]; m[9] = r[5];
  m[2] = r[6]; m[6] = r[7]; m[10] = r[8];
  m[12] = tf.t[0]; m[13] = tf.t[1]; m[14] = tf.t[2];
  return m;
}

export function applyRigid(tf: RigidTransform, p: Vec3): Vec3 {
  const r = tf.r;
  const t = tf.t;
  return [
    r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0],
    r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1],
    r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2],
  ];
}

export function invertRigid(tf: RigidTransform): RigidTransform {
  const r = tf.r;
  // 正交矩阵逆 = 转置
  const rt = [r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]];
  const t: Vec3 = [
    -(rt[0] * tf.t[0] + rt[1] * tf.t[1] + rt[2] * tf.t[2]),
    -(rt[3] * tf.t[0] + rt[4] * tf.t[1] + rt[5] * tf.t[2]),
    -(rt[6] * tf.t[0] + rt[7] * tf.t[1] + rt[8] * tf.t[2]),
  ];
  return { r: rt, t };
}

export function identityRigid(): RigidTransform {
  return { r: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0] };
}

export function mulRigid(a: RigidTransform, b: RigidTransform): RigidTransform {
  const ar = a.r, br = b.r;
  const r = new Array<number>(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      r[i * 3 + j] =
        ar[i * 3] * br[j] + ar[i * 3 + 1] * br[3 + j] + ar[i * 3 + 2] * br[6 + j];
    }
  }
  return { r, t: applyRigid(a, b.t) };
}

/** 由三组对应点求刚体变换（把 b 的点变到 a） */
export function rigidFromCorrespondences(
  pa: [Vec3, Vec3, Vec3],
  pb: [Vec3, Vec3, Vec3],
): RigidTransform {
  const ca = centroid3(pa);
  const cb = centroid3(pb);
  let h00 = 0, h01 = 0, h02 = 0, h10 = 0, h11 = 0, h12 = 0, h20 = 0, h21 = 0, h22 = 0;
  for (let i = 0; i < 3; i++) {
    const qa = v3.sub(pa[i], ca);
    const qb = v3.sub(pb[i], cb);
    h00 += qb[0] * qa[0]; h01 += qb[0] * qa[1]; h02 += qb[0] * qa[2];
    h10 += qb[1] * qa[0]; h11 += qb[1] * qa[1]; h12 += qb[1] * qa[2];
    h20 += qb[2] * qa[0]; h21 += qb[2] * qa[1]; h22 += qb[2] * qa[2];
  }
  const R = rotationFromH([[h00, h01, h02], [h10, h11, h12], [h20, h21, h22]]);
  // t = ca - R cb
  const Rcb: Vec3 = [
    R[0] * cb[0] + R[1] * cb[1] + R[2] * cb[2],
    R[3] * cb[0] + R[4] * cb[1] + R[5] * cb[2],
    R[6] * cb[0] + R[7] * cb[1] + R[8] * cb[2],
  ];
  return { r: R, t: v3.sub(ca, Rcb) };
}

function centroid3(p: [Vec3, Vec3, Vec3]): Vec3 {
  return [
    (p[0][0] + p[1][0] + p[2][0]) / 3,
    (p[0][1] + p[1][1] + p[2][1]) / 3,
    (p[0][2] + p[1][2] + p[2][2]) / 3,
  ];
}

/** 由协方差类矩阵 H = sum qb*qa^T 求旋转 R（使 R*qb 对齐 qa），行主序返回 */
function rotationFromH(h: number[][]): number[] {
  // 四元数法（Horn）：构造 4×4 对称矩阵 N
  const Sxx = h[0][0], Sxy = h[0][1], Sxz = h[0][2];
  const Syx = h[1][0], Syy = h[1][1], Syz = h[1][2];
  const Szx = h[2][0], Szy = h[2][1], Szz = h[2][2];
  const trace = Sxx + Syy + Szz;
  const N = [
    trace, Syz - Szy, Szx - Sxz, Sxy - Syx,
    Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz,
    Szx - Sxz, Sxy + Syx, Syy - Sxx - Szz, Syz + Szy,
    Sxy - Syx, Szx + Sxz, Syz + Szy, Szz - Sxx - Syy,
  ];
  // 求最大特征值对应的特征向量（Jacobi 迭代，4×4 足够快）
  const q = dominantEigenvector(N);
  const [w, x, y, z] = q;
  // 四元数 -> 旋转矩阵（行主序）
  return [
    w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z,
  ];
}

function dominantEigenvector(m: number[]): number[] {
  // 幂迭代
  let v = [1, 0, 0, 0];
  for (let iter = 0; iter < 64; iter++) {
    const o = [0, 0, 0, 0];
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) o[r] += m[r * 4 + c] * v[c];
    }
    const l = Math.hypot(o[0], o[1], o[2], o[3]) || 1;
    v = o.map((x) => x / l);
  }
  return v;
}

/** 最近邻搜索（均匀网格哈希） */
export class GridHash {
  private cell = new Map<number, number[]>();
  private pts: Float32Array;
  private inv: number;

  private constructor(points: Float32Array, cellSize: number) {
    this.pts = points;
    this.inv = 1 / cellSize;
  }

  static build(points: Float32Array, cellSize: number): GridHash {
    const g = new GridHash(points, cellSize);
    const n = points.length / 3;
    for (let i = 0; i < n; i++) {
      const key = g.keyOf(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]);
      let arr = g.cell.get(key);
      if (!arr) g.cell.set(key, (arr = []));
      arr.push(i);
    }
    return g;
  }

  private keyOf(x: number, y: number, z: number): number {
    const cx = Math.floor(x * this.inv);
    const cy = Math.floor(y * this.inv);
    const cz = Math.floor(z * this.inv);
    return ((cx * 73856093) ^ (cy * 19349663) ^ (cz * 83492791)) | 0;
  }

  /** 返回最近点索引，无则 -1 */
  nearest(x: number, y: number, z: number, maxDist: number): number {
    let best = -1;
    let bestD2 = maxDist * maxDist;
    const cx = Math.floor(x * this.inv);
    const cy = Math.floor(y * this.inv);
    const cz = Math.floor(z * this.inv);
    const r = Math.max(1, Math.ceil(maxDist * this.inv));
    for (let ix = cx - r; ix <= cx + r; ix++) {
      for (let iy = cy - r; iy <= cy + r; iy++) {
        for (let iz = cz - r; iz <= cz + r; iz++) {
          const key = ((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)) | 0;
          const arr = this.cell.get(key);
          if (!arr) continue;
          for (let k = 0; k < arr.length; k++) {
            const j = arr[k];
            const px = this.pts[j * 3], py = this.pts[j * 3 + 1], pz = this.pts[j * 3 + 2];
            const d2 = (px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2;
            if (d2 < bestD2) {
              bestD2 = d2;
              best = j;
            }
          }
        }
      }
    }
    return best;
  }

  /** 半径内全部点（去重），调用方提供输出数组 */
  radius(x: number, y: number, z: number, radius: number, out: number[]): number[] {
    out.length = 0;
    const cx = Math.floor(x * this.inv);
    const cy = Math.floor(y * this.inv);
    const cz = Math.floor(z * this.inv);
    const r = Math.max(1, Math.ceil(radius * this.inv));
    const r2 = radius * radius;
    for (let ix = cx - r; ix <= cx + r; ix++) {
      for (let iy = cy - r; iy <= cy + r; iy++) {
        for (let iz = cz - r; iz <= cz + r; iz++) {
          const key = ((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)) | 0;
          const arr = this.cell.get(key);
          if (!arr) continue;
          for (let k = 0; k < arr.length; k++) {
            const j = arr[k];
            const px = this.pts[j * 3], py = this.pts[j * 3 + 1], pz = this.pts[j * 3 + 2];
            const d2 = (px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2;
            if (d2 <= r2) out.push(j);
          }
        }
      }
    }
    return out;
  }
}
