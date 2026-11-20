// 共享类型定义 —— 主线程 / Worker / Node 脚本三端通用

export type Vec3 = [number, number, number];

/** 顶点所属表面类型 */
export const enum SurfaceLabel {
  Outer = 0, // 釉面（外壁）
  Inner = 1, // 胎面（内壁）
  Fracture = 2, // 断口
  Base = 3, // 底足/底平面
  Fill = 4, // 补配部分
}

export interface Mesh {
  /** 顶点坐标，长度 = 顶点数 × 3 */
  positions: Float32Array;
  /** 三角形索引，长度 = 三角面数 × 3 */
  indices: Uint32Array;
  /** 每顶点表面标签，长度 = 顶点数 */
  labels: Uint8Array;
}

export interface FracturePatch {
  /** 断口顶点（世界坐标） */
  points: Float32Array;
  /** 断口法向（朝外） */
  normals: Float32Array;
  /** 该点处胎壁厚度 (mm) */
  thickness: Float32Array;
  /** 厚度梯度方向（在断口局部平面内，指向增厚方向） */
  thickDir: Float32Array;
  /** FPFH 特征，长度 = 点数 × descDim */
  desc: Float32Array;
  descDim: number;
  /** 断口质心指向釉面（外壁）的平均方向（碎片局部坐标） */
  outerDir: Vec3;
  /** 断口质心指向胎面（内壁）的平均方向 */
  innerDir: Vec3;
  /** 断口质心（碎片局部坐标） */
  centroid: Vec3;
}

export interface Shard {
  id: number;
  /** 出土编号，如 T01-H07-012 */
  code: string;
  mesh: Mesh;
  /** 当前位姿（把碎片坐标变到器物整体坐标） */
  transform: RigidTransform;
  /** 配对前分析结果（Worker 回填） */
  fracture?: FracturePatch;
  /** 面积 (mm²) */
  area: number;
  /** 质心（碎片局部坐标） */
  centroid: Vec3;
}

export interface RigidTransform {
  /** 旋转 3×3，行主序 */
  r: number[];
  t: Vec3;
}

export interface MatchCandidate {
  a: number;
  b: number;
  /** 综合得分 0~1，越大越好 */
  score: number;
  /** 平均缝隙 (mm)，ICP 后点到平面距离均值 */
  gap: number;
  /** 把 b 变到 a 坐标系的刚体变换 */
  transformBtoA: RigidTransform;
  /** 重叠断口点数 */
  overlap: number;
  /** 厚度梯度一致性 [0,1] */
  thicknessConsistency: number;
}

export interface Dataset {
  name: string;
  unit: 'mm';
  profile: PotProfile;
  shards: Shard[];
  /** 真值邻接（仅合成数据，用于自检） */
  groundTruthAdjacency?: Array<[number, number]>;
  /** 被挖掉的碎片编号（用于补配演示） */
  missingIds?: number[];
}

export interface PotProfile {
  height: number;
  /** 外轮廓 (r, z) 采样点，自底足到口沿 */
  outer: Array<[number, number]>;
  /** 内轮廓 (r, z) 采样点 */
  inner: Array<[number, number]>;
}

export interface MeasureResult {
  mouthDiameter: number;
  bellyDiameter: number;
  baseDiameter: number;
  height: number;
  axis: { origin: Vec3; dir: Vec3 };
  /** 外轮廓线 (r, z) */
  outline: Array<[number, number]>;
  /** 与器形库最接近的类型 */
  bestMatch: { type: string; distance: number };
}

// ---------- Worker 通信协议 ----------

export type WorkerRequest =
  | { kind: 'analyze'; shards: Array<{ id: number; mesh: Mesh }> }
  | {
      kind: 'matchAll';
      shards: Array<{ id: number; fracture: FracturePatch }>;
      options: MatchOptions;
    }
  | { kind: 'matchPair'; a: FracturePatch; b: FracturePatch; options: MatchOptions };

export interface MatchOptions {
  /** RANSAC 采样迭代次数倍数 */
  iterationsFactor: number;
  /** 缝隙合格阈值 (mm) */
  gapThreshold: number;
  /** 厚度梯度方向一致性最低分 */
  minThicknessConsistency: number;
}

export type WorkerResponse =
  | { kind: 'analyze:done'; fractures: Map<number, FracturePatch> | Array<[number, FracturePatch]> }
  | { kind: 'matchAll:done'; candidates: MatchCandidate[]; elapsedMs: number }
  | { kind: 'matchPair:done'; candidate: MatchCandidate | null }
  | { kind: 'progress'; done: number; total: number; current: string }
  | { kind: 'error'; message: string };
