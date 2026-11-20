// 数据集序列化：TypedArray <-> 普通 JSON 对象
import type { Dataset, FracturePatch, Mesh, Shard } from './types';
import { identityRigid } from './math3d';

export interface SerializedMesh {
  positions: number[];
  indices: number[];
  labels: number[];
}
export interface SerializedFracture {
  points: number[];
  normals: number[];
  thickness: number[];
  thickDir: number[];
  desc: number[];
  descDim: number;
  outerDir: [number, number, number];
  innerDir: [number, number, number];
  centroid: [number, number, number];
}
export interface SerializedDataset {
  name: string;
  unit: 'mm';
  profile: Dataset['profile'];
  groundTruthAdjacency?: Array<[number, number]>;
  missingIds?: number[];
  shards: Array<{
    id: number;
    code: string;
    area: number;
    centroid: [number, number, number];
    mesh: SerializedMesh;
    fracture?: SerializedFracture;
  }>;
}

export function serializeMesh(mesh: Mesh): SerializedMesh {
  return {
    positions: Array.from(mesh.positions),
    indices: Array.from(mesh.indices),
    labels: Array.from(mesh.labels),
  };
}

export function deserializeMesh(s: SerializedMesh): Mesh {
  return {
    positions: Float32Array.from(s.positions),
    indices: Uint32Array.from(s.indices),
    labels: Uint8Array.from(s.labels),
  };
}

export function serializeFracture(f: FracturePatch): SerializedFracture {
  return {
    points: Array.from(f.points),
    normals: Array.from(f.normals),
    thickness: Array.from(f.thickness),
    thickDir: Array.from(f.thickDir),
    desc: Array.from(f.desc),
    descDim: f.descDim,
    outerDir: f.outerDir,
    innerDir: f.innerDir,
    centroid: f.centroid,
  };
}

export function deserializeFracture(s: SerializedFracture): FracturePatch {
  return {
    points: Float32Array.from(s.points),
    normals: Float32Array.from(s.normals),
    thickness: Float32Array.from(s.thickness),
    thickDir: Float32Array.from(s.thickDir),
    desc: Float32Array.from(s.desc),
    descDim: s.descDim,
    outerDir: s.outerDir,
    innerDir: s.innerDir,
    centroid: s.centroid,
  };
}

export function deserializeDataset(s: SerializedDataset): Dataset {
  const shards: Shard[] = s.shards.map((x) => ({
    id: x.id,
    code: x.code,
    mesh: deserializeMesh(x.mesh),
    transform: identityRigid(),
    area: x.area,
    centroid: x.centroid,
    fracture: x.fracture ? deserializeFracture(x.fracture) : undefined,
  }));
  return {
    name: s.name,
    unit: s.unit,
    profile: s.profile,
    shards,
    groundTruthAdjacency: s.groundTruthAdjacency,
    missingIds: s.missingIds,
  };
}

export function serializeDataset(d: Dataset): SerializedDataset {
  return {
    name: d.name,
    unit: d.unit,
    profile: d.profile,
    groundTruthAdjacency: d.groundTruthAdjacency,
    missingIds: d.missingIds,
    shards: d.shards.map((s) => ({
      id: s.id,
      code: s.code,
      area: s.area,
      centroid: s.centroid,
      mesh: serializeMesh(s.mesh),
      fracture: s.fracture ? serializeFracture(s.fracture) : undefined,
    })),
  };
}
