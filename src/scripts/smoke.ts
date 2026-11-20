// UI 全流程冒烟测试：模拟界面对核心模块的调用顺序
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { deserializeDataset, type SerializedDataset } from '../core/serialize';
import { analyzeShard, subPatchesOf } from '../core/fracture';
import { greedyAssembly, matchAll, DEFAULT_OPTIONS } from '../core/matcher';
import { globalICP } from '../core/global-icp';
import { verifyGaps } from '../core/gap';
import { measureVessel, exportLineSVG, exportMeasureJSON } from '../core/measure';
import { fillMissingBySymmetry, mergeFillMesh } from '../core/fill';
import { identityRigid, applyRigid } from '../core/math3d';
import type { Mesh, RigidTransform } from '../core/types';
import { SurfaceLabel } from '../core/types';

const here = dirname(fileURLToPath(import.meta.url));
const json = JSON.parse(readFileSync(resolve(here, '../../public/datasets/pot30.json'), 'utf8')) as SerializedDataset;
const ds = deserializeDataset(json);
console.log(`载入数据集：${ds.name}，${ds.shards.length} 片`);

// 1. 断口分析（Worker 同款函数）
for (const s of ds.shards) s.fracture = analyzeShard(s.mesh) ?? undefined;
const subCounts = ds.shards.map((s) => (s.fracture ? subPatchesOf(s.fracture).length : 0));
console.log(`断口分析：边缝子块数 min/max = ${Math.min(...subCounts)}/${Math.max(...subCounts)}`);

// 2. 配对
const ready = ds.shards.filter((s) => s.fracture);
const { candidates, elapsedMs } = matchAll(ready.map((s) => ({ id: s.id, fracture: s.fracture! })), DEFAULT_OPTIONS);
console.log(`配对：${candidates.length} 候选，${elapsedMs.toFixed(0)} ms`);

// 3. 一键拼合 + ICP
const centroids = new Map(ds.shards.map((s) => [s.id, s.centroid]));
const assembly = greedyAssembly(ds.shards.map((s) => s.id), candidates, 4, centroids);
for (const s of ds.shards) s.transform = assembly.transforms.get(s.id) ?? identityRigid();
globalICP(ready, assembly.accepted, 3);
const gap = verifyGaps(ds.shards, assembly.accepted.map((c) => [c.a, c.b] as [number, number]));
console.log(`拼合：接受 ${assembly.accepted.length} 边，平均缝隙 ${gap.meanGap.toFixed(3)} mm`);

// 4. 补配
const world = ds.shards.map((s) => ({ positions: xform(s.mesh.positions, s.transform), labels: s.mesh.labels }));
const regions = (ds.missingIds ?? []).map((id) => {
  const col = id % 6, row = Math.floor(id / 6);
  const cuts = [26, 52, 80, 106];
  return {
    theta: ((col + 0.5) / 6) * Math.PI * 2,
    thetaWidth: (Math.PI * 2) / 6,
    z0: row === 0 ? 0 : cuts[row - 1],
    z1: row === 4 ? ds.profile.height : cuts[row],
  };
});
const fills = fillMissingBySymmetry(world, ds.profile, regions);
let merged = mergeAll(ds);
for (const f of fills) merged = mergeFillMesh(merged, f);
const fillVerts = merged.positions.length / 3;
const hasFill = merged.labels.some((l) => l === SurfaceLabel.Fill);
console.log(`补配：${fills.length} 处，含 Fill 标签=${hasFill}，合并后顶点 ${fillVerts}`);

// 5. 测量与导出
const m = measureVessel(merged);
console.log(`测量：口径 ${m.mouthDiameter.toFixed(1)} 腹径 ${m.bellyDiameter.toFixed(1)} 底径 ${m.baseDiameter.toFixed(1)} 高 ${m.height.toFixed(1)}，器形=${m.bestMatch.type}`);
const svg = exportLineSVG(m);
const mj = exportMeasureJSON(m);
console.log(`导出：SVG ${svg.length} 字节，JSON ${mj.length} 字节，SVG 头=${svg.slice(0, 5)}`);

function xform(p: Float32Array, tf: RigidTransform): Float32Array {
  const out = new Float32Array(p.length);
  for (let i = 0; i < p.length / 3; i++) {
    const w = applyRigid(tf, [p[i * 3], p[i * 3 + 1], p[i * 3 + 2]]);
    out[i * 3] = w[0]; out[i * 3 + 1] = w[1]; out[i * 3 + 2] = w[2];
  }
  return out;
}
function mergeAll(d: typeof ds): Mesh {
  let pos = new Float32Array(0), lab = new Uint8Array(0), idx = new Uint32Array(0);
  for (const s of d.shards) {
    const tp = xform(s.mesh.positions, s.transform);
    const np = new Float32Array(pos.length + tp.length); np.set(pos); np.set(tp, pos.length);
    const nl = new Uint8Array(lab.length + s.mesh.labels.length); nl.set(lab); nl.set(s.mesh.labels, lab.length);
    const base = pos.length / 3;
    const ni = new Uint32Array(idx.length + s.mesh.indices.length); ni.set(idx);
    for (let i = 0; i < s.mesh.indices.length; i++) ni[idx.length + i] = s.mesh.indices[i] + base;
    pos = np; lab = nl; idx = ni;
  }
  return { positions: pos, labels: lab, indices: idx };
}
console.log('\n冒烟测试通过：UI 调用链全部可运行');
