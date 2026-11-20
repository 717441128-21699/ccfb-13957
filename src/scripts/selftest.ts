// 端到端自检：
// 1) 生成 30 片破罐（含噪声）
// 2) Worker 同款算法在主线程跑断口分析 + 全量配对
// 3) 贪心拼合 -> 验证平均缝隙 ≤ 0.3mm、30 片配对 ≤ 20s
// 4) 真值邻接召回率 / 候选准确率
// 5) 测量 + 补配 + 线图导出冒烟测试
import { generatePot } from '../core/generator';
import { analyzeShard } from '../core/fracture';
import { greedyAssembly, matchAll, pairPreFilter } from '../core/matcher';
import { verifyGaps } from '../core/gap';
import { measureVessel, exportLineSVG, VESSEL_LIBRARY } from '../core/measure';
import { fillMissingBySymmetry, mergeFillMesh } from '../core/fill';
import { applyRigid, identityRigid, mulRigid } from '../core/math3d';
import type { Dataset, Mesh, Shard, SurfaceLabel } from '../core/types';
import { SurfaceLabel as SL } from '../core/types';

function fmt(n: number, d = 2): string {
  return n.toFixed(d);
}

async function main() {
  const tStart = performance.now();
  console.log('=== 陶瓷碎片虚拟拼接自检 ===\n');

  // ---------- 1. 生成数据 ----------
  const missingIds = [9, 15, 23];
  const gen = generatePot({ columns: 6, rowCuts: [26, 52, 80, 106], seed: 20261120, noiseMm: 0.08, missingIds });
  const present = gen.shards.filter((s) => !missingIds.includes(s.id));
  console.log(`[1] 生成破罐：共 ${gen.shards.length} 片，现存 ${present.length} 片，挖除 ${missingIds.length} 片`);
  const verts = present.reduce((a, s) => a + s.mesh.positions.length / 3, 0);
  console.log(`    总顶点 ${verts}，真值邻接边 ${gen.adjacency.length} 条`);

  // ---------- 2. 断口分析 ----------
  const tAna0 = performance.now();
  const shards: Shard[] = present.map((s) => {
    const fracture = analyzeShard(s.mesh, { maxFracturePoints: 380 });
    return { ...s, fracture: fracture ?? undefined };
  });
  const noFrac = shards.filter((s) => !s.fracture);
  console.log(`[2] 断口分析完成，用时 ${fmt(performance.now() - tAna0)} ms，无断口片: ${noFrac.length}`);
  if (noFrac.length) console.log(`    警告: ${noFrac.map((s) => s.id).join(',')}`);

  // ---------- 3. 全量配对 ----------
  const analyzable = shards.filter((s) => s.fracture);
  let preFiltered = 0;
  for (let i = 0; i < analyzable.length; i++) {
    for (let j = i + 1; j < analyzable.length; j++) {
      if (!pairPreFilter(analyzable[i].fracture!, analyzable[j].fracture!)) preFiltered++;
    }
  }
  const tMatch0 = performance.now();
  const { candidates, elapsedMs } = matchAll(
    analyzable.map((s) => ({ id: s.id, fracture: s.fracture! })),
    { iterationsFactor: 1.0, gapThreshold: 1.5, minThicknessConsistency: 0.35 },
  );
  console.log(`[3] 全量配对用时 ${fmt(elapsedMs)} ms（要求 ≤ 20000 ms），粗筛跳过 ${preFiltered} 对，产出候选 ${candidates.length} 条`);

  // ---------- 4. 真值比对 ----------
  const presentSet = new Set(present.map((s) => s.id));
  const gtEdges = new Set(
    gen.adjacency
      .filter(([a, b]) => presentSet.has(a) && presentSet.has(b))
      .map(([a, b]) => (a < b ? `${a}-${b}` : `${b}-${a}`)),
  );
  const candEdges = new Set(candidates.map((c) => (c.a < c.b ? `${c.a}-${c.b}` : `${c.b}-${c.a}`)));
  let hit = 0;
  for (const e of gtEdges) if (candEdges.has(e)) hit++;
  const recall = hit / gtEdges.size;
  const precision = candidates.length ? hit / candidates.length : 0;
  console.log(`[4] 真值邻接 ${gtEdges.size} 条，命中 ${hit} 条，召回率 ${fmt(recall * 100, 1)}%，候选准确率 ${fmt(precision * 100, 1)}%`);

  // ---------- 5. 贪心拼合 + 全局 ICP + 缝隙 ----------
  const centroidMap = new Map(analyzable.map((s) => [s.id, s.centroid]));
  const assembly = greedyAssembly(
    analyzable.map((s) => s.id),
    candidates,
    4,
    centroidMap,
  );
  for (const s of analyzable) {
    s.transform = assembly.transforms.get(s.id) ?? identityRigid();
  }
  const { globalICP } = await import('../core/global-icp');
  const refine = globalICP(analyzable, assembly.accepted, 3);
  // 由接受边直接求连通分量
  const adjMap = new Map<number, number[]>();
  for (const s of analyzable) adjMap.set(s.id, []);
  for (const e of assembly.accepted) {
    adjMap.get(e.a)?.push(e.b);
    adjMap.get(e.b)?.push(e.a);
  }
  const seenC = new Set<number>();
  let compCount = 0;
  const compSizes: number[] = [];
  for (const id of adjMap.keys()) {
    if (seenC.has(id)) continue;
    compCount++;
    let size = 0;
    const stk = [id];
    seenC.add(id);
    while (stk.length) {
      const x = stk.pop()!;
      size++;
      for (const n of adjMap.get(x)!) if (!seenC.has(n)) { seenC.add(n); stk.push(n); }
    }
    compSizes.push(size);
  }
  const adjacencyPresent = gen.adjacency.filter(([a, b]) => presentSet.has(a) && presentSet.has(b));
  const gap = verifyGaps(analyzable, adjacencyPresent);
  console.log(`[5] 拼合接受边 ${assembly.accepted.length} 条，连通分量 ${compCount} 个（大小 ${compSizes.sort((a, b) => b - a).join('/')}），全局 ICP ${refine.iterations} 轮、最大位移 ${refine.maxDelta.toFixed(3)} mm`);
  console.log(`    缝隙验证：平均 ${fmt(gap.meanGap, 3)} mm（要求 ≤ 0.3 mm），合格边占比 ${fmt(gap.passRatio * 100, 1)}%，可测邻接边 ${gap.edgeCount} 条`);
  const worst = [...gap.perEdge].sort((a, b) => b.meanGap - a.meanGap).slice(0, 3);
  for (const w of worst) console.log(`    最差边 #${w.a}-#${w.b}: 平均 ${fmt(w.meanGap, 3)} mm，最大 ${fmt(w.maxGap, 3)} mm`);

  // ---------- 6. 器形测量 ----------
  const merged = mergeShards(analyzable);
  const measure = measureVessel(merged);
  const truth = gen.profile;
  console.log(`[6] 测量结果（器形库匹配: ${measure.bestMatch.type}，库距 ${fmt(measure.bestMatch.distance, 3)}）`);
  console.log(`    口径 ${fmt(measure.mouthDiameter)} mm（真值约 ${fmt(truth.outer[truth.outer.length - 1][0] * 2)}）`);
  console.log(`    腹径 ${fmt(measure.bellyDiameter)} mm（真值约 ${fmt(Math.max(...truth.outer.map((p) => p[0])) * 2)}）`);
  console.log(`    底径 ${fmt(measure.baseDiameter)} mm（真值约 ${fmt(truth.outer[0][0] * 2)}）`);
  console.log(`    高   ${fmt(measure.height)} mm（真值 ${truth.height}）`);
  const svg = exportLineSVG(measure);
  console.log(`    线图 SVG 导出 ${svg.length} 字节；器形库内置 ${VESSEL_LIBRARY.length} 类`);

  // ---------- 7. 对称补配 ----------
  const presentWorld = analyzable.map((s) => ({
    positions: transformedPositions(s),
    labels: s.mesh.labels,
  }));
  const missing = gen.shards.filter((s) => missingIds.includes(s.id));
  const axis = measure.axis;
  const regions = missing.map((m) => {
    const d = [m.centroid[0] - axis.origin[0], m.centroid[1] - axis.origin[1], m.centroid[2] - axis.origin[2]] as const;
    const z = d[0] * axis.dir[0] + d[1] * axis.dir[1] + d[2] * axis.dir[2];
    const rx = d[0] - axis.dir[0] * z, ry = d[1] - axis.dir[1] * z, rz = d[2] - axis.dir[2] * z;
    return { theta: Math.atan2(rz, rx), thetaWidth: (Math.PI * 2) / 6, z0: z - 13, z1: z + 13 };
  });
  const fills = fillMissingBySymmetry(presentWorld, truth, regions);
  let fillMesh: Mesh = merged;
  for (const f of fills) fillMesh = mergeFillMesh(fillMesh, f);
  const fillVerts = fills.reduce((a, f) => a + f.mesh.positions.length / 3, 0);
  console.log(`[7] 对称补配 ${fills.length} 处，补配顶点 ${fillVerts}，标签 Fill=${SL.Fill}`);

  // ---------- 汇总 ----------
  const total = performance.now() - tStart;
  const okTime = elapsedMs <= 20000;
  const okGap = gap.edgeCount >= 10 && gap.meanGap <= 0.3;
  const okRecall = recall >= 0.85;
  console.log(`\n=== 结论 ===`);
  console.log(`配对耗时 ${fmt(elapsedMs)} ms / 20000 ms：${okTime ? '通过' : '不通过'}`);
  console.log(`平均缝隙 ${fmt(gap.meanGap, 3)} mm / 0.3 mm：${okGap ? '通过' : '不通过'}`);
  console.log(`真值召回 ${fmt(recall * 100, 1)}% / 85%：${okRecall ? '通过' : '不通过'}`);
  console.log(`端到端总用时 ${fmt(total)} ms`);
  if (!(okTime && okGap && okRecall)) process.exitCode = 1;
}

function mergeShards(shards: Shard[]): Mesh {
  let positions = new Float32Array(0);
  let labels = new Uint8Array(0);
  let indices = new Uint32Array(0);
  for (const s of shards) {
    const tp = transformedPositions(s);
    const newPos = new Float32Array(positions.length + tp.length);
    newPos.set(positions, 0);
    newPos.set(tp, positions.length);
    const newLab = new Uint8Array(labels.length + s.mesh.labels.length);
    newLab.set(labels, 0);
    newLab.set(s.mesh.labels, labels.length);
    const base = positions.length / 3;
    const newIdx = new Uint32Array(indices.length + s.mesh.indices.length);
    newIdx.set(indices, 0);
    for (let i = 0; i < s.mesh.indices.length; i++) newIdx[indices.length + i] = s.mesh.indices[i] + base;
    positions = newPos;
    labels = newLab;
    indices = newIdx;
  }
  return { positions, labels, indices };
}

function transformedPositions(s: Shard): Float32Array {
  const out = new Float32Array(s.mesh.positions.length);
  for (let i = 0; i < s.mesh.positions.length / 3; i++) {
    const w = applyRigid(s.transform, [s.mesh.positions[i * 3], s.mesh.positions[i * 3 + 1], s.mesh.positions[i * 3 + 2]]);
    out[i * 3] = w[0]; out[i * 3 + 1] = w[1]; out[i * 3 + 2] = w[2];
  }
  return out;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
