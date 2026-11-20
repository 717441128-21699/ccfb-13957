// 数据集生成脚本：产出 public/datasets/pot30.json
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { generatePot } from '../core/generator';
import { serializeDataset } from '../core/serialize';
import type { Dataset } from '../core/types';
import { identityRigid } from '../core/math3d';

const __dirname = dirname(fileURLToPath(import.meta.url));
const outPath = resolve(__dirname, '../../public/datasets/pot30.json');

// 选 3 片作为缺失（补配演示）：腹部一片、肩部一片、近口沿一片
const missingIds = [9, 15, 23];

const ds = generatePot({
  columns: 6,
  rowCuts: [26, 52, 80, 106],
  seed: 20261120,
  noiseMm: 0.08,
  missingIds,
});

const dataset: Dataset = {
  name: '合成陶罐 30 片（含扫描噪声 0.08mm）',
  unit: 'mm',
  profile: ds.profile,
  groundTruthAdjacency: ds.adjacency,
  missingIds,
  shards: ds.shards
    .filter((s) => !missingIds.includes(s.id))
    .map((s) => ({ ...s, transform: identityRigid() })),
};

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, JSON.stringify(serializeDataset(dataset)));

const totalVerts = dataset.shards.reduce((a, s) => a + s.mesh.positions.length / 3, 0);
const totalTris = dataset.shards.reduce((a, s) => a + s.mesh.indices.length / 3, 0);
console.log(`已生成 ${outPath}`);
console.log(`碎片数: ${dataset.shards.length}（另挖除 ${missingIds.length} 片待补配）`);
console.log(`总顶点: ${totalVerts}，总三角面: ${totalTris}`);
console.log(`真值邻接边: ${ds.adjacency.length} 条`);
console.log(`文件体积: ${(JSON.stringify(serializeDataset(dataset)).length / 1024 / 1024).toFixed(2)} MB`);
