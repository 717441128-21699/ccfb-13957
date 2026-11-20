// 极简 OBJ 网格导入（v / f，三角或四边面），标签默认走自动分类
import type { Mesh } from './types';
import { SurfaceLabel } from './types';

export function parseOBJ(text: string): Mesh {
  const positions: number[] = [];
  const indices: number[] = [];
  const faceVerts: number[][] = [];

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const parts = line.split(/\s+/);
    if (parts[0] === 'v') {
      positions.push(Number(parts[1]), Number(parts[2]), Number(parts[3] ?? 0));
    } else if (parts[0] === 'f') {
      const verts: number[] = [];
      for (let i = 1; i < parts.length; i++) {
        const idx = Number(parts[i].split('/')[0]);
        verts.push(idx > 0 ? idx - 1 : positions.length / 3 + idx);
      }
      faceVerts.push(verts);
    }
  }
  for (const f of faceVerts) {
    for (let i = 1; i < f.length - 1; i++) {
      indices.push(f[0], f[i], f[i + 1]);
    }
  }
  return {
    positions: Float32Array.from(positions),
    indices: Uint32Array.from(indices),
    labels: new Uint8Array(positions.length / 3).fill(SurfaceLabel.Fracture),
  };
}
