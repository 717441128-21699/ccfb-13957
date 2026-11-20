// 碎片配对 Web Worker：断口分析 + 全量 RANSAC/ICP 配对均在后台线程
import type { FracturePatch, MatchOptions, WorkerRequest, WorkerResponse } from '../core/types';
import { analyzeShard } from '../core/fracture';
import { matchAll } from '../core/matcher';

function post(msg: WorkerResponse) {
  (self as unknown as Worker).postMessage(msg);
}

self.onmessage = (ev: MessageEvent<WorkerRequest>) => {
  const req = ev.data;
  try {
    if (req.kind === 'analyze') {
      const fractures: Array<[number, FracturePatch]> = [];
      let i = 0;
      for (const s of req.shards) {
        const f = analyzeShard(s.mesh);
        if (f) fractures.push([s.id, f]);
        i++;
        post({ kind: 'progress', done: i, total: req.shards.length, current: `分析 #${s.id}` });
      }
      post({ kind: 'analyze:done', fractures });
      return;
    }
    if (req.kind === 'matchAll') {
      const options: MatchOptions = req.options;
      const result = matchAll(
        req.shards.map((s) => ({ id: s.id, fracture: s.fracture })),
        options,
        (done, total, current) => post({ kind: 'progress', done, total, current }),
      );
      post({ kind: 'matchAll:done', candidates: result.candidates, elapsedMs: result.elapsedMs });
      return;
    }
    if (req.kind === 'matchPair') {
      // 动态引入避免循环打包问题
      import('../core/matcher').then(({ matchPair }) => {
        const r = matchPair(req.a, req.b, req.options);
        post({
          kind: 'matchPair:done',
          candidate: r
            ? { a: -1, b: -1, ...r }
            : null,
        });
      });
      return;
    }
  } catch (err) {
    post({ kind: 'error', message: (err as Error).message ?? String(err) });
  }
};
