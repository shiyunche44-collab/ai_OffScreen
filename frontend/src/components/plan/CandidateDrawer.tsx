import { useState } from "react";
import { useCandidates } from "../../api/queries";
import type { Candidate, ShotsView } from "../../api/types";
import { formatDuration } from "../../lib/format";
import { intervalProblem, parseSeconds } from "../../lib/plan";
import { ShotThumb } from "./ShotThumb";

const FACTORS: [keyof Candidate["factors"], string][] = [
  ["embedding", "图文"],
  ["caption", "描述"],
  ["character", "人物"],
  ["quality", "画质"],
  ["reuse", "未重复"],
  ["time_order", "顺序"],
  ["exclusions", "非片头尾"],
];

/**
 * The best footage for one segment, best first, with the factors of each score. "用这个" swaps
 * it in for the clip being replaced (or adds it, when no clip is picked); the second form takes
 * any stretch of the film.
 */
export function CandidateDrawer({
  projectId,
  segmentId,
  planVersion,
  index,
  shots,
  filmMs,
  busy,
  onPick,
  onInterval,
  onClose,
}: {
  projectId: string;
  segmentId: string;
  planVersion: number;
  index: number | null;
  shots: ShotsView | undefined;
  filmMs: number;
  busy: boolean;
  onPick: (shotId: string) => void;
  onInterval: (inMs: number, outMs: number) => void;
  onClose: () => void;
}) {
  const [limit, setLimit] = useState(12);
  const cands = useCandidates(projectId, segmentId, planVersion, limit);
  const [inText, setInText] = useState("");
  const [outText, setOutText] = useState("");
  const inMs = parseSeconds(inText);
  const outMs = parseSeconds(outText);
  const problem = intervalProblem(inMs, outMs, filmMs);

  return (
    <aside aria-label="候选镜头" className="space-y-2 rounded border border-sky-300 p-3 text-sm dark:border-sky-800">
      <div className="flex items-start gap-2">
        <p className="min-w-0 flex-1 text-xs text-slate-500">
          {segmentId} · {index === null ? "添加镜头" : `替换第 ${index + 1} 个镜头`}
        </p>
        <button type="button" onClick={onClose} aria-label="关闭候选" className="text-slate-500 hover:text-slate-900">
          ×
        </button>
      </div>
      {cands.isPending ? <p className="text-xs text-slate-500">加载中…</p> : null}
      {cands.error ? (
        <p role="alert" className="text-xs text-rose-600">
          {cands.error.message}
        </p>
      ) : null}
      {cands.data ? (
        <>
          <p className="text-xs text-slate-500">
            共 {cands.data.total} 个可用镜头{cands.data.vector_search ? "" : "（未启用向量检索，只按场景和描述排序）"}
          </p>
          <ol className="space-y-2" aria-label="候选列表">
            {cands.data.candidates.map((c, rank) => (
              <li key={c.shot_id} aria-label={`候选 ${rank + 1}`} className="flex gap-2 rounded border border-slate-200 p-2 dark:border-slate-800">
                <ShotThumb view={shots} shotId={c.shot_id} width={96} />
                <div className="min-w-0 flex-1 space-y-1">
                  <p className="flex flex-wrap items-baseline gap-x-2 text-xs">
                    <span className="font-medium tabular-nums">{Math.round(c.score * 100)} 分</span>
                    <span className="text-slate-500">
                      {c.shot_id} · {formatDuration(c.start_ms)}–{formatDuration(c.end_ms)}
                      {c.scene_id ? ` · ${c.scene_id}` : ""}
                    </span>
                    {c.current ? <span className="rounded bg-sky-100 px-1 text-sky-700 dark:bg-sky-900 dark:text-sky-200">正在使用</span> : null}
                    {c.used_by ? <span className="rounded bg-amber-100 px-1 text-amber-700 dark:bg-amber-900 dark:text-amber-200">已用于 {c.used_by}</span> : null}
                  </p>
                  {c.caption ? <p className="line-clamp-2 text-xs leading-snug">{c.caption}</p> : null}
                  <p className="text-[11px] text-slate-500" title="各项得分（0–100）">
                    {FACTORS.map(([k, name]) => `${name} ${Math.round(c.factors[k] * 100)}`).join(" · ")}
                  </p>
                </div>
                <button
                  type="button"
                  disabled={busy || c.current}
                  onClick={() => onPick(c.shot_id)}
                  aria-label={`用候选 ${rank + 1}`}
                  className="self-start rounded border border-sky-500 px-2 py-0.5 text-xs text-sky-700 disabled:opacity-50 dark:text-sky-300"
                >
                  用这个
                </button>
              </li>
            ))}
          </ol>
          {cands.data.total > limit ? (
            <button type="button" onClick={() => setLimit((n) => n + 12)} className="text-xs text-sky-600 hover:underline">
              再显示 12 个
            </button>
          ) : null}
        </>
      ) : null}
      <details className="border-t border-slate-200 pt-2 dark:border-slate-800">
        <summary className="cursor-pointer text-xs">用影片的任意一段</summary>
        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
          <label>
            入点
            <input
              value={inText}
              onChange={(e) => setInText(e.target.value)}
              aria-label="区间入点（秒）"
              inputMode="decimal"
              className="ml-1 w-20 rounded border border-slate-300 bg-transparent px-1 dark:border-slate-700"
            />
          </label>
          <label>
            出点
            <input
              value={outText}
              onChange={(e) => setOutText(e.target.value)}
              aria-label="区间出点（秒）"
              inputMode="decimal"
              className="ml-1 w-20 rounded border border-slate-300 bg-transparent px-1 dark:border-slate-700"
            />
          </label>
          <button
            type="button"
            disabled={problem !== null || busy}
            onClick={() => inMs !== null && outMs !== null && onInterval(inMs, outMs)}
            className="rounded border border-sky-500 px-2 py-0.5 text-sky-700 disabled:opacity-50 dark:text-sky-300"
          >
            用这一段
          </button>
          {inText || outText ? <span className="text-amber-600">{problem}</span> : null}
        </div>
      </details>
    </aside>
  );
}
