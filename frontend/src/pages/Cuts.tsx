import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useAsset, useCutEvaluation, useCuts, useSaveCuts, useShots } from "../api/queries";
import { QueryState } from "../components/Query";
import { cutsFromShots, fpsOf, frameAt, neighbours, sameCuts, timeOfFrame, withCut, withoutCut } from "../lib/cuts";
import { formatDuration } from "../lib/format";
import { fileUrl } from "../lib/project";

const button = "rounded border border-slate-300 px-3 py-1 text-sm disabled:opacity-50 dark:border-slate-700";

/**
 * Mark the real hard cuts of a test film: step through the video frame by frame, press "mark" on
 * the first frame of each new shot. The marks are the ground truth that shot detection is scored
 * against (precision / recall / F1 within a few frames). Keys: `,` / `.` step a frame, `m` marks.
 */
export function CutsPage() {
  const { assetId = "" } = useParams();
  const asset = useAsset(assetId);
  const shots = useShots(assetId);
  const saved = useCuts(assetId);
  const save = useSaveCuts(assetId);
  const [tolerance, setTolerance] = useState(2);
  const evaluation = useCutEvaluation(assetId, tolerance, !!saved.data?.marked);

  const video = useRef<HTMLVideoElement>(null);
  const [frame, setFrame] = useState(0);
  const [draft, setDraft] = useState<number[] | null>(null);
  const cuts = draft ?? saved.data?.cuts ?? [];
  const dirty = draft !== null && !sameCuts(draft, saved.data?.cuts ?? []);
  const fps = asset.data ? fpsOf(asset.data.asset.video.fps.num, asset.data.asset.video.fps.den) : 24;
  const last = neighbours(cuts, frame);

  const goto = useCallback(
    (f: number) => {
      const target = Math.max(0, f);
      if (video.current) video.current.currentTime = timeOfFrame(target, fps);
      setFrame(target);
    },
    [fps],
  );
  const mark = useCallback(() => setDraft((d) => withCut(d ?? saved.data?.cuts ?? [], frame)), [frame, saved.data]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
      if (e.key === ",") goto(frame - 1);
      else if (e.key === ".") goto(frame + 1);
      else if (e.key === "m") mark();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [frame, goto, mark]);

  const marked = cuts.includes(frame);
  return (
    <section>
      <QueryState isPending={asset.isPending || saved.isPending} error={asset.error ?? saved.error}>
        {asset.data ? (
          <div className="space-y-6">
            <header>
              <p className="text-xs">
                <Link className="text-sky-600 hover:underline dark:text-sky-400" to={`/library/${assetId}`}>
                  ← 影片分析
                </Link>
              </p>
              <h1 className="text-xl font-semibold">切点标注：{asset.data.asset.title}</h1>
              <p className="mt-1 text-xs text-slate-500">
                逐帧找到每个新镜头的第一帧并标记；{fps.toFixed(3)} 帧/秒。快捷键：「,」「.」逐帧，「m」标记。
              </p>
            </header>

            <QueryState isPending={shots.isPending} error={shots.error}>
              {shots.data ? (
                <div className="space-y-3">
                  <video
                    ref={video}
                    controls
                    preload="auto"
                    src={fileUrl(shots.data.video)}
                    aria-label="代理视频"
                    onSeeked={(e) => setFrame(frameAt(e.currentTarget.currentTime, fps))}
                    onTimeUpdate={(e) => setFrame(frameAt(e.currentTarget.currentTime, fps))}
                    className="w-full max-w-3xl rounded bg-black"
                  />
                  <div className="flex flex-wrap items-center gap-2">
                    <button type="button" className={button} onClick={() => goto(frame - 1)} aria-label="上一帧">
                      ‹ 帧
                    </button>
                    <button type="button" className={button} onClick={() => goto(frame + 1)} aria-label="下一帧">
                      帧 ›
                    </button>
                    <span className="tabular-nums text-sm" aria-label="当前帧">
                      帧 {frame}（{formatDuration(Math.round((frame / fps) * 1000))}）
                    </span>
                    <button
                      type="button"
                      className="rounded bg-sky-600 px-3 py-1 text-sm text-white disabled:opacity-50"
                      disabled={frame < 1 || marked}
                      onClick={mark}
                    >
                      {marked ? "已标记" : "标记为切点"}
                    </button>
                    <button type="button" className={button} disabled={last.prev === null} onClick={() => last.prev !== null && goto(last.prev)}>
                      上一个切点
                    </button>
                    <button type="button" className={button} disabled={last.next === null} onClick={() => last.next !== null && goto(last.next)}>
                      下一个切点
                    </button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      className={button}
                      onClick={() => setDraft(cutsFromShots(shots.data.shots.slice(1).map((s) => s.start_ms), fps))}
                    >
                      载入检测结果作为起点
                    </button>
                  </div>
                </div>
              ) : null}
            </QueryState>

            <div>
              <div className="mb-2 flex flex-wrap items-center gap-3">
                <h2 className="text-lg font-medium">已标记 {cuts.length} 个切点</h2>
                <button
                  type="button"
                  className="rounded bg-emerald-600 px-3 py-1 text-sm text-white disabled:opacity-50"
                  disabled={!dirty || save.isPending}
                  onClick={() => save.mutate(cuts, { onSuccess: () => setDraft(null) })}
                >
                  保存
                </button>
                {dirty ? <span className="text-xs text-amber-600">有未保存的修改</span> : null}
                {dirty ? (
                  <button type="button" className="text-xs text-slate-500 hover:underline" onClick={() => setDraft(null)}>
                    放弃修改
                  </button>
                ) : null}
              </div>
              {save.error ? (
                <p role="alert" className="mb-2 text-sm text-rose-600">
                  {save.error.message}
                </p>
              ) : null}
              {cuts.length ? (
                <ul aria-label="切点" className="flex flex-wrap gap-2">
                  {cuts.map((c) => (
                    <li key={c} className="flex items-center gap-1 rounded border border-slate-200 px-2 py-0.5 text-xs dark:border-slate-800">
                      <button type="button" className="tabular-nums hover:underline" onClick={() => goto(c)}>
                        帧 {c}
                      </button>
                      <button
                        type="button"
                        aria-label={`删除切点 ${c}`}
                        className="text-slate-500 hover:text-rose-600"
                        onClick={() => setDraft(withoutCut(cuts, c))}
                      >
                        ×
                      </button>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-slate-500">还没有标记。</p>
              )}
            </div>

            <div aria-label="评估">
              <h2 className="mb-2 text-lg font-medium">检测效果</h2>
              {!saved.data?.marked ? (
                <p className="text-sm text-slate-500">保存标记后，这里给出镜头检测的准确率、召回率和 F1。</p>
              ) : (
                <>
                  <label className="mb-2 flex items-center gap-2 text-sm">
                    容差
                    <input
                      type="number"
                      min={0}
                      max={50}
                      value={tolerance}
                      onChange={(e) => setTolerance(Math.min(50, Math.max(0, Number(e.target.value) || 0)))}
                      className="w-16 rounded border border-slate-300 bg-transparent px-2 py-0.5 dark:border-slate-700"
                    />
                    帧
                  </label>
                  <QueryState isPending={evaluation.isPending} error={evaluation.error}>
                    {evaluation.data ? (
                      <div className="space-y-1 text-sm">
                        <p>
                          标记 {evaluation.data.marked} · 检测 {evaluation.data.detected} · 命中 {evaluation.data.true_positives}
                        </p>
                        <p>
                          准确率 {evaluation.data.precision.toFixed(3)} · 召回率 {evaluation.data.recall.toFixed(3)} · F1{" "}
                          <strong>{evaluation.data.f1.toFixed(3)}</strong>
                        </p>
                        <p className="text-xs text-slate-500">多检测的（帧）：{evaluation.data.false_positives.join("、") || "无"}</p>
                        <p className="text-xs text-slate-500">漏掉的（帧）：{evaluation.data.false_negatives.join("、") || "无"}</p>
                      </div>
                    ) : null}
                  </QueryState>
                </>
              )}
            </div>
          </div>
        ) : null}
      </QueryState>
    </section>
  );
}
