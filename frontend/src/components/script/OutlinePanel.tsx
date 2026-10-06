import { useEffect, useState } from "react";
import { useGenerateOutline, useOutline, useResetOutline, useSaveOutline } from "../../api/queries";
import { ApiError } from "../../api/client";
import type { Job, OutlineBeat, Scene } from "../../api/types";
import { move } from "../../lib/script";
import { SceneChips } from "./SceneChips";

export const MIN_BEAT_S = 3;

/** Why this outline cannot be saved (the backend refuses the same things). */
export function outlineProblems(beats: readonly OutlineBeat[], knownScenes: readonly string[]): string[] {
  const out: string[] = [];
  const names = new Set<string>();
  beats.forEach((b, i) => {
    const n = i + 1;
    const name = b.beat.trim();
    if (!name) out.push(`第 ${n} 个节拍没有名字`);
    else if (names.has(name)) out.push(`节拍名「${name}」重复`);
    names.add(name);
    if (b.scene_refs.length === 0) out.push(`第 ${n} 个节拍没有场景`);
    if (knownScenes.length > 0 && b.scene_refs.some((r) => !knownScenes.includes(r))) out.push(`第 ${n} 个节拍引用了不存在的场景`);
    if (!Number.isInteger(b.target_s) || b.target_s < MIN_BEAT_S) out.push(`第 ${n} 个节拍至少 ${MIN_BEAT_S} 秒`);
  });
  if (beats.length === 0) out.push("大纲不能为空");
  return out;
}

export function OutlinePanel({
  projectId,
  targetSeconds,
  scenes,
  generating,
  onPreview,
}: {
  projectId: string;
  targetSeconds: number;
  scenes: readonly Scene[] | undefined;
  /** The outline job, while one is queued or running. */
  generating: Job | undefined;
  onPreview: (sceneId: string) => void;
}) {
  const outline = useOutline(projectId);
  const save = useSaveOutline(projectId);
  const reset = useResetOutline(projectId);
  const generate = useGenerateOutline(projectId);
  const served = outline.data?.outline.beats;
  const [draft, setDraft] = useState<OutlineBeat[]>([]);
  const servedJson = JSON.stringify(served ?? []);
  useEffect(() => setDraft(served ?? []), [servedJson]); // eslint-disable-line react-hooks/exhaustive-deps

  const dirty = JSON.stringify(draft) !== servedJson;
  const problems = outlineProblems(draft, (scenes ?? []).map((s) => s.id));
  const total = draft.reduce((n, b) => n + (Number.isFinite(b.target_s) ? b.target_s : 0), 0);
  const edit = (i: number, change: Partial<OutlineBeat>) => setDraft((d) => d.map((b, j) => (j === i ? { ...b, ...change } : b)));
  const busy = generate.isPending || generating !== undefined;
  const missing = outline.error instanceof ApiError && outline.error.status === 404;

  return (
    <section aria-label="大纲" className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-lg font-medium">大纲</h2>
        {outline.data?.edited ? (
          <span className="rounded bg-sky-100 px-2 py-0.5 text-xs text-sky-800 dark:bg-sky-950 dark:text-sky-200">已手动修改</span>
        ) : null}
        <button
          type="button"
          disabled={busy}
          onClick={() => generate.mutate()}
          className="ml-auto rounded border border-sky-500 px-3 py-1 text-sm text-sky-700 disabled:opacity-50 dark:text-sky-300"
        >
          {busy ? "生成中…" : served ? "重新生成大纲" : "生成大纲"}
        </button>
      </div>
      {generate.error ? <p role="alert" className="text-sm text-rose-600">{generate.error.message}</p> : null}

      {outline.isPending ? <p className="text-sm text-slate-500">加载中…</p> : null}
      {missing ? <p className="text-sm text-slate-500">还没有大纲。生成后可以在写正文之前调整节拍、场景和时长。</p> : null}
      {outline.error && !missing ? <p role="alert" className="text-sm text-rose-600">出错了：{outline.error.message}</p> : null}

      {served ? (
        <>
          <ol className="space-y-2" aria-label="节拍">
            {draft.map((b, i) => (
              <li key={i} className="space-y-2 rounded border border-slate-200 p-3 dark:border-slate-800">
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <span className="text-xs text-slate-500">{i + 1}</span>
                  <input
                    aria-label={`节拍 ${i + 1} 名称`}
                    value={b.beat}
                    onChange={(e) => edit(i, { beat: e.target.value })}
                    className="w-32 rounded border border-slate-300 bg-transparent px-2 py-0.5 dark:border-slate-700"
                  />
                  <label className="flex items-center gap-1">
                    <input
                      aria-label={`节拍 ${i + 1} 秒数`}
                      type="number"
                      min={MIN_BEAT_S}
                      value={Number.isFinite(b.target_s) ? b.target_s : ""}
                      onChange={(e) => edit(i, { target_s: e.target.value === "" ? Number.NaN : Number(e.target.value) })}
                      className="w-20 rounded border border-slate-300 bg-transparent px-2 py-0.5 dark:border-slate-700"
                    />
                    秒
                  </label>
                  <span className="ml-auto flex gap-1 text-xs">
                    <button type="button" aria-label={`上移节拍 ${i + 1}`} disabled={i === 0} onClick={() => setDraft((d) => move(d, i, i - 1))} className="rounded border px-2 py-0.5 disabled:opacity-40">
                      ↑
                    </button>
                    <button type="button" aria-label={`下移节拍 ${i + 1}`} disabled={i === draft.length - 1} onClick={() => setDraft((d) => move(d, i, i + 1))} className="rounded border px-2 py-0.5 disabled:opacity-40">
                      ↓
                    </button>
                    <button type="button" aria-label={`删除节拍 ${i + 1}`} onClick={() => setDraft((d) => d.filter((_, j) => j !== i))} className="rounded border px-2 py-0.5 text-rose-600">
                      删除
                    </button>
                  </span>
                </div>
                <input
                  aria-label={`节拍 ${i + 1} 要讲什么`}
                  value={b.focus ?? ""}
                  placeholder="这一节要讲什么"
                  onChange={(e) => edit(i, { focus: e.target.value })}
                  className="w-full rounded border border-slate-300 bg-transparent px-2 py-1 text-sm dark:border-slate-700"
                />
                <SceneChips ids={b.scene_refs} scenes={scenes} onPreview={onPreview} onChange={(ids) => edit(i, { scene_refs: ids })} />
              </li>
            ))}
          </ol>

          <div className="flex flex-wrap items-center gap-3 text-sm">
            <button
              type="button"
              onClick={() =>
                setDraft((d) => [
                  ...d,
                  { beat: `beat_${d.length + 1}`, focus: "", scene_refs: scenes?.[0] ? [scenes[0].id] : [], target_s: MIN_BEAT_S * 2 },
                ])
              }
              className="rounded border px-3 py-1"
            >
              ＋ 添加节拍
            </button>
            <span className={Math.abs(total - targetSeconds) > targetSeconds * 0.2 ? "text-amber-600" : "text-slate-500"}>
              合计 {total} 秒 · 目标 {targetSeconds} 秒
            </span>
            <span className="ml-auto flex gap-2">
              <button
                type="button"
                disabled={!dirty}
                onClick={() => setDraft(served)}
                className="rounded border px-3 py-1 disabled:opacity-50"
              >
                放弃修改
              </button>
              {outline.data?.edited ? (
                <button
                  type="button"
                  disabled={reset.isPending}
                  onClick={() => reset.mutate()}
                  className="rounded border px-3 py-1"
                >
                  恢复为生成版
                </button>
              ) : null}
              <button
                type="button"
                disabled={!dirty || problems.length > 0 || save.isPending}
                onClick={() => save.mutate(draft)}
                className="rounded bg-sky-600 px-3 py-1 text-white disabled:opacity-50"
              >
                保存大纲
              </button>
            </span>
          </div>
          {dirty && problems.length > 0 ? (
            <ul className="text-xs text-amber-600" aria-label="大纲的问题">
              {problems.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>
          ) : null}
          {save.error ? <p role="alert" className="text-sm text-rose-600">{save.error.message}</p> : null}
          {reset.error ? <p role="alert" className="text-sm text-rose-600">{reset.error.message}</p> : null}
          <p className="text-xs text-slate-500">大纲的修改会用在下一次「生成文案」里；已经写好的文案不会因此改变。</p>
        </>
      ) : null}
    </section>
  );
}
