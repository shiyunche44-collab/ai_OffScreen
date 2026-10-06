import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError } from "../api/client";
import {
  useJobs,
  useProject,
  useRestoreScript,
  useRewriteSegment,
  useRunStep,
  useSaveScript,
  useScenes,
  useScript,
  useTranscript,
} from "../api/queries";
import type { Job, ProjectDetail, ScriptSegment } from "../api/types";
import { OutlinePanel } from "../components/script/OutlinePanel";
import { ParamsForm } from "../components/script/ParamsForm";
import { ScenePreview } from "../components/script/ScenePreview";
import { SegmentCard, type RewriteState } from "../components/script/SegmentCard";
import { VersionPanel } from "../components/script/VersionPanel";
import { QueryState } from "../components/Query";
import { formatDuration } from "../lib/format";
import { optionsOf } from "../lib/project";
import {
  contentOf,
  isDirty,
  move,
  newSegmentId,
  notesOf,
  problemsOf,
  settle,
  totalStatus,
  type ScriptContent,
} from "../lib/script";

const active = (j: Job) => j.status === "queued" || j.status === "running";

export function ScriptEditorPage() {
  const { projectId = "" } = useParams();
  const project = useProject(projectId);
  return (
    <section>
      <QueryState isPending={project.isPending} error={project.error}>
        {project.data ? <Editor detail={project.data} /> : null}
      </QueryState>
    </section>
  );
}

function Editor({ detail }: { detail: ProjectDetail }) {
  const projectId = detail.project.id;
  const assetId = detail.project.asset_id;
  const options = optionsOf(detail);
  const targetSeconds = Math.round(options.minutes * 60);

  const jobs = useJobs().data;
  const script = useScript(projectId, true);
  const scenes = useScenes(assetId).data?.scenes;
  const lines = useTranscript(assetId).data?.lines;
  const save = useSaveScript(projectId);
  const restore = useRestoreScript(projectId);
  const rewrite = useRewriteSegment(projectId);
  const run = useRunStep(detail.project);

  const server = script.data;
  const [draft, setDraft] = useState<ScriptContent | null>(null);
  const [original, setOriginal] = useState<ScriptContent | null>(null);
  const [base, setBase] = useState<number | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [viewing, setViewing] = useState<number | null>(null);

  const dirty = draft !== null && original !== null && isDirty(draft, original);
  const adopt = (s: NonNullable<typeof server>) => {
    const content = contentOf(s);
    setDraft(content);
    setOriginal(content);
    setBase(s.version);
  };

  // A new current version (a generation, a rewrite, a restore) replaces the text on screen,
  // unless there are unsaved edits: those are kept and the page says the ground moved.
  useEffect(() => {
    if (server && !dirty) adopt(server);
  }, [server?.version]); // eslint-disable-line react-hooks/exhaustive-deps
  const stale = dirty && server !== undefined && base !== null && server.version !== base;

  useEffect(() => {
    if (!dirty) return;
    const warn = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  const problems = useMemo(() => (draft ? problemsOf(draft) : []), [draft]);
  const total = draft ? totalStatus(draft.segments, targetSeconds) : null;
  const sceneList = scenes ?? [];
  const previewScene = sceneList.find((s) => s.id === preview);
  const beatNames = draft?.outline.map((b) => b.beat) ?? [];

  const edit = (change: (c: ScriptContent) => ScriptContent) => setDraft((d) => (d ? change(d) : d));
  const setSegment = (i: number, next: ScriptSegment) =>
    edit((c) => ({ ...c, segments: c.segments.map((s, j) => (j === i ? next : s)) }));

  const rewriteState = (segmentId: string): RewriteState => {
    const mine = (jobs ?? [])
      .filter((j) => j.stage === "creation.rewrite" && j.scope["project_id"] === projectId && j.scope["segment_id"] === segmentId)
      .sort((a, b) => b.created_at.localeCompare(a.created_at));
    if (mine.some(active) || (rewrite.isPending && rewrite.variables?.segmentId === segmentId)) return { kind: "busy" };
    return mine[0]?.status === "failed" ? { kind: "failed", message: mine[0].error ?? "未知错误" } : { kind: "idle" };
  };

  const generating = (jobs ?? []).find(
    (j) => j.stage === "creation.outline" && j.scope["asset_id"] === assetId && active(j),
  );
  const scriptJob = (jobs ?? []).find((j) => j.stage === "creation.script" && j.scope["asset_id"] === assetId && active(j));
  const noScript = script.error instanceof ApiError && script.error.status === 404;

  const onSave = () => {
    if (!draft || !original) return;
    save.mutate(
      { content: settle(draft, original), baseVersion: base },
      { onSuccess: (saved) => adopt(saved) },
    );
  };

  return (
    <div className="space-y-8">
      <header>
        <p className="text-xs">
          <Link className="text-sky-600 hover:underline dark:text-sky-400" to={`/projects/${projectId}`}>
            ← {detail.project.name}
          </Link>
        </p>
        <h1 className="text-xl font-semibold">文案编辑</h1>
        <p className="mt-1 text-xs text-slate-500">
          {options.minutes} 分钟 · 风格 {options.style} · {options.spoil_ending ? "含结局" : "不剧透结局"}
        </p>
      </header>

      <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_22rem]">
        <div className="min-w-0 space-y-8">
          <details open={noScript} className="rounded border border-slate-200 px-4 py-3 dark:border-slate-800">
            <summary className="cursor-pointer text-lg font-medium">参数</summary>
            <div className="mt-3">
              <ParamsForm detail={detail} />
            </div>
          </details>

          <OutlinePanel
            projectId={projectId}
            targetSeconds={targetSeconds}
            scenes={scenes}
            generating={generating}
            onPreview={setPreview}
          />

          <section aria-label="文案" className="space-y-3">
            <div className="flex flex-wrap items-center gap-3">
              <h2 className="text-lg font-medium">文案</h2>
              {server ? <span className="text-xs text-slate-500">当前 v{server.version}</span> : null}
              <button
                type="button"
                disabled={scriptJob !== undefined || run.isPending}
                onClick={() => {
                  if (!server || window.confirm("会在现有版本之上新增一个 AI 生成的版本，已有的版本都保留。继续？")) run.mutate("creation.script");
                }}
                className="ml-auto rounded border border-sky-500 px-3 py-1 text-sm text-sky-700 disabled:opacity-50 dark:text-sky-300"
              >
                {scriptJob ? `生成中 ${Math.round(scriptJob.progress * 100)}%` : server ? "重新生成文案" : "生成文案"}
              </button>
            </div>
            {run.error ? <p role="alert" className="text-sm text-rose-600">{run.error.message}</p> : null}

            {script.isPending ? <p className="text-sm text-slate-500">加载中…</p> : null}
            {noScript ? <p className="text-sm text-slate-500">还没有文案。先生成大纲（可选），再点「生成文案」。</p> : null}
            {script.error && !noScript ? <p role="alert" className="text-sm text-rose-600">出错了：{script.error.message}</p> : null}

            {draft && total ? (
              <>
                <p
                  className={`text-sm ${total.state === "ok" ? "text-slate-500" : "text-amber-600"}`}
                  aria-label="字数与时长"
                >
                  共 {draft.segments.length} 段 · {total.chars} 字 · 预计 {formatDuration(total.seconds * 1000)} · 目标{" "}
                  {formatDuration(total.targetSeconds * 1000)}
                  {total.state === "short" ? "（偏短）" : total.state === "long" ? "（偏长）" : ""}
                </p>

                {stale ? (
                  <div role="alert" className="flex flex-wrap items-center gap-3 rounded border border-amber-400 px-3 py-2 text-sm text-amber-700 dark:text-amber-300">
                    <span>文案已有新版本 v{server?.version}（你是基于 v{base} 修改的）。现在保存会被拒绝。</span>
                    <button type="button" onClick={() => server && adopt(server)} className="rounded border border-amber-500 px-2 py-0.5">
                      载入最新版（放弃我的修改）
                    </button>
                  </div>
                ) : null}

                <ol className="space-y-3" aria-label="段落">
                  {draft.segments.map((segment, i) => (
                    <SegmentCard
                      key={segment.id}
                      index={i}
                      count={draft.segments.length}
                      segment={segment}
                      notes={notesOf(draft, segment.id)}
                      scenes={scenes}
                      lines={lines}
                      beats={beatNames}
                      onChange={(next) => setSegment(i, next)}
                      onMove={(delta) => edit((c) => ({ ...c, segments: move(c.segments, i, i + delta) }))}
                      onDelete={() => edit((c) => ({ ...c, segments: c.segments.filter((_, j) => j !== i) }))}
                      onInsertAfter={() =>
                        edit((c) => {
                          const fresh: ScriptSegment = {
                            id: newSegmentId(c.segments),
                            kind: "narration",
                            beat: segment.beat ?? null,
                            text: "",
                            scene_refs: [...segment.scene_refs],
                            line_refs: [],
                          };
                          return { ...c, segments: [...c.segments.slice(0, i + 1), fresh, ...c.segments.slice(i + 1)] };
                        })
                      }
                      onPreview={setPreview}
                      rewrite={{
                        blocked: dirty ? "先保存或放弃当前的修改，再让 AI 改写" : stale ? "文案已有新版本" : null,
                        state: rewriteState(segment.id),
                        submit: (instruction) => {
                          if (base !== null) rewrite.mutate({ segmentId: segment.id, instruction, baseVersion: base });
                        },
                      }}
                    />
                  ))}
                </ol>
                {rewrite.error ? <p role="alert" className="text-sm text-rose-600">{rewrite.error.message}</p> : null}

                <div className="flex flex-wrap items-center gap-3">
                  <button
                    type="button"
                    onClick={() =>
                      edit((c) => ({
                        ...c,
                        segments: [
                          ...c.segments,
                          { id: newSegmentId(c.segments), kind: "narration", beat: null, text: "", scene_refs: [], line_refs: [] },
                        ],
                      }))
                    }
                    className="rounded border px-3 py-1 text-sm"
                  >
                    ＋ 添加一段
                  </button>
                  <span className="ml-auto flex gap-2">
                    <button
                      type="button"
                      disabled={!dirty}
                      onClick={() => original && setDraft(original)}
                      className="rounded border px-3 py-1 text-sm disabled:opacity-50"
                    >
                      放弃修改
                    </button>
                    <button
                      type="button"
                      disabled={!dirty || problems.length > 0 || save.isPending}
                      onClick={onSave}
                      className="rounded bg-sky-600 px-3 py-1 text-sm text-white disabled:opacity-50"
                    >
                      保存为新版本
                    </button>
                  </span>
                </div>
                {dirty && problems.length > 0 ? (
                  <ul aria-label="文案的问题" className="text-xs text-amber-600">
                    {problems.map((p) => (
                      <li key={p}>{p}</li>
                    ))}
                  </ul>
                ) : null}
                {save.error ? (
                  <p role="alert" className="text-sm text-rose-600">
                    {save.error instanceof ApiError && save.error.status === 409
                      ? "文档已被更新，没有保存。请先载入最新版。"
                      : `保存失败：${save.error.message}`}
                  </p>
                ) : null}
              </>
            ) : null}
          </section>

          {server ? (
            <VersionPanel
              projectId={projectId}
              current={server.version}
              selected={viewing}
              onSelect={setViewing}
              scenes={scenes}
              lines={lines}
              onPreview={setPreview}
              restoring={restore.isPending}
              restoreBlocked={dirty ? "先保存或放弃当前的修改，再恢复旧版本" : null}
              onRestore={(version) =>
                restore.mutate({ version, baseVersion: server.version }, { onSuccess: () => setViewing(null) })
              }
            />
          ) : null}
          {restore.error ? <p role="alert" className="text-sm text-rose-600">{restore.error.message}</p> : null}
        </div>

        <div className="space-y-3 lg:sticky lg:top-4 lg:self-start">
          {previewScene ? (
            <ScenePreview assetId={assetId} scene={previewScene} onClose={() => setPreview(null)} />
          ) : (
            <p className="rounded border border-dashed border-slate-300 p-3 text-xs text-slate-500 dark:border-slate-700">
              点击任何一个场景编号（如 sc_001），在这里预览那一段影片。
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
