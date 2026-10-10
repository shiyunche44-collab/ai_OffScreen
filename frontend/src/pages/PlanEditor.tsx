import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError } from "../api/client";
import { useEditPlan, useJobs, usePlan, usePreviewSegment, useProject, useRunStep, useShots, useTranscript } from "../api/queries";
import type { Job, PlanOp, ProjectDetail, SegmentPreview } from "../api/types";
import { CandidateDrawer } from "../components/plan/CandidateDrawer";
import { ClipTrimmer } from "../components/plan/ClipTrimmer";
import { OriginalPicker } from "../components/plan/OriginalPicker";
import { SegmentRow } from "../components/plan/SegmentRow";
import { QueryState } from "../components/Query";
import { formatDuration } from "../lib/format";
import { movedIndex, planMs } from "../lib/plan";
import { fileUrl } from "../lib/project";

const active = (j: Job) => j.status === "queued" || j.status === "running";

type Tool =
  | { kind: "candidates"; segmentId: string; index: number | null }
  | { kind: "trim"; segmentId: string; index: number }
  | { kind: "original"; after: string | null };

export function PlanEditorPage() {
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
  const plan = usePlan(projectId);
  const shots = useShots(assetId);
  const lines = useTranscript(assetId).data?.lines;
  const jobs = useJobs().data;
  const edit = useEditPlan(projectId);
  const previewer = usePreviewSegment(projectId);
  const run = useRunStep(detail.project);

  const [tool, setTool] = useState<Tool | null>(null);
  const [preview, setPreview] = useState<{ segmentId: string; result: SegmentPreview } | null>(null);

  const doc = plan.data;
  const noPlan = plan.error instanceof ApiError && plan.error.status === 404;
  const building = (jobs ?? []).find((j) => j.stage === "creation.plan" && j.scope["asset_id"] === assetId && active(j));
  const staleCount = doc?.segments.filter((s) => s.stale).length ?? 0;
  const filmMs = shots.data ? Math.max(0, ...shots.data.shots.map((s) => s.end_ms)) : undefined;

  const apply = (ops: PlanOp[], then?: () => void) => {
    if (!doc) return;
    edit.mutate({ ops, baseVersion: doc.version }, { onSuccess: () => then?.() });
  };
  const busy = edit.isPending;

  const toolSegment = tool && tool.kind !== "original" ? doc?.segments.find((s) => s.id === tool.segmentId) : undefined;
  const toolClip = tool?.kind === "trim" ? toolSegment?.clips[tool.index] : undefined;

  return (
    <div className="space-y-6">
      <header>
        <p className="text-xs">
          <Link className="text-sky-600 hover:underline dark:text-sky-400" to={`/projects/${projectId}`}>
            ← {detail.project.name}
          </Link>
        </p>
        <h1 className="text-xl font-semibold">剪辑计划</h1>
        <p className="mt-1 text-xs text-slate-500">
          每段的配音和镜头。改动立即存成新版本；改文字请到{" "}
          <Link className="text-sky-600 hover:underline dark:text-sky-400" to={`/projects/${projectId}/script`}>
            文案编辑
          </Link>
          。
        </p>
      </header>

      <QueryState isPending={plan.isPending && !noPlan} error={noPlan ? null : plan.error}>
        {noPlan ? (
          <p className="text-sm text-slate-500">
            还没有剪辑计划。先生成文案，再{" "}
            <button type="button" disabled={run.isPending || building !== undefined} onClick={() => run.mutate("creation.plan")} className="text-sky-600 hover:underline">
              构建计划
            </button>
            。
          </p>
        ) : null}

        {doc ? (
          <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_24rem]">
            <div className="min-w-0 space-y-3">
              <div className="flex flex-wrap items-center gap-3 text-sm">
                <span aria-label="计划概况" className="text-slate-500">
                  当前 v{doc.version}（{doc.author === "human" ? "人工编辑" : "自动构建"}）· 共 {doc.segments.length} 段 · {formatDuration(planMs(doc))}
                </span>
                <button
                  type="button"
                  disabled={run.isPending || building !== undefined}
                  onClick={() => run.mutate("creation.plan")}
                  className="ml-auto rounded border border-sky-500 px-3 py-1 text-sky-700 disabled:opacity-50 dark:text-sky-300"
                >
                  {building ? `构建中 ${Math.round(building.progress * 100)}%` : "重建计划"}
                </button>
              </div>
              {staleCount > 0 ? (
                <p role="status" className="rounded border border-amber-400 px-3 py-2 text-sm text-amber-700 dark:text-amber-300">
                  有 {staleCount} 段改了音色或语速，点「重建计划」重新配音；你锁定的镜头和手排的顺序都会保留。
                </p>
              ) : null}
              {run.error ? (
                <p role="alert" className="text-sm text-rose-600">
                  {run.error.message}
                </p>
              ) : null}
              {edit.error ? (
                <p role="alert" className="text-sm text-rose-600">
                  {edit.error instanceof ApiError && edit.error.status === 409 ? "计划已有新版本，没有改动。已为你载入最新版，请再试一次。" : `没有改成：${edit.error.message}`}
                </p>
              ) : null}
              {previewer.error ? (
                <p role="alert" className="text-sm text-rose-600">
                  预览失败：{previewer.error.message}
                </p>
              ) : null}

              <button
                type="button"
                disabled={busy}
                onClick={() => setTool({ kind: "original", after: null })}
                className="rounded border border-dashed px-2 py-0.5 text-xs"
              >
                ＋ 在开头插入原声
              </button>
              <ol aria-label="段落" className="space-y-3">
                {doc.segments.map((segment, i) => (
                  <SegmentRow
                    key={segment.id}
                    projectId={projectId}
                    index={i}
                    count={doc.segments.length}
                    segment={segment}
                    shots={shots.data}
                    selected={tool && tool.kind !== "original" && tool.segmentId === segment.id ? tool.index : null}
                    busy={busy}
                    previewing={previewer.isPending && previewer.variables?.segmentId === segment.id}
                    actions={{
                      swap: (index) => setTool({ kind: "candidates", segmentId: segment.id, index }),
                      add: () => setTool({ kind: "candidates", segmentId: segment.id, index: null }),
                      trim: (index) => setTool({ kind: "trim", segmentId: segment.id, index }),
                      remove: (index) => apply([{ op: "remove_clip", segment_id: segment.id, index }]),
                      lock: (index, locked) => apply([{ op: "set_locked", segment_id: segment.id, index, locked }]),
                      move: (delta) => {
                        const to = movedIndex(i, delta, doc.segments.length);
                        if (to !== null) apply([{ op: "move_segment", segment_id: segment.id, to_index: to }]);
                      },
                      delete: () => apply([{ op: "delete_segment", segment_id: segment.id }], () => setTool(null)),
                      insertOriginal: () => setTool({ kind: "original", after: segment.id }),
                      preview: () =>
                        previewer.mutate(
                          { segmentId: segment.id, version: doc.version },
                          { onSuccess: (result) => setPreview({ segmentId: segment.id, result }) },
                        ),
                      setVoice: (voice_id, speed) => apply([{ op: "set_voice", segment_id: segment.id, voice_id, speed }]),
                    }}
                  />
                ))}
              </ol>
            </div>

            <div className="space-y-3 lg:sticky lg:top-4 lg:self-start">
              {tool?.kind === "candidates" && toolSegment ? (
                <CandidateDrawer
                  projectId={projectId}
                  segmentId={tool.segmentId}
                  planVersion={doc.version}
                  index={tool.index}
                  shots={shots.data}
                  filmMs={filmMs ?? Number.MAX_SAFE_INTEGER}
                  busy={busy}
                  onPick={(shot_id) =>
                    apply(
                      [
                        tool.index === null
                          ? { op: "add_clip", segment_id: tool.segmentId, to: { shot_id, speed: 1 } }
                          : { op: "swap_clip", segment_id: tool.segmentId, index: tool.index, to: { shot_id, speed: 1 } },
                      ],
                      () => setTool(null),
                    )
                  }
                  onInterval={(src_in_ms, src_out_ms) =>
                    apply(
                      [
                        tool.index === null
                          ? { op: "add_clip", segment_id: tool.segmentId, to: { src_in_ms, src_out_ms, speed: 1 } }
                          : { op: "swap_clip", segment_id: tool.segmentId, index: tool.index, to: { src_in_ms, src_out_ms, speed: 1 } },
                      ],
                      () => setTool(null),
                    )
                  }
                  onClose={() => setTool(null)}
                />
              ) : null}
              {tool?.kind === "trim" && toolClip && shots.data ? (
                <ClipTrimmer
                  segmentId={tool.segmentId}
                  index={tool.index}
                  clip={toolClip}
                  shots={shots.data}
                  filmMs={filmMs ?? Number.MAX_SAFE_INTEGER}
                  busy={busy}
                  onApply={(src_in_ms, src_out_ms, speed) =>
                    apply([{ op: "trim_clip", segment_id: tool.segmentId, index: tool.index, src_in_ms, src_out_ms, speed }], () => setTool(null))
                  }
                  onClose={() => setTool(null)}
                />
              ) : null}
              {tool?.kind === "original" ? (
                <OriginalPicker
                  after={tool.after}
                  lines={lines}
                  busy={busy}
                  onInsert={(line_refs) => apply([{ op: "insert_original", line_refs, after: tool.after }], () => setTool(null))}
                  onClose={() => setTool(null)}
                />
              ) : null}

              {preview ? (
                <aside aria-label="段落预览" className="space-y-2 rounded border border-slate-200 p-3 text-sm dark:border-slate-800">
                  <div className="flex items-start gap-2">
                    <p className="min-w-0 flex-1 text-xs text-slate-500">
                      {preview.segmentId} 的预览（v{preview.result.plan_version}，{preview.result.cached ? "已有缓存" : "新渲染"}）
                    </p>
                    <button type="button" onClick={() => setPreview(null)} aria-label="关闭段落预览" className="text-slate-500 hover:text-slate-900">
                      ×
                    </button>
                  </div>
                  <video key={preview.result.file} controls autoPlay src={fileUrl(preview.result.file)} aria-label="段落预览视频" className="w-full rounded bg-black" />
                </aside>
              ) : null}
              {!tool && !preview ? (
                <p className="rounded border border-dashed border-slate-300 p-3 text-xs text-slate-500 dark:border-slate-700">
                  点镜头下的「换镜」看候选，「裁剪」调入出点，「预览此段」试看一段。换过或裁过的镜头会自动锁定，重建计划时保留。
                </p>
              ) : null}
            </div>
          </div>
        ) : null}
      </QueryState>
    </div>
  );
}
