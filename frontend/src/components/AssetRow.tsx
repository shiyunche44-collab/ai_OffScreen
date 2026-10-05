import { Link, useNavigate } from "react-router-dom";
import { useAnalyzeAsset, useCreateProject } from "../api/queries";
import type { AssetDetail, Job } from "../api/types";
import { analysisState } from "../lib/analysis";
import { formatDuration } from "../lib/format";

function Status({ state }: { state: ReturnType<typeof analysisState> }) {
  switch (state.kind) {
    case "done":
      return <span className="text-emerald-600">已分析</span>;
    case "queued":
      return <span className="text-slate-500">排队中</span>;
    case "running":
      return (
        <span className="flex items-center gap-2 text-sky-600">
          分析中 {Math.round(state.job.progress * 100)}%
          <progress className="h-1.5 w-24" max={1} value={state.job.progress} aria-label="分析进度" />
        </span>
      );
    case "failed":
      return (
        <span role="alert" className="text-rose-600" title={state.job.error ?? undefined}>
          分析失败{state.job.error ? `：${state.job.error}` : ""}
        </span>
      );
    case "none":
      return <span className="text-slate-500">未分析</span>;
  }
}

export function AssetRow({ detail, jobs }: { detail: AssetDetail; jobs: Job[] | undefined }) {
  const { asset } = detail;
  const state = analysisState(detail, jobs);
  const analyze = useAnalyzeAsset();
  const createProject = useCreateProject();
  const navigate = useNavigate();
  const busy = state.kind === "queued" || state.kind === "running" || analyze.isPending;

  return (
    <li className="px-4 py-3" aria-label={asset.title}>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="min-w-0 flex-1">
          <div className="truncate font-medium">
            {state.kind === "done" ? (
              <Link className="hover:underline" to={`/library/${asset.id}`}>
                {asset.title}
              </Link>
            ) : (
              asset.title
            )}
          </div>
          <div className="text-xs text-slate-500">
            {formatDuration(asset.duration_ms)} · {asset.video.width}×{asset.video.height}
            {asset.subtitles_external ? " · 有外挂字幕" : " · 无外挂字幕"}
          </div>
        </div>
        <div className="text-sm">
          <Status state={state} />
        </div>
        <div className="flex gap-2">
          {state.kind !== "done" ? (
            <button
              type="button"
              disabled={busy}
              onClick={() => analyze.mutate(asset.id)}
              className="rounded border border-slate-300 px-3 py-1 text-sm disabled:opacity-50 dark:border-slate-700"
            >
              {state.kind === "failed" ? "重新分析" : "分析"}
            </button>
          ) : null}
          <button
            type="button"
            disabled={createProject.isPending}
            onClick={() =>
              createProject.mutate(asset.id, { onSuccess: (p) => void navigate(`/projects/${p.id}`) })
            }
            className="rounded bg-sky-600 px-3 py-1 text-sm text-white disabled:opacity-50"
          >
            新建项目
          </button>
        </div>
      </div>
      {analyze.error || createProject.error ? (
        <p role="alert" className="mt-2 text-sm text-rose-600">
          {(analyze.error ?? createProject.error)?.message}
        </p>
      ) : null}
    </li>
  );
}
