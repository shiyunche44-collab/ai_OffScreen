import type { Job, ProjectDetail } from "../api/types";
import { useRunStep } from "../api/queries";
import { STEPS, stepState, type StepState } from "../lib/project";

function Status({ state }: { state: StepState }) {
  switch (state.kind) {
    case "done":
      return <span className="text-emerald-600">✓ 已完成</span>;
    case "queued":
      return <span className="text-slate-500">排队中</span>;
    case "running":
      return (
        <span className="flex items-center gap-2 text-sky-600">
          进行中 {Math.round(state.job.progress * 100)}%
          <progress className="h-1.5 w-24" max={1} value={state.job.progress} aria-label="进度" />
        </span>
      );
    case "failed":
      return (
        <span role="alert" className="text-rose-600">
          失败{state.job.error ? `：${state.job.error}` : ""}
        </span>
      );
    case "none":
      return <span className="text-slate-500">未开始</span>;
  }
}

export function StepList({ detail, jobs }: { detail: ProjectDetail; jobs: Job[] | undefined }) {
  const run = useRunStep(detail.project);
  return (
    <ol className="divide-y divide-slate-200 rounded border border-slate-200 dark:divide-slate-800 dark:border-slate-800" aria-label="流程">
      {STEPS.map(({ stage, label, action }) => {
        const state = stepState(stage, detail, jobs);
        const busy = state.kind === "queued" || state.kind === "running" || (run.isPending && run.variables === stage);
        return (
          <li key={stage} className="flex flex-wrap items-center gap-x-4 gap-y-1 px-4 py-3" aria-label={label}>
            <span className="min-w-0 flex-1 font-medium">{label}</span>
            <span className="text-sm">
              <Status state={state} />
            </span>
            {state.kind !== "done" ? (
              <button
                type="button"
                disabled={busy}
                onClick={() => run.mutate(stage)}
                className="rounded bg-sky-600 px-3 py-1 text-sm text-white disabled:opacity-50"
              >
                {state.kind === "failed" ? "重试" : action}
              </button>
            ) : null}
          </li>
        );
      })}
      {run.error ? (
        <li role="alert" className="px-4 py-2 text-sm text-rose-600">
          {run.error.message}
        </li>
      ) : null}
    </ol>
  );
}
