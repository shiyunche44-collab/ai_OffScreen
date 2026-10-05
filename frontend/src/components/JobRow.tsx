import { useState } from "react";
import { useCancelJob, useRetryJob } from "../api/queries";
import type { Job } from "../api/types";
import { isActive, runtime, stageLabel, statusLabel } from "../lib/jobs";
import { JobLog } from "./JobLog";

const tone: Record<Job["status"], string> = {
  queued: "text-slate-500",
  running: "text-sky-600",
  succeeded: "text-emerald-600",
  failed: "text-rose-600",
  canceled: "text-slate-500",
};

export function JobRow({ job, assetTitle }: { job: Job; assetTitle: string | undefined }) {
  const [showLog, setShowLog] = useState(false);
  const cancel = useCancelJob();
  const retry = useRetryJob();
  const took = runtime(job);
  // A running job stops at its next checkpoint: the request was accepted but it is still running.
  const canceling = cancel.isSuccess && job.status === "running";
  const failure = cancel.error ?? retry.error;

  return (
    <li className="rounded border border-slate-200 px-4 py-3 dark:border-slate-800" aria-label={`${stageLabel(job.stage)} ${job.id}`}>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <div className="min-w-0 flex-1">
          <span className="font-medium">{stageLabel(job.stage)}</span>
          {assetTitle ? <span className="ml-2 text-sm text-slate-500">{assetTitle}</span> : null}
          <div className="text-xs text-slate-500">
            {job.id} · {job.lane}
            {job.attempt > 1 ? ` · 第 ${job.attempt} 次尝试` : ""}
            {took ? ` · 用时 ${took}` : ""}
          </div>
        </div>
        <span className={`text-sm ${tone[job.status]}`}>{canceling ? "取消中…" : statusLabel(job.status)}</span>
        <div className="flex gap-2">
          {isActive(job) ? (
            <button
              type="button"
              disabled={cancel.isPending || canceling}
              onClick={() => cancel.mutate(job.id)}
              className="rounded border border-slate-300 px-3 py-1 text-sm disabled:opacity-50 dark:border-slate-700"
            >
              取消
            </button>
          ) : null}
          {job.status === "failed" ? (
            <button
              type="button"
              disabled={retry.isPending}
              onClick={() => retry.mutate(job.id)}
              className="rounded bg-sky-600 px-3 py-1 text-sm text-white disabled:opacity-50"
            >
              重试
            </button>
          ) : null}
          <button
            type="button"
            aria-expanded={showLog}
            onClick={() => setShowLog((v) => !v)}
            className="rounded border border-slate-300 px-3 py-1 text-sm dark:border-slate-700"
          >
            日志
          </button>
        </div>
      </div>
      {job.status === "running" || job.status === "queued" ? (
        <progress className="mt-2 h-1.5 w-full" max={1} value={job.progress} aria-label="进度" />
      ) : null}
      {job.message ? <p className="mt-1 text-xs text-slate-500">{job.message}</p> : null}
      {job.error ? (
        <p role="alert" className="mt-1 text-sm text-rose-600">
          {job.error}
        </p>
      ) : null}
      {failure ? (
        <p role="alert" className="mt-1 text-sm text-rose-600">
          {failure.message}
        </p>
      ) : null}
      {showLog ? (
        <div className="mt-3">
          <JobLog jobId={job.id} live={job.status === "running"} />
        </div>
      ) : null}
    </li>
  );
}
