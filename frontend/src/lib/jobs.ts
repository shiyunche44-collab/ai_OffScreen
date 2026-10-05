import type { Job } from "../api/types";

const STAGE_LABELS: Record<string, string> = {
  "analysis.story": "分析影片",
  "creation.script": "生成文案",
  "creation.plan": "构建剪辑计划",
  "output.render": "渲染成片",
};

export const stageLabel = (stage: string): string => STAGE_LABELS[stage] ?? stage;

const STATUS_LABELS: Record<Job["status"], string> = {
  queued: "排队中",
  running: "运行中",
  succeeded: "已完成",
  failed: "失败",
  canceled: "已取消",
};

export const statusLabel = (status: Job["status"]): string => STATUS_LABELS[status];

export const isActive = (job: Job): boolean => job.status === "queued" || job.status === "running";

export type JobFilter = "all" | "active" | "failed";

export function filterJobs(jobs: Job[], filter: JobFilter): Job[] {
  if (filter === "active") return jobs.filter(isActive);
  if (filter === "failed") return jobs.filter((j) => j.status === "failed");
  return jobs;
}

/** How long a finished job ran ("42 秒", "3 分 05 秒"), or null if it has not finished. */
export function runtime(job: Job): string | null {
  if (!job.started_at || !job.finished_at) return null;
  const seconds = Math.max(0, Math.round((Date.parse(job.finished_at) - Date.parse(job.started_at)) / 1000));
  if (seconds < 60) return `${seconds} 秒`;
  const m = Math.floor(seconds / 60);
  return `${m} 分 ${String(seconds % 60).padStart(2, "0")} 秒`;
}
