import type { AssetDetail, Job } from "../api/types";

/** The stage that analysis runs up to (what the "analyze" button queues). */
export const ANALYSIS_STAGE = "analysis.story";

export type AnalysisState =
  | { kind: "done" }
  | { kind: "queued"; job: Job }
  | { kind: "running"; job: Job }
  | { kind: "failed"; job: Job }
  | { kind: "none" };

const newest = (a: Job, b: Job) =>
  a.created_at === b.created_at ? b.id.localeCompare(a.id) : b.created_at.localeCompare(a.created_at);

/**
 * Where an asset stands: built (every analysis stage is cached), being worked on (an active
 * analysis job), failed (its latest analysis job failed), or not started. Jobs come from the live
 * job list, so progress moves without refetching the assets.
 */
export function analysisState(detail: AssetDetail, jobs: Job[] | undefined): AnalysisState {
  if (detail.stages.length > 0 && detail.stages.every((s) => s.cached)) return { kind: "done" };
  const mine = (jobs ?? [])
    .filter((j) => j.stage === ANALYSIS_STAGE && j.scope["asset_id"] === detail.asset.id)
    .sort(newest);
  const active = mine.find((j) => j.status === "running" || j.status === "queued");
  if (active) return { kind: active.status === "running" ? "running" : "queued", job: active };
  const latest = mine[0];
  if (latest?.status === "failed") return { kind: "failed", job: latest };
  return { kind: "none" };
}
