import { useMemo, useState } from "react";
import { useAssets, useJobs } from "../api/queries";
import { JobRow } from "../components/JobRow";
import { QueryState } from "../components/Query";
import { filterJobs, type JobFilter } from "../lib/jobs";

const filters: { id: JobFilter; label: string }[] = [
  { id: "all", label: "全部" },
  { id: "active", label: "进行中" },
  { id: "failed", label: "失败" },
];

export function Jobs() {
  const { data, isPending, error } = useJobs();
  const assets = useAssets();
  const [filter, setFilter] = useState<JobFilter>("all");
  const titles = useMemo(
    () => new Map((assets.data ?? []).map((d) => [d.asset.id, d.asset.title])),
    [assets.data],
  );
  const shown = filterJobs(data ?? [], filter);

  return (
    <section>
      <div className="mb-4 flex items-center gap-4">
        <h1 className="text-xl font-semibold">作业中心</h1>
        <div className="flex gap-1 text-sm" role="group" aria-label="筛选">
          {filters.map(({ id, label }) => (
            <button
              key={id}
              type="button"
              aria-pressed={filter === id}
              onClick={() => setFilter(id)}
              className={`rounded px-3 py-1 ${filter === id ? "bg-sky-600 text-white" : "border border-slate-300 dark:border-slate-700"}`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      <QueryState isPending={isPending} error={error}>
        {shown.length ? (
          <ul className="space-y-2">
            {shown.map((job) => {
              const assetId = job.scope["asset_id"];
              return (
                <JobRow
                  key={job.id}
                  job={job}
                  assetTitle={typeof assetId === "string" ? titles.get(assetId) : undefined}
                />
              );
            })}
          </ul>
        ) : (
          <p className="text-sm text-slate-500">{filter === "all" ? "还没有作业。" : "没有符合条件的作业。"}</p>
        )}
      </QueryState>
    </section>
  );
}
