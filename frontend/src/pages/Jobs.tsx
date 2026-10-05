import { useJobs } from "../api/queries";
import { QueryState } from "../components/Query";

// Skeleton: the job center proper (log viewer, retry / cancel) is M2-09.
export function Jobs() {
  const { data, isPending, error } = useJobs();
  return (
    <section>
      <h1 className="mb-4 text-xl font-semibold">作业中心</h1>
      <QueryState isPending={isPending} error={error}>
        {data?.length ? (
          <ul className="space-y-2">
            {data.map((job) => (
              <li key={job.id} className="rounded border border-slate-200 px-4 py-3 dark:border-slate-800">
                <div className="flex items-baseline gap-3">
                  <span className="font-medium">{job.stage}</span>
                  <span className="text-xs text-slate-500">{job.status}</span>
                </div>
                <progress className="mt-2 h-1.5 w-full" max={1} value={job.progress} aria-label="进度" />
                {job.message ? <p className="mt-1 text-xs text-slate-500">{job.message}</p> : null}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-slate-500">没有作业。</p>
        )}
      </QueryState>
    </section>
  );
}
