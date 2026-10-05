import { useAssets, useJobs } from "../api/queries";
import { AssetRow } from "../components/AssetRow";
import { ImportPanel } from "../components/ImportPanel";
import { QueryState } from "../components/Query";

export function Library() {
  const { data, isPending, error } = useAssets();
  const jobs = useJobs(); // kept live by /api/events; drives the per-asset analysis status
  return (
    <section>
      <h1 className="mb-4 text-xl font-semibold">素材库</h1>
      <ImportPanel />
      <QueryState isPending={isPending} error={error}>
        {data?.length ? (
          <ul className="divide-y divide-slate-200 rounded border border-slate-200 dark:divide-slate-800 dark:border-slate-800">
            {data.map((detail) => (
              <AssetRow key={detail.asset.id} detail={detail} jobs={jobs.data} />
            ))}
          </ul>
        ) : (
          <p className="text-sm text-slate-500">还没有素材。输入路径或从媒体目录里选一部电影导入。</p>
        )}
      </QueryState>
    </section>
  );
}
