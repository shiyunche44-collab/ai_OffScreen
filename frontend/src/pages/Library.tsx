import { Link } from "react-router-dom";
import { useAssets } from "../api/queries";
import { QueryState } from "../components/Query";

// Skeleton: the library page proper (import form, analysis status) is M2-08.
export function Library() {
  const { data, isPending, error } = useAssets();
  return (
    <section>
      <h1 className="mb-4 text-xl font-semibold">素材库</h1>
      <QueryState isPending={isPending} error={error}>
        {data?.length ? (
          <ul className="divide-y divide-slate-200 rounded border border-slate-200 dark:divide-slate-800 dark:border-slate-800">
            {data.map((asset) => (
              <li key={asset.id} className="px-4 py-3">
                <span className="font-medium">{asset.title}</span>
                <span className="ml-3 text-xs text-slate-500">{asset.id}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-slate-500">还没有素材。导入电影后会显示在这里。</p>
        )}
      </QueryState>
      <p className="mt-6 text-sm">
        <Link className="text-sky-600 hover:underline dark:text-sky-400" to="/jobs">
          查看作业中心 →
        </Link>
      </p>
    </section>
  );
}
