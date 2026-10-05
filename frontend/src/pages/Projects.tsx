import { Link } from "react-router-dom";
import { useAssets, useProjects } from "../api/queries";
import { QueryState } from "../components/Query";

export function Projects() {
  const { data, isPending, error } = useProjects();
  const assets = useAssets();
  const titles = new Map((assets.data ?? []).map((d) => [d.asset.id, d.asset.title]));
  return (
    <section>
      <h1 className="mb-4 text-xl font-semibold">项目</h1>
      <QueryState isPending={isPending} error={error}>
        {data?.length ? (
          <ul className="divide-y divide-slate-200 rounded border border-slate-200 dark:divide-slate-800 dark:border-slate-800">
            {data.map((project) => (
              <li key={project.id}>
                <Link to={`/projects/${project.id}`} className="block px-4 py-3 hover:bg-slate-100 dark:hover:bg-slate-900">
                  <span className="font-medium">{project.name}</span>
                  <span className="ml-3 text-xs text-slate-500">{titles.get(project.asset_id) ?? project.asset_id}</span>
                </Link>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-slate-500">还没有项目。在素材库里选一部电影，点“新建项目”。</p>
        )}
      </QueryState>
    </section>
  );
}
