import { useParams } from "react-router-dom";
import { useProject } from "../api/queries";
import { QueryState } from "../components/Query";

// Skeleton: the project page proper (one button per step, script text, video) is M2-10.
export function ProjectPage() {
  const { projectId = "" } = useParams();
  const { data, isPending, error } = useProject(projectId);
  return (
    <section>
      <QueryState isPending={isPending} error={error}>
        {data ? (
          <>
            <h1 className="mb-4 text-xl font-semibold">{data.project.name}</h1>
            <ul className="space-y-1 text-sm">
              {data.stages.map((s) => (
                <li key={s.stage}>
                  {s.cached ? "✓" : "·"} {s.stage}
                </li>
              ))}
            </ul>
          </>
        ) : null}
      </QueryState>
    </section>
  );
}
