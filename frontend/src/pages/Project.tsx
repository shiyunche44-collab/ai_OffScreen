import { Link, useParams } from "react-router-dom";
import { useJobs, useProject } from "../api/queries";
import { QueryState } from "../components/Query";
import { ScriptView } from "../components/ScriptView";
import { StepList } from "../components/StepList";
import { fileUrl, optionsOf } from "../lib/project";

export function ProjectPage() {
  const { projectId = "" } = useParams();
  const { data, isPending, error } = useProject(projectId);
  const jobs = useJobs(); // live: drives each step's progress

  return (
    <section>
      <QueryState isPending={isPending} error={error}>
        {data ? (
          <div className="space-y-8">
            <header>
              <h1 className="text-xl font-semibold">{data.project.name}</h1>
              {(() => {
                const o = optionsOf(data);
                return (
                  <p className="mt-1 text-xs text-slate-500">
                    {o.minutes} 分钟 · 风格 {o.style} · 音色 {o.voice ?? "默认"} · {o.spoil_ending ? "含结局" : "不剧透结局"}
                  </p>
                );
              })()}
            </header>

            <StepList detail={data} jobs={jobs.data} />

            {!data.stages.find((s) => s.stage === "creation.script")?.cached ? (
              <p className="text-sm">
                <Link className="text-sky-600 hover:underline dark:text-sky-400" to={`/projects/${projectId}/script`}>
                  调整参数、大纲，编辑文案 →
                </Link>
              </p>
            ) : null}

            {data.stages.find((s) => s.stage === "creation.script")?.cached ? (
              <div>
                <div className="mb-3 flex items-baseline gap-3">
                  <h2 className="text-lg font-medium">解说文案</h2>
                  <Link className="text-sm text-sky-600 hover:underline dark:text-sky-400" to={`/projects/${projectId}/script`}>
                    编辑文案 →
                  </Link>
                </div>
                <ScriptView projectId={projectId} />
              </div>
            ) : null}

            {data.video ? (
              <div>
                <h2 className="mb-3 text-lg font-medium">成片</h2>
                <video
                  controls
                  preload="metadata"
                  src={fileUrl(data.video)}
                  aria-label="成片"
                  className="w-full max-w-3xl rounded bg-black"
                />
              </div>
            ) : null}

            <p className="text-sm">
              <Link className="text-sky-600 hover:underline dark:text-sky-400" to="/jobs">
                查看作业详情与日志 →
              </Link>
            </p>
          </div>
        ) : null}
      </QueryState>
    </section>
  );
}
