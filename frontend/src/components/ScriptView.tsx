import { useScript } from "../api/queries";
import { CHARS_PER_SECOND } from "../lib/project";
import { formatDuration } from "../lib/format";
import { QueryState } from "./Query";

const BEATS: Record<string, string> = { hook: "钩子", setup: "铺垫", development: "发展", climax: "高潮", ending: "结尾" };

/** The commentary text, read-only (editing arrives with the script editor, M4). */
export function ScriptView({ projectId }: { projectId: string }) {
  const { data, isPending, error } = useScript(projectId, true);
  return (
    <section aria-label="文案">
      <QueryState isPending={isPending} error={error}>
        {data ? (
          <>
            <p className="mb-3 text-xs text-slate-500">
              共 {data.segments.length} 段 · 约 {data.segments.reduce((n, s) => n + s.text.length, 0)} 字 · 预计{" "}
              {formatDuration((data.segments.reduce((n, s) => n + s.text.length, 0) / CHARS_PER_SECOND) * 1000)}
            </p>
            <ol className="space-y-3">
              {data.segments.map((segment) => (
                <li key={segment.id} className="rounded border border-slate-200 px-4 py-3 dark:border-slate-800">
                  <div className="mb-1 text-xs text-slate-500">
                    {segment.id}
                    {segment.beat ? ` · ${BEATS[segment.beat] ?? segment.beat}` : ""}
                    {segment.kind === "original" ? " · 原声" : ""}
                  </div>
                  <p className="leading-relaxed">{segment.text}</p>
                </li>
              ))}
            </ol>
          </>
        ) : null}
      </QueryState>
    </section>
  );
}
