import type { Scene } from "../api/types";
import { formatDuration } from "../lib/format";
import { indexAt } from "../lib/timeline";

export function SceneList({
  scenes,
  currentMs,
  onSeek,
}: {
  scenes: Scene[];
  currentMs: number;
  onSeek: (ms: number) => void;
}) {
  const current = indexAt(scenes, currentMs);
  return (
    <ol aria-label="场景" className="space-y-2">
      {scenes.map((scene, i) => (
        <li key={scene.id} aria-current={i === current ? "true" : undefined}>
          <button
            type="button"
            onClick={() => onSeek(scene.start_ms)}
            className={`w-full rounded border px-3 py-2 text-left text-sm hover:border-sky-400 ${
              i === current
                ? "border-sky-500 bg-sky-50 dark:bg-sky-950"
                : "border-slate-200 dark:border-slate-800"
            }`}
          >
            <div className="mb-0.5 flex flex-wrap gap-x-3 text-xs text-slate-500">
              <span className="tabular-nums">
                {formatDuration(scene.start_ms)}–{formatDuration(scene.end_ms)}
              </span>
              <span>{scene.id}</span>
              {scene.location ? <span>{scene.location}</span> : null}
              <span>重要度 {Math.round((scene.importance ?? 0.5) * 100)}%</span>
            </div>
            <p className="leading-relaxed">{scene.summary}</p>
          </button>
        </li>
      ))}
    </ol>
  );
}
