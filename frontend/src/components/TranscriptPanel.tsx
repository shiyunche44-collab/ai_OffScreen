import { useEffect, useRef } from "react";
import type { TranscriptLine } from "../api/types";
import { formatDuration } from "../lib/format";
import { indexAt } from "../lib/timeline";

/** The dialogue, the current line highlighted while the film plays; a line seeks the player. */
export function TranscriptPanel({
  lines,
  currentMs,
  onSeek,
}: {
  lines: TranscriptLine[];
  currentMs: number;
  onSeek: (ms: number) => void;
}) {
  const current = indexAt(lines, currentMs);
  const active = useRef<HTMLLIElement>(null);
  useEffect(() => {
    active.current?.scrollIntoView?.({ block: "nearest" });
  }, [current]);

  if (lines.length === 0) return <p className="text-sm text-slate-500">没有台词。</p>;
  return (
    <ol aria-label="台词" className="max-h-96 overflow-y-auto rounded border border-slate-200 dark:border-slate-800">
      {lines.map((line, i) => (
        <li key={line.id} ref={i === current ? active : undefined} aria-current={i === current ? "true" : undefined}>
          <button
            type="button"
            onClick={() => onSeek(line.start_ms)}
            className={`flex w-full gap-3 px-3 py-1.5 text-left text-sm hover:bg-slate-100 dark:hover:bg-slate-900 ${
              i === current ? "bg-sky-100 dark:bg-sky-950" : ""
            }`}
          >
            <span className="w-12 shrink-0 tabular-nums text-xs leading-5 text-slate-500">
              {formatDuration(line.start_ms)}
            </span>
            <span>{line.text}</span>
          </button>
        </li>
      ))}
    </ol>
  );
}
