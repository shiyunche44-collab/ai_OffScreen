import { useMemo, useState } from "react";
import type { TranscriptLine } from "../../api/types";
import { formatDuration } from "../../lib/format";

const SHOWN = 100;

/** Pick transcript lines to play as original sound after a segment. */
export function OriginalPicker({
  after,
  lines,
  busy,
  onInsert,
  onClose,
}: {
  after: string | null;
  lines: readonly TranscriptLine[] | undefined;
  busy: boolean;
  onInsert: (lineIds: string[]) => void;
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const [picked, setPicked] = useState<string[]>([]);
  const shown = useMemo(() => {
    const q = query.trim();
    return (lines ?? []).filter((l) => !q || l.text.includes(q));
  }, [lines, query]);
  const toggle = (id: string) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]));
  // Lines play in film order, whatever order they were ticked in.
  const ordered = (lines ?? []).filter((l) => picked.includes(l.id)).map((l) => l.id);

  return (
    <aside aria-label="插入原声" className="space-y-2 rounded border border-sky-300 p-3 text-sm dark:border-sky-800">
      <div className="flex items-start gap-2">
        <p className="min-w-0 flex-1 text-xs text-slate-500">在 {after ?? "开头"} 之后插入原声段：勾选要播放的台词</p>
        <button type="button" onClick={onClose} aria-label="关闭插入原声" className="text-slate-500 hover:text-slate-900">
          ×
        </button>
      </div>
      {lines === undefined ? <p className="text-xs text-slate-500">加载台词…</p> : null}
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="搜索台词"
        aria-label="搜索台词"
        className="w-full rounded border border-slate-300 bg-transparent px-2 py-1 text-xs dark:border-slate-700"
      />
      <ul aria-label="台词" className="max-h-64 space-y-1 overflow-y-auto text-xs">
        {shown.slice(0, SHOWN).map((l) => (
          <li key={l.id}>
            <label className="flex cursor-pointer items-start gap-2">
              <input type="checkbox" checked={picked.includes(l.id)} onChange={() => toggle(l.id)} className="mt-0.5" />
              <span className="shrink-0 tabular-nums text-slate-500">{formatDuration(l.start_ms)}</span>
              <span>{l.text}</span>
            </label>
          </li>
        ))}
      </ul>
      {shown.length > SHOWN ? <p className="text-xs text-slate-500">只显示前 {SHOWN} 条，搜索可缩小范围。</p> : null}
      <div className="flex items-center justify-end gap-2">
        <span className="text-xs text-slate-500">已选 {picked.length} 条</span>
        <button
          type="button"
          disabled={ordered.length === 0 || busy}
          onClick={() => onInsert(ordered)}
          className="rounded bg-sky-600 px-3 py-1 text-sm text-white disabled:opacity-50"
        >
          插入
        </button>
      </div>
    </aside>
  );
}
